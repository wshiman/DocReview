from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    out: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"):
            out.append(ch)
    return "".join(out)


def _build_comment(issue: Dict[str, Any]) -> str:
    issue_no = str(issue.get("issue_no") or issue.get("issue_id") or "Q")
    category = str(issue.get("category", "")).strip()
    original_text = str(issue.get("original_text", "")).strip()
    modified_text = str(issue.get("modified_text", "")).strip()
    review_round = str(issue.get("review_round", "")).strip()
    reason = str(issue.get("reason", "")).strip()
    suggestion = str(issue.get("suggestion", "")).strip()
    return (
        f"[Issue] {issue_no}\n"
        f"[Round] {review_round}\n"
        f"[Category] {category}\n"
        f"[Original Text] {original_text}\n"
        f"[Modified Text] {modified_text}\n"
        f"[Reason] {reason}\n"
        f"[Suggestion] {suggestion}"
    )


def _is_visual_issue(issue: Dict[str, Any]) -> bool:
    anchor = issue.get("anchor", {})
    return (
        issue.get("review_round") == 0
        or str(issue.get("category", "")).strip() == "layout"
        or (isinstance(anchor, dict) and bool(anchor.get("visual_category")))
    )


def _is_page_level_issue(issue: Dict[str, Any]) -> bool:
    anchor = issue.get("anchor", {})
    return isinstance(anchor, dict) and bool(anchor.get("page_level_annotation"))


@dataclass
class LocateResult:
    page_index: int
    rect: fitz.Rect
    total_hits_on_page: int


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _score_page_for_issue(page: fitz.Page, issue: Dict[str, Any], page_index: int) -> Tuple[float, float]:
    # IMPORTANT:
    # page_no from OCR-derived paragraphs can drift after postprocess/re-chunking.
    # So we prioritize text evidence, and use page_no only as a weak prior.
    evidence = 0.0
    page_prior = 0.0
    page_text = page.get_text("text")
    n_page = _normalize_text(page_text)

    target_page = issue.get("page_no")
    if isinstance(target_page, int) and target_page >= 1 and target_page == page_index + 1:
        page_prior += 40.0

    snippet = str(issue.get("snippet", "")).strip()
    if snippet:
        n_snip = _normalize_text(snippet)
        if n_snip and n_snip in n_page:
            evidence += 260.0
        elif n_snip:
            evidence += SequenceMatcher(None, n_snip[:500], n_page[:3500]).ratio() * 120.0

    original = str(issue.get("original_text", "")).strip()
    if original:
        if original in page_text:
            evidence += 130.0
        elif _normalize_text(original) in _normalize_text(page_text):
            evidence += 35.0

    # Optional strong evidence: the exact paragraph sent to DeepSeek for this issue.
    review_input_text = str(issue.get("review_input_text", "")).strip()
    if review_input_text:
        n_review_input = _normalize_text(review_input_text)
        if n_review_input and n_review_input in n_page:
            evidence += 320.0
        elif n_review_input:
            evidence += SequenceMatcher(None, n_review_input[:800], n_page[:5000]).ratio() * 160.0
    return evidence, page_prior


def _choose_best_rect(page: fitz.Page, rects: List[fitz.Rect], issue: Dict[str, Any]) -> fitz.Rect:
    if len(rects) == 1:
        return rects[0]

    snippet = str(issue.get("snippet", "")).strip()
    original = str(issue.get("original_text", "")).strip()
    if not snippet or not original:
        return _choose_rect_by_reading_flow(rects)

    pos = snippet.find(original)
    prefix = snippet[max(0, pos - 30) : pos].strip() if pos >= 0 else ""
    suffix = snippet[pos + len(original) : pos + len(original) + 30].strip() if pos >= 0 else ""

    anchor: Optional[Tuple[float, float]] = None
    # Prefer exact prefix/suffix anchors.
    if prefix:
        pfx_rects = page.search_for(prefix)
        if pfx_rects:
            r = pfx_rects[-1]
            anchor = (r.x1, (r.y0 + r.y1) / 2.0)
    if anchor is None and suffix:
        sfx_rects = page.search_for(suffix)
        if sfx_rects:
            r = sfx_rects[0]
            anchor = (r.x0, (r.y0 + r.y1) / 2.0)
    if anchor is None:
        return _choose_rect_by_reading_flow(rects)

    ax, ay = anchor
    # Minimize weighted 2D distance to anchor to avoid same-line wrong match.
    return min(rects, key=lambda r: abs((r.x0 + r.x1) / 2.0 - ax) * 0.35 + abs((r.y0 + r.y1) / 2.0 - ay))


def _choose_rect_by_reading_flow(rects: List[fitz.Rect]) -> fitz.Rect:
    # Stable fallback: pick by normal reading order (top-to-bottom then left-to-right).
    return sorted(rects, key=lambda r: (round(r.y0, 2), round(r.x0, 2)))[0]


def _build_search_candidates(original: str) -> List[str]:
    base = (original or "").strip()
    if not base:
        return []
    cands = [base]
    nows = _normalize_text(base)
    if nows and nows != base:
        cands.append(nows)
    return cands


def _issue_bbox_rect(page: fitz.Page, issue: Dict[str, Any]) -> Optional[fitz.Rect]:
    bbox = issue.get("bbox")
    if not bbox:
        anchor = issue.get("anchor", {})
        if isinstance(anchor, dict):
            bbox = anchor.get("bbox")
    if not bbox and isinstance(issue.get("bbox_list"), list) and issue["bbox_list"]:
        bbox = issue["bbox_list"][0]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        rect = fitz.Rect([float(x) for x in bbox])
    except Exception:
        return None
    page_rect = page.rect
    rect.x0 = max(page_rect.x0, min(page_rect.x1, rect.x0))
    rect.x1 = max(page_rect.x0, min(page_rect.x1, rect.x1))
    rect.y0 = max(page_rect.y0, min(page_rect.y1, rect.y0))
    rect.y1 = max(page_rect.y0, min(page_rect.y1, rect.y1))
    if rect.width <= 0 or rect.height <= 0:
        return None
    return rect


def _page_normalized_char_index(page: fitz.Page) -> Tuple[str, List[fitz.Rect]]:
    raw = page.get_text("rawdict")
    norm_chars: List[str] = []
    boxes: List[fitz.Rect] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars", [])
                if chars:
                    for ch_info in chars:
                        c = str(ch_info.get("c", ""))
                        nc = _normalize_for_match(c)
                        if not nc:
                            continue
                        try:
                            box = fitz.Rect(ch_info.get("bbox"))
                        except Exception:
                            continue
                        for nchar in nc:
                            norm_chars.append(nchar)
                            boxes.append(box)
                    continue
                # Fallback when char-level info is unavailable.
                text = str(span.get("text", ""))
                ntext = _normalize_for_match(text)
                if not ntext:
                    continue
                try:
                    sbox = fitz.Rect(span.get("bbox"))
                except Exception:
                    continue
                for nchar in ntext:
                    norm_chars.append(nchar)
                    boxes.append(sbox)
    return "".join(norm_chars), boxes


def _find_rects_by_normalized_match(page: fitz.Page, query: str) -> List[fitz.Rect]:
    n_query = _normalize_for_match(query)
    if len(n_query) < 2:
        return []
    n_page, boxes = _page_normalized_char_index(page)
    if not n_page or not boxes or n_query not in n_page:
        return []
    rects: List[fitz.Rect] = []
    start = 0
    while True:
        idx = n_page.find(n_query, start)
        if idx < 0:
            break
        end = idx + len(n_query)
        span_boxes = boxes[idx:end]
        if span_boxes:
            x0 = min(b.x0 for b in span_boxes)
            y0 = min(b.y0 for b in span_boxes)
            x1 = max(b.x1 for b in span_boxes)
            y1 = max(b.y1 for b in span_boxes)
            rects.append(fitz.Rect(x0, y0, x1, y1))
        start = idx + 1
    return rects


def _locate_issue(doc: fitz.Document, issue: Dict[str, Any]) -> Optional[LocateResult]:
    target_page = issue.get("page_no")
    if isinstance(target_page, int) and target_page >= 1 and target_page <= doc.page_count:
        page = doc[target_page - 1]
        if _is_page_level_issue(issue):
            rect = fitz.Rect(page.rect.x0 + 36, page.rect.y0 + 36, page.rect.x0 + 180, page.rect.y0 + 72)
            return LocateResult(page_index=target_page - 1, rect=rect, total_hits_on_page=1)
        bbox_rect = _issue_bbox_rect(page, issue)
        if bbox_rect is not None:
            return LocateResult(page_index=target_page - 1, rect=bbox_rect, total_hits_on_page=1)

    original = str(issue.get("original_text", "")).strip()
    if not original:
        return None

    search_cands = _build_search_candidates(original)
    if not search_cands:
        return None

    candidates: List[Tuple[float, float, int, List[fitz.Rect]]] = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        rects: List[fitz.Rect] = []
        for cand in search_cands:
            rects = page.search_for(cand)
            if rects:
                break
        if not rects:
            # Fallback for OCR spacing / punctuation variance between review text and source PDF.
            rects = _find_rects_by_normalized_match(page, original)
        if not rects:
            continue
        evidence, page_prior = _score_page_for_issue(page, issue, page_index)
        # Penalize short ambiguous strings that appear many times.
        if len(_normalize_text(original)) <= 6 and len(rects) > 1:
            evidence -= min(45.0, (len(rects) - 1) * 12.0)
        candidates.append((evidence, page_prior, page_index, rects))

    if not candidates:
        return None

    # Rank by textual evidence first, then weak page prior.
    candidates.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    _, _, best_page_index, best_rects = candidates[0]
    page = doc[best_page_index]
    rect = _choose_best_rect(page, best_rects, issue)
    return LocateResult(page_index=best_page_index, rect=rect, total_hits_on_page=len(best_rects))


def _choose_note_point(
    page_rect: fitz.Rect,
    highlight_rect: fitz.Rect,
    occupied_notes: List[fitz.Rect],
) -> Tuple[fitz.Point, fitz.Rect, str]:
    icon_size = 18.0
    pad = 6.0
    edge_pad = 8.0
    text_gap = 3.0
    page_width = float(page_rect.width)
    page_height = float(page_rect.height)

    left_space = max(0.0, highlight_rect.x0 - page_rect.x0)
    right_space = max(0.0, page_rect.x1 - highlight_rect.x1)
    if right_space >= icon_size + pad:
        side = "right"
        x = min(highlight_rect.x1 + pad, page_rect.x1 - icon_size - edge_pad)
    elif left_space >= icon_size + pad:
        side = "left"
        x = max(page_rect.x0 + edge_pad, highlight_rect.x0 - icon_size - pad)
    elif right_space >= left_space:
        side = "right-edge"
        x = page_rect.x1 - icon_size - edge_pad
    else:
        side = "left-edge"
        x = page_rect.x0 + edge_pad

    # Anchor to the highlighted line. Only nudge when this icon would overlap
    # another icon on the same page; do not accumulate page-level slot offsets.
    base_y = _clamp(
        (highlight_rect.y0 + highlight_rect.y1) / 2.0 - icon_size / 2.0,
        page_rect.y0 + edge_pad,
        page_rect.y1 - icon_size - edge_pad,
    )

    candidate_offsets = [0.0]
    for step in range(1, 12):
        delta = step * (icon_size + 2.0)
        candidate_offsets.extend([delta, -delta])

    best_y = base_y
    for offset in candidate_offsets:
        y = _clamp(base_y + offset, page_rect.y0 + edge_pad, page_rect.y1 - icon_size - edge_pad)
        rect = fitz.Rect(x, y, x + icon_size, y + icon_size)
        if not any(rect.intersects(existing) for existing in occupied_notes):
            occupied_notes.append(rect)
            return fitz.Point(x, y), rect, side
        best_y = y

    # Last-resort compact stacking near the original line.
    rect = fitz.Rect(x, best_y, x + icon_size, best_y + icon_size)
    occupied_notes.append(rect)
    return fitz.Point(x, best_y), rect, side


def _choose_visual_note_point(
    page_rect: fitz.Rect,
    anchor_rect: fitz.Rect,
    occupied_notes: List[fitz.Rect],
) -> Tuple[fitz.Point, fitz.Rect, str]:
    icon_size = 18.0
    pad = 8.0
    edge_pad = 8.0
    if anchor_rect.x0 - page_rect.x0 >= icon_size + pad:
        side = "left"
        x = max(page_rect.x0 + edge_pad, anchor_rect.x0 - icon_size - pad)
    else:
        side = "left-edge"
        x = page_rect.x0 + edge_pad

    base_y = _clamp(
        (anchor_rect.y0 + anchor_rect.y1) / 2.0 - icon_size / 2.0,
        page_rect.y0 + edge_pad,
        page_rect.y1 - icon_size - edge_pad,
    )
    candidate_offsets = [0.0]
    for step in range(1, 16):
        delta = step * (icon_size + 2.0)
        candidate_offsets.extend([delta, -delta])
    best_y = base_y
    for offset in candidate_offsets:
        y = _clamp(base_y + offset, page_rect.y0 + edge_pad, page_rect.y1 - icon_size - edge_pad)
        rect = fitz.Rect(x, y, x + icon_size, y + icon_size)
        if not any(rect.intersects(existing) for existing in occupied_notes):
            occupied_notes.append(rect)
            return fitz.Point(x, y), rect, side
        best_y = y
    rect = fitz.Rect(x, best_y, x + icon_size, best_y + icon_size)
    occupied_notes.append(rect)
    return fitz.Point(x, best_y), rect, side


def annotate_pdf_from_review_json(
    input_pdf: str | Path,
    review_json_path: str | Path,
    output_pdf: str | Path,
) -> Dict[str, Any]:
    input_pdf = Path(input_pdf).resolve()
    review_json_path = Path(review_json_path).resolve()
    output_pdf = Path(output_pdf).resolve()

    payload = json.loads(review_json_path.read_text(encoding="utf-8"))
    issues = []
    for key in ("issues", "visual_structure_issues", "word_helper_issues"):
        raw = payload.get(key, [])
        if isinstance(raw, list):
            issues.extend(item for item in raw if isinstance(item, dict))
    return annotate_pdf_from_issues(input_pdf=input_pdf, issues=issues, output_pdf=output_pdf)


def annotate_pdf_from_issues(
    input_pdf: str | Path,
    issues: List[Dict[str, Any]],
    output_pdf: str | Path,
) -> Dict[str, Any]:
    if not isinstance(issues, list):
        raise ValueError("Invalid issues payload: `issues` must be a list.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if output_pdf.exists():
        output_pdf.unlink()
    shutil.copy2(str(input_pdf), str(output_pdf))

    doc = fitz.open(str(output_pdf))
    applied = 0
    skipped: List[Dict[str, str]] = []
    placements: List[Dict[str, Any]] = []
    page_note_rects: Dict[int, List[fitz.Rect]] = {}

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        issue_no = str(issue.get("issue_no") or issue.get("issue_id") or f"Q{applied+1}")
        located = _locate_issue(doc, issue)
        if located is None:
            skipped.append({"issue_no": issue_no, "reason": "original_text_not_found"})
            continue

        page = doc[located.page_index]
        comment = _build_comment(issue)
        rect = located.rect
        is_visual = _is_visual_issue(issue)
        is_page_level = _is_page_level_issue(issue)

        page_rect = page.rect
        if is_visual or is_page_level:
            note_point, note_rect, note_side = _choose_visual_note_point(
                page_rect,
                rect,
                page_note_rects.setdefault(located.page_index, []),
            )
        else:
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1.0, 0.86, 0.10))
            highlight.set_opacity(0.38)
            highlight.set_info(
                title=f"DocReview {issue_no}",
                # Keep highlight purely visual to avoid duplicate detail panes.
                content="",
                subject=str(issue.get("category", "")),
            )
            highlight.update()
            note_point, note_rect, note_side = _choose_note_point(
                page_rect,
                rect,
                page_note_rects.setdefault(located.page_index, []),
            )
        text_note = page.add_text_annot(note_point, comment)
        text_note.set_info(
            title=f"{'WordHelper' if is_page_level else 'VisualReview' if is_visual else 'DocReview'} {issue_no}",
            # Keep full review details in side note only.
            content=comment,
            subject=str(issue.get("category", "")),
        )
        # Use neutral comment icon color to avoid aggressive red square impression.
        text_note.set_colors(
            stroke=(0.20, 0.60, 0.25)
            if is_page_level
            else (0.95, 0.45, 0.10)
            if is_visual
            else (0.12, 0.45, 0.85)
        )
        # Prefer comment/note icon instead of square marker.
        try:
            text_note.set_name("Comment" if is_page_level else "Key" if is_visual else "Comment")
        except Exception:
            pass
        text_note.update()

        applied += 1
        placements.append(
            {
                "issue_no": issue_no,
                "page_no": located.page_index + 1,
                "hit_count_on_page": located.total_hits_on_page,
                "rect": [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)],
                "note_rect": [round(note_rect.x0, 2), round(note_rect.y0, 2), round(note_rect.x1, 2), round(note_rect.y1, 2)],
                "note_side": note_side,
                "original_text": str(issue.get("original_text", "")),
                "locate_strategy": (
                    "page_level_note"
                    if is_page_level
                    else
                    "visual_bbox_note_only"
                    if is_visual and _issue_bbox_rect(page, issue) is not None
                    else "bbox"
                    if _issue_bbox_rect(page, issue) is not None
                    else "text"
                ),
            }
        )

    doc.saveIncr()
    doc.close()

    return {
        "input_pdf": str(input_pdf),
        "review_json": None,
        "output_pdf": str(output_pdf),
        "total_issues": len(issues),
        "applied_issues": applied,
        "skipped_issues": skipped,
        "placements": placements,
    }
