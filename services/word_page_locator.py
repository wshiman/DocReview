from __future__ import annotations

import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

try:
    from ..models import IssueItem, ParseResult, ParagraphUnit
    from .office import find_soffice
except ImportError:  # pragma: no cover
    from models import IssueItem, ParseResult, ParagraphUnit
    from services.office import find_soffice


@dataclass
class PageMatch:
    page_no: int
    confidence: str
    score: float
    margin: float
    reason: str


def _normalize_for_match(text: str) -> str:
    out: List[str] = []
    for ch in (text or "").lower():
        if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"):
            out.append(ch)
    return "".join(out)


def _context_around(text: str, needle: str, radius: int = 80) -> str:
    if not text or not needle:
        return ""
    pos = text.find(needle)
    if pos < 0:
        return ""
    start = max(0, pos - radius)
    end = min(len(text), pos + len(needle) + radius)
    return text[start:end]


def _candidate_texts(issue: IssueItem, paragraph: Optional[ParagraphUnit]) -> List[Tuple[str, float, str]]:
    candidates: List[Tuple[str, float, str]] = []

    original = (issue.original_text or "").strip()
    review_input = (issue.review_input_text or "").strip()
    snippet = (issue.snippet or "").strip()
    para_text = (paragraph.text if paragraph else "").strip()

    for label, text, weight in (
        ("review_input_context", _context_around(review_input, original), 360.0),
        ("snippet_context", _context_around(snippet, original), 320.0),
        ("review_input_text", review_input, 260.0),
        ("paragraph_text", para_text, 250.0),
        ("snippet", snippet, 220.0),
        ("original_text", original, 160.0),
    ):
        norm = _normalize_for_match(text)
        if len(norm) < 4:
            continue
        # Very long spans often cross page boundaries; keep a centered window.
        if len(norm) > 900:
            norm = norm[:900]
        candidates.append((norm, weight, label))

    dedup: List[Tuple[str, float, str]] = []
    seen = set()
    for norm, weight, label in candidates:
        if norm in seen:
            continue
        seen.add(norm)
        dedup.append((norm, weight, label))
    return dedup


def _score_pages(
    *,
    issue: IssueItem,
    paragraph: Optional[ParagraphUnit],
    page_texts: List[str],
) -> List[Tuple[float, int, List[str]]]:
    candidates = _candidate_texts(issue, paragraph)
    scores: List[Tuple[float, int, List[str]]] = []
    for page_index, page_text in enumerate(page_texts):
        score = 0.0
        reasons: List[str] = []
        for query, weight, label in candidates:
            if query in page_text:
                adjusted = weight
                if len(query) <= 8:
                    adjusted *= 0.55
                score += adjusted
                reasons.append(label)
                continue
            if len(query) >= 80:
                ratio = SequenceMatcher(None, query[:600], page_text[:5000]).ratio()
                if ratio >= 0.72:
                    score += weight * ratio * 0.5
                    reasons.append(f"{label}_fuzzy")
        if score > 0:
            scores.append((score, page_index, reasons))
    scores.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scores


def _locate_issue_page(
    *,
    issue: IssueItem,
    paragraph: Optional[ParagraphUnit],
    page_texts: List[str],
) -> Optional[PageMatch]:
    scores = _score_pages(issue=issue, paragraph=paragraph, page_texts=page_texts)
    if not scores:
        return None

    best_score, best_page_index, reasons = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    margin = best_score - second_score

    if best_score >= 320.0 and (margin >= 120.0 or len(scores) == 1):
        confidence = "high"
    elif best_score >= 160.0 and (margin >= 80.0 or len(scores) == 1):
        confidence = "medium"
    else:
        return None

    return PageMatch(
        page_no=best_page_index + 1,
        confidence=confidence,
        score=round(best_score, 2),
        margin=round(margin, 2),
        reason=",".join(reasons[:4]),
    )


def _convert_docx_to_pdf(docx_path: Path, output_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    office_cmd = find_soffice()
    if not office_cmd:
        return None, "office_converter_not_found"

    out_dir = output_dir / "word_page_locator" / f"{docx_path.stem}_{docx_path.stat().st_mtime_ns}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        office_cmd,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(docx_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "convert_failed"
        return None, msg

    pdf_path = out_dir / f"{docx_path.stem}.pdf"
    if not pdf_path.exists():
        return None, "converted_pdf_missing"
    return pdf_path.resolve(), None


def locate_word_issue_pages(
    *,
    parse_result: ParseResult,
    issues: List[IssueItem],
    output_dir: str | Path,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "enabled": parse_result.source_type == "docx",
        "converted_pdf": None,
        "error": None,
        "located_count": 0,
        "unlocated_count": 0,
        "confidence_distribution": {"high": 0, "medium": 0, "unknown": 0},
    }
    if parse_result.source_type != "docx" or not issues:
        return meta

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_path, error = _convert_docx_to_pdf(parse_result.normalized_path, output_root)
    if pdf_path is None:
        meta["error"] = error or "convert_failed"
        meta["unlocated_count"] = len(issues)
        meta["confidence_distribution"] = {"high": 0, "medium": 0, "unknown": len(issues)}
        for issue in issues:
            issue.anchor = dict(issue.anchor or {})
            issue.anchor["page_no_confidence"] = "unknown"
            issue.anchor["page_no_source"] = "word_rendered_pdf"
            issue.anchor["page_no_reason"] = str(meta["error"])
        return meta

    meta["converted_pdf"] = str(pdf_path)
    paragraphs_by_id = {p.para_id: p for p in parse_result.paragraphs}
    with fitz.open(str(pdf_path)) as doc:
        page_texts = [_normalize_for_match(page.get_text("text")) for page in doc]

    located = 0
    unknown = 0
    confidence_dist = {"high": 0, "medium": 0, "unknown": 0}
    for issue in issues:
        paragraph = paragraphs_by_id.get(issue.para_id)
        match = _locate_issue_page(issue=issue, paragraph=paragraph, page_texts=page_texts)
        issue.anchor = dict(issue.anchor or {})
        issue.anchor["page_no_source"] = "word_rendered_pdf"
        if match is None:
            unknown += 1
            confidence_dist["unknown"] += 1
            issue.anchor["page_no_confidence"] = "unknown"
            issue.anchor["page_no_reason"] = "no_unique_text_match"
            continue
        located += 1
        confidence_dist[match.confidence] += 1
        issue.page_no = match.page_no
        issue.anchor["page_no"] = match.page_no
        issue.anchor["page_no_confidence"] = match.confidence
        issue.anchor["page_no_score"] = match.score
        issue.anchor["page_no_margin"] = match.margin
        issue.anchor["page_no_reason"] = match.reason

    meta["located_count"] = located
    meta["unlocated_count"] = unknown
    meta["confidence_distribution"] = confidence_dist
    return meta
