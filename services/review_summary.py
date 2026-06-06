from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF


def _safe_page_no(raw: Any) -> Optional[int]:
    try:
        val = int(raw)
    except Exception:
        return None
    return val if val >= 1 else None


def _infer_page_no_from_bbox(issue: Dict[str, Any], pdf_path: Path) -> Optional[int]:
    bbox = issue.get("bbox")
    if not bbox and isinstance(issue.get("bbox_list"), list) and issue["bbox_list"]:
        bbox = issue["bbox_list"][0]
    if not bbox:
        anchor = issue.get("anchor", {})
        if isinstance(anchor, dict):
            bbox = anchor.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None

    rect = fitz.Rect(x0, y0, x1, y1)
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page_idx in range(doc.page_count):
                page_rect = doc[page_idx].rect
                if (
                    rect.x0 >= page_rect.x0
                    and rect.y0 >= page_rect.y0
                    and rect.x1 <= page_rect.x1
                    and rect.y1 <= page_rect.y1
                ):
                    return page_idx + 1
    except Exception:
        return None
    return None


def _resolve_page_no(issue: Dict[str, Any], pdf_path: Optional[Path]) -> Optional[int]:
    direct = _safe_page_no(issue.get("page_no"))
    if direct:
        return direct
    anchor = issue.get("anchor", {})
    if isinstance(anchor, dict):
        anchor_page = _safe_page_no(anchor.get("page_no"))
        if anchor_page:
            return anchor_page
    if pdf_path is not None and pdf_path.exists():
        by_bbox = _infer_page_no_from_bbox(issue, pdf_path)
        if by_bbox:
            return by_bbox
        original = str(issue.get("original_text", "")).strip()
        if original:
            try:
                with fitz.open(str(pdf_path)) as doc:
                    for page_idx in range(doc.page_count):
                        if doc[page_idx].search_for(original):
                            return page_idx + 1
            except Exception:
                return None
    return None


def _severity_tag_cn(severity: str) -> str:
    val = (severity or "").lower()
    if val == "high":
        return "高"
    if val == "medium":
        return "中"
    return "低"


def _source_label(issue: Dict[str, Any]) -> tuple[str, str, str]:
    origin = str(issue.get("origin") or "").strip()
    anchor = issue.get("anchor", {})
    if str(issue.get("evidence_source") or "") == "word_main":
        return "Word 主审查", "#5a2ca0", "word_main"
    if origin == "word_helper" or str(issue.get("evidence_source") or "") == "word_helper":
        return "Word 辅助审查", "#1f8a3b", "word"
    if origin == "visual" or (isinstance(anchor, dict) and anchor.get("visual_category")):
        return "Qwen 视觉审查", "#c45100", "visual"
    return "DeepSeek 文本审查", "#1d5fd1", "text"


def _is_word_main_payload(issue: Dict[str, Any], review_payload: Dict[str, Any]) -> bool:
    meta = review_payload.get("meta", {})
    stats = review_payload.get("stats", {})
    return bool(
        (isinstance(meta, dict) and meta.get("word_main_review"))
        or (isinstance(stats, dict) and stats.get("word_main_review"))
    )


def _word_location_label(issue: Dict[str, Any], fallback_para_label: str) -> str:
    anchor = issue.get("anchor", {})
    if isinstance(anchor, dict):
        location = str(anchor.get("word_location") or "").strip()
        if location:
            return f"Word位置 · {location}"
    return f"Word位置 · {fallback_para_label}"


def _collect_summary_issues(review_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for key in ("issues", "visual_structure_issues", "word_helper_issues"):
        raw = review_payload.get(key) or []
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                copied = dict(item)
                copied.setdefault("_summary_source_key", key)
                collected.append(copied)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in collected:
        key = str(item.get("issue_id") or item.get("issue_no") or id(item))
        source = _source_label(item)[2]
        dedup_key = f"{source}|{key}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped.append(item)
    deduped.sort(key=lambda x: (_source_label(x)[2], str(x.get("issue_no") or ""), str(x.get("issue_id") or "")))
    return deduped


def build_review_summary_markdown(
    *,
    review_payload: Dict[str, Any],
    pdf_path: Optional[Path],
    title: str = "问题摘要报告",
) -> str:
    issues = _collect_summary_issues(review_payload)

    lines: List[str] = [f"# {title}", ""]
    if not issues:
        lines.extend(["未发现问题。", ""])
        return "\n".join(lines).rstrip() + "\n"

    current_group = ""
    for idx, issue in enumerate(issues, start=1):
        if not isinstance(issue, dict):
            continue
        source_label, color, group_key = _source_label(issue)
        if group_key != current_group:
            current_group = group_key
            lines.append(f"## <span style=\"color:{color}\">{source_label}</span>")
            lines.append("")
        issue_no = str(issue.get("issue_no") or f"Q{idx}")
        severity = _severity_tag_cn(str(issue.get("severity", "medium")))
        category = str(issue.get("category", "unknown"))
        page_no = _resolve_page_no(issue, pdf_path)
        para_label = f"段落 {str(issue.get('para_id') or '').strip() or issue_no}"
        if page_no:
            page_label = f"第{page_no}页"
        elif group_key == "word_main" or _is_word_main_payload(issue, review_payload):
            page_label = _word_location_label(issue, para_label)
        else:
            page_label = "页码未知"
        reason = str(issue.get("reason", "")).strip()
        suggestion = str(issue.get("suggestion", "")).strip()
        original = str(issue.get("original_text", "")).strip()

        lines.append(
            f"### <span style=\"color:{color}\">{severity} 问题 {issue_no}</span> · {page_label} · {category}"
        )
        if original:
            lines.append("")
            lines.append("原文：")
            lines.append("")
            lines.append(f"> {original}")
        if reason:
            lines.append("")
            lines.append(f"问题：{reason}")
        if suggestion:
            lines.append("")
            lines.append(f"建议：{suggestion}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_review_summary_markdown(
    *,
    review_json_path: Path,
    output_path: Path,
    pdf_path: Optional[Path] = None,
    title: str = "问题摘要报告",
) -> Path:
    payload = json.loads(review_json_path.read_text(encoding="utf-8"))
    content = build_review_summary_markdown(review_payload=payload, pdf_path=pdf_path, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
