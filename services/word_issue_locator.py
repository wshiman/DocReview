from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

try:
    from ..models import IssueItem, ParagraphUnit
except ImportError:  # pragma: no cover
    from models import IssueItem, ParagraphUnit


def _int_or_none(value: object) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _style_label(anchor: Dict[str, object]) -> str:
    meta = anchor.get("docx_meta")
    style = ""
    if isinstance(meta, dict):
        style = str(meta.get("style_name") or "").strip()
    style = style or str(anchor.get("style_name") or "").strip()
    return f" · 样式 {style}" if style else ""


def _estimate_line_no(issue: IssueItem, paragraph: ParagraphUnit) -> Optional[int]:
    original = (issue.original_text or "").strip()
    haystack = paragraph.anchor.get("raw_text") or paragraph.text
    text = str(haystack or "")
    if not original or not text:
        return None
    pos = text.find(original)
    if pos < 0:
        compact_text = "".join(text.split())
        compact_original = "".join(original.split())
        pos = compact_text.find(compact_original) if compact_original else -1
        if pos < 0:
            return None
        # Chinese Word documents often wrap around 28-40 chars. This is only a locator hint.
        return max(1, math.floor(pos / 36) + 1)
    return max(1, text[:pos].count("\n") + math.floor(len(text[:pos].replace("\n", "")) / 36) + 1)


def _location_label(paragraph: ParagraphUnit, issue: IssueItem) -> str:
    anchor = dict(paragraph.anchor or {})
    kind = str(anchor.get("kind") or "paragraph")
    para_no = _int_or_none(str(paragraph.para_id).rsplit("-", 1)[-1])
    base = f"段落 {para_no}" if para_no else f"段落 {paragraph.para_id}"

    if "table_cell" in kind:
        table_no = (_int_or_none(anchor.get("table_index")) or 0) + 1
        row_no = (_int_or_none(anchor.get("row_index")) or 0) + 1
        col_no = (_int_or_none(anchor.get("col_index")) or 0) + 1
        cell_para_no = (_int_or_none(anchor.get("paragraph_index")) or 0) + 1
        base = f"表格 {table_no} · 第 {row_no} 行第 {col_no} 列 · 单元格段落 {cell_para_no} · {base}"
    else:
        raw_index = _int_or_none(anchor.get("paragraph_index"))
        xml_index = _int_or_none(anchor.get("paragraph_xml_index"))
        docx_no = (raw_index if raw_index is not None else xml_index)
        if docx_no is not None:
            base = f"正文第 {docx_no + 1} 段 · {base}"

    line_no = _estimate_line_no(issue, paragraph)
    if line_no:
        base += f" · 估算段内第 {line_no} 行"
    base += _style_label(anchor)
    return base


def attach_word_issue_locations(issues: Iterable[IssueItem], paragraphs: List[ParagraphUnit]) -> None:
    paragraphs_by_id = {p.para_id: p for p in paragraphs}
    for issue in issues:
        paragraph = paragraphs_by_id.get(issue.para_id)
        if not paragraph or paragraph.source_type != "docx":
            continue
        issue.anchor = dict(issue.anchor or {})
        issue.anchor["word_location"] = _location_label(paragraph, issue)
        issue.anchor["word_location_source"] = "docx_structure"
        issue.anchor["paragraph_text"] = paragraph.text[:500]
        issue.anchor["paragraph_anchor"] = dict(paragraph.anchor or {})
