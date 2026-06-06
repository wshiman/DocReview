from __future__ import annotations

import re
from typing import Dict, List, Tuple

try:
    from ..models import IssueItem, ParagraphUnit, ReviewConfig
except ImportError:  # pragma: no cover
    from models import IssueItem, ParagraphUnit, ReviewConfig

WORD_HELPER_CATEGORIES = {
    "layout_spacing",
    "line_break",
    "paragraph_split",
    "quote_style",
    "case_style",
    "punctuation_visual",
}


def _norm_space(text: str) -> str:
    return re.sub(r"[ \t\u00a0\u3000]+", " ", (text or "").strip())


def _make_issue(
    para: ParagraphUnit,
    idx: int,
    *,
    category: str,
    reason: str,
    suggestion: str,
    original_text: str,
    modified_text: str,
    confidence: float = 0.95,
) -> IssueItem:
    return IssueItem(
        issue_id=f"{para.para_id}-word-{idx:03d}",
        issue_no="",
        para_id=para.para_id,
        category=category,
        severity="low" if category in {"layout_spacing", "quote_style", "case_style", "punctuation_visual"} else "medium",
        reason=reason,
        suggestion=suggestion,
        confidence=confidence,
        page_no=para.page_no,
        original_text=original_text,
        modified_text=modified_text,
        snippet=para.text[:240],
        review_input_text=para.text,
        review_round=0,
        anchor={"evidence_source": "word_helper"},
        origin="word_helper",
        evidence_source="word_helper",
    )


def _has_visible_extra_space(raw_text: str) -> bool:
    text = raw_text or ""
    return bool(re.search(r"(?<=[\u4e00-\u9fff])[ \t\u00a0\u3000]+(?=[\u4e00-\u9fff])", text)) or bool(
        re.search(r"[ \t\u00a0\u3000]{2,}", text)
    )


def _space_issue_span(raw_text: str) -> str:
    match = re.search(r".{0,24}(?<=[\u4e00-\u9fff])[ \t\u00a0\u3000]+(?=[\u4e00-\u9fff]).{0,24}", raw_text or "")
    if not match:
        match = re.search(r".{0,24}[ \t\u00a0\u3000]{2,}.{0,24}", raw_text or "")
    return match.group(0).strip() if match else (raw_text or "").strip()


def _has_manual_break(paragraph: ParagraphUnit) -> bool:
    raw_runs = paragraph.anchor.get("raw_runs_text") or []
    if isinstance(raw_runs, list):
        for item in raw_runs:
            if isinstance(item, str) and ("\n" in item or "\v" in item):
                return True
    return bool(paragraph.anchor.get("hard_break_count") or 0)


def _is_paragraph_split(paragraph: ParagraphUnit, next_paragraph: ParagraphUnit | None = None) -> bool:
    text = paragraph.text.strip()
    if bool(re.search(r"[，,、；;：:]$", text)):
        return True
    if next_paragraph is not None:
        next_text = next_paragraph.text.strip()
        if next_text and bool(re.search(r"^[，,、；;：:。.!?】》）)]", next_text)):
            return True
    return False


def review_word_helper_document(
    paragraphs: List[ParagraphUnit],
    config: ReviewConfig | None = None,
) -> Tuple[List[IssueItem], List[Dict[str, object]]]:
    del config
    issues: List[IssueItem] = []
    filtered: List[Dict[str, object]] = []

    for para_index, para in enumerate(paragraphs):
        next_para = paragraphs[para_index + 1] if para_index + 1 < len(paragraphs) else None
        raw_text = str(para.anchor.get("docx_meta", {}).get("raw_text") or para.text)
        if _has_visible_extra_space(raw_text):
            original = _space_issue_span(raw_text)
            issues.append(
                _make_issue(
                    para,
                    len(issues) + 1,
                    category="layout_spacing",
                    reason="Word 原始结构中存在中文字符之间的异常空格或连续空格。",
                    suggestion="删除中文字符之间不必要的空格；英文词间正常空格不处理。",
                    original_text=original,
                    modified_text="",
                )
            )
        if _has_manual_break(para):
            issues.append(
                _make_issue(
                    para,
                    len(issues) + 1,
                    category="line_break",
                    reason="Word 段内存在人工硬换行痕迹。",
                    suggestion="改为自然换行或合并到同一段。",
                    original_text=para.text,
                    modified_text="",
                )
            )
        if _is_paragraph_split(para, next_para):
            issues.append(
                _make_issue(
                    para,
                    len(issues) + 1,
                    category="paragraph_split",
                    reason="Word 段落以连接性标点结尾或下一段以标点开头，疑似错误另起自然段。",
                    suggestion="检查该处是否应与下一自然段合并。",
                    original_text=(para.text + ("\n" + next_para.text if next_para is not None else ""))[:240],
                    modified_text="",
                )
            )

    normalized: List[IssueItem] = []
    for idx, issue in enumerate(issues, start=1):
        issue.issue_id = f"{issue.para_id}-r0-issue-{idx:03d}"
        issue.issue_no = f"W{idx}"
        normalized.append(issue)
    return normalized, filtered
