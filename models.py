from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

SourceType = Literal["docx", "pdf"]
IssueOrigin = Literal["text", "visual", "word_helper"]
SeverityType = Literal["low", "medium", "high"]
PdfParseBackend = Literal["paddle", "doc2x"]


@dataclass
class ReviewConfig:
    model: str = "deepseek-v4-pro"
    visual_model: Optional[str] = None
    pdf_parse_backend: PdfParseBackend = "paddle"
    ignore_cjk_punctuation_width: bool = False
    review_word_formulas: bool = False
    enable_visual_structure: bool = True
    language_mode: Literal["follow_source", "zh", "en"] = "follow_source"
    window_size: int = 2
    max_retries: int = 3
    request_timeout: int = 90
    severity_threshold: SeverityType = "low"
    max_concurrency: int = 2
    api_base_url: str = "https://api.deepseek.com"
    api_key: Optional[str] = None
    visual_api_base_url: Optional[str] = None
    visual_api_key: Optional[str] = None
    # 0 means auto mode: keep reviewing until no issues or no safe progress.
    max_review_rounds: int = 0


@dataclass
class ParagraphUnit:
    doc_id: str
    para_id: str
    page_no: Optional[int]
    text: str
    bbox_list: List[List[float]] = field(default_factory=list)
    source_type: SourceType = "docx"
    anchor: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IssueItem:
    issue_id: str
    para_id: str
    issue_no: str
    category: str
    severity: SeverityType
    reason: str
    suggestion: str
    confidence: float
    page_no: Optional[int] = None
    original_text: str = ""
    modified_text: str = ""
    snippet: str = ""
    review_input_text: str = ""
    review_round: int = 1
    bbox_list: List[List[float]] = field(default_factory=list)
    anchor: Dict[str, Any] = field(default_factory=dict)
    origin: IssueOrigin = "text"
    evidence_source: Optional[str] = None


@dataclass
class ParseResult:
    source_type: SourceType
    original_path: Path
    normalized_path: Path
    doc_id: str
    paragraphs: List[ParagraphUnit]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewResult:
    parse_result: ParseResult
    issues: List[IssueItem]
    annotated_file_path: Optional[Path] = None
    annotated_markdown_path: Optional[Path] = None
    review_report_path: Optional[Path] = None
    summary_markdown_path: Optional[Path] = None
    stats: Dict[str, Any] = field(default_factory=dict)


SEVERITY_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
}
