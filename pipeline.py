from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .models import IssueItem, ParagraphUnit, ReviewConfig, ReviewResult
    from .services.annotators import apply_annotations
    from .services.layout_regions import extract_layout_regions
    from .services.ocr_postprocess import preprocess_parse_result
    from .services.parsers import parse_document, parse_word_helper_document
    from .services.reviewer import filter_issues_by_artifact_rules, review_paragraphs_with_audit
    from .services.review_summary import write_review_summary_markdown
    from .services.visual_region_renderer import group_regions_for_render, render_region_groups
    from .services.visual_structure_reviewer import review_visual_regions
    from .services.word_issue_locator import attach_word_issue_locations
    from .services.word_page_locator import locate_word_issue_pages
    from .services.word_helper_reviewer import review_word_helper_document
except ImportError:  # pragma: no cover
    from models import IssueItem, ParagraphUnit, ReviewConfig, ReviewResult
    from services.annotators import apply_annotations
    from services.layout_regions import extract_layout_regions
    from services.ocr_postprocess import preprocess_parse_result
    from services.parsers import parse_document, parse_word_helper_document
    from services.reviewer import filter_issues_by_artifact_rules, review_paragraphs_with_audit
    from services.review_summary import write_review_summary_markdown
    from services.visual_region_renderer import group_regions_for_render, render_region_groups
    from services.visual_structure_reviewer import review_visual_regions
    from services.word_issue_locator import attach_word_issue_locations
    from services.word_page_locator import locate_word_issue_pages
    from services.word_helper_reviewer import review_word_helper_document


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_QWEN_VISUAL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_VISUAL_MODEL = "qwen3.7-plus"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _env_enabled(name: str, default: str = "off") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _log_visual(msg: str) -> None:
    print(f"[VisualReview] {msg}", flush=True)


def _serialize_issue(issue: IssueItem, adopted: bool = True) -> Dict[str, object]:
    return {
        "issue_id": issue.issue_id,
        "issue_no": issue.issue_no,
        "para_id": issue.para_id,
        "page_no": issue.page_no,
        "category": issue.category,
        "severity": issue.severity,
        "confidence": round(issue.confidence, 4),
        "reason": issue.reason,
        "suggestion": issue.suggestion,
        "original_text": issue.original_text,
        "modified_text": issue.modified_text,
        "snippet": issue.snippet,
        "review_input_text": issue.review_input_text,
        "review_round": issue.review_round,
        "bbox_list": issue.bbox_list,
        "anchor": issue.anchor,
        "origin": issue.origin,
        "evidence_source": issue.evidence_source,
        "adopted": adopted,
    }


def _normalize_text_for_dedup(text: str) -> str:
    return "".join((text or "").split()).lower()


def _dedup_visual_vs_text_issues(
    text_issues: List[IssueItem],
    visual_issues: List[IssueItem],
) -> Tuple[List[IssueItem], List[Dict[str, object]]]:
    kept: List[IssueItem] = []
    dropped: List[Dict[str, object]] = []
    text_keys = {
        f"{issue.page_no}|{_normalize_text_for_dedup(issue.original_text)}"
        for issue in text_issues
        if issue.page_no and issue.original_text.strip()
    }
    for issue in visual_issues:
        key = f"{issue.page_no}|{_normalize_text_for_dedup(issue.original_text)}"
        if issue.page_no and issue.original_text.strip() and key in text_keys:
            payload = _serialize_issue(issue, adopted=False)
            payload["filter_reason"] = "duplicate_text_issue"
            dropped.append(payload)
            continue
        kept.append(issue)
    return kept, dropped


def _tag_visual_issue(issue: IssueItem) -> IssueItem:
    issue.origin = "visual"
    issue.evidence_source = issue.evidence_source or "visual_structure"
    return issue


def _prepare_word_helper_annotations(
    helper_issues: List[IssueItem],
    *,
    helper_paragraph_count: int,
    pdf_page_count: int,
) -> List[IssueItem]:
    if not helper_issues or pdf_page_count <= 0:
        return []
    denom = max(1, helper_paragraph_count)
    out: List[IssueItem] = []
    for issue in helper_issues:
        try:
            para_idx = int(str(issue.para_id).rsplit("-", 1)[-1])
        except Exception:
            para_idx = 1
        page_no = min(pdf_page_count, max(1, int(((para_idx - 1) / denom) * pdf_page_count) + 1))
        issue.page_no = page_no
        issue.anchor = dict(issue.anchor or {})
        issue.anchor["page_level_annotation"] = True
        issue.anchor["annotation_scope"] = "page"
        issue.anchor["page_mapping"] = "word_paragraph_ratio"
        out.append(issue)
    return out


def build_config(
    model: str = "deepseek-v4-pro",
    language_mode: str = "zh",
    window_size: int = 2,
    max_retries: int = 3,
    severity_threshold: str = "low",
    max_review_rounds: int | None = None,
    enable_visual_structure: bool | None = None,
    pdf_parse_backend: str = "paddle",
) -> ReviewConfig:
    resolved_max_review_rounds = (
        max(0, int(max_review_rounds))
        if max_review_rounds is not None
        else max(0, _env_int("DOC_REVIEW_MAX_ROUNDS", 0))
    )
    return ReviewConfig(
        model=model,
        visual_model=_env_str("DOC_REVIEW_VISUAL_MODEL", DEFAULT_QWEN_VISUAL_MODEL),
        pdf_parse_backend="doc2x" if str(pdf_parse_backend).strip().lower() == "doc2x" else "paddle",
        enable_visual_structure=(
            bool(enable_visual_structure)
            if enable_visual_structure is not None
            else _env_enabled("DOC_REVIEW_ENABLE_VISUAL_STRUCTURE", default="on")
        ),
        language_mode=language_mode,  # type: ignore[arg-type]
        window_size=max(1, int(window_size)),
        max_retries=max(1, int(max_retries)),
        request_timeout=_env_int("DOC_REVIEW_TIMEOUT", 90),
        severity_threshold=severity_threshold,  # type: ignore[arg-type]
        max_concurrency=max(1, _env_int("DOC_REVIEW_MAX_CONCURRENCY", 2)),
        api_base_url=_env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        visual_api_base_url=_env_str("DOC_REVIEW_VISUAL_BASE_URL", DEFAULT_QWEN_VISUAL_BASE_URL),
        visual_api_key=(
            os.getenv("DOC_REVIEW_VISUAL_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
        ),
        max_review_rounds=resolved_max_review_rounds,
    )


def process_document(
    file_path: str | Path,
    config: ReviewConfig,
    output_dir: str | Path,
    word_helper_path: Optional[str | Path] = None,
) -> ReviewResult:
    total_started = time.monotonic()
    parse_started = time.monotonic()
    parse_result = parse_document(file_path, pdf_parse_backend=config.pdf_parse_backend)
    parse_result = preprocess_parse_result(parse_result, config)
    if parse_result.source_type == "pdf":
        config = replace(config, ignore_cjk_punctuation_width=True)
    elif parse_result.source_type == "docx":
        config = replace(config, review_word_formulas=True)
    parse_elapsed = time.monotonic() - parse_started
    deepseek_started = time.monotonic()
    text_issues, filtered_issues = review_paragraphs_with_audit(parse_result.paragraphs, config)
    deepseek_elapsed = time.monotonic() - deepseek_started
    qwen_elapsed = 0.0
    is_word_main = parse_result.source_type == "docx"
    helper_issues: List[IssueItem] = []
    helper_filtered_payload: List[Dict[str, object]] = []
    helper_report_issues: List[Dict[str, object]] = []
    helper_annotation_issues: List[IssueItem] = []
    helper_paragraph_count = 0
    helper_path = None

    visual_issues: List[IssueItem] = []
    visual_regions_payload: List[Dict[str, object]] = []
    visual_filtered_payload: List[Dict[str, object]] = []
    visual_audit_payload: List[Dict[str, object]] = []
    visual_review_region_count = 0
    visual_structure_error = ""
    word_page_locator_meta: Dict[str, object] = {}
    visual_enabled = bool(getattr(config, "enable_visual_structure", True))
    visual_region_limit = max(1, _env_int("DOC_REVIEW_VISUAL_MAX_REGIONS", 120))
    visual_dpi = max(72, _env_int("DOC_REVIEW_VISUAL_DPI", 200))

    if is_word_main:
        helper_issues, helper_filtered_payload = review_word_helper_document(parse_result.paragraphs, config)
        helper_paragraph_count = len(parse_result.paragraphs)
        for idx, issue in enumerate(helper_issues, start=1):
            issue.issue_id = f"{issue.para_id}-word-main-{idx:03d}"
            issue.issue_no = f"WH{idx}"
            issue.anchor = dict(issue.anchor or {})
            issue.anchor["word_helper_main"] = True
            issue.anchor["structure_source"] = "word_main"
            issue.evidence_source = "word_main"
        helper_report_issues = [_serialize_issue(issue, adopted=True) for issue in helper_issues]
        parse_result.metadata["word_main_structure"] = {
            "paragraph_count": helper_paragraph_count,
            "issue_count": len(helper_issues),
        }
    elif word_helper_path:
        helper_path = Path(word_helper_path).expanduser().resolve()
    if helper_path is not None and helper_path.exists():
        try:
            helper_parse_result = parse_word_helper_document(helper_path)
            helper_issues, helper_filtered_payload = review_word_helper_document(helper_parse_result.paragraphs, config)
            helper_paragraph_count = len(helper_parse_result.paragraphs)
            parse_result.metadata["word_helper"] = {
                "source_file": str(helper_parse_result.original_path),
                "normalized_file": str(helper_parse_result.normalized_path),
                "paragraph_count": helper_paragraph_count,
            }
        except Exception as exc:
            parse_result.metadata["word_helper_error"] = str(exc)
            helper_issues = []
            helper_filtered_payload = []
            helper_report_issues = []

    if visual_enabled and parse_result.source_type == "pdf":
        _log_visual(
            f"enabled=true backend={os.getenv('DOC_REVIEW_LAYOUT_BACKEND', 'mineru')} "
            f"max_regions={visual_region_limit} dpi={visual_dpi} model={config.visual_model or config.model}"
        )
        try:
            layout_regions, layout_meta = extract_layout_regions(
                parse_result.normalized_path,
                max_regions=visual_region_limit,
            )
            _log_visual(
                f"layout used_backend={layout_meta.get('used_backend')} regions={len(layout_regions)} "
                f"errors={len(layout_meta.get('errors', []))}"
            )
            groups = group_regions_for_render(layout_regions)
            _log_visual(f"grouped groups={len(groups)}")
            group_manifest = render_region_groups(
                parse_result.normalized_path,
                doc_id=parse_result.doc_id,
                groups=groups[:visual_region_limit],
                dpi=visual_dpi,
                output_root=output_dir,
                pad_pt=6.0,
            )
            visual_crop_manifest = list(group_manifest.values())
            _log_visual(f"rendered crops={len(visual_crop_manifest)}")
            review_regions: List[Dict[str, object]] = []
            for group in groups[:visual_region_limit]:
                info = group_manifest.get(group.group_id, {})
                if not info:
                    continue
                for region in group.regions:
                    region.image_path = str(info.get("image_path", "")).strip()
                    region.anchor["group_bbox"] = info.get("bbox")
                    region.anchor["group_id"] = group.group_id
                    visual_regions_payload.append(region.to_dict())
                    if not region.reviewable:
                        continue
                reviewable_regions = [region for region in group.regions if region.reviewable]
                if not reviewable_regions:
                    continue
                group_page_no = int(getattr(group, "page_no", reviewable_regions[0].page_no) or reviewable_regions[0].page_no)
                group_bbox = getattr(group, "bbox", reviewable_regions[0].bbox)
                group_type = getattr(group, "group_type", "region")
                boundary_prev_text = str(getattr(group, "boundary_prev_text", "") or "").strip()
                boundary_next_text = str(getattr(group, "boundary_next_text", "") or "").strip()
                review_regions.append(
                    {
                        "region_id": group.group_id,
                        "group_id": group.group_id,
                        "group_type": group_type,
                        "page_no": group_page_no,
                        "type": group_type,
                        "bbox": info.get("bbox") or group_bbox,
                        "group_bbox": info.get("bbox") or group_bbox,
                        "text_hint": "\n\n".join(region.text_hint for region in reviewable_regions if region.text_hint)[:1200],
                        "boundary_text_hint": {
                            "prev_tail": boundary_prev_text[-120:],
                            "next_head": boundary_next_text[:120],
                        },
                        "boundary_bbox": {
                            "prev": getattr(group, "boundary_prev_bbox", []) or [],
                            "next": getattr(group, "boundary_next_bbox", []) or [],
                        },
                        "image_path": str(info.get("image_path", "")).strip(),
                        "source_backend": reviewable_regions[0].source_backend,
                        "child_regions": [
                            {
                                "region_id": region.region_id,
                                "type": region.region_type,
                                "bbox": region.bbox,
                                "text_hint": region.text_hint[:240],
                            }
                            for region in reviewable_regions
                        ],
                    }
                )
            parse_result.metadata["visual_layout_meta"] = layout_meta
            parse_result.metadata["visual_region_images"] = visual_crop_manifest
            visual_review_region_count = len(review_regions[:visual_region_limit])
            _log_visual(f"review_regions={visual_review_region_count}")
        except Exception as exc:
            visual_structure_error = f"layout_or_render_failed: {exc}"
            parse_result.metadata["visual_structure_error"] = visual_structure_error
            _log_visual(f"error {visual_structure_error}")
            review_regions = []

        if review_regions:
            qwen_started: Optional[float] = None
            try:
                qwen_started = time.monotonic()
                visual_issues, visual_audit_payload = review_visual_regions(
                    regions=review_regions[:visual_region_limit],
                    config=config,
                    language="zh" if config.language_mode in {"zh", "follow_source"} else "en",
                )
                qwen_elapsed += time.monotonic() - qwen_started
                visual_issues = [_tag_visual_issue(issue) for issue in visual_issues]
                _log_visual(
                    f"model_done audit={len(visual_audit_payload)} raw_issues={len(visual_issues)}"
                )
                paragraphs_by_id: Dict[str, ParagraphUnit] = {
                    issue.para_id: ParagraphUnit(
                        doc_id=parse_result.doc_id,
                        para_id=issue.para_id,
                        page_no=issue.page_no,
                        text=issue.review_input_text or issue.snippet or issue.original_text,
                        source_type=parse_result.source_type,
                        bbox_list=issue.bbox_list,
                        anchor=issue.anchor,
                    )
                    for issue in visual_issues
                }
                visual_issues, dropped_pairs = filter_issues_by_artifact_rules(visual_issues, paragraphs_by_id)
                visual_filtered_payload.extend(
                    [
                        {
                            **_serialize_issue(issue, adopted=False),
                            "filter_reason": reason,
                        }
                        for issue, reason in dropped_pairs
                    ]
                )
            except Exception as exc:
                if qwen_started is not None:
                    qwen_elapsed += time.monotonic() - qwen_started
                visual_structure_error = f"visual_model_failed: {exc}"
                parse_result.metadata["visual_structure_error"] = visual_structure_error
                _log_visual(f"error {visual_structure_error}")
    elif visual_enabled:
        _log_visual(f"enabled=true skipped source_type={parse_result.source_type}")
    else:
        _log_visual("enabled=false")

    visual_issues, visual_dedup_filtered = _dedup_visual_vs_text_issues(text_issues, visual_issues)
    visual_filtered_payload.extend(visual_dedup_filtered)

    issues = [*text_issues, *visual_issues]
    if not is_word_main:
        helper_annotation_issues = _prepare_word_helper_annotations(
            helper_issues,
            helper_paragraph_count=helper_paragraph_count,
            pdf_page_count=int(parse_result.metadata.get("page_count") or 0),
        )
        helper_report_issues = [_serialize_issue(issue, adopted=True) for issue in helper_annotation_issues]
    filtered_issues_all = [*filtered_issues, *visual_filtered_payload, *helper_filtered_payload]
    annotated_pdf: Optional[Path] = None
    annotated_md: Optional[Path] = None
    if not is_word_main:
        annotated_pdf, annotated_md = apply_annotations(parse_result, [*issues, *helper_annotation_issues], output_dir)
    else:
        attach_word_issue_locations([*issues, *helper_issues], parse_result.paragraphs)
        word_page_locator_meta = locate_word_issue_pages(
            parse_result=parse_result,
            issues=[*issues, *helper_issues],
            output_dir=output_dir,
        )
        helper_report_issues = [_serialize_issue(issue, adopted=True) for issue in helper_issues]
        annotated_pdf, annotated_md = apply_annotations(parse_result, [*issues, *helper_issues], output_dir)
    review_round_values = [x.review_round for x in issues]
    total_elapsed = time.monotonic() - total_started
    timing_stats = {
        "parse_seconds": round(parse_elapsed, 3),
        "deepseek_seconds": round(deepseek_elapsed, 3),
        "qwen_seconds": round(qwen_elapsed, 3),
        "total_seconds": round(total_elapsed, 3),
    }
    print(
        "[Timing] "
        f"parse={timing_stats['parse_seconds']:.3f}s "
        f"deepseek={timing_stats['deepseek_seconds']:.3f}s "
        f"qwen={timing_stats['qwen_seconds']:.3f}s "
        f"total={timing_stats['total_seconds']:.3f}s",
        flush=True,
    )

    stats: Dict[str, object] = {
        "source_type": parse_result.source_type,
        "paragraph_count": len(parse_result.paragraphs),
        "issue_count": len(issues),
        "text_issue_count": len(text_issues),
        "visual_issue_count": len(visual_issues),
        "word_helper_issue_count": len(helper_issues),
        "word_helper_filtered_count": len(helper_filtered_payload),
        "visual_region_count": len(visual_regions_payload),
        "visual_crop_count": len(parse_result.metadata.get("visual_region_images", []) or []),
        "visual_review_region_count": visual_review_region_count,
        "visual_review_audit_count": len(visual_audit_payload),
        "visual_structure_error": visual_structure_error,
        "word_helper_enabled": helper_path is not None,
        "word_helper_source": str(helper_path) if helper_path is not None else None,
        "word_helper_error": parse_result.metadata.get("word_helper_error"),
        "word_helper_annotation_policy": "page_level_note" if not is_word_main else "disabled_for_word_main",
        "word_helper_filter_policy": "disabled" if not is_word_main else "not_applicable",
        "word_main_review": is_word_main,
        "word_structure_issue_count": len(helper_issues) if is_word_main else 0,
        "word_page_locator": word_page_locator_meta,
        "ocr_used": bool(parse_result.metadata.get("ocr_used", False)),
        "postprocessed": bool(parse_result.metadata.get("postprocessed", False)),
        "pdf_parse_backend": parse_result.metadata.get("pdf_parser_backend"),
        "timing": timing_stats,
        **timing_stats,
        "issue_distribution": {
            "high": sum(1 for x in issues if x.severity == "high"),
            "medium": sum(1 for x in issues if x.severity == "medium"),
            "low": sum(1 for x in issues if x.severity == "low"),
        },
        "filtered_issue_count": len(filtered_issues_all),
        "adopted_issue_count": len(issues),
        "review_round_distribution": (
            {
                str(round_no): sum(1 for x in issues if x.review_round == round_no)
                for round_no in range(min(review_round_values), max(review_round_values) + 1)
            }
            if review_round_values
            else {}
        ),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{parse_result.original_path.stem}.review.json"
    report_payload = {
        "meta": {
            "source_type": parse_result.source_type,
            "doc_id": parse_result.doc_id,
            "input_file": str(parse_result.original_path),
            "normalized_file": str(parse_result.normalized_path),
            "model": config.model,
            "language_mode": config.language_mode,
            "window_size": config.window_size,
            "max_retries": config.max_retries,
            "severity_threshold": config.severity_threshold,
            "max_review_rounds": config.max_review_rounds,
            "visual_model": config.visual_model or config.model,
            "pdf_parse_backend": config.pdf_parse_backend,
            "pdf_parser_backend": parse_result.metadata.get("pdf_parser_backend"),
            "visual_api_base_url": config.visual_api_base_url or config.api_base_url,
            "visual_structure_enabled": visual_enabled,
            "visual_max_regions": visual_region_limit,
            "visual_dpi": visual_dpi,
            "visual_structure_error": visual_structure_error,
            "word_helper_enabled": helper_path is not None,
            "word_helper_source": str(helper_path) if helper_path is not None else None,
            "word_helper_error": parse_result.metadata.get("word_helper_error"),
            "word_helper_annotation_policy": "page_level_note" if not is_word_main else "disabled_for_word_main",
            "word_helper_filter_policy": "disabled" if not is_word_main else "not_applicable",
            "word_main_review": is_word_main,
            "word_structure_issue_count": len(helper_issues) if is_word_main else 0,
            "word_page_locator": word_page_locator_meta,
            "review_mode": "auto_until_clean" if config.max_review_rounds == 0 else "fixed_rounds",
        },
        "stats": stats,
        "issues": [_serialize_issue(issue, adopted=True) for issue in issues],
        "visual_structure_issues": [_serialize_issue(issue, adopted=True) for issue in visual_issues],
        "word_helper_issues": helper_report_issues,
        "visual_regions": visual_regions_payload,
        "visual_review_audit": visual_audit_payload,
        "filtered_issues": filtered_issues_all,
        "issues_all": [
            *[_serialize_issue(issue, adopted=True) for issue in issues],
            *filtered_issues_all,
            *helper_report_issues,
        ],
        "artifacts": {
            "annotated_pdf": str(annotated_pdf) if annotated_pdf else None,
            "annotated_markdown_bundle": str(annotated_md) if annotated_md else None,
            "word_rendered_pdf": word_page_locator_meta.get("converted_pdf") if word_page_locator_meta else None,
        },
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path = output_dir / f"{parse_result.original_path.stem}.review.summary.md"
    try:
        write_review_summary_markdown(
            review_json_path=report_path,
            output_path=summary_md_path,
            pdf_path=(
                parse_result.normalized_path
                if parse_result.source_type == "pdf"
                else Path(str(word_page_locator_meta["converted_pdf"]))
                if word_page_locator_meta.get("converted_pdf")
                else None
            ),
            title=f"{parse_result.original_path.stem} 问题摘要报告",
        )
        report_payload["artifacts"]["review_summary_markdown"] = str(summary_md_path)
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        summary_md_path = None
    filtered_report_path = output_dir / f"{parse_result.original_path.stem}.review.filtered.json"
    filtered_payload = {
        "meta": report_payload["meta"],
        "stats": {
            **stats,
            "issue_count": len(filtered_issues_all),
            "adopted_issue_count": len(issues),
        },
        "issues": filtered_issues_all,
        "artifacts": report_payload["artifacts"],
    }
    filtered_report_path.write_text(json.dumps(filtered_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return ReviewResult(
        parse_result=parse_result,
        issues=issues,
        annotated_file_path=annotated_pdf,
        annotated_markdown_path=annotated_md,
        review_report_path=report_path,
        summary_markdown_path=summary_md_path,
        stats=stats,
    )


def summarize_result(result: ReviewResult) -> str:
    stats = result.stats
    return (
        f"Review completed. source={stats.get('source_type')} | paragraphs={stats.get('paragraph_count')} "
        f"| issues={stats.get('issue_count')} | ocr_used={stats.get('ocr_used')} "
        f"| postprocessed={stats.get('postprocessed')} | dist={stats.get('issue_distribution')}"
    )


def issues_to_rows(result: ReviewResult):
    rows = []
    for issue in result.issues:
        rows.append(
            {
                "issue_id": issue.issue_id,
                "issue_no": issue.issue_no,
                "para_id": issue.para_id,
                "page_no": issue.page_no,
                "category": issue.category,
                "severity": issue.severity,
                "confidence": round(issue.confidence, 4),
                "reason": issue.reason,
                "suggestion": issue.suggestion,
                "original_text": issue.original_text,
                "modified_text": issue.modified_text,
                "snippet": issue.snippet,
                "review_input_text": issue.review_input_text,
                "review_round": issue.review_round,
            }
        )
    return rows


def issues_to_table_rows(issue_rows):
    table_rows = []
    for row in issue_rows:
        table_rows.append(
            [
                row["issue_id"],
                row["issue_no"],
                row["para_id"],
                row["page_no"],
                row["category"],
                row["severity"],
                row["confidence"],
                row["reason"],
                row["suggestion"],
                row["original_text"],
                row["modified_text"],
                row["review_round"],
                row["snippet"],
            ]
        )
    return table_rows


def locate_issue_detail(result: ReviewResult, issue_id: str) -> Tuple[str, str]:
    issue = next((x for x in result.issues if x.issue_id == issue_id), None)
    if issue is None:
        return "Issue not found.", ""

    para = next((p for p in result.parse_result.paragraphs if p.para_id == issue.para_id), None)
    src_text = para.text if para else ""
    detail = (
        f"Issue: {issue.issue_id} ({issue.issue_no})\n"
        f"Paragraph: {issue.para_id}\n"
        f"Page: {issue.page_no}\n"
        f"Category: {issue.category}\n"
        f"Severity: {issue.severity}\n"
        f"Confidence: {issue.confidence:.2f}\n\n"
        f"Reason:\n{issue.reason}\n\n"
        f"Suggestion:\n{issue.suggestion}\n\n"
        f"Original Text:\n{issue.original_text}\n"
        f"Modified Text:\n{issue.modified_text}\n"
        f"Review Round:\n{issue.review_round}\n"
    )
    return detail, src_text
