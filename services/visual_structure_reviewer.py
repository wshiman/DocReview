from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    from ..models import IssueItem, ReviewConfig
except ImportError:  # pragma: no cover
    from models import IssueItem, ReviewConfig

VISUAL_CATEGORIES = {
    "layout_spacing",
    "quote_style",
    "case_style",
    "line_break",
    "page_split_artifact",
    "table_layout",
    "punctuation_visual",
    "equation_format",
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class VisualReviewContext:
    doc_id: str
    model: str
    api_base_url: str
    api_key: str
    request_timeout: int
    max_retries: int


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    src = (text or "").strip()
    if not src:
        return None
    try:
        out = json.loads(src)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", src, flags=re.S)
    for frag in fenced:
        try:
            out = json.loads(frag)
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", src, flags=re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def _img_to_data_uri(image_path: Path) -> str:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _build_system_prompt(language: str) -> str:
    if language == "zh":
        return (
            "你是专业PDF视觉结构审查器，只判断截图中可直接看见的版面排版和OCR可视格式问题。"
            "审查边界：完全不审核语义、逻辑、事实、术语、内容优劣，不做内容润色、学术修改或错别字判断。"
            "允许问题类别仅限：layout_spacing, quote_style, case_style, line_break, page_split_artifact, table_layout, punctuation_visual, equation_format。"
            "硬性豁免：PDF页面宽度内正常自动换行、段落自然行距、两端对齐导致的字距拉伸，全部视为正常格式，禁止报错。"
            "layout_spacing 只适用于人工多余连续空格或OCR错误插入空格；正常排版拉伸、行距间距一律豁免。"
            "必须逐行逐字符扫描可见标点，不得只看语义是否通顺：中文正文、中文短语、中文段落结尾中夹杂英文半角逗号/句号/分号/冒号/问号/感叹号/括号，"
            "以及中英文标点重复或连续混用，如“篇.”、“项.”、“等.”、“图1.”、“，,”、“。.”、“(中文)”等，属于 punctuation_visual；"
            "英文缩写、英文期刊名、h-index、NeurIPS、TPAMI、数字小数点和URL中的英文标点除外。"
            "普通单段截图禁止报告 line_break；页面内自然换行不属于错误。"
            "若截图区域是公式或包含公式，只检查可见公式排版问题：公式是否断裂/重叠/溢出、编号括号与位置是否规范、同页可见编号是否明显重复或顺序异常、公式与编号是否对齐；类别用 equation_format。"
            "若证据不确定或可能是正常PDF排版，必须不输出 issue。输出必须是 JSON 对象。"
        )
    return (
        "You are a professional PDF visual-structure reviewer. Only judge directly visible layout, "
        "typesetting, and OCR visual-format issues in the screenshot. Do not review semantics, logic, facts, "
        "terminology, content quality, writing style, academic wording, or typos. Allowed categories only: "
        "layout_spacing, quote_style, case_style, line_break, page_split_artifact, table_layout, punctuation_visual, equation_format. "
        "Hard exemptions: normal PDF line wrapping within page width, natural paragraph line spacing, and justified "
        "text spacing are normal formatting and must not be reported. layout_spacing only applies to explicit extra "
        "consecutive spaces or OCR-inserted spaces. Carefully scan visible punctuation line by line and character by "
        "character; do not skip punctuation just because the sentence is semantically readable. ASCII comma/period/"
        "semicolon/colon/question/exclamation/parentheses mixed into Chinese prose, Chinese phrases, or Chinese paragraph "
        "endings, duplicated mixed punctuation such as '，,' or '。.', and ASCII period after Chinese measure words "
        "such as '篇.', '项.', '等.', or '图1.' are punctuation_visual. Exempt English abbreviations, "
        "journal names, h-index, conference names, decimal numbers, and URLs. Single-paragraph crops must not report line_break. "
        "If the crop is an equation or contains equations, check only visible equation formatting: broken/overlapped/"
        "overflowing equations, equation number parentheses and placement, visibly duplicated or out-of-order numbers on the same page, "
        "and equation-number alignment; use category equation_format. If evidence is uncertain or may be normal PDF formatting, output no issue. Output must be a JSON object."
    )


def _build_user_prompt(region_payload: Dict[str, Any], language: str) -> Dict[str, Any]:
    if language == "zh":
        instruction = "\n".join(
            [
                "只根据截图和 region 元数据做可视结构审查，按以下流程执行：",
                "1. 先读取 region.group_type。",
                "2. 若 group_type == adjacent_paragraphs：这是 A 段 + B 段的相邻段截图，A、B 各自段内的标点、空格、格式已由单段截图检查过；此处只检查段间关系和边界标点，只检查 A 结尾、B 开头、两段之间边界区域的问题，包括错误换行、误拆分分段、跨页残片、本应合并的相邻段落、边界处承接标点异常；只允许输出 line_break、page_split_artifact 或 punctuation_visual。",
                "3. 若 group_type != adjacent_paragraphs：只检查单段/单块内部的 OCR 可视格式问题；禁止输出 line_break。",
                "4. 对所有截图先执行豁免判断：PDF页面宽度内正常自动换行、段落自然行距、两端对齐造成的字距拉伸，都必须视为正常，不得输出 issue。",
                "5. layout_spacing 只有在能明确看到人工多余连续空格或 OCR 错误插入空格时才可报告；不要因为文字被两端对齐拉开、行距较大、字距均匀变宽而报告。",
                "6. punctuation_visual 必须逐行逐字符检查，不能只抽查首尾，不能因为句子语义可读就跳过：中文正文、中文短语、中文段落结尾中出现英文半角逗号、句号、分号、冒号、问号、感叹号、括号，或中英文标点连续混用/重复，均要报告。",
                "7. 但若 group_type == adjacent_paragraphs，punctuation_visual 只允许报告 boundary_text_hint.prev_tail 和 boundary_text_hint.next_head 附近的边界标点问题；禁止报告 A 段内部或 B 段内部已经能由单段截图发现的标点问题，以免重复标注。",
                "8. punctuation_visual 重点检查这些高频错误：中文量词/中文词后接英文句点，如“篇.”、“项.”、“等.”；中文图表编号或标题后误用英文句点，如“图1.”；中文标点后又接英文标点，如“，,”、“。.”；中文内容使用英文括号，如“(中文)”。",
                "9. punctuation_visual 的豁免：英文缩写、英文期刊/会议名、h-index、TPAMI、NeurIPS、URL、邮箱、版本号、小数点、英文列表中的英文标点，不要报告。",
                "10. 对 punctuation_visual，original_text 填错误标点附近 4-20 个可见字符即可；modified_text 填替换为中文全角标点后的片段；bbox 可使用整段/整块 bbox。",
                "11. table_layout 只报告可视列错位、单元格挤压、文字溢出、明显表格线/内容错位；不要判断表格数值或内容逻辑。",
                "12. 若 region.type == equation 或截图中包含公式：只检查公式可视格式，包含公式断裂、上下标（注意上标命名时的前后一致性问题）/符号重叠、公式溢出、公式编号括号/位置不规范、同页可见编号明显重复或顺序异常、公式与编号未对齐；类别使用 equation_format。不要判断公式数学内容是否正确。",
                "13. 若证据不确定、需要阅读上下文才能判断、或可能是正常 PDF 排版，输出空 issues。",
                "14. 每张截图整体判断一次；bbox 可使用 region.bbox 表示整段/整块区域，不需要定位到具体字符。",
                "15. 每条 issue 必须包含 page_no 与 bbox（[x0,y0,x1,y1]）；若无法定位到截图对应区域，则不要输出该 issue。",
            ]
        )
    else:
        instruction = "\n".join(
            [
                "Review only visible structure/OCR-format issues from the screenshot and region metadata. Follow this process:",
                "1. First read region.group_type.",
                "2. If group_type == adjacent_paragraphs: this is an A+B adjacent-paragraph crop. Punctuation, spacing, and format issues inside A or inside B are already reviewed by single-paragraph crops. Here only inspect the A ending, B beginning, and the boundary between them, including wrong paragraph split, erroneous paragraph break, page-split artifact, adjacent paragraphs that should be merged, or boundary punctuation continuation errors. Only output line_break, page_split_artifact, or punctuation_visual.",
                "3. If group_type != adjacent_paragraphs: check only visual OCR/layout issues inside the single paragraph/block. Do not output line_break.",
                "4. Apply exemptions before reporting: normal PDF line wrapping within page width, natural paragraph line spacing, and justified text spacing are normal and must not be reported.",
                "5. layout_spacing is allowed only when explicit extra consecutive spaces or OCR-inserted spaces are clearly visible. Do not report uniformly stretched justified text, large line spacing, or normal character spacing.",
                "6. For punctuation_visual, scan line by line and character by character, not just paragraph starts/ends, and do not skip punctuation because the sentence is readable: ASCII comma, period, semicolon, colon, question mark, exclamation mark, or parentheses inside Chinese prose, Chinese phrases, or Chinese paragraph endings, plus duplicated mixed punctuation, must be reported.",
                "7. But if group_type == adjacent_paragraphs, punctuation_visual may only report boundary punctuation issues near boundary_text_hint.prev_tail and boundary_text_hint.next_head. Do not report punctuation issues inside A or inside B that single-paragraph crops can already detect, to avoid duplicate annotations.",
                "8. Prioritize these frequent punctuation_visual errors: ASCII period after Chinese measure words or Chinese terms such as '篇.', '项.', '等.'; ASCII period after Chinese figure/table labels or headings such as '图1.'; mixed duplicated punctuation such as '，,' or '。.'; ASCII parentheses around Chinese content such as '(中文)'.",
                "9. Exempt punctuation in English abbreviations, journal/conference names, h-index, TPAMI, NeurIPS, URLs, emails, versions, decimal numbers, and English lists.",
                "10. For punctuation_visual, original_text may be a 4-20 character visible span around the bad punctuation; modified_text should be the same span with Chinese full-width punctuation; bbox may be the whole paragraph/block bbox.",
                "11. table_layout only covers visible column misalignment, squeezed/overflowing cell text, or table line/content misalignment; do not judge table values or logic.",
                "12. If region.type == equation or the crop contains equations: check only visible equation formatting, including broken equations, overlapping superscripts/subscripts/symbols, overflow, nonstandard equation-number parentheses/placement, visibly duplicated or out-of-order numbers on the same page, and equation-number misalignment; use equation_format. Do not judge mathematical correctness.",
                "13. If evidence is uncertain, requires broader context, or may be normal PDF formatting, return an empty issues list.",
                "14. Review each screenshot once. bbox may be region.bbox for the whole paragraph/block; character-level localization is not required.",
                "15. Each issue must include page_no and bbox [x0,y0,x1,y1]. If no screenshot-level region can be localized, omit the issue.",
            ]
        )
    schema = {
        "issues": [
            {
                "category": "layout_spacing|quote_style|case_style|line_break|page_split_artifact|table_layout|punctuation_visual|equation_format",
                "severity": "low|medium|high",
                "reason": "string",
                "suggestion": "string",
                "original_text": "string",
                "modified_text": "string",
                "page_no": 1,
                "bbox": [0, 0, 0, 0],
            }
        ]
    }
    return {
        "instruction": instruction,
        "region": region_payload,
        "required_schema": schema,
    }


def _call_vlm(
    *,
    ctx: VisualReviewContext,
    image_path: Path,
    region_payload: Dict[str, Any],
    language: str,
) -> Dict[str, Any]:
    url = f"{ctx.api_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {ctx.api_key}", "Content-Type": "application/json"}
    system_prompt = _build_system_prompt(language)
    user_json = _build_user_prompt(region_payload, language)
    image_data_uri = _img_to_data_uri(image_path)
    user_text = json.dumps(user_json, ensure_ascii=False)
    payload = {
        "model": ctx.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, ctx.max_retries) + 1):
        started = time.monotonic()
        try:
            print(
                f"[VisualReview] call model={ctx.model} region={region_payload.get('region_id')} "
                f"group={region_payload.get('group_id')} attempt={attempt}",
                flush=True,
            )
            resp = requests.post(url, headers=headers, json=payload, timeout=ctx.request_timeout)
            elapsed = time.monotonic() - started
            if resp.status_code >= 400:
                text = resp.text[:600]
                if "unknown variant `image_url`" in text or "expected `text`" in text:
                    raise RuntimeError(
                        "visual review endpoint rejected image input. "
                        "The current API base URL appears text-only; set DOC_REVIEW_VISUAL_BASE_URL/API_KEY "
                        "to a vision-capable OpenAI-compatible gateway such as Qwen/DashScope, "
                        f"or use a vision-capable model endpoint. raw={text}"
                    )
                raise RuntimeError(f"visual review request failed {resp.status_code}: {text}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = _safe_json_extract(content)
            if not isinstance(parsed, dict):
                raise RuntimeError("visual review response not json object")
            print(
                f"[VisualReview] success region={region_payload.get('region_id')} "
                f"issues={len(parsed.get('issues', []) if isinstance(parsed.get('issues'), list) else [])} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            return parsed
        except Exception as exc:
            elapsed = time.monotonic() - started
            last_err = exc
            print(
                f"[VisualReview] failed region={region_payload.get('region_id')} "
                f"attempt={attempt}/{ctx.max_retries} elapsed={elapsed:.1f}s error={exc}",
                flush=True,
            )
            time.sleep(1.0)
    raise RuntimeError(f"visual reviewer failed after retries: {last_err}")


def _normalize_bbox(bbox: Any) -> List[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    try:
        vals = [float(x) for x in bbox]
    except Exception:
        return []
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return []
    return vals


def _bbox_inside(outer: Sequence[float], inner: Sequence[float], tolerance: float = 2.0) -> bool:
    if len(outer) != 4 or len(inner) != 4:
        return False
    return (
        float(inner[0]) >= float(outer[0]) - tolerance
        and float(inner[1]) >= float(outer[1]) - tolerance
        and float(inner[2]) <= float(outer[2]) + tolerance
        and float(inner[3]) <= float(outer[3]) + tolerance
    )


def _severity(raw: Any) -> str:
    val = str(raw or "medium").strip().lower()
    return val if val in {"low", "medium", "high"} else "medium"


def _visual_cat(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    return val if val in VISUAL_CATEGORIES else "layout_spacing"


def _issue_key(issue: IssueItem) -> str:
    material = f"{issue.page_no}|{issue.original_text.strip().lower()}|{issue.anchor.get('visual_category','')}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _has_explicit_extra_space_evidence(original_text: str, reason: str, suggestion: str) -> bool:
    material = f"{original_text}\n{reason}\n{suggestion}"
    if re.search(r"[A-Za-z0-9]\s{2,}[A-Za-z0-9]", material):
        return True
    return any(token in material for token in ["连续空格", "多个空格", "多打了空格", "OCR插入空格", "extra consecutive spaces"])


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _boundary_text_from_region(region_payload: Dict[str, Any]) -> Dict[str, str]:
    raw = region_payload.get("boundary_text_hint")
    if not isinstance(raw, dict):
        return {}
    return {
        "prev_tail": str(raw.get("prev_tail", "") or "").strip(),
        "next_head": str(raw.get("next_head", "") or "").strip(),
    }


def _is_adjacent_boundary_punctuation_issue(original_text: str, boundary_text_hint: Dict[str, str]) -> bool:
    """
    Adjacent paragraph crops overlap with single-paragraph crops. Keep punctuation issues only
    when the evidence is located at the A/B boundary, not inside A or B.
    """

    original = _compact_text(original_text)
    if not original:
        return False
    prev_tail = _compact_text(boundary_text_hint.get("prev_tail", ""))
    next_head = _compact_text(boundary_text_hint.get("next_head", ""))
    if not prev_tail and not next_head:
        return True

    boundary_window = (prev_tail[-80:] + next_head[:80])
    if original in boundary_window:
        return True

    prefix = original[: min(len(original), 8)]
    suffix = original[-min(len(original), 8) :]
    prev_edge = prev_tail[-80:]
    next_edge = next_head[:80]
    return bool((suffix and suffix in prev_edge) or (prefix and prefix in next_edge))


def parse_visual_issues_to_issue_items(
    *,
    raw_payload: Dict[str, Any],
    para_id: str,
    fallback_page_no: int,
    fallback_bbox: Sequence[float],
    group_bbox: Sequence[float] | None = None,
    review_input_text: str,
    group_type: str = "",
    boundary_text_hint: Dict[str, str] | None = None,
) -> List[IssueItem]:
    raw_issues = raw_payload.get("issues", []) if isinstance(raw_payload, dict) else []
    if not isinstance(raw_issues, list):
        return []
    out: List[IssueItem] = []
    for idx, raw in enumerate(raw_issues, start=1):
        if not isinstance(raw, dict):
            continue
        region_bbox = _normalize_bbox(fallback_bbox)
        crop_bbox = _normalize_bbox(group_bbox) or region_bbox
        raw_bbox = _normalize_bbox(raw.get("bbox"))
        # The model sees a cropped image, so bbox coordinates can be ambiguous.
        # Only trust model bbox when it is clearly in original-PDF coordinates.
        bbox = raw_bbox if raw_bbox and _bbox_inside(crop_bbox, raw_bbox) else region_bbox
        page_no = int(raw.get("page_no") or fallback_page_no or 0)
        if page_no <= 0 or not bbox:
            continue
        visual_category = _visual_cat(raw.get("category"))
        if group_type == "adjacent_paragraphs" and visual_category not in {
            "line_break",
            "page_split_artifact",
            "punctuation_visual",
        }:
            continue
        reason = str(raw.get("reason", "")).strip()
        suggestion = str(raw.get("suggestion", "")).strip()
        if not reason or not suggestion:
            continue
        original_text = str(raw.get("original_text", "")).strip()
        modified_text = str(raw.get("modified_text", "")).strip()
        if visual_category == "line_break" and group_type != "adjacent_paragraphs":
            continue
        if (
            visual_category == "punctuation_visual"
            and group_type == "adjacent_paragraphs"
            and not _is_adjacent_boundary_punctuation_issue(original_text, boundary_text_hint or {})
        ):
            continue
        if visual_category == "layout_spacing" and not _has_explicit_extra_space_evidence(original_text, reason, suggestion):
            continue
        item = IssueItem(
            issue_id=f"{para_id}-visual-{idx:03d}",
            issue_no="",
            para_id=para_id,
            category="layout",
            severity=_severity(raw.get("severity")),
            reason=reason,
            suggestion=suggestion,
            confidence=0.6,
            page_no=page_no,
            original_text=original_text,
            modified_text=modified_text,
            snippet=review_input_text[:240],
            review_input_text=review_input_text,
            review_round=0,
            bbox_list=[bbox],
            anchor={"visual_category": visual_category, "bbox": bbox},
        )
        out.append(item)

    dedup: Dict[str, IssueItem] = {}
    for issue in out:
        key = _issue_key(issue)
        if key not in dedup:
            dedup[key] = issue
    return list(dedup.values())


def build_visual_context(config: ReviewConfig) -> VisualReviewContext:
    api_key = (
        config.visual_api_key
        or os.getenv("DOC_REVIEW_VISUAL_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or os.getenv("QWEN_API_KEY", "").strip()
        or config.api_key
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("Missing API key for visual structure review.")
    model = (config.visual_model or config.model or "").strip()
    if not model:
        raise RuntimeError("Missing visual model for visual structure review.")
    api_base_url = (
        config.visual_api_base_url
        or os.getenv("DOC_REVIEW_VISUAL_BASE_URL", "").strip()
        or config.api_base_url
    )
    return VisualReviewContext(
        doc_id="",
        model=model,
        api_base_url=api_base_url,
        api_key=api_key,
        request_timeout=config.request_timeout,
        max_retries=config.max_retries,
    )


def review_visual_regions(
    *,
    regions: List[Dict[str, Any]],
    config: ReviewConfig,
    language: str = "zh",
) -> Tuple[List[IssueItem], List[Dict[str, Any]]]:
    """
    Review visual region screenshots and return visual IssueItems plus raw audit payload.
    """

    if not regions:
        return [], []
    ctx = build_visual_context(config)
    max_workers = max(1, min(len(regions), _env_int("DOC_REVIEW_VISUAL_MAX_CONCURRENCY", 5)))
    print(f"[VisualReview] batch regions={len(regions)} concurrency={max_workers}", flush=True)

    def _review_one(idx: int, region: Dict[str, Any]) -> Tuple[int, List[IssueItem], Dict[str, Any]]:
        image_path = Path(str(region.get("image_path", "")).strip())
        if not image_path.exists():
            return idx, [], {
                "region_id": region.get("region_id"),
                "group_id": region.get("group_id"),
                "raw": {},
                "issue_count": 0,
                "error": f"image not found: {image_path}",
            }
        page_no = int(region.get("page_no") or 0)
        bbox = region.get("bbox") or []
        text_hint = str(region.get("text_hint", "")).strip()
        para_id = f"visual-{idx:05d}"
        payload = {
            "region_id": region.get("region_id"),
            "group_id": region.get("group_id"),
            "group_type": region.get("group_type"),
            "page_no": page_no,
            "bbox": bbox,
            "child_regions": region.get("child_regions") or [],
            "boundary_text_hint": _boundary_text_from_region(region),
            "boundary_bbox": region.get("boundary_bbox") or {},
            "type": region.get("type", "unknown"),
            "text_hint": text_hint[:1200],
        }
        raw = _call_vlm(
            ctx=ctx,
            image_path=image_path,
            region_payload=payload,
            language=language,
        )
        parsed = parse_visual_issues_to_issue_items(
            raw_payload=raw,
            para_id=para_id,
            fallback_page_no=page_no,
            fallback_bbox=bbox,
            group_bbox=region.get("group_bbox") or bbox,
            review_input_text=text_hint,
            group_type=str(region.get("group_type") or ""),
            boundary_text_hint=_boundary_text_from_region(region),
        )
        audit = {
            "region_id": region.get("region_id"),
            "group_id": region.get("group_id"),
            "raw": raw,
            "issue_count": len(parsed),
        }
        return idx, parsed, audit

    results: List[Tuple[int, List[IssueItem], Dict[str, Any]]] = []
    if max_workers == 1:
        results = [_review_one(idx, region) for idx, region in enumerate(regions, start=1)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_review_one, idx, region)
                for idx, region in enumerate(regions, start=1)
            ]
            for future in as_completed(futures):
                results.append(future.result())

    issues: List[IssueItem] = []
    audit_rows: List[Dict[str, Any]] = []
    for _idx, parsed, audit in sorted(results, key=lambda row: row[0]):
        audit_rows.append(audit)
        for item in parsed:
            item.issue_id = f"{item.para_id}-r0-issue-{len(issues) + 1:03d}"
            item.issue_no = f"V{len(issues) + 1}"
            issues.append(item)
    return issues, audit_rows
