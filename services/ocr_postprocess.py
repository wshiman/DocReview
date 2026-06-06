from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

try:
    from ..models import ParagraphUnit, ParseResult, ReviewConfig
except ImportError:  # pragma: no cover
    from models import ParagraphUnit, ParseResult, ReviewConfig


def _log_enabled() -> bool:
    raw = os.getenv("DOC_REVIEW_OCR_POSTPROC_LOG", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _log(msg: str) -> None:
    if _log_enabled():
        print(f"[OcrPostprocess] {msg}", flush=True)


def _env_enabled(name: str, default: str = "off") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _strong_page_merge_enabled() -> bool:
    return _env_enabled("DOC_REVIEW_OCR_STRONG_PAGE_MERGE", default="on")


def _normalize_quote_style(s: str) -> str:
    # Unify quote style to reduce style-only diffs in downstream review.
    return s.translate(str.maketrans({"“": '"', "”": '"', "＂": '"', "‘": "'", "’": "'"}))


def _normalize_cjk_noise_spaces(s: str) -> str:
    # Best-effort OCR cleanup for accidental CJK char splitting.
    return re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", s)


def _normalize_spaces(s: str) -> str:
    # Keep English word spacing, compress obvious OCR spacing noise.
    s = s.replace("\u00a0", " ").replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    # Remove spaces around punctuation in CJK context only.
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+([，。；：！？、])", r"\1", s)
    s = re.sub(r"([（【《“‘])\s+", r"\1", s)
    s = re.sub(r"\s+([）】》”’])", r"\1", s)
    s = _normalize_quote_style(s)
    return s.strip()


def _is_english_dominant(s: str) -> bool:
    if not s:
        return False
    en = sum(1 for ch in s if ("a" <= ch.lower() <= "z"))
    zh = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return en >= max(8, zh * 2)


def _normalize_formula_spacing(s: str) -> str:
    # Normalize broken latex-like fragments: "$ \Delta $clbA" -> "$ \\Delta clbA $"
    s = re.sub(r"\$\s*\\Delta\s*\$\s*([A-Za-z0-9_+-]+)", r"$ \\Delta \1 $", s)
    s = re.sub(r"\$\s*\\gamma\s*\$", r"$ \\gamma $", s)
    s = re.sub(r"\$\s*\\([A-Za-z]+)\s*\$\s*([A-Za-z0-9_+-]+)", r"$ \\\1 \2 $", s)
    return s


def _normalize_ocr_punctuation(s: str) -> str:
    # Conservative punctuation normalization:
    # - never change English full-stop/comma to Chinese punctuation
    # - only collapse duplicated punctuation artifacts and strip accidental spaces.
    if not s:
        return s
    text = s
    text = re.sub(r"([,，.。;；:：!?！？、])\s+\1", r"\1", text)
    text = re.sub(r"([,，.。;；:：!?！？、]){3,}", lambda m: m.group(0)[:2], text)
    # Keep punctuation style as-is for English-dominant lines.
    if _is_english_dominant(text):
        return text
    return text


def _split_inline_heading(text: str) -> List[str]:
    s = text.strip()
    if not s:
        return []
    if "\n" in s:
        out: List[str] = []
        for line in s.splitlines():
            out.extend(_split_inline_heading(line))
        return out

    m = re.match(r"^(#{1,6}\s*)?(\d+(?:\.\d+)*\s+[\u4e00-\u9fffA-Za-z][^，。；：！？.!?]{1,24})\s+(.{20,})$", s)
    if not m:
        return [s]
    heading_prefix = m.group(1) or ""
    heading = (heading_prefix + m.group(2)).strip()
    rest = m.group(3).strip()
    if not rest or _is_toc_noise_line(rest):
        return [s]
    if re.search(r"[，。；：！？.!?]|[\u4e00-\u9fff]\(", rest):
        return [heading, rest]
    return [s]


def _is_toc_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("目录"):
        return True
    # Typical TOC pattern: title ...... 3
    if re.search(r"\.{3,}\s*\d+\s*$", s):
        return True
    # Dense section indices in a single line.
    if len(re.findall(r"\b\d+(?:\.\d+)+\b", s)) >= 2 and len(s) < 120:
        return True
    return False


def _looks_like_section_title(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) <= 80 and re.match(r"^(#{1,6}\s*)?(第[一二三四五六七八九十0-9]+[章节部分])", s):
        return True
    if len(s) <= 90 and re.match(r"^(#{1,6}\s*)?\d+(?:\.\d+)*\s+[\u4e00-\u9fffA-Za-z]", s):
        return True
    if len(s) <= 80 and re.match(r"^(Aim\s*\d+|Abstract|摘要|目录)", s, flags=re.I):
        return True
    return False


def _is_markdown_image(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r"^!\[[^\]]*\]\([^)]+\)\s*$", s)) or s.startswith("data:image/")


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|")


def _is_cover_like_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _looks_like_section_title(s):
        return True
    if len(s) <= 120 and re.search(r"(作品\s*ID|参赛学生类型|参加赛道|组别|自选项目组|请勿|全国大学生|创新设计竞赛)", s):
        return True
    if len(s) <= 80 and re.match(r"^(花为媒|药为信|内容完整性自查表|说明)\b", s):
        return True
    return False


def _is_cjk_open_fragment(text: str) -> bool:
    if not text:
        return False
    if text[-1] in "，、（(《“‘-":
        return True
    if text[-1] in "。！？.!?;；：:":
        return False
    # Long prose ending without punctuation is often a page-break fragment.
    return len(text) >= 40 and bool(re.search(r"[\u4e00-\u9fff]", text[-12:]))


def _looks_like_numeric_unit_continuation(a: str, b: str) -> bool:
    a_s = (a or "").strip()
    b_s = (b or "").lstrip()
    if not a_s or not b_s:
        return False
    if not re.search(r"\d(?:[\d,.]*\d)?$", a_s):
        return False
    units = (
        "次迭代",
        "次",
        "轮",
        "个",
        "年",
        "月",
        "日",
        "小时",
        "分钟",
        "秒",
        "阶段",
        "章",
        "节",
        "项",
        "条",
        "倍",
        "%",
    )
    return any(b_s.startswith(u) for u in units)


def _looks_like_short_page_continuation(a: str, b: str) -> bool:
    a_s = (a or "").strip()
    b_s = (b or "").strip()
    if not a_s or not b_s or len(b_s) > 28:
        return False
    continuation_heads = (
        "次",
        "并",
        "且",
        "而",
        "并且",
        "从而",
        "因此",
        "随后",
        "之后",
        "再",
        "再次",
        "这表明",
        "说明",
        "可见",
        "表明",
        "使得",
        "以便",
        "并在",
        "并将",
        "并可",
    )
    if any(b_s.startswith(h) for h in continuation_heads):
        return True
    if b_s[0] in "，、；：)）】》":
        return True
    return bool(re.search(r"[A-Za-z0-9]$", a_s) and re.match(r"^[a-z]", b_s))


def _should_merge(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if "\n" in a or "\n" in b:
        return False
    if _looks_like_section_title(a) or _looks_like_section_title(b):
        return False
    if _is_cover_like_line(a) or _is_cover_like_line(b):
        return False
    if _is_markdown_image(a) or _is_markdown_image(b):
        return False
    if _is_table_line(a) or _is_table_line(b):
        return False
    if _is_toc_noise_line(a) or _is_toc_noise_line(b):
        return False

    a_end = a[-1]
    if _strong_page_merge_enabled():
        if _looks_like_numeric_unit_continuation(a, b):
            return True
        if a_end not in "。！？.!?;；：" and _looks_like_short_page_continuation(a, b):
            return True
    # Handle mid-token page break: "n1p" + "I..."
    if re.search(r"[A-Za-z0-9]$", a) and re.match(r"^[A-Za-z0-9]", b):
        return True
    # If a sentence is visibly unfinished, merge as a page-break continuation.
    if _is_cjk_open_fragment(a):
        return True
    return False


def _merge_paragraph_texts(paragraphs: List[ParagraphUnit]) -> List[Tuple[str, List[ParagraphUnit]]]:
    groups: List[Tuple[str, List[ParagraphUnit]]] = []
    current_text = ""
    current_src: List[ParagraphUnit] = []

    for p in paragraphs:
        if p.anchor.get("reviewable") is False:
            if current_text:
                groups.append((current_text, current_src))
                current_text = ""
                current_src = []
            groups.append((p.text.strip(), [p]))
            continue
        normalized = _normalize_formula_spacing(_normalize_spaces(p.text))
        normalized = _normalize_ocr_punctuation(normalized)
        for t in _split_inline_heading(normalized):
            if not t:
                continue
            t = _normalize_cjk_noise_spaces(t)
            if not current_text:
                current_text = t
                current_src = [p]
                continue

            if _should_merge(current_text, t):
                if _looks_like_numeric_unit_continuation(current_text, t):
                    joiner = ""
                else:
                    joiner = "" if (re.search(r"[A-Za-z0-9]$", current_text) and re.match(r"^[A-Za-z0-9]", t)) else " "
                current_text = (current_text + joiner + t).strip()
                current_src.append(p)
            else:
                groups.append((current_text, current_src))
                current_text = t
                current_src = [p]

    if current_text:
        groups.append((current_text, current_src))
    return groups


def _filter_toc_noise(groups: List[Tuple[str, List[ParagraphUnit]]]) -> List[Tuple[str, List[ParagraphUnit]]]:
    out: List[Tuple[str, List[ParagraphUnit]]] = []
    for text, src in groups:
        # Drop obvious TOC lines. Keep only a standalone title like "目录".
        if _is_toc_noise_line(text):
            s = text.strip()
            if s in {"目录", "# 目录"}:
                out.append((text, src))
                continue
            if re.search(r"\.{3,}|\d", s):
                continue
            if not _looks_like_section_title(s):
                continue
        out.append((text, src))
    return out


def _pick_group_page_no(src_group: List[ParagraphUnit]) -> Optional[int]:
    for unit in src_group:
        if unit.page_no:
            return int(unit.page_no)
        anchor = unit.anchor if isinstance(unit.anchor, dict) else {}
        raw_page = anchor.get("page_no")
        try:
            page_no = int(raw_page)
        except Exception:
            page_no = 0
        if page_no > 0:
            return page_no
    return None


def _safe_json_extract(text: str) -> Optional[Dict]:
    text = text.strip()
    try:
        x = json.loads(text)
        if isinstance(x, dict):
            return x
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        x = json.loads(m.group(0))
        return x if isinstance(x, dict) else None
    except json.JSONDecodeError:
        return None


def _llm_refine_chunk(chunk_text: str, config: ReviewConfig) -> str:
    api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return chunk_text
    if api_key.strip().lower() in {"dummy", "test", "none", "null"}:
        return chunk_text

    prompt = {
        "instruction": (
            "你是OCR后处理助手。只做版式整合，不做内容改写。"
            "任务：合并被分页断开的段落、清理目录/页码噪声、修复明显空格噪声与断裂公式。"
            "必须保留封面/标题页的原始换行；章节标题、小节标题必须单独成段，不能和正文合并。"
            "禁止改变事实、禁止新增观点、禁止语义改写。"
        ),
        "input_markdown": chunk_text,
        "output_schema": {"clean_markdown": "string"},
    }
    req = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    url = f"{config.api_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=req, timeout=max(60, config.request_timeout))
    if resp.status_code >= 400:
        raise RuntimeError(f"llm refine http {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = _safe_json_extract(content)
    refined = str((parsed or {}).get("clean_markdown", "")).strip()
    return refined if refined else chunk_text


def _stitch_with_overlap(prev_text: str, next_text: str, overlap_chars: int) -> str:
    if not prev_text:
        return next_text
    if not next_text:
        return prev_text
    max_k = min(len(prev_text), len(next_text), overlap_chars)
    # Prefer larger overlap for stable merge.
    for k in range(max_k, 79, -1):
        if prev_text[-k:] == next_text[:k]:
            return prev_text + next_text[k:]
    return prev_text.rstrip() + "\n\n" + next_text.lstrip()


def _optional_llm_refine(text: str, config: ReviewConfig) -> str:
    mode = os.getenv("DOC_REVIEW_OCR_LLM_REFINE", "off").strip().lower()
    if mode not in {"on", "1", "true"}:
        return text

    max_chars = int(os.getenv("DOC_REVIEW_OCR_LLM_REFINE_MAX_CHARS", "14000"))
    overlap_chars = int(os.getenv("DOC_REVIEW_OCR_LLM_REFINE_OVERLAP", "1200"))
    overlap_chars = max(200, min(overlap_chars, max_chars // 2))
    step = max(1000, max_chars - overlap_chars)

    try:
        started = time.time()
        if len(text) <= max_chars:
            refined = _llm_refine_chunk(text, config)
            _log(f"llm_refine applied windows=1 elapsed={time.time() - started:.2f}s chars={len(text)}")
            return refined

        chunks: List[str] = []
        for start in range(0, len(text), step):
            chunks.append(text[start:start + max_chars])
            if start + max_chars >= len(text):
                break

        merged = ""
        for idx, chunk in enumerate(chunks, start=1):
            refined_chunk = _llm_refine_chunk(chunk, config)
            merged = _stitch_with_overlap(merged, refined_chunk, overlap_chars)
            _log(f"llm_refine window={idx}/{len(chunks)} in_chars={len(chunk)} out_chars={len(refined_chunk)}")
        _log(
            f"llm_refine applied windows={len(chunks)} overlap={overlap_chars} "
            f"elapsed={time.time() - started:.2f}s total_chars={len(text)}"
        )
        return merged
    except Exception as exc:
        _log(f"llm_refine skipped due to error={exc}")
        return text


def preprocess_parse_result(parse_result: ParseResult, config: ReviewConfig) -> ParseResult:
    if parse_result.source_type != "pdf":
        return parse_result
    if not parse_result.metadata.get("markdown_used"):
        return parse_result

    groups = _merge_paragraph_texts(parse_result.paragraphs)
    groups = _filter_toc_noise(groups)
    if not groups:
        return parse_result

    merged_text = "\n\n".join([g[0] for g in groups]).strip()
    merged_text = _optional_llm_refine(merged_text, config)

    final_blocks = [x.strip() for x in re.split(r"\n\s*\n+", merged_text) if x.strip()]
    block_page_map: Dict[str, int] = {}
    source_page_candidates: List[tuple[str, int]] = []
    for text, src_group in groups:
        page_no = _pick_group_page_no(src_group)
        if page_no:
            block_page_map[text.strip()] = page_no
            source_page_candidates.append((text.strip(), page_no))

    new_paras: List[ParagraphUnit] = []
    for i, block in enumerate(final_blocks, start=1):
        page_no = block_page_map.get(block)
        if page_no is None:
            compact_block = "".join(block.split())
            for src_text, src_page_no in source_page_candidates:
                compact_src = "".join(src_text.split())
                if compact_block and (compact_block in compact_src or compact_src in compact_block):
                    page_no = src_page_no
                    break
        new_paras.append(
            ParagraphUnit(
                doc_id=parse_result.doc_id,
                para_id=f"para-{i:05d}",
                page_no=page_no,
                text=block,
                bbox_list=[],
                source_type="pdf",
                anchor={
                    "kind": "pdf_paddle_postprocessed_image" if _is_markdown_image(block) else "pdf_paddle_postprocessed",
                    "paragraph_index": i - 1,
                    "page_no": page_no,
                    "reviewable": not _is_markdown_image(block),
                },
            )
        )

    parse_result.paragraphs = new_paras
    parse_result.metadata = dict(parse_result.metadata)
    parse_result.metadata["postprocessed"] = True
    parse_result.metadata["postprocess_paragraph_count"] = len(new_paras)
    parse_result.metadata["raw_markdown"] = merged_text
    _log(f"postprocess done paragraph_count={len(new_paras)}")
    return parse_result
