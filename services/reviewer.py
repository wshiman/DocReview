from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

try:
    from ..models import IssueItem, ParagraphUnit, ReviewConfig, SEVERITY_ORDER
except ImportError:  # pragma: no cover
    from models import IssueItem, ParagraphUnit, ReviewConfig, SEVERITY_ORDER

_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_LOCK = threading.Lock()
_PROMPT_CACHE_VERSION = "content-review-v11-word-formula-openxml-note"


def _log_level() -> str:
    level = os.getenv("DOC_REVIEW_LOG_LEVEL", "verbose").strip().lower()
    if level not in {"verbose", "normal", "error"}:
        return "verbose"
    return level


def _should_log(level: str) -> bool:
    current = _log_level()
    order = {"error": 1, "normal": 2, "verbose": 3}
    return order[current] >= order[level]


def _log(level: str, msg: str) -> None:
    if _should_log(level):
        print(msg, flush=True)


def _env_enabled(name: str, default: str = "off") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _quote_style_filter_enabled() -> bool:
    return _env_enabled("DOC_REVIEW_FILTER_QUOTE_STYLE_ISSUES", default="on")


def _formula_filter_enabled() -> bool:
    return _env_enabled("DOC_REVIEW_FILTER_FORMULA_ISSUES", default="on")


def _split_artifact_filter_enabled() -> bool:
    return _env_enabled("DOC_REVIEW_FILTER_SPLIT_ARTIFACTS", default="on")


def _detect_language(text: str) -> str:
    zh_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en_chars = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    return "zh" if zh_chars >= en_chars else "en"


def _severity_pass(item: IssueItem, threshold: str) -> bool:
    return SEVERITY_ORDER[item.severity] >= SEVERITY_ORDER[threshold]


def _safe_json_extract(text: str) -> Optional[Dict]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    for chunk in fenced:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue

    brace = re.search(r"\{.*\}", text, flags=re.S)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _build_cache_key(config: ReviewConfig, paragraph: ParagraphUnit, context_payload: Dict) -> str:
    material = {
        "prompt_version": _PROMPT_CACHE_VERSION,
        "model": config.model,
        "lang": config.language_mode,
        "ignore_cjk_punctuation_width": bool(getattr(config, "ignore_cjk_punctuation_width", False)),
        "review_word_formulas": bool(getattr(config, "review_word_formulas", False)),
        "para_id": paragraph.para_id,
        "text": paragraph.text,
        "context": context_payload,
    }
    payload = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_get(cache_key: str) -> Optional[Dict]:
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    with cache_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cache_set(cache_key: str, data: Dict) -> None:
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    with _CACHE_LOCK:
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _build_system_prompt(
    language: str,
    ignore_cjk_punctuation_width: bool = False,
    review_word_formulas: bool = False,
) -> str:
    zh_width_guard = (
        "PDF文本审查规则：不要仅因中文语境中使用英文半角逗号、句号、冒号、分号、括号，"
        "或英文语境中使用中文全角标点而报告问题；中英文标点风格/全半角差异交给视觉审查处理。"
        "但仍必须报告明确错误符号，例如连续两个逗号/句号、重复混用标点、缺少句末标点、明显破坏句意的标点。\n"
    )
    en_width_guard = (
        "PDF text-review rule: do not report issues solely because Chinese prose uses ASCII punctuation "
        "or English prose uses full-width/CJK punctuation; punctuation-width/style differences are handled by visual review. "
        "Still report clearly faulty punctuation such as duplicated commas/periods, repeated mixed punctuation, missing final punctuation, "
        "or punctuation that clearly breaks meaning.\n"
    )
    zh_formula_rule = (
        "Word公式审查规则：段落中的 [公式: ...] 来自 DOCX/OpenXML(OMML) 公式文本化解析，"
        "可能丢失二维排版、括号、上下标视觉层级或编号位置。允许结合上下文理解公式大意，"
        "只报告高置信的明显可见格式问题：上标/下标标记明显缺失或混乱、编号不一致、符号前后不一致、公式文本明显残缺；"
        "不要判断数学推导是否正确，不要把疑似解析伪影当成确定错误。\n"
        if review_word_formulas
        else "公式 token 与 LaTeX 符号默认不审，不要基于公式字符给出纠错。"
    )
    en_formula_rule = (
        "Word formula rule: [Formula: ...] or [公式: ...] text is serialized from DOCX/OpenXML(OMML) equations, "
        "so two-dimensional layout, brackets, superscript/subscript visual hierarchy, or equation-number position may be lost. "
        "You may use context to understand the formula roughly, but report only high-confidence obvious visible/textual format issues: "
        "broken superscript/subscript notation, equation-number inconsistency, symbol inconsistency, or clearly malformed formula text. "
        "Do not judge mathematical correctness and do not treat likely parser artifacts as certain document errors. "
        if review_word_formulas
        else "Skip formula-token-level proofreading (LaTeX/math symbols). "
    )
    if language == "zh":
        return (
            "你是高标准文档审校助手。必须做细粒度审查：逐句阅读、逐项核对，不漏报同段内多个独立问题。"
            "仅报告会影响语义、专业性、可读性、前后一致性的实质问题。"
            "允许类别：typo, grammar, logic, terminology, punctuation。"
            "标点审查必须包含：句末标点缺失、重复标点、明显错用标点、并列/停顿关系标点不当。"
            + (zh_width_guard if ignore_cjk_punctuation_width else "")
            + "强约束：不要把排版/OCR/Markdown/LaTeX带来的无害空白当错误；"
            "如跨页断行、目录残留、公式周围空格、全半角空白差异，默认不报。"
            "跨页断段/分页残句默认不做补全审查；仅在同一段内有充分证据时，才允许报缺失。"
            "引号样式（“”/\"\"/‘’/''）差异默认不报。"
            f"{zh_formula_rule}"
            "仅当空格或标点实质改变含义或破坏术语时，才允许报告。"
            "输出必须是 JSON 对象，禁止额外解释文本。"
        )
    return (
        "You are a high-precision document reviewer. Perform fine-grained sentence-level checks and report "
        "all distinct substantive issues in the paragraph. Allowed categories: typo, grammar, logic, "
        "terminology, punctuation. Punctuation checks must include missing sentence-final punctuation, "
        "duplicated punctuation, clear punctuation misuse, and incorrect pause/coordination punctuation. "
        + (en_width_guard if ignore_cjk_punctuation_width else "")
        + "Do not report harmless OCR/layout/Markdown/LaTeX whitespace artifacts (line/page breaks, TOC "
        "remnants, formula spacing, full-width/half-width whitespace) unless meaning is changed or a "
        "domain term is broken. Do not treat cross-page split fragments as completion tasks unless sufficient "
        "evidence exists in the same paragraph. Ignore quote-style-only differences by default "
        f"(“”/\"\"/‘’/''), and {en_formula_rule}"
        "Output must be a JSON object only."
    )


def _build_user_prompt(
    paragraph: ParagraphUnit,
    context_before: Sequence[str],
    context_after: Sequence[str],
    rolling_summary: str,
    glossary: Dict[str, str],
    language: str,
    ignore_cjk_punctuation_width: bool = False,
    review_word_formulas: bool = False,
) -> str:
    glossary_json = json.dumps(glossary, ensure_ascii=False)
    schema = {
        "issues": [
            {
                "category": "typo|grammar|logic|terminology|punctuation",
                "reason": "string",
                "suggestion": "string",
                "original_text": "string",
                "modified_text": "string",
            }
        ],
        "summary_update": "string",
        "glossary_updates": {"term": "preferred_term"},
    }

    if language == "zh":
        punctuation_width_rule = (
            "【PDF标点规则】不要仅因中英文标点全半角/风格差异报错，例如中文句子里出现英文逗号、句号、括号本身不作为文本审查问题；"
            "这类差异交给视觉审查。仍需报告明确错误符号，如连续两个逗号/句号、重复混用标点、缺少句末标点、标点明显破坏句意。\n"
            if ignore_cjk_punctuation_width
            else ""
        )
        formula_rule = (
            "【Word公式规则】段落中的 [公式: ...] 来自 DOCX/OpenXML(OMML) 公式文本化解析，"
            "可能丢失二维排版、括号、上下标视觉层级或编号位置。你可以结合上下文理解公式大意，"
            "但只报告高置信明显问题：上标/下标标记明显缺失或混乱、公式编号前后不一致、符号命名前后不一致、公式文本明显残缺。"
            "不要判断数学推导或公式本身是否正确，不要把疑似解析伪影当成确定错误。\n"
            if review_word_formulas
            else "【公式规则】含 LaTeX/数学符号的 token 默认不审，不基于公式字符给出纠错。\n"
        )
        instruction = (
            "你仅输出标准JSON，禁止多余解释、备注、markdown标记。\n"
            "【审校粒度】逐句逐项检查，且同一段可返回多条错误；不要在发现一条后停止。\n"
            "【必查清单】\n"
            "1) 笔误与错别字；\n"
            "2) 语法/搭配/句式不通顺；\n"
            "3) 段内及与上下文的逻辑矛盾、指代不清、时态/语气冲突；\n"
            "4) 专业术语、缩写、专名前后不一致；\n"
            "5) 标点问题：句末应有而缺失、重复标点、明显错用、并列停顿关系错误。\n"
            "【豁免规则】忽略排版/OCR/格式造成的无害空白：换行、分页、目录页码残留、Markdown/LaTeX空格、标题字间空格、公式两侧空格。仅当其改变语义或破坏术语时才可报告。\n"
            "【跨页规则】不要把跨页断段或分页残句当作补全任务；只有在当前段落内部证据充分时，才允许提出缺失补全。\n"
            "【引号规则】“”/\"\"/‘’/'' 的样式差异默认不报。\n"
            f"{formula_rule}"
            f"{punctuation_width_rule}"
            "【原文片段规则】每条错误必填 original_text，且必须是当前段落中的连续原文最小错误片段。若仅2~4字易重复，扩充至6~20字连续原文以便唯一定位。禁止改写、禁止编造。\n"
            "【可替换文本规则】每条错误必填 modified_text，只写可直接替换 original_text 的修正文案；不能写“删除/改为/建议”等说明文字，不能包含与 original_text 无关的上下文。\n"
            "例如 original_text 为“IEEE 多媒体 IEEE MultiMedia 国际期刊”时，modified_text 只能是“IEEE MultiMedia 国际期刊”，不能是“删除重复中文名称，改为……”。\n"
            "【标点与空格替换】若标点重复/错用，original_text 必须包含错误标点本身，modified_text 必须包含修正后的标点；"
            "若缺少句末标点，original_text 应包含句末附近连续原文，modified_text 是补齐标点后的同一片段；"
            "若确有语义性空格错误，original_text/modified_text 必须保留必要空格差异，但普通 OCR 空白噪声不得上报。\n"
            "【输出完整性】issues 覆盖该段所有高置信问题；不要把多个独立错误合并成一条笼统描述。\n"
            "若无问题，issues 返回空数组。"
        )
    else:
        punctuation_width_rule = (
            "[PDF punctuation rule] Do not report punctuation issues solely because of ASCII-vs-CJK/full-width punctuation style differences; visual review handles those. Still report clearly faulty punctuation such as duplicated commas/periods, repeated mixed punctuation, missing final punctuation, or punctuation that clearly breaks meaning.\n"
            if ignore_cjk_punctuation_width
            else ""
        )
        formula_rule = (
            "[Word Formula Rule] [Formula: ...] or [公式: ...] text is serialized from DOCX/OpenXML(OMML) equations. "
            "Two-dimensional layout, brackets, superscript/subscript visual hierarchy, or equation-number position may be lost. "
            "Use context only to understand the formula roughly, and report only high-confidence obvious issues: broken superscript/subscript notation, "
            "equation-number inconsistency, symbol inconsistency, or clearly malformed formula text. "
            "Do not judge mathematical correctness and do not treat likely parser artifacts as certain document errors.\n"
            if review_word_formulas
            else "[Formula Rule] Skip formula-token-level proofreading for LaTeX/math symbols.\n"
        )
        instruction = (
            "Output only standard JSON with no extra text, comments or markdown formatting.\n"
            "[Granularity] Review sentence by sentence and report multiple distinct issues if present; do not stop after the first one.\n"
            "[Checklist] Check five categories only: typo, grammar, logic, terminology consistency, punctuation (missing sentence-final punctuation, duplicated punctuation, punctuation misuse, wrong coordination punctuation).\n"
            "[Context] Use context_before/context_after to detect cross-sentence and cross-paragraph inconsistencies (reference, tense/mood, claim contradiction, terminology drift).\n"
            "[Exemption Rule] Ignore harmless OCR/layout whitespace artifacts: line/page breaks, TOC residues, Markdown/LaTeX spaces, spaces between title characters, spaces around formulas. Report whitespace only if meaning changes or a proper/domain term is broken.\n"
            "[Cross-page Rule] Do not treat cross-page split fragments as completion tasks. Suggest completion only with sufficient evidence within the same paragraph.\n"
            "[Quote Rule] Ignore quote-style-only differences (\"\"/“”/''/‘’) by default.\n"
            f"{formula_rule}"
            f"{punctuation_width_rule}"
            "[Original Text Rule] original_text must be an exact contiguous verbatim span from current paragraph, minimal faulty range. If 2-4 chars are ambiguous, expand to 6-20 contiguous chars for unique localization. Never paraphrase or fabricate.\n"
            "[Replacement Rule] modified_text is required for every issue and must be the exact replacement for original_text only. Do not include instructions such as delete/change to/suggest; do not include unrelated context.\n"
            "Example: if original_text is 'IEEE Multimedia IEEE MultiMedia journal', modified_text must be 'IEEE MultiMedia journal', not an explanatory sentence. For missing final punctuation, original_text should include the sentence ending and modified_text should be the same ending with punctuation added.\n"
            "[Punctuation/Whitespace Replacement] For duplicated or misused punctuation, original_text must include the faulty punctuation and modified_text must contain the corrected punctuation. For semantic whitespace errors, preserve the exact required whitespace difference in original_text/modified_text; never report ordinary OCR spacing noise.\n"
            "[Completeness] issues should cover all high-confidence errors in this paragraph, not a single summarized issue.\n"
            "If no issues, return an empty issues list."
        )
    formula_parse_note = ""
    if review_word_formulas:
        kind = str(paragraph.anchor.get("kind", ""))
        parser_name = "openxml_fallback" if kind == "paragraph_xml" else "python-docx/openxml"
        formula_texts = paragraph.anchor.get("docx_meta", {}).get("formula_texts", [])
        if formula_texts:
            formula_parse_note = (
                f"Formula text was generated by DOCX OpenXML/OMML serialization ({parser_name}); "
                "treat it as a lossy text representation, not as the exact visual equation."
            )

    payload = {
        "instruction": instruction,
        "context_before": list(context_before),
        "current_paragraph": paragraph.text,
        "formula_parse_note": formula_parse_note,
        "context_after": list(context_after),
        "rolling_summary": rolling_summary,
        "glossary": glossary,
        "required_schema": schema,
    }
    return f"{json.dumps(payload, ensure_ascii=False, indent=2)}\nGlossary Snapshot: {glossary_json}"


def _call_deepseek(
    prompt: str,
    system_prompt: str,
    config: ReviewConfig,
    session: requests.Session,
) -> Dict:
    api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or config.api_key.")

    url = f"{config.api_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    last_err: Optional[Exception] = None
    for attempt in range(config.max_retries):
        started = time.time()
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=config.request_timeout)
            if resp.status_code >= 500:
                raise RuntimeError(f"DeepSeek server error {resp.status_code}: {resp.text[:400]}")
            if resp.status_code >= 400:
                raise RuntimeError(f"DeepSeek request failed {resp.status_code}: {resp.text[:400]}")

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = _safe_json_extract(content)
            if not isinstance(parsed, dict) or "issues" not in parsed:
                raise ValueError("Model returned invalid JSON schema")
            cost = time.time() - started
            raw_preview = content if len(content) <= 800 else f"{content[:800]}...(truncated)"
            _log("verbose", f"[DeepSeek] success model={config.model} attempt={attempt + 1} elapsed={cost:.2f}s raw={raw_preview}")
            return parsed
        except Exception as exc:
            last_err = exc
            cost = time.time() - started
            _log("error", f"[DeepSeek] failed model={config.model} attempt={attempt + 1}/{config.max_retries} elapsed={cost:.2f}s error={exc}")
            sleep_s = min(8.0, 2 ** attempt)
            time.sleep(sleep_s)

    raise RuntimeError(f"DeepSeek call failed after retries: {last_err}")


def _normalize_issue(
    raw: Dict,
    paragraph: ParagraphUnit,
    idx: int,
    review_round: int = 1,
) -> Optional[IssueItem]:
    try:
        category = str(raw.get("category", "grammar")).strip().lower()
        if category not in {"typo", "grammar", "logic", "terminology", "punctuation"}:
            category = "grammar"

        reason = str(raw.get("reason", "")).strip()
        suggestion = str(raw.get("suggestion", "")).strip()
        original_text = str(raw.get("original_text", "")).strip()
        modified_text = str(raw.get("modified_text", "")).strip()
        if not reason or not suggestion or not original_text or not modified_text:
            _log(
                "normal",
                f"[Review] para={paragraph.para_id} drop_issue reason=missing_required_field "
                f"has_reason={bool(reason)} has_suggestion={bool(suggestion)} "
                f"has_original={bool(original_text)} has_modified={bool(modified_text)}",
            )
            return None

        if original_text not in paragraph.text:
            _log(
                "normal",
                f"[Review] para={paragraph.para_id} drop_issue reason=original_text_not_in_paragraph original_text={original_text[:80]}",
            )
            return None

        return IssueItem(
            issue_id=f"{paragraph.para_id}-issue-{idx:03d}",
            issue_no="",
            para_id=paragraph.para_id,
            category=category,
            severity="medium",
            reason=reason,
            suggestion=suggestion,
            confidence=0.5,
            page_no=paragraph.page_no,
            original_text=original_text,
            modified_text=modified_text,
            snippet=_build_issue_snippet(paragraph.text, original_text),
            review_input_text=paragraph.text,
            review_round=review_round,
        )
    except Exception:
        return None


def _contains_formula_token(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"\$[^$]*\$", t):
        return True
    if re.search(r"\\[A-Za-z]+", t):
        return True
    if re.search(r"[A-Za-z0-9]+_[A-Za-z0-9]+|[A-Za-z0-9]+\^[A-Za-z0-9]+", t):
        return True
    if re.search(r"[∑∏√∞≈≠≤≥α-ωΑ-Ω]", t):
        return True
    # OCR-confusable short math-like token, e.g. 1/l, l_j, O(n).
    if re.fullmatch(r"[A-Za-z0-9]{1,3}[/=+\-][A-Za-z0-9]{1,3}", t):
        return True
    return False


def _normalize_quote_style_text(text: str) -> str:
    t = (text or "").strip()
    return t.translate(str.maketrans({"“": '"', "”": '"', "＂": '"', "‘": "'", "’": "'"}))


def _is_quote_style_only_change(original_text: str, modified_text: str) -> bool:
    o = (original_text or "").strip()
    m = (modified_text or "").strip()
    if not o or not m:
        return False
    if o == m:
        return False
    if _normalize_quote_style_text(o) != _normalize_quote_style_text(m):
        return False
    return o != m


def _looks_like_orphan_split_fragment(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if len(t) <= 6 and bool(re.search(r"[。！？.!?]$", t)):
        return True
    if len(t) <= 10 and t[0] in "，、；：)）】》":
        return True
    return False


def _looks_like_split_completion_issue(issue: IssueItem, paragraphs_by_id: Dict[str, ParagraphUnit]) -> bool:
    para = paragraphs_by_id.get(issue.para_id)
    if para is None:
        return False
    original = issue.original_text.strip()
    modified = issue.modified_text.strip()
    if not original or not modified:
        return False
    if original == modified:
        return False
    # Classic completion shape: modified extends original and keeps prefix.
    if not modified.startswith(original):
        return False
    suffix = modified[len(original):].strip()
    if not suffix:
        return False
    # If completion starts with boundary-like token, treat as split artifact.
    boundary_heads = ("次", "并", "且", "而", "因此", "随后", "之后", "这表明", "说明", "可见", "表明")
    if any(suffix.startswith(x) for x in boundary_heads):
        return True
    if original.endswith(tuple("0123456789")) and re.match(r"^(次|轮|个|年|月|日|小时|分钟|秒|%)", suffix):
        return True
    # Paragraph boundary hint: issue points to head/tail tiny fragment.
    text = para.text
    pos = text.find(original)
    if pos >= 0:
        near_tail = pos + len(original) >= len(text) - 6
        near_head = pos <= 6
        if (near_tail or near_head) and (len(original) <= 12 or _looks_like_orphan_split_fragment(original)):
            return True
    return False


def _is_issue_at_paragraph_head(issue: IssueItem, paragraph: ParagraphUnit, max_offset: int = 12) -> bool:
    text = paragraph.text or ""
    original = issue.original_text.strip()
    if not text or not original:
        return False
    pos = text.find(original)
    return pos >= 0 and pos <= max_offset


def _looks_like_completion_language(issue: IssueItem) -> bool:
    material = f"{issue.reason} {issue.suggestion}".lower()
    zh_tokens = (
        "缺少",
        "缺失",
        "遗漏",
        "漏写",
        "不完整",
        "补全",
        "补齐",
        "补写",
        "省略",
        "主语缺失",
        "宾语缺失",
        "对象缺失",
        "补充对象",
        "缺少对象",
        "补充主语",
        "补充宾语",
        "句子不完整",
        "语义不完整",
        "多余",
        "残留",
        "疑似残留",
        "删除",
    )
    en_tokens = (
        "missing",
        "incomplete",
        "supplement object",
        "complete the sentence",
        "subject missing",
        "object missing",
    )
    if any(tok in material for tok in zh_tokens):
        return True
    if any(tok in material for tok in en_tokens):
        return True
    original = issue.original_text.strip()
    modified = issue.modified_text.strip()
    if original and modified and modified.startswith(original) and len(modified) > len(original):
        return True
    return False


def _looks_like_head_orphan_trim_issue(issue: IssueItem) -> bool:
    original = issue.original_text.strip()
    modified = issue.modified_text.strip()
    if not original or not modified or original == modified:
        return False
    if not original.endswith(modified):
        return False
    removed = original[: len(original) - len(modified)].strip()
    if not removed:
        return False
    if len(removed) > 20:
        return False
    # Require clear sentence-boundary trimming, avoid swallowing punctuation-fix issues.
    if not bool(re.search(r"[，。；：!?！？]$", removed)):
        return False
    return bool(re.search(r"[。！？!?].+", original))


def _looks_like_head_prefix_completion_issue(issue: IssueItem) -> bool:
    original = issue.original_text.strip()
    modified = issue.modified_text.strip()
    if not original or not modified or original == modified:
        return False
    if not modified.endswith(original):
        return False
    prefix = modified[: len(modified) - len(original)].strip()
    if not prefix:
        return False
    if len(prefix) > 32:
        return False
    if original[0] in "（([【{<0123456789":
        return True
    return bool(re.match(r"^(与|和|在|对|将|由|从|向|就|跟)", prefix))


def _looks_like_head_completion_issue(issue: IssueItem, paragraphs_by_id: Dict[str, ParagraphUnit]) -> bool:
    if issue.category in {"punctuation", "terminology"}:
        return False
    para = paragraphs_by_id.get(issue.para_id)
    if para is None:
        return False
    if not _is_issue_at_paragraph_head(issue, para):
        return False
    original = issue.original_text.strip()
    modified = issue.modified_text.strip()
    if not _looks_like_completion_language(issue):
        return False
    if _looks_like_head_orphan_trim_issue(issue):
        return True
    if _looks_like_head_prefix_completion_issue(issue):
        return True
    # Strong suppression for head short-fragment completion, typical OCR page split.
    if len(original) <= 18 and (original.endswith("。") or original.endswith(".") or original.endswith("，")):
        return True
    if _looks_like_orphan_split_fragment(original):
        return True
    if original and modified and modified.startswith(original) and len(modified) - len(original) <= 24:
        return True
    return False


def _filter_issue_artifacts(
    issues: List[IssueItem],
    paragraphs_by_id: Dict[str, ParagraphUnit],
    allow_formula_issues: bool = False,
) -> Tuple[List[IssueItem], List[Tuple[IssueItem, str]]]:
    kept: List[IssueItem] = []
    dropped: List[Tuple[IssueItem, str]] = []
    for issue in issues:
        if not allow_formula_issues and _formula_filter_enabled() and (
            _contains_formula_token(issue.original_text) or _contains_formula_token(issue.modified_text)
        ):
            _log("normal", f"[Review] para={issue.para_id} drop_issue reason=formula_artifact issue={issue.issue_id}")
            dropped.append((issue, "formula_artifact"))
            continue
        if _quote_style_filter_enabled() and _is_quote_style_only_change(issue.original_text, issue.modified_text):
            _log("normal", f"[Review] para={issue.para_id} drop_issue reason=quote_style_only issue={issue.issue_id}")
            dropped.append((issue, "quote_style_only"))
            continue
        if _split_artifact_filter_enabled():
            if _looks_like_head_completion_issue(issue, paragraphs_by_id):
                _log(
                    "normal",
                    f"[Review] para={issue.para_id} drop_issue reason=head_completion_artifact issue={issue.issue_id}",
                )
                dropped.append((issue, "head_completion_artifact"))
                continue
            if _looks_like_split_completion_issue(issue, paragraphs_by_id):
                _log(
                    "normal",
                    f"[Review] para={issue.para_id} drop_issue reason=split_completion_artifact issue={issue.issue_id}",
                )
                dropped.append((issue, "split_completion_artifact"))
                continue
            if _looks_like_orphan_split_fragment(issue.original_text) and len(issue.modified_text.strip()) <= 24:
                _log(
                    "normal",
                    f"[Review] para={issue.para_id} drop_issue reason=orphan_fragment_artifact issue={issue.issue_id}",
                )
                dropped.append((issue, "orphan_fragment_artifact"))
                continue
        kept.append(issue)
    return kept, dropped


def _coalesce_filtered_issues(entries: List[Tuple[IssueItem, str]]) -> List[Tuple[IssueItem, str]]:
    dedup: Dict[str, Tuple[IssueItem, str]] = {}
    for issue, reason in entries:
        key_material = (
            f"{issue.para_id}|{issue.category}|{issue.original_text.strip()}|"
            f"{issue.modified_text.strip()}|{reason}|{issue.review_round}"
        )
        key = hashlib.sha1(key_material.encode("utf-8")).hexdigest()
        if key not in dedup:
            dedup[key] = (issue, reason)
    return list(dedup.values())


def _build_issue_snippet(paragraph_text: str, original_text: str, radius: int = 120) -> str:
    text = paragraph_text or ""
    original = original_text or ""
    if not text:
        return ""
    if not original:
        return text[:180]
    pos = text.find(original)
    if pos < 0:
        return text[:180]
    start = max(0, pos - radius)
    end = min(len(text), pos + len(original) + radius)
    return text[start:end].strip()


def _coalesce_issues(issues: List[IssueItem]) -> List[IssueItem]:
    dedup: Dict[str, IssueItem] = {}
    for item in issues:
        key_material = (
            f"{item.para_id}|{item.category}|{item.original_text.strip()}|"
            f"{item.modified_text.strip()}|{item.reason.strip().lower()}"
        )
        key = hashlib.sha1(key_material.encode("utf-8")).hexdigest()
        if key not in dedup:
            dedup[key] = item

    # Re-index for stable IDs after dedupe.
    grouped: Dict[str, List[IssueItem]] = {}
    for issue in dedup.values():
        grouped.setdefault(issue.para_id, []).append(issue)

    normalized: List[IssueItem] = []
    for para_id, group in grouped.items():
        for idx, issue in enumerate(sorted(group, key=lambda x: (x.category, x.issue_id)), start=1):
            issue.issue_id = f"{para_id}-issue-{idx:03d}"
            normalized.append(issue)
    return normalized


def _apply_glossary_normalization(text: str, glossary: Dict[str, str]) -> str:
    normalized = text
    for term, preferred in glossary.items():
        if not term or not preferred:
            continue
        if re.search(r"[A-Za-z]", term):
            pattern = re.compile(rf"\b{re.escape(term)}\b")
            normalized = pattern.sub(preferred, normalized)
        else:
            normalized = normalized.replace(term, preferred)
    return normalized


def _consistency_review(issues: List[IssueItem], glossary: Dict[str, str]) -> List[IssueItem]:
    consistent = []
    for issue in issues:
        issue.reason = _apply_glossary_normalization(issue.reason, glossary)
        issue.suggestion = _apply_glossary_normalization(issue.suggestion, glossary)
        consistent.append(issue)
    return consistent


def _review_paragraphs_once(
    paragraphs: List[ParagraphUnit],
    config: ReviewConfig,
    review_round: int = 1,
    filtered_sink: Optional[List[Tuple[IssueItem, str]]] = None,
) -> List[IssueItem]:
    paragraphs = [p for p in paragraphs if p.anchor.get("reviewable", True) is not False and p.text.strip()]
    if not paragraphs:
        return []
    paragraphs_by_id = {p.para_id: p for p in paragraphs}

    language = "zh" if config.language_mode == "zh" else "en"
    if config.language_mode == "follow_source":
        language = _detect_language("\n".join(p.text for p in paragraphs[: min(20, len(paragraphs))]))

    ignore_cjk_punctuation_width = bool(getattr(config, "ignore_cjk_punctuation_width", False))
    review_word_formulas = bool(getattr(config, "review_word_formulas", False))
    system_prompt = _build_system_prompt(
        language,
        ignore_cjk_punctuation_width=ignore_cjk_punctuation_width,
        review_word_formulas=review_word_formulas,
    )
    rolling_summary = ""
    glossary: Dict[str, str] = {}

    results: List[IssueItem] = []
    lock = threading.Lock()

    def _build_task(index: int):
        paragraph = paragraphs[index]
        ws = config.window_size
        before = [p.text for p in paragraphs[max(0, index - ws): index]]
        after = [p.text for p in paragraphs[index + 1: index + 1 + ws]]
        context_payload = {
            "before": before,
            "after": after,
            "rolling_summary": rolling_summary,
            "glossary": glossary,
        }
        cache_key = _build_cache_key(config, paragraph, context_payload)
        return paragraph, before, after, cache_key

    session = requests.Session()

    # Maintain rolling memory in sequence while allowing API calls in a small pool.
    for batch_start in range(0, len(paragraphs), max(1, config.max_concurrency)):
        batch_indexes = list(range(batch_start, min(len(paragraphs), batch_start + config.max_concurrency)))
        futures = []
        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            for idx in batch_indexes:
                paragraph, before, after, cache_key = _build_task(idx)
                cached = _cache_get(cache_key)
                if cached is not None:
                    futures.append((idx, paragraph, cached, True))
                    continue

                prompt = _build_user_prompt(
                    paragraph,
                    before,
                    after,
                    rolling_summary,
                    glossary,
                    language,
                    ignore_cjk_punctuation_width=ignore_cjk_punctuation_width,
                    review_word_formulas=review_word_formulas,
                )
                _log("normal", f"[Review] para={paragraph.para_id} page={paragraph.page_no} begin model={config.model}")
                future = executor.submit(_call_deepseek, prompt, system_prompt, config, session)
                futures.append((idx, paragraph, future, False))

            resolved = []
            for idx, paragraph, payload, is_cached in futures:
                if is_cached:
                    raw = payload
                    _log("normal", f"[Review] para={paragraph.para_id} hit_cache=true")
                else:
                    raw = payload.result()
                    before = [p.text for p in paragraphs[max(0, idx - config.window_size): idx]]
                    after = [p.text for p in paragraphs[idx + 1: idx + 1 + config.window_size]]
                    cache_key = _build_cache_key(
                        config,
                        paragraph,
                        {
                            "before": before,
                            "after": after,
                            "rolling_summary": rolling_summary,
                            "glossary": glossary,
                        },
                    )
                    _cache_set(cache_key, raw)
                    _log("normal", f"[Review] para={paragraph.para_id} hit_cache=false")
                resolved.append((idx, paragraph, raw))

            # Update rolling memory by original order for consistency.
            for idx, paragraph, raw in sorted(resolved, key=lambda x: x[0]):
                raw_issues = raw.get("issues", []) if isinstance(raw, dict) else []
                parsed_items = [
                    _normalize_issue(item, paragraph, i, review_round=review_round)
                    for i, item in enumerate(raw_issues, start=1)
                    if isinstance(item, dict)
                ]
                parsed_items = [item for item in parsed_items if item is not None]
                parsed_items, dropped_items = _filter_issue_artifacts(
                    parsed_items,
                    paragraphs_by_id,
                    allow_formula_issues=review_word_formulas,
                )
                if filtered_sink is not None and dropped_items:
                    filtered_sink.extend(dropped_items)
                if _should_log("verbose"):
                    raw_text = json.dumps(raw, ensure_ascii=False)
                    if len(raw_text) > 1200:
                        raw_text = f"{raw_text[:1200]}...(truncated)"
                    _log("verbose", f"[Review] para={paragraph.para_id} raw_parsed={raw_text}")

                with lock:
                    results.extend(parsed_items)
                if parsed_items:
                    for issue in parsed_items:
                        _log(
                            "normal",
                            f"[Review] para={paragraph.para_id} issue={issue.category} "
                            f"reason={issue.reason[:120]} suggestion={issue.suggestion[:120]}",
                        )
                else:
                    _log("normal", f"[Review] para={paragraph.para_id} no_issue")

                summary_update = str(raw.get("summary_update", "")).strip() if isinstance(raw, dict) else ""
                if summary_update:
                    joined = f"{rolling_summary}\n{summary_update}".strip()
                    rolling_summary = joined[-2200:]

                glossary_updates = raw.get("glossary_updates", {}) if isinstance(raw, dict) else {}
                if isinstance(glossary_updates, dict):
                    for k, v in glossary_updates.items():
                        kk = str(k).strip()
                        vv = str(v).strip()
                        if kk and vv:
                            glossary[kk] = vv

    consistent = _consistency_review(_coalesce_issues(results), glossary)
    filtered = [issue for issue in consistent if _severity_pass(issue, config.severity_threshold)]
    return sorted(filtered, key=lambda x: (x.para_id, x.category, x.issue_id))


def _is_safe_modified_text(original_text: str, modified_text: str) -> bool:
    original = (original_text or "").strip()
    modified = (modified_text or "").strip()
    if not original or not modified or original == modified:
        return False
    if "\n" in modified or "\r" in modified:
        return False
    instruction_noise = ("改为", "修改为", "替换为", "删除", "建议", "应当", "should", "replace", "change to")
    if any(token.lower() in modified.lower() for token in instruction_noise):
        return False
    return len(modified) <= max(80, len(original) * 4)


def _apply_safe_replacements(paragraphs: List[ParagraphUnit], issues: List[IssueItem]) -> tuple[List[ParagraphUnit], int]:
    issue_map: Dict[str, List[IssueItem]] = {}
    for issue in issues:
        if _is_safe_modified_text(issue.original_text, issue.modified_text):
            issue_map.setdefault(issue.para_id, []).append(issue)

    updated: List[ParagraphUnit] = []
    applied = 0
    for para in paragraphs:
        text = para.text
        replacements = []
        for issue in issue_map.get(para.para_id, []):
            if text.count(issue.original_text) != 1:
                continue
            start = text.find(issue.original_text)
            end = start + len(issue.original_text)
            replacements.append((start, end, issue.modified_text, issue))
        replacements.sort(key=lambda x: x[0], reverse=True)

        occupied: List[tuple[int, int]] = []
        for start, end, modified, issue in replacements:
            if any(not (end <= a or start >= b) for a, b in occupied):
                continue
            text = text[:start] + modified + text[end:]
            occupied.append((start, start + len(modified)))
            applied += 1
            _log("normal", f"[Review] round={issue.review_round} para={para.para_id} apply_fix issue={issue.issue_id}")
        updated.append(replace(para, text=text))
    return updated, applied


def _issue_key(issue: IssueItem) -> str:
    material = f"{issue.para_id}|{issue.category}|{issue.original_text}|{issue.modified_text}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _review_paragraphs_core(
    paragraphs: List[ParagraphUnit],
    config: ReviewConfig,
) -> Tuple[List[IssueItem], List[Tuple[IssueItem, str]]]:
    source_paragraphs = [p for p in paragraphs if p.anchor.get("reviewable", True) is not False and p.text.strip()]
    if not source_paragraphs:
        return [], []

    configured_rounds = int(getattr(config, "max_review_rounds", 0) or 0)
    auto_until_clean = configured_rounds <= 0
    if auto_until_clean:
        try:
            max_rounds = max(1, int(os.getenv("DOC_REVIEW_HARD_MAX_ROUNDS", "50")))
        except ValueError:
            max_rounds = 50
        round_label = f"auto_until_clean hard_max={max_rounds}"
    else:
        max_rounds = max(1, configured_rounds)
        round_label = str(max_rounds)
    current_paragraphs = [replace(p) for p in source_paragraphs]
    original_text_by_para = {p.para_id: p.text for p in source_paragraphs}
    all_issues: List[IssueItem] = []
    filtered_issues: List[Tuple[IssueItem, str]] = []
    seen: set[str] = set()
    seen_states: set[str] = set()

    for review_round in range(1, max_rounds + 1):
        state_material = "\n\n".join(f"{p.para_id}:{p.text}" for p in current_paragraphs)
        state_hash = hashlib.sha1(state_material.encode("utf-8")).hexdigest()
        if state_hash in seen_states:
            _log("normal", f"[Review] iterative_round={review_round} stop reason=repeated_text_state")
            break
        seen_states.add(state_hash)

        _log("normal", f"[Review] iterative_round={review_round}/{round_label} begin")
        round_issues = _review_paragraphs_once(
            current_paragraphs,
            config,
            review_round=review_round,
            filtered_sink=filtered_issues,
        )
        locatable_issues: List[IssueItem] = []
        for issue in round_issues:
            original_para_text = original_text_by_para.get(issue.para_id, "")
            if issue.original_text not in original_para_text:
                _log(
                    "normal",
                    f"[Review] round={review_round} skip_final_issue reason=not_in_original "
                    f"para={issue.para_id} original_text={issue.original_text[:80]}",
                )
                continue
            key = _issue_key(issue)
            if key in seen:
                continue
            seen.add(key)
            locatable_issues.append(
                replace(issue, snippet=_build_issue_snippet(original_para_text, issue.original_text))
            )
        all_issues.extend(locatable_issues)

        current_paragraphs, applied = _apply_safe_replacements(current_paragraphs, round_issues)
        _log(
            "normal",
            f"[Review] iterative_round={review_round}/{round_label} issues={len(round_issues)} "
            f"new_locatable={len(locatable_issues)} fixes_applied={applied}",
        )
        if not round_issues:
            _log("normal", f"[Review] iterative_round={review_round} stop reason=no_issues")
            break
        if applied == 0:
            _log("normal", f"[Review] iterative_round={review_round} stop reason=no_safe_replacement")
            break
    else:
        _log("normal", f"[Review] stop reason=hard_max_rounds_reached rounds={max_rounds}")

    normalized = _coalesce_issues(all_issues)
    sorted_issues = sorted(normalized, key=lambda x: (x.para_id, x.review_round, x.category, x.issue_id))
    for idx, issue in enumerate(sorted_issues, start=1):
        issue.issue_no = f"Q{idx}"
        issue.issue_id = f"{issue.para_id}-r{issue.review_round}-issue-{idx:03d}"
    return sorted_issues, _coalesce_filtered_issues(filtered_issues)


def review_paragraphs_with_audit(
    paragraphs: List[ParagraphUnit],
    config: ReviewConfig,
) -> Tuple[List[IssueItem], List[Dict[str, object]]]:
    adopted_issues, filtered = _review_paragraphs_core(paragraphs, config)
    filtered_payload: List[Dict[str, object]] = []
    for idx, (issue, reason) in enumerate(filtered, start=1):
        filtered_payload.append(
            {
                "issue_id": issue.issue_id or f"{issue.para_id}-filtered-{idx:03d}",
                "issue_no": issue.issue_no or f"F{idx}",
                "para_id": issue.para_id,
                "page_no": issue.page_no,
                "category": issue.category,
                "severity": issue.severity,
                "confidence": issue.confidence,
                "reason": issue.reason,
                "suggestion": issue.suggestion,
                "original_text": issue.original_text,
                "modified_text": issue.modified_text,
                "snippet": issue.snippet,
                "review_input_text": issue.review_input_text,
                "review_round": issue.review_round,
                "adopted": False,
                "filter_reason": reason,
            }
        )
    return adopted_issues, filtered_payload


def review_paragraphs(paragraphs: List[ParagraphUnit], config: ReviewConfig) -> List[IssueItem]:
    issues, _ = _review_paragraphs_core(paragraphs, config)
    return issues


def filter_issues_by_artifact_rules(
    issues: List[IssueItem],
    paragraphs_by_id: Dict[str, ParagraphUnit],
    allow_formula_issues: bool = False,
) -> Tuple[List[IssueItem], List[Tuple[IssueItem, str]]]:
    """
    Public wrapper so non-text review stages can reuse the same artifact filters.
    """

    return _filter_issue_artifacts(issues, paragraphs_by_id, allow_formula_issues=allow_formula_issues)
