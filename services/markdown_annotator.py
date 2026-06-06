from __future__ import annotations

import html
import base64
import mimetypes
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
from urllib.parse import urlparse

try:
    import mistune
except Exception:  # pragma: no cover
    mistune = None

try:
    import bleach
except Exception:  # pragma: no cover
    bleach = None

try:
    from ..models import IssueItem, ParseResult
    from .temp_paths import workspace_temp_root
except ImportError:  # pragma: no cover
    from models import IssueItem, ParseResult
    from services.temp_paths import workspace_temp_root


def _log_enabled() -> bool:
    raw = os.getenv("DOC_REVIEW_MARKDOWN_AUDIT_LOG", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _log(msg: str) -> None:
    if _log_enabled():
        print(f"[MarkdownAudit] {msg}", flush=True)


def _extract_quoted_terms(text: str) -> List[str]:
    if not text:
        return []
    patterns = [
        r"'([^'\n]{1,80})'",
        r'"([^"\n]{1,80})"',
        r"‘([^’\n]{1,80})’",
        r"“([^”\n]{1,80})”",
    ]
    out: List[str] = []
    for p in patterns:
        out.extend(re.findall(p, text))
    return out


def _normalize_candidate(candidate: str) -> str:
    s = candidate.strip()
    s = s.replace("\\Delta", "Δ")
    s = re.sub(r"[`$*\[\]{}]", "", s)
    s = s.replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_compact_locator(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if len(s) > 36:
        return False
    if "\n" in s:
        return False
    if re.search(r"[。！？；，、,.!?;:]", s):
        return False
    return True


def _extract_issue_candidates(text: str) -> List[str]:
    if not text:
        return []

    cands = _extract_quoted_terms(text)

    # Common pattern in Chinese suggestions: 将 'X' 改为 'Y'
    m = re.search(r"将\s*['\"“‘]?(.+?)['\"”’]?\s*改为", text)
    if m:
        cands.append(m.group(1))

    # If the model says one character is extra, keep the surrounding quoted phrase
    # when available. The single character alone is too ambiguous in Chinese text.
    m = re.search(r"['\"“‘]([^'\"“”‘’\n]{2,80})['\"”’]中多[了出]?一个", text)
    if m:
        cands.insert(0, m.group(1))

    return cands


def _candidate_phrases(issue: IssueItem) -> List[str]:
    cands: List[str] = []

    if issue.original_text:
        cands.append(issue.original_text)

    for text in (issue.suggestion, issue.reason):
        cands.extend(_extract_issue_candidates(text))

    # Reviewer snippets are often full paragraph prefixes. They are useful for
    # reports, but unsafe as highlight locators unless they are compact anchors.
    if issue.snippet and _is_compact_locator(issue.snippet):
        cands.append(issue.snippet.strip()[:80])

    normalized = [_normalize_candidate(x) for x in cands]
    dedup: List[str] = []
    seen = set()
    for cand in normalized:
        if not cand or len(cand) < 2:
            continue
        key = cand.casefold()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(cand)
    return dedup


def _is_html_like(segment: str) -> bool:
    return bool(re.search(r"</?[A-Za-z][^>]*>", segment or ""))


def _is_render_safe_span(segment: str) -> bool:
    s = segment.strip()
    if not s:
        return False
    if "\n" in s:
        return False
    if len(s) > 80:
        return False
    if _is_html_like(s):
        return False
    # Avoid corrupting formula / command fragments.
    if "$" in s or "\\" in s:
        return False
    if "==" in s:
        return False
    return True


def _protect_html_blocks(text: str) -> tuple[str, Dict[str, str]]:
    blocks: Dict[str, str] = {}

    def _repl(match: re.Match[str]) -> str:
        token = f"@@DOCREVIEW_HTML_BLOCK_{len(blocks)}@@"
        blocks[token] = match.group(0)
        return token

    protected = re.sub(r"<table\b.*?</table>", _repl, text, flags=re.I | re.S)
    protected = re.sub(r"<img\b[^>]*>", _repl, protected, flags=re.I | re.S)
    return protected, blocks


def _restore_html_blocks(text: str, blocks: Dict[str, str]) -> str:
    for token, block in blocks.items():
        text = text.replace(token, block)
    return text


def _repair_highlight_damage(markdown_text: str) -> str:
    text = markdown_text
    text = text.replace("text==-align", "text-align")
    text = re.sub(r"==\s*(<table\b)", r"\1", text, flags=re.I)
    text = re.sub(r"(</table>)\s*==", r"\1", text, flags=re.I)
    return text


def _convert_safe_text_highlights(text: str, to_html_mark: bool) -> str:
    replacement = r"<mark>\1</mark>" if to_html_mark else r"@@DOCREVIEW_MARK_START@@\1@@DOCREVIEW_MARK_END@@"
    return re.sub(r"==([^=<>\n]{1,200}?)==", replacement, text)


def _convert_highlights_inside_html_blocks(markdown_text: str, to_html_mark: bool = True) -> str:
    def _block_repl(match: re.Match[str]) -> str:
        block = match.group(0)
        parts = re.split(r"(<[^>]+>)", block)
        for i, part in enumerate(parts):
            if part.startswith("<") and part.endswith(">"):
                continue
            parts[i] = _convert_safe_text_highlights(part, to_html_mark=to_html_mark)
        return "".join(parts)

    return re.sub(r"<table\b.*?</table>", _block_repl, markdown_text, flags=re.I | re.S)


def _repair_heading_highlights(markdown_text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        hashes = match.group(1)
        title = match.group(2).strip()
        return f"{hashes} <mark>{title}</mark>"

    return re.sub(r"^==(#{1,6})\s+(.+?)==\s*$", _repl, markdown_text, flags=re.M)


def _normalize_formula_markup(markdown_text: str) -> str:
    text = markdown_text
    text = re.sub(r"\$\s*\^\{\[([0-9,\-\s]+)\]\}\s*\$", r"<sup>[\1]</sup>", text)
    text = re.sub(r"\$\s*\^\{\+\}\s*\$", r"<sup>+</sup>", text)
    text = re.sub(r"\$\s*\^\{([0-9A-Za-z+\-]+)\}\s*\$", r"<sup>\1</sup>", text)
    text = re.sub(r"IFN-\s*\$\s*\\beta\s*\$", "IFN-β", text)
    text = re.sub(r"IFN-\s*\$\s*\\gamma\s*\$", "IFN-γ", text)
    text = re.sub(r"\$\s*MnO_\{2\}\s*\$", "MnO₂", text)
    text = re.sub(r"\$\s*O_\{2\}\s*\$", "O₂", text)
    text = re.sub(r"\$\s*Mn\^\{2\+\}\s*\$", "Mn²⁺", text)
    text = re.sub(r"\s+([。；，、])", r"\1", text)
    return text


def normalize_annotated_markdown(markdown_text: str) -> str:
    text = _repair_highlight_damage(markdown_text)
    protected, blocks = _protect_html_blocks(text)
    protected, stats = _audit_markdown_highlights(protected)
    text = _restore_html_blocks(protected, blocks)
    text = _convert_highlights_inside_html_blocks(text, to_html_mark=True)
    text = _repair_heading_highlights(text)
    text = _normalize_formula_markup(text)
    _log(
        "normalize markdown "
        f"pairs_found={stats['pairs_found']} pairs_kept={stats['pairs_kept']} "
        f"pairs_removed={stats['pairs_removed']} orphan_tokens={stats['orphan_mark_tokens']}"
    )
    return text.strip() + "\n"


def _find_span(text: str, candidate: str) -> Optional[Tuple[int, int]]:
    idx = text.find(candidate)
    if idx >= 0:
        return idx, idx + len(candidate)

    low_text = text.casefold()
    low_cand = candidate.casefold()
    idx = low_text.find(low_cand)
    if idx >= 0:
        return idx, idx + len(candidate)
    return None


def _best_span(text: str, issue: IssueItem) -> Optional[Tuple[int, int]]:
    if issue.original_text:
        strict = _find_span(text, _normalize_candidate(issue.original_text))
        if strict:
            seg = text[strict[0]: strict[1]]
            if _is_render_safe_span(seg):
                return strict

    for cand in _candidate_phrases(issue):
        span = _find_span(text, cand)
        if not span:
            continue
        seg = text[span[0]: span[1]]
        if _is_render_safe_span(seg):
            return span

    _log(f"skip highlight issue={issue.issue_no or issue.issue_id} para={issue.para_id} reason=no reliable locator")
    return None


def _apply_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text
    merged: List[Tuple[int, int]] = []
    for s, e in sorted(spans, key=lambda x: (x[0], x[1])):
        if s < 0 or e <= s:
            continue
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    out: List[str] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            out.append(text[cursor:s])
        seg = text[s:e]
        if _is_render_safe_span(seg):
            out.append(f"=={seg}==")
        else:
            out.append(seg)
        cursor = e
    if cursor < len(text):
        out.append(text[cursor:])
    return "".join(out)


def _audit_markdown_highlights(markdown_text: str) -> tuple[str, Dict[str, int]]:
    stats = {
        "pairs_found": 0,
        "pairs_kept": 0,
        "pairs_removed": 0,
        "orphan_mark_tokens": 0,
    }

    pair_pat = re.compile(r"==([^=\n]{1,200}?)==")

    def _repl(match: re.Match[str]) -> str:
        stats["pairs_found"] += 1
        segment = match.group(1)
        if _is_render_safe_span(segment):
            stats["pairs_kept"] += 1
            return f"=={segment}=="
        stats["pairs_removed"] += 1
        return segment

    audited = pair_pat.sub(_repl, markdown_text)

    # Count leftovers that may indicate OCR debris or malformed mark syntax.
    leftovers = len(re.findall(r"==", audited)) - (stats["pairs_kept"] * 2)
    stats["orphan_mark_tokens"] = max(0, leftovers)
    return audited, stats


def build_annotated_markdown(parse_result: ParseResult, issues: List[IssueItem]) -> str:
    issue_map: Dict[str, List[IssueItem]] = {}
    for issue in issues:
        issue_map.setdefault(issue.para_id, []).append(issue)

    blocks: List[str] = []
    for para in parse_result.paragraphs:
        para_issues = issue_map.get(para.para_id, [])
        if not para_issues:
            blocks.append(para.text)
            continue

        spans: List[Tuple[int, int]] = []
        for issue in para_issues:
            span = _best_span(para.text, issue)
            if span:
                spans.append(span)
        highlighted = _apply_spans(para.text, spans)
        blocks.append(highlighted)

        for issue in para_issues:
            blocks.append(
                f"> [!NOTE] {issue.issue_no} | {issue.category} | {issue.severity}\n"
                f"> 原因: {issue.reason}\n"
                f"> 建议: {issue.suggestion}\n"
                f"> 原文: {issue.original_text}\n"
                f"> 修正: {issue.modified_text}\n"
                f"> 轮次: {issue.review_round}"
            )

    raw_markdown = "\n\n".join(blocks).strip() + "\n"
    return normalize_annotated_markdown(raw_markdown)


def _render_note_blocks(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("> [!NOTE]"):
            note_lines = [line[len("> "):]]
            i += 1
            while i < len(lines) and lines[i].startswith("> "):
                note_lines.append(lines[i][2:])
                i += 1
            block_html = [
                "<div class='note-block'>",
                f"<div class='note-title'>{html.escape(note_lines[0])}</div>",
            ]
            for x in note_lines[1:]:
                block_html.append(f"<div class='note-line'>{html.escape(x)}</div>")
            block_html.append("</div>")
            out.append("\n".join(block_html))
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _rewrite_markdown_images_to_html(markdown_text: str) -> str:
    # Convert markdown image syntax to explicit html img blocks before markdown rendering,
    # so we can preserve local file paths for PDF rendering.
    pat = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def _repl(match: re.Match[str]) -> str:
        alt = html.escape(match.group(1).strip() or "image")
        src = match.group(2).strip()
        parsed = urlparse(src)
        if parsed.scheme in {"file", "data"}:
            src_attr = src
        elif parsed.scheme in {"http", "https"}:
            src_attr = src
        else:
            p = Path(src).expanduser()
            src_attr = p.as_uri() if p.is_absolute() else src
        return (
            "<p class='image-block'>"
            f"<img class='doc-image' src=\"{html.escape(src_attr, quote=True)}\" alt=\"{alt}\"/>"
            "</p>"
        )

    return pat.sub(_repl, markdown_text)


def _materialize_data_uri_images(markdown_text: str) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    if "data:image/" not in markdown_text:
        return markdown_text, None

    tmp_dir = tempfile.TemporaryDirectory(prefix="doc_review_md_images_", dir=str(workspace_temp_root()))
    tmp_path = Path(tmp_dir.name)
    counter = 0
    pat = re.compile(r"!\[([^\]]*)\]\((data:image/([a-zA-Z0-9.+-]+);base64,([^)]+))\)")

    def _repl(match: re.Match[str]) -> str:
        nonlocal counter
        alt = match.group(1)
        ext = (match.group(3) or "png").lower().split("+", 1)[0]
        payload = match.group(4).strip()
        counter += 1
        img_name = f"image_{counter:04d}.{ext}"
        img_path = tmp_path / img_name
        try:
            img_path.write_bytes(base64.b64decode(payload, validate=False))
            return f"![{alt}]({img_name})"
        except Exception:
            return match.group(0)

    converted = pat.sub(_repl, markdown_text)
    if counter == 0:
        tmp_dir.cleanup()
        return markdown_text, None
    return converted, tmp_dir


def _sanitize_rendered_html(body: str) -> str:
    if bleach is None:
        return body
    allowed_tags = [
        "a",
        "blockquote",
        "br",
        "code",
        "div",
        "em",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "img",
        "li",
        "mark",
        "ol",
        "p",
        "pre",
        "span",
        "strong",
        "sub",
        "sup",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
    ]
    allowed_attrs = {
        "*": ["class"],
        "a": ["href", "title"],
        "img": ["src", "alt", "class"],
        "table": ["border"],
        "td": ["colspan", "rowspan", "align"],
        "th": ["colspan", "rowspan", "align"],
    }
    return bleach.clean(
        body,
        tags=allowed_tags,
        attributes=allowed_attrs,
        protocols=["http", "https", "file", "data", "mailto"],
        strip=True,
    )


def _replace_mark_syntax_with_tokens(markdown_text: str) -> str:
    pat = re.compile(r"==([^=\n]{1,200}?)==")
    start_token = "@@DOCREVIEW_MARK_START@@"
    end_token = "@@DOCREVIEW_MARK_END@@"

    def _repl(match: re.Match[str]) -> str:
        segment = match.group(1)
        if not _is_render_safe_span(segment):
            return segment
        return f"{start_token}{segment}{end_token}"

    return pat.sub(_repl, markdown_text)


def _replace_mark_tags_with_tokens(markdown_text: str) -> str:
    pat = re.compile(r"<mark>(.*?)</mark>", flags=re.I | re.S)

    def _repl(match: re.Match[str]) -> str:
        segment = match.group(1)
        if "\n" in segment or _is_html_like(segment):
            return segment
        return f"@@DOCREVIEW_MARK_START@@{segment}@@DOCREVIEW_MARK_END@@"

    return pat.sub(_repl, markdown_text)


def _restore_mark_tokens(html_text: str) -> str:
    # Tokens are plain text and safe to restore post markdown-render.
    highlight_style = (
        "background-color:#fff200; color:#b91c1c; font-weight:700; "
        "text-decoration:underline; text-decoration-thickness:1.2px; "
        "text-underline-offset:2px; padding:0.04em 0.08em;"
    )
    html_text = html_text.replace(
        "@@DOCREVIEW_MARK_START@@",
        f"<span class='review-highlight' style='{highlight_style}'>",
    )
    html_text = html_text.replace("@@DOCREVIEW_MARK_END@@", "</span>")
    return html_text


def _markdown_to_html(md: str) -> str:
    prepared = _render_note_blocks(md)
    prepared = _rewrite_markdown_images_to_html(prepared)
    prepared = _replace_mark_syntax_with_tokens(prepared)
    prepared = _replace_mark_tags_with_tokens(prepared)

    if mistune is not None:
        # Raw OCR html has been sanitized / normalized before this point.
        # Keep escape=False so our explicit <img> blocks can render.
        if hasattr(mistune, "create_markdown"):
            markdown = mistune.create_markdown(escape=False, hard_wrap=False)
            body = markdown(prepared)
        elif hasattr(mistune, "Markdown") and hasattr(mistune, "Renderer"):
            # mistune 0.x compatibility
            renderer = mistune.Renderer(escape=False, hard_wrap=False)
            markdown = mistune.Markdown(renderer=renderer)
            body = markdown(prepared)
        elif hasattr(mistune, "markdown"):
            body = mistune.markdown(prepared, escape=False, hard_wrap=False)
        else:
            escaped = html.escape(prepared)
            body = "\n".join([f"<p>{line}</p>" for line in escaped.splitlines() if line.strip()])
    else:
        # Fallback renderer: keep simple paragraphs.
        escaped = html.escape(prepared)
        body = "\n".join([f"<p>{line}</p>" for line in escaped.splitlines() if line.strip()])
    body = _sanitize_rendered_html(body)
    body = _restore_mark_tokens(body)

    css = """
    <style>
      body { font-family: 'PingFang SC', 'Microsoft YaHei', Helvetica, Arial, sans-serif; color:#1f2937; }
      .doc { font-size: 13.5px; line-height: 1.65; }
      p { margin: 0 0 9px 0; }
      h1 { font-size: 24px; margin: 16px 0 10px 0; }
      h2 { font-size: 20px; margin: 14px 0 9px 0; }
      h3 { font-size: 17px; margin: 12px 0 8px 0; }
      h4 { font-size: 15px; margin: 10px 0 7px 0; }
      mark, .review-highlight {
        background-color: #fff200;
        color: #b91c1c;
        font-weight: 700;
        text-decoration: underline;
        text-decoration-thickness: 1.2px;
        text-underline-offset: 2px;
        padding: 0.04em 0.08em;
      }
      .image-block { text-align: center; margin: 8px 0 12px 0; }
      .doc-image { max-width: 88%; height: auto; }
      .note-block { background:#f8fafc; border-left:5px solid #1d4ed8; padding:10px 12px; margin:12px 0 16px 0; }
      .note-title { font-weight: 700; margin-bottom: 6px; }
      .note-line { margin: 2px 0; }
      ul, ol { margin: 0 0 14px 24px; }
      code { background:#eef2f7; padding:2px 4px; border-radius:4px; }
    </style>
    """
    return f"<html><head>{css}</head><body><div class='doc'>{body}</div></body></html>"


def _mime_ext(mime: str) -> str:
    guessed = mimetypes.guess_extension(mime)
    if guessed:
        return guessed.lstrip(".")
    if mime.endswith("jpeg"):
        return "jpg"
    if mime.endswith("png"):
        return "png"
    return "bin"


def externalize_markdown_images(markdown_text: str, asset_dir: Path) -> str:
    asset_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    pat = re.compile(r"!\[([^\]]*)\]\((data:image/([a-zA-Z0-9.+-]+);base64,([^)]+))\)")

    def _repl(match: re.Match[str]) -> str:
        nonlocal counter
        alt = match.group(1).strip() or "image"
        mime = f"image/{match.group(3)}"
        payload = match.group(4).strip()
        counter += 1
        ext = _mime_ext(mime)
        img_path = asset_dir / f"image_{counter:04d}.{ext}"
        try:
            img_path.write_bytes(base64.b64decode(payload, validate=False))
            return f"![{alt}]({asset_dir.name}/{img_path.name})"
        except Exception:
            return match.group(0)

    return pat.sub(_repl, markdown_text)


def convert_highlights_for_typora(markdown_text: str) -> str:
    protected, blocks = _protect_html_blocks(markdown_text)
    protected = re.sub(r"==([^=\n]{1,200}?)==", r"<mark>\1</mark>", protected)
    return _restore_html_blocks(protected, blocks)


def _copy_bundle_extra_files(extra_files: Sequence[Path], asset_dir: Path, subdir: str) -> List[Tuple[Path, str]]:
    copied: List[Tuple[Path, str]] = []
    if not extra_files:
        return copied
    target_dir = asset_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    seen: Dict[str, int] = {}
    for src in extra_files:
        if not src.exists() or not src.is_file():
            continue
        name = src.name
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{src.stem}_{count}{src.suffix}"
        dst = target_dir / name
        shutil.copy2(src, dst)
        copied.append((dst, f"{asset_dir.name}/{subdir}/{name}"))
    return copied


def write_markdown_bundle(markdown_text: str, md_path: Path, extra_files: Sequence[str | Path] | None = None) -> tuple[Path, Path]:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = md_path.with_suffix("").with_name(f"{md_path.stem}.assets")
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    bundle_md = normalize_annotated_markdown(markdown_text)
    bundle_md = convert_highlights_for_typora(bundle_md)
    bundle_md = externalize_markdown_images(bundle_md, asset_dir)
    extra_paths = [Path(p).expanduser().resolve() for p in (extra_files or []) if str(p).strip()]
    visual_assets = _copy_bundle_extra_files(extra_paths, asset_dir, "visual_regions")
    if visual_assets:
        lines = ["", "## Visual Region Screenshots", ""]
        for _, rel in visual_assets:
            lines.append(f"![visual region]({rel})")
        bundle_md = bundle_md.rstrip() + "\n\n" + "\n\n".join(lines).strip() + "\n"
    md_path.write_text(bundle_md, encoding="utf-8")

    zip_path = md_path.with_suffix(".bundle.zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(md_path, arcname=md_path.name)
        if asset_dir.exists():
            for file in sorted(asset_dir.rglob("*")):
                if file.is_file():
                    zf.write(file, arcname=f"{asset_dir.name}/{file.relative_to(asset_dir).as_posix()}")
    return md_path, zip_path


def markdown_to_pdf(markdown_text: str, out_pdf: Path) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf_markdown, tmp_images = _materialize_data_uri_images(markdown_text)
    html_text = _markdown_to_html(pdf_markdown)

    page_rect = fitz.paper_rect("a4")
    content_rect = fitz.Rect(50, 56, page_rect.width - 50, page_rect.height - 62)

    writer = fitz.DocumentWriter(str(out_pdf))
    archive = fitz.Archive(tmp_images.name) if tmp_images is not None else None
    story = fitz.Story(html=html_text, archive=archive)

    def rectfn(rect_num: int, filled: fitz.Rect):
        return page_rect, content_rect, None

    try:
        story.write(writer, rectfn)
        writer.close()
    finally:
        if tmp_images is not None:
            tmp_images.cleanup()
    return out_pdf
