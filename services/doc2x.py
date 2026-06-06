from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import fitz  # PyMuPDF
import requests

try:
    from .layout_regions import LayoutRegion
except ImportError:  # pragma: no cover
    from layout_regions import LayoutRegion

DEFAULT_API_BASE = "https://v2.doc2x.noedgeai.com"
DEFAULT_MODEL = "v3-2026"


@dataclass
class Doc2XPageMarkdown:
    page_no: int
    markdown_text: str


@dataclass
class Doc2XParseResult:
    markdown_text: str
    pages: List[Doc2XPageMarkdown] = field(default_factory=list)
    layout_regions: List[LayoutRegion] = field(default_factory=list)
    raw_result: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def _api_token() -> str:
    token = os.getenv("DOC_REVIEW_DOC2X_API_TOKEN", "").strip() or os.getenv("DOC2X_API_KEY", "").strip()
    if not token:
        raise RuntimeError("Missing DOC_REVIEW_DOC2X_API_TOKEN or DOC2X_API_KEY for Doc2X API.")
    return token


def _api_base() -> str:
    return os.getenv("DOC_REVIEW_DOC2X_API_BASE", DEFAULT_API_BASE).strip().rstrip("/") or DEFAULT_API_BASE


def _timeout_sec() -> int:
    raw = os.getenv("DOC_REVIEW_DOC2X_TIMEOUT", "900").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return 900


def _poll_interval_sec() -> float:
    raw = os.getenv("DOC_REVIEW_DOC2X_POLL_INTERVAL", "3").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 3.0


def _model_name() -> str:
    return os.getenv("DOC_REVIEW_DOC2X_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _log_enabled() -> bool:
    raw = os.getenv("DOC_REVIEW_DOC2X_LOG", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _log(msg: str) -> None:
    if _log_enabled():
        print(f"[Doc2X] {msg}", flush=True)


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_api_token()}"}


def _first_str(data: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _find_preupload_fields(payload: Any) -> tuple[str, str]:
    if isinstance(payload, dict):
        uid = _first_str(payload, ("uid", "uuid", "task_id", "taskId", "id"))
        url = _first_str(payload, ("url", "upload_url", "uploadUrl", "presigned_url", "presignedUrl"))
        if uid and url:
            return uid, url
        for value in payload.values():
            uid, url = _find_preupload_fields(value)
            if uid and url:
                return uid, url
    elif isinstance(payload, list):
        for item in payload:
            uid, url = _find_preupload_fields(item)
            if uid and url:
                return uid, url
    return "", ""


def _submit_preupload(session: requests.Session, path: Path) -> tuple[str, str, Dict[str, Any]]:
    url = f"{_api_base()}/api/v2/parse/preupload"
    payload = {"model": _model_name()}
    _log(f"preupload start file={path.name} model={payload['model']}")
    resp = session.post(url, headers={**_auth_headers(), "Content-Type": "application/json"}, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Doc2X preupload failed {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    uid, upload_url = _find_preupload_fields(data)
    if not uid or not upload_url:
        raise RuntimeError(f"Doc2X preupload response missing uid/url: {str(data)[:500]}")
    _log(f"preupload done uid={uid}")
    return uid, upload_url, data


def _upload_file(session: requests.Session, upload_url: str, path: Path) -> None:
    _log(f"upload start size={path.stat().st_size}")
    with path.open("rb") as fp:
        resp = session.put(upload_url, data=fp, headers={"Content-Type": "application/pdf"}, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Doc2X upload failed {resp.status_code}: {resp.text[:500]}")
    _log("upload done")


def _status_value(payload: Dict[str, Any]) -> str:
    direct = _first_str(payload, ("status", "state"))
    if direct:
        return direct.lower()
    data = payload.get("data")
    if isinstance(data, dict):
        return _status_value(data)
    return ""


def _find_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("result", "output", "document", "json"):
            value = data.get(key)
            if isinstance(value, dict) and _has_doc2x_pages(value):
                return value
        if _has_doc2x_pages(data):
            return data
    for key in ("result", "output", "document", "json"):
        value = payload.get(key)
        if isinstance(value, dict) and _has_doc2x_pages(value):
            return value
    return payload


def _poll_status(session: requests.Session, uid: str, timeout_sec: int, poll_interval_sec: float) -> Dict[str, Any]:
    url = f"{_api_base()}/api/v2/parse/status"
    started = time.time()
    last_payload: Dict[str, Any] = {}
    _log(f"poll start uid={uid} timeout={timeout_sec}s interval={poll_interval_sec}s")
    while True:
        resp = session.get(url, headers=_auth_headers(), params={"uid": uid}, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"Doc2X status failed {resp.status_code}: {resp.text[:500]}")
        payload = resp.json()
        if isinstance(payload, dict):
            last_payload = payload
        status = _status_value(last_payload)
        progress = ""
        data = last_payload.get("data")
        if isinstance(data, dict):
            progress = str(data.get("progress") or data.get("percent") or "").strip()
        _log(f"poll status={status or 'unknown'} progress={progress or '-'}")
        if status in {"success", "done", "finished", "succeeded"}:
            return _find_result_payload(last_payload)
        if status in {"failed", "fail", "error", "timeout"}:
            raise RuntimeError(f"Doc2X parse failed: status={status}")
        if time.time() - started > timeout_sec:
            raise RuntimeError(f"Doc2X parse timeout after {timeout_sec}s, last_status={status or 'unknown'}")
        time.sleep(poll_interval_sec)


def _has_doc2x_pages(payload: Dict[str, Any]) -> bool:
    pages = payload.get("pages") or payload.get("page")
    return isinstance(pages, list)


def _pages_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = result.get("pages") or result.get("page")
    return pages if isinstance(pages, list) else []


def _page_no(page: Dict[str, Any], fallback_index: int) -> int:
    for key in ("page_no", "pageNo"):
        if key in page:
            try:
                value = int(page[key])
            except Exception:
                break
            return value if value > 0 else fallback_index
    for key in ("page_idx", "pageIndex"):
        if key in page:
            try:
                value = int(page[key])
            except Exception:
                break
            return value + 1 if value >= 0 else fallback_index
    return fallback_index


def _page_markdown(page: Dict[str, Any]) -> str:
    for key in ("md", "markdown", "markdown_text", "markdownText"):
        value = page.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def _layout_blocks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    layout = page.get("layout")
    if isinstance(layout, dict) and isinstance(layout.get("blocks"), list):
        return [x for x in layout["blocks"] if isinstance(x, dict)]
    blocks = page.get("blocks")
    if isinstance(blocks, list):
        return [x for x in blocks if isinstance(x, dict)]
    return []


def _block_bbox(block: Dict[str, Any]) -> Optional[List[float]]:
    bbox = block.get("bbox") or block.get("box")
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x", "y", "w", "h")):
            return [float(bbox["x"]), float(bbox["y"]), float(bbox["x"]) + float(bbox["w"]), float(bbox["y"]) + float(bbox["h"])]
        if all(k in bbox for k in ("left", "top", "right", "bottom")):
            return [float(bbox["left"]), float(bbox["top"]), float(bbox["right"]), float(bbox["bottom"])]
        if all(k in bbox for k in ("x0", "y0", "x1", "y1")):
            return [float(bbox["x0"]), float(bbox["y0"]), float(bbox["x1"]), float(bbox["y1"])]
    if isinstance(bbox, (list, tuple)):
        vals = [float(x) for x in bbox]
        if len(vals) == 4:
            return vals
        if len(vals) == 8:
            xs = vals[0::2]
            ys = vals[1::2]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _doc2x_region_type(block: Dict[str, Any]) -> str:
    raw = str(block.get("type") or block.get("block_type") or block.get("label") or "").strip().lower()
    mapping = {
        "text": "paragraph",
        "plain_text": "paragraph",
        "paragraph": "paragraph",
        "title": "title",
        "heading": "title",
        "abstract": "abstract",
        "table": "table",
        "tablegroup": "table",
        "table_group": "table",
        "figure": "figure",
        "figuregroup": "figure",
        "figure_group": "figure",
        "image": "figure",
        "caption": "paragraph",
        "figurecaption": "paragraph",
        "figure_caption": "paragraph",
        "equation": "equation",
        "formula": "equation",
        "code": "paragraph",
        "header": "header_footer",
        "footer": "header_footer",
    }
    return mapping.get(raw.replace(" ", "").replace("-", "_"), mapping.get(raw, "unknown"))


def _doc2x_text_reviewable_type(block: Dict[str, Any]) -> bool:
    raw_type = str(block.get("type") or block.get("block_type") or block.get("label") or "").strip().lower()
    key = raw_type.replace(" ", "").replace("-", "_")
    region_type = _doc2x_region_type(block)
    if region_type in {"figure", "equation", "header_footer"}:
        return False
    return key not in {"image", "figure", "figuregroup", "figure_group", "equation", "formula"}


def _block_text(block: Dict[str, Any]) -> str:
    for key in ("text", "content", "md", "markdown", "html"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    table_data = block.get("table_data") or block.get("tableData")
    if isinstance(table_data, dict):
        html = table_data.get("html")
        if isinstance(html, str) and html.strip():
            return html.strip()
    return ""


def _page_review_markdown(page: Dict[str, Any]) -> str:
    blocks = _layout_blocks(page)
    if not blocks:
        return _page_markdown(page)
    chunks: List[str] = []
    for block in blocks:
        if _block_is_boilerplate(block) or not _doc2x_text_reviewable_type(block):
            continue
        text = _block_text(block)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks).strip() or _page_markdown(page)


def _page_size_hints(page: Dict[str, Any]) -> tuple[float, float]:
    for key in ("page_width", "pageWidth", "width"):
        width = page.get(key)
        for hkey in ("page_height", "pageHeight", "height"):
            height = page.get(hkey)
            try:
                w = float(width)
                h = float(height)
            except Exception:
                continue
            if w > 0 and h > 0:
                return w, h
    size = page.get("page_size") or page.get("pageSize") or page.get("size")
    if isinstance(size, dict):
        try:
            w = float(size.get("width") or size.get("w"))
            h = float(size.get("height") or size.get("h"))
        except Exception:
            return 0.0, 0.0
        if w > 0 and h > 0:
            return w, h
    return 0.0, 0.0


def _scale_bbox_to_pdf(raw_bbox: List[float], page_rect: fitz.Rect, page_width: float, page_height: float) -> Optional[List[float]]:
    x0, y0, x1, y1 = raw_bbox
    if x1 <= x0 or y1 <= y0:
        return None
    if page_width > 0 and page_height > 0:
        sx = page_rect.width / page_width
        sy = page_rect.height / page_height
        x0 *= sx
        x1 *= sx
        y0 *= sy
        y1 *= sy
    elif 0 <= x0 <= 1 and 0 <= x1 <= 1 and 0 <= y0 <= 1 and 0 <= y1 <= 1:
        x0 *= page_rect.width
        x1 *= page_rect.width
        y0 *= page_rect.height
        y1 *= page_rect.height
    x0 = max(page_rect.x0, min(page_rect.x1, x0))
    x1 = max(page_rect.x0, min(page_rect.x1, x1))
    y0 = max(page_rect.y0, min(page_rect.y1, y0))
    y1 = max(page_rect.y0, min(page_rect.y1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _block_is_boilerplate(block: Dict[str, Any]) -> bool:
    attrs = block.get("attributes")
    if isinstance(attrs, dict):
        value = attrs.get("is_boilerplate") or attrs.get("isBoilerplate") or attrs.get("boilerplate")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _extract_layout_regions(result: Dict[str, Any], pdf_path: Path) -> List[LayoutRegion]:
    regions: List[LayoutRegion] = []
    pages = _pages_from_result(result)
    with fitz.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(pages):
            page_no = _page_no(page, page_index + 1)
            if page_no < 1 or page_no > doc.page_count:
                continue
            page_rect = doc[page_no - 1].rect
            page_width, page_height = _page_size_hints(page)
            for block_index, block in enumerate(_layout_blocks(page), start=1):
                raw_bbox = _block_bbox(block)
                if raw_bbox is None:
                    continue
                bbox = _scale_bbox_to_pdf(raw_bbox, page_rect, page_width, page_height)
                if bbox is None:
                    continue
                region_type = _doc2x_region_type(block)
                boilerplate = _block_is_boilerplate(block)
                if boilerplate:
                    region_type = "header_footer"
                regions.append(
                    LayoutRegion(
                        region_id=f"r-{page_no:03d}-{block_index:04d}",
                        page_no=page_no,
                        region_type=region_type,
                        bbox=bbox,
                        text_hint=_block_text(block)[:300],
                        source_backend="doc2x",
                        reviewable=region_type not in {"figure", "header_footer"},
                        anchor={
                            "doc2x_block_id": block.get("id") or block.get("block_id") or block.get("blockId"),
                            "doc2x_raw_type": block.get("type") or block.get("block_type") or block.get("label"),
                            "doc2x_reading_order": block.get("reading_order") or block.get("readingOrder"),
                            "doc2x_parent_id": block.get("parent_id") or block.get("parentId"),
                            "doc2x_is_boilerplate": boilerplate,
                        },
                    )
                )
    return regions


def parse_pdf_to_doc2x(path: str | Path) -> Doc2XParseResult:
    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"Doc2X input file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Doc2X parser only supports PDF files.")

    with requests.Session() as session:
        uid, upload_url, preupload_payload = _submit_preupload(session, pdf_path)
        _upload_file(session, upload_url, pdf_path)
        result_payload = _poll_status(session, uid, _timeout_sec(), _poll_interval_sec())

    pages: List[Doc2XPageMarkdown] = []
    md_parts: List[str] = []
    for idx, page in enumerate(_pages_from_result(result_payload), start=1):
        page_no = _page_no(page, idx)
        md = _page_review_markdown(page)
        if not md:
            continue
        pages.append(Doc2XPageMarkdown(page_no=page_no, markdown_text=md))
        md_parts.append(md)

    markdown_text = "\n\n".join(md_parts).strip()
    if not markdown_text:
        raise RuntimeError("Doc2X produced no usable page markdown.")

    regions = _extract_layout_regions(result_payload, pdf_path)
    meta = {
        "uid": uid,
        "model": _model_name(),
        "preupload": preupload_payload,
        "page_count": len(_pages_from_result(result_payload)),
        "layout_region_count": len(regions),
    }
    return Doc2XParseResult(
        markdown_text=markdown_text,
        pages=pages,
        layout_regions=regions,
        raw_result=result_payload,
        meta=meta,
    )
