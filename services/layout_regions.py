from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import requests

REGION_TYPES = {
    "paragraph",
    "title",
    "abstract",
    "table",
    "figure",
    "cover",
    "header_footer",
    "unknown",
}
DEFAULT_SKIP_TYPES = {"figure", "header_footer"}


@dataclass
class LayoutRegion:
    region_id: str
    page_no: int
    region_type: str
    bbox: List[float]
    text_hint: str
    source_backend: str
    reviewable: bool = True
    image_path: str = ""
    anchor: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "region_id": self.region_id,
            "page_no": self.page_no,
            "type": self.region_type,
            "bbox": [round(float(x), 2) for x in self.bbox],
            "text_hint": self.text_hint,
            "image_path": self.image_path,
            "source_backend": self.source_backend,
            "reviewable": self.reviewable,
        }
        if self.anchor:
            payload["anchor"] = self.anchor
        return payload


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _clamp_bbox(bbox: Iterable[float], page_rect: fitz.Rect) -> Optional[List[float]]:
    try:
        x0, y0, x1, y1 = [float(x) for x in bbox]
    except Exception:
        return None
    x0 = max(page_rect.x0, min(page_rect.x1, x0))
    x1 = max(page_rect.x0, min(page_rect.x1, x1))
    y0 = max(page_rect.y0, min(page_rect.y1, y0))
    y1 = max(page_rect.y0, min(page_rect.y1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _bbox_from_raw(raw_bbox: Any, page_rect: fitz.Rect, page_width_hint: float = 0.0, page_height_hint: float = 0.0) -> Optional[List[float]]:
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, dict):
        if all(k in raw_bbox for k in ("x", "y", "w", "h")):
            vals = [raw_bbox["x"], raw_bbox["y"], raw_bbox["x"] + raw_bbox["w"], raw_bbox["y"] + raw_bbox["h"]]
        elif all(k in raw_bbox for k in ("left", "top", "right", "bottom")):
            vals = [raw_bbox["left"], raw_bbox["top"], raw_bbox["right"], raw_bbox["bottom"]]
        elif all(k in raw_bbox for k in ("x0", "y0", "x1", "y1")):
            vals = [raw_bbox["x0"], raw_bbox["y0"], raw_bbox["x1"], raw_bbox["y1"]]
        else:
            return None
    else:
        try:
            vals = [float(x) for x in raw_bbox]
        except Exception:
            return None
        if len(vals) == 8:
            xs = vals[0::2]
            ys = vals[1::2]
            vals = [min(xs), min(ys), max(xs), max(ys)]
        elif len(vals) != 4:
            return None

    try:
        x0, y0, x1, y1 = [float(x) for x in vals]
    except Exception:
        return None

    # Some APIs return [x, y, width, height].
    if x1 <= x0 or y1 <= y0:
        if x1 > 0 and y1 > 0:
            x1 = x0 + x1
            y1 = y0 + y1

    # Normalized coordinates.
    if 0 <= x0 <= 1 and 0 <= x1 <= 1 and 0 <= y0 <= 1 and 0 <= y1 <= 1:
        x0 *= page_rect.width
        x1 *= page_rect.width
        y0 *= page_rect.height
        y1 *= page_rect.height

    # Pixel coordinates, scaled back to PDF points using page size hints when present.
    if page_width_hint > 0 and page_height_hint > 0:
        sx = page_rect.width / page_width_hint
        sy = page_rect.height / page_height_hint
        x0 *= sx
        x1 *= sx
        y0 *= sy
        y1 *= sy

    return _clamp_bbox([x0, y0, x1, y1], page_rect)


def _guess_region_type(page_no: int, page_count: int, text: str, bbox: List[float], page_rect: fitz.Rect) -> str:
    content = (text or "").strip()
    lower = content.lower()
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    page_w = max(1.0, float(page_rect.width))
    page_h = max(1.0, float(page_rect.height))
    top_ratio = y0 / page_h
    bottom_ratio = y1 / page_h
    width_ratio = width / page_w

    if not content:
        return "figure"
    if page_no == 1 and top_ratio < 0.22 and len(content) <= 120 and width_ratio >= 0.45:
        return "cover"
    if any(k in lower for k in ("abstract", "摘要")):
        return "abstract"
    if len(content) <= 120 and width_ratio >= 0.55 and height <= 72:
        # Uppercase-heavy lines are usually titles/headings in OCR blocks.
        alpha = sum(1 for ch in content if "a" <= ch.lower() <= "z")
        upper = sum(1 for ch in content if "A" <= ch <= "Z")
        if alpha > 0 and upper / max(1, alpha) >= 0.5:
            return "title"
        if content.startswith("#"):
            return "title"
    if "|" in content or "\t" in content or "  " in content:
        return "table"
    if (top_ratio < 0.06 or bottom_ratio > 0.94) and len(content) <= 80:
        return "header_footer"
    if len(content) <= 2:
        return "unknown"
    if page_count > 0 and page_no > page_count:
        return "unknown"
    return "paragraph"


def _region_type_from_raw(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    mapping = {
        "text": "paragraph",
        "paragraph": "paragraph",
        "plain_text": "paragraph",
        "title": "title",
        "heading": "title",
        "section_title": "title",
        "abstract": "abstract",
        "summary": "abstract",
        "table": "table",
        "figure": "figure",
        "image": "figure",
        "cover": "cover",
        "header": "header_footer",
        "footer": "header_footer",
        "header_footer": "header_footer",
    }
    return mapping.get(raw, "unknown")


def _extract_blocks_from_pymupdf(pdf_path: Path, max_regions: int) -> List[LayoutRegion]:
    regions: List[LayoutRegion] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            page_no = page_index + 1
            page_rect = page.rect
            raw = page.get_text("dict")
            for block_index, block in enumerate(raw.get("blocks", []), start=1):
                if len(regions) >= max_regions:
                    return regions
                block_type = int(block.get("type", 0))
                bbox_raw = block.get("bbox")
                if not bbox_raw:
                    continue
                bbox = _bbox_from_raw(bbox_raw, page_rect)
                if bbox is None:
                    continue
                if block_type == 1:
                    region_type = "figure"
                    text_hint = ""
                else:
                    lines = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = str(span.get("text", "")).strip()
                            if text:
                                lines.append(text)
                    text_hint = " ".join(lines).strip()
                    region_type = _guess_region_type(page_no, doc.page_count, text_hint, bbox, page_rect)
                if region_type not in REGION_TYPES:
                    region_type = "unknown"
                region_id = f"r-{page_no:03d}-{block_index:04d}"
                regions.append(
                    LayoutRegion(
                        region_id=region_id,
                        page_no=page_no,
                        region_type=region_type,
                        bbox=bbox,
                        text_hint=text_hint[:300],
                        source_backend="pymupdf",
                        reviewable=region_type not in DEFAULT_SKIP_TYPES,
                    )
                )
    return regions


def _iter_mineru_blocks(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    stack: List[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if any(k in node for k in ("bbox", "box")) and any(
                k in node for k in ("type", "block_type", "category", "label")
            ):
                yield node
            for value in node.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)


def _mineru_create_url() -> str:
    return os.getenv("DOC_REVIEW_MINERU_CREATE_URL", "https://mineru.net/api/v4/extract/task").strip()


def _mineru_result_url_template() -> str:
    return os.getenv(
        "DOC_REVIEW_MINERU_RESULT_URL_TEMPLATE",
        "https://mineru.net/api/v1/agent/parse/{task_id}",
    ).strip()


def _mine_task_id(data: Any) -> str:
    if isinstance(data, dict):
        for key in ("task_id", "taskId", "id"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for val in data.values():
            got = _mine_task_id(val)
            if got:
                return got
    elif isinstance(data, list):
        for item in data:
            got = _mine_task_id(item)
            if got:
                return got
    return ""


def _page_size_hint(block: Dict[str, Any]) -> Tuple[float, float]:
    for key in ("page_size", "pageSize", "size"):
        value = block.get(key)
        if isinstance(value, dict):
            width = float(value.get("width") or value.get("w") or 0)
            height = float(value.get("height") or value.get("h") or 0)
            if width > 0 and height > 0:
                return width, height
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                width, height = float(value[0]), float(value[1])
            except Exception:
                continue
            if width > 0 and height > 0:
                return width, height
    width = float(block.get("page_width") or block.get("pageWidth") or 0)
    height = float(block.get("page_height") or block.get("pageHeight") or 0)
    return width, height


def _call_mineru_layout(pdf_path: Path, max_regions: int) -> List[LayoutRegion]:
    token = os.getenv("MINERU_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing MINERU_API_TOKEN")
    timeout = _env_int("DOC_REVIEW_MINERU_TIMEOUT", 180)
    poll = _env_float("DOC_REVIEW_MINERU_POLL_INTERVAL", 3.0)
    headers = {"Authorization": f"Bearer {token}"}
    create_url = _mineru_create_url()
    result_tpl = _mineru_result_url_template()

    with requests.Session() as session:
        with pdf_path.open("rb") as fp:
            resp = session.post(
                create_url,
                headers=headers,
                files={"file": fp},
                data={"model_version": "vlm"},
                timeout=60,
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"MinerU create task failed {resp.status_code}: {resp.text[:300]}")
        create_payload = resp.json()
        task_id = _mine_task_id(create_payload)
        if not task_id:
            raise RuntimeError("MinerU create response missing task id")

        started = time.time()
        while True:
            url = result_tpl.format(task_id=task_id)
            poll_resp = session.get(url, headers=headers, timeout=60)
            if poll_resp.status_code >= 400:
                raise RuntimeError(f"MinerU poll failed {poll_resp.status_code}: {poll_resp.text[:300]}")
            payload = poll_resp.json()
            state = str(
                payload.get("state")
                or payload.get("status")
                or payload.get("data", {}).get("state")
                or payload.get("data", {}).get("status")
                or ""
            ).strip().lower()
            if state in {"done", "finished", "success", "succeeded"}:
                break
            if state in {"failed", "error", "timeout"}:
                raise RuntimeError(f"MinerU parse failed: state={state}")
            if time.time() - started >= timeout:
                raise RuntimeError("MinerU parse timeout")
            time.sleep(max(0.5, poll))

    regions: List[LayoutRegion] = []
    with fitz.open(str(pdf_path)) as doc:
        for idx, block in enumerate(_iter_mineru_blocks(payload), start=1):
            if len(regions) >= max_regions:
                break
            page_no = int(block.get("page_no") or block.get("page") or block.get("page_index", 0) + 1 or 1)
            if page_no < 1 or page_no > doc.page_count:
                continue
            page = doc[page_no - 1]
            bbox_raw = block.get("bbox") or block.get("box")
            page_w, page_h = _page_size_hint(block)
            bbox = _bbox_from_raw(bbox_raw, page.rect, page_width_hint=page_w, page_height_hint=page_h)
            if bbox is None:
                continue
            region_type = _region_type_from_raw(
                block.get("type") or block.get("block_type") or block.get("category") or block.get("label")
            )
            text_hint = str(block.get("text") or block.get("content") or "").strip()
            regions.append(
                LayoutRegion(
                    region_id=f"r-{page_no:03d}-{idx:04d}",
                    page_no=page_no,
                    region_type=region_type,
                    bbox=bbox,
                    text_hint=text_hint[:300],
                    source_backend="mineru",
                    reviewable=region_type not in DEFAULT_SKIP_TYPES,
                    anchor={"mineru_raw_type": str(block.get("type") or block.get("block_type") or "")},
                )
            )
    return regions


def extract_layout_regions(pdf_path: str | Path, max_regions: int = 120) -> Tuple[List[LayoutRegion], Dict[str, Any]]:
    """
    Extract layout regions from a PDF with backend preference:
    mineru -> paddle -> pymupdf.

    `paddle` currently reuses local PyMuPDF block extraction as fallback geometry source.
    """

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    requested_backend = os.getenv("DOC_REVIEW_LAYOUT_BACKEND", "mineru").strip().lower()
    if requested_backend not in {"mineru", "paddle", "pymupdf"}:
        requested_backend = "mineru"

    backend_chain = [requested_backend]
    if requested_backend != "pymupdf":
        backend_chain.append("pymupdf")

    errors: List[Dict[str, str]] = []
    for backend in backend_chain:
        try:
            if backend == "mineru":
                regions = _call_mineru_layout(path, max_regions=max_regions)
            elif backend == "paddle":
                # Paddle OCR markdown currently does not expose robust block bboxes here.
                # Keep this backend configurable while relying on local geometry extraction.
                regions = _extract_blocks_from_pymupdf(path, max_regions=max_regions)
                for region in regions:
                    region.source_backend = "paddle"
            else:
                regions = _extract_blocks_from_pymupdf(path, max_regions=max_regions)
            return regions, {
                "requested_backend": requested_backend,
                "used_backend": backend,
                "errors": errors,
                "region_count": len(regions),
            }
        except Exception as exc:
            errors.append({"backend": backend, "error": str(exc)})
            continue
    return [], {
        "requested_backend": requested_backend,
        "used_backend": "none",
        "errors": errors,
        "region_count": 0,
    }
