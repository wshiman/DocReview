from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import fitz  # PyMuPDF

try:
    from .layout_regions import LayoutRegion
except ImportError:  # pragma: no cover
    from layout_regions import LayoutRegion


@dataclass
class RegionGroup:
    group_id: str
    page_no: int
    bbox: List[float]
    region_ids: List[str]
    regions: List[LayoutRegion]
    group_type: str = "region"
    boundary_prev_text: str = ""
    boundary_next_text: str = ""
    boundary_prev_bbox: List[float] = field(default_factory=list)
    boundary_next_bbox: List[float] = field(default_factory=list)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _merge_bbox(boxes: Iterable[Sequence[float]]) -> List[float]:
    box_list = list(boxes)
    if not box_list:
        raise ValueError("boxes must not be empty")
    x0 = min(float(b[0]) for b in box_list)
    y0 = min(float(b[1]) for b in box_list)
    x1 = max(float(b[2]) for b in box_list)
    y1 = max(float(b[3]) for b in box_list)
    return [x0, y0, x1, y1]


def _dist_y(a: Sequence[float], b: Sequence[float]) -> float:
    if a[3] < b[1]:
        return float(b[1] - a[3])
    if b[3] < a[1]:
        return float(a[1] - b[3])
    return 0.0


def _same_column(a: Sequence[float], b: Sequence[float]) -> bool:
    overlap = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    min_width = max(1.0, min(float(a[2]) - float(a[0]), float(b[2]) - float(b[0])))
    centers_close = abs(((float(a[0]) + float(a[2])) / 2) - ((float(b[0]) + float(b[2])) / 2)) <= 90
    return overlap / min_width >= 0.35 or centers_close


def _flush_group(groups: List[RegionGroup], page_no: int, chunk: List[LayoutRegion], group_type: str) -> None:
    if not chunk:
        return
    bbox = _merge_bbox(r.bbox for r in chunk)
    group_id = f"g-{page_no:03d}-{len(groups) + 1:04d}"
    groups.append(
        RegionGroup(
            group_id=group_id,
            page_no=page_no,
            bbox=bbox,
            region_ids=[r.region_id for r in chunk],
            regions=list(chunk),
            group_type=group_type,
        )
    )


def _build_adjacency_groups(page_no: int, base_groups: List[RegionGroup], groups: List[RegionGroup]) -> None:
    candidates = [g for g in base_groups if g.group_type == "paragraph" and len(g.regions) <= 8]
    for prev, cur in zip(candidates, candidates[1:]):
        if prev.page_no != cur.page_no:
            continue
        if _dist_y(prev.bbox, cur.bbox) > 72:
            continue
        if not _same_column(prev.bbox, cur.bbox):
            continue
        merged_regions = [*prev.regions, *cur.regions]
        if len(merged_regions) > 12:
            continue
        bbox = _merge_bbox(r.bbox for r in merged_regions)
        if bbox[3] - bbox[1] > 340:
            continue
        group_id = f"g-{page_no:03d}-{len(groups) + 1:04d}"
        groups.append(
            RegionGroup(
                group_id=group_id,
                page_no=page_no,
                bbox=bbox,
                region_ids=[r.region_id for r in merged_regions],
                regions=merged_regions,
                group_type="adjacent_paragraphs",
                boundary_prev_text=" ".join(r.text_hint for r in prev.regions if r.text_hint).strip(),
                boundary_next_text=" ".join(r.text_hint for r in cur.regions if r.text_hint).strip(),
                boundary_prev_bbox=list(prev.bbox),
                boundary_next_bbox=list(cur.bbox),
            )
        )


def _is_equation_label(region: LayoutRegion) -> bool:
    text = (region.text_hint or "").strip()
    return bool(re.search(r"(^|\s)(公式|方程|等式|equation|formula)\s*\d*", text, flags=re.I))


def _is_equation_body(region: LayoutRegion) -> bool:
    text = (region.text_hint or "").strip()
    if region.region_type == "equation":
        return True
    if not text:
        return False
    if "�" in text or any(ch in text for ch in "ℒ∑∏√∞≈≠≤≥Δθ∼∣"):
        return True
    if re.search(r"[\^_=].*[\(\[]?\d+[\)\]]?$", text):
        return True
    if re.search(r"[\(\[]\s*\d+\s*[\)\]]\s*$", text) and len(text) <= 180:
        return True
    return False


def _build_equation_context_groups(page_no: int, ordered: List[LayoutRegion], groups: List[RegionGroup]) -> None:
    equation_like = [r for r in ordered if _is_equation_label(r) or _is_equation_body(r)]
    if not equation_like:
        return

    cluster: List[LayoutRegion] = []
    for region in equation_like:
        if not cluster:
            cluster = [region]
            continue
        gap = _dist_y(cluster[-1].bbox, region.bbox)
        # Equation labels are often left-aligned while equations are centered;
        # vertical proximity is the stable signal for equation context.
        if gap <= 42:
            cluster.append(region)
            continue
        _flush_equation_context_group(page_no, cluster, groups)
        cluster = [region]
    _flush_equation_context_group(page_no, cluster, groups)


def _flush_equation_context_group(page_no: int, cluster: List[LayoutRegion], groups: List[RegionGroup]) -> None:
    bodies = [r for r in cluster if _is_equation_body(r)]
    if not bodies:
        return
    # Single equation blocks are already rendered by the normal paragraph grouping.
    if len(cluster) < 2 and len(bodies) < 2:
        return
    bbox = _merge_bbox(r.bbox for r in cluster)
    if bbox[3] - bbox[1] > 220:
        return
    group_id = f"g-{page_no:03d}-{len(groups) + 1:04d}"
    groups.append(
        RegionGroup(
            group_id=group_id,
            page_no=page_no,
            bbox=bbox,
            region_ids=[r.region_id for r in cluster],
            regions=list(cluster),
            group_type="equation_context",
        )
    )


def group_regions_for_render(regions: List[LayoutRegion], max_merge_gap: float = 18.0, max_group_size: int = 8) -> List[RegionGroup]:
    groups: List[RegionGroup] = []
    by_page: Dict[int, List[LayoutRegion]] = {}
    reviewable_types = {"paragraph", "title", "abstract", "table", "cover", "unknown"}
    for region in regions:
        if region.region_type not in reviewable_types:
            continue
        by_page.setdefault(region.page_no, []).append(region)

    for page_no, page_regions in sorted(by_page.items()):
        ordered = sorted(page_regions, key=lambda r: (r.bbox[1], r.bbox[0]))
        page_base_start = len(groups)
        chunk: List[LayoutRegion] = []
        for region in ordered:
            if not chunk:
                chunk = [region]
                continue
            last = chunk[-1]
            text_like = region.region_type in {"paragraph", "title", "abstract", "cover", "unknown"} and last.region_type in {
                "paragraph",
                "title",
                "abstract",
                "cover",
                "unknown",
            }
            close = _dist_y(last.bbox, region.bbox) <= max_merge_gap
            same_col = _same_column(last.bbox, region.bbox)
            next_bbox = _merge_bbox([*(r.bbox for r in chunk), region.bbox])
            max_height_ok = next_bbox[3] - next_bbox[1] <= 260
            if text_like and close and same_col and len(chunk) < max_group_size and max_height_ok:
                chunk.append(region)
                continue
            _flush_group(groups, page_no, chunk, "paragraph" if chunk[0].region_type != "table" else "table")
            chunk = [region]
        if chunk:
            _flush_group(groups, page_no, chunk, "paragraph" if chunk[0].region_type != "table" else "table")
        _build_adjacency_groups(page_no, groups[page_base_start:], groups)
        _build_equation_context_groups(page_no, ordered, groups)
    return groups


def _expand_bbox(bbox: Sequence[float], page_rect: fitz.Rect, pad_pt: float) -> fitz.Rect:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    rect = fitz.Rect(x0 - pad_pt, y0 - pad_pt, x1 + pad_pt, y1 + pad_pt)
    rect.x0 = max(page_rect.x0, rect.x0)
    rect.y0 = max(page_rect.y0, rect.y0)
    rect.x1 = min(page_rect.x1, rect.x1)
    rect.y1 = min(page_rect.y1, rect.y1)
    return rect


def render_region_groups(
    pdf_path: str | Path,
    doc_id: str,
    groups: List[RegionGroup],
    dpi: int = 200,
    pad_pt: float = 6.0,
    output_root: str | Path | None = None,
) -> Dict[str, Dict[str, object]]:
    """
    Render grouped region crops to PNGs.

    Returns a map:
    {group_id: {"image_path": "...", "bbox": [...], "page_no": n, "region_ids": [...]}}
    """

    path = Path(pdf_path).expanduser().resolve()
    out_root = Path(output_root).resolve() if output_root else Path(__file__).resolve().parents[1] / "outputs"
    out_dir = out_root / "visual_regions" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = max(72, int(dpi)) / 72.0
    mat = fitz.Matrix(zoom, zoom)
    manifest: Dict[str, Dict[str, object]] = {}

    with fitz.open(str(path)) as doc:
        for group in groups:
            if group.page_no < 1 or group.page_no > doc.page_count:
                continue
            page = doc[group.page_no - 1]
            clip = _expand_bbox(group.bbox, page.rect, pad_pt=pad_pt)
            if clip.width <= 1 or clip.height <= 1:
                continue
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            out_path = out_dir / f"{group.group_id}.png"
            pix.save(str(out_path))
            manifest[group.group_id] = {
                "image_path": str(out_path),
                "bbox": [round(float(clip.x0), 2), round(float(clip.y0), 2), round(float(clip.x1), 2), round(float(clip.y1), 2)],
                "page_no": group.page_no,
                "region_ids": list(group.region_ids),
                "group_type": group.group_type,
            }
    return manifest
