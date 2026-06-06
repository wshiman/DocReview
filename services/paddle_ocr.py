from __future__ import annotations

import json
import base64
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

DEFAULT_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_MODEL = "PaddleOCR-VL-1.6"


@dataclass
class PaddlePageMarkdown:
    page_no: int
    markdown_text: str
    layout_index: int


@dataclass
class PaddleParseResult:
    markdown_text: str
    pages: List[PaddlePageMarkdown] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OCR_ASSET_ROOT = PROJECT_ROOT / "outputs" / "ocr_assets"


def _log_enabled() -> bool:
    raw = os.getenv("DOC_REVIEW_PADDLE_LOG", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _log(msg: str) -> None:
    if _log_enabled():
        print(f"[PaddleOCR] {msg}", flush=True)


def _api_token() -> str:
    token = os.getenv("DOC_REVIEW_PADDLE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing DOC_REVIEW_PADDLE_API_TOKEN for Paddle OCR API.")
    return token


def _job_url() -> str:
    return os.getenv("DOC_REVIEW_PADDLE_API_JOB_URL", DEFAULT_JOB_URL).strip() or DEFAULT_JOB_URL


def _model_name() -> str:
    return os.getenv("DOC_REVIEW_PADDLE_API_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _timeout_sec() -> int:
    raw = os.getenv("DOC_REVIEW_PADDLE_API_TIMEOUT", "900").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return 900


def _poll_interval_sec() -> float:
    raw = os.getenv("DOC_REVIEW_PADDLE_API_POLL_INTERVAL", "3").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 3.0


def _result_timeout_sec() -> int:
    raw = os.getenv("DOC_REVIEW_PADDLE_API_RESULT_TIMEOUT", "600").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return 600


def _result_retries() -> int:
    raw = os.getenv("DOC_REVIEW_PADDLE_API_RESULT_RETRIES", "3").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _optional_payload() -> Dict[str, object]:
    return {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }


def _safe_rel_path(raw_path: str) -> Path:
    raw = (raw_path or "").strip().replace("\\", "/")
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        # URL-like path should not be treated as local relative path.
        return Path("image.bin")
    parts = [p for p in raw.split("/") if p and p not in {".", ".."}]
    if not parts:
        return Path("image.bin")
    return Path(*parts)


def _image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _download_markdown_images(
    images: Dict[str, str],
    session: requests.Session,
    asset_dir: Path,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    asset_dir.mkdir(parents=True, exist_ok=True)

    for raw_key, raw_url in images.items():
        key = str(raw_key or "").strip()
        url = str(raw_url or "").strip()
        if not key or not url or not url.startswith(("http://", "https://")):
            continue
        rel_path = _safe_rel_path(key)
        out_path = (asset_dir / rel_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code != 200:
                continue
            out_path.write_bytes(resp.content)
            mapping[key] = _image_data_uri(out_path)
        except Exception:
            continue
    return mapping


def _rewrite_markdown_image_refs(md_text: str, image_uri_map: Dict[str, str]) -> str:
    if not md_text:
        return md_text
    rewritten = md_text

    # Replace explicit paths in html / markdown links.
    for key, uri in sorted(image_uri_map.items(), key=lambda x: len(x[0]), reverse=True):
        rewritten = rewritten.replace(f'src="{key}"', f'src="{uri}"')
        rewritten = rewritten.replace(f"src='{key}'", f"src='{uri}'")
        rewritten = rewritten.replace(f"({key})", f"({uri})")
        rewritten = rewritten.replace(f"({Path(key).name})", f"({uri})")

    # Replace [image] placeholders by available images in sequence.
    placeholder_pat = re.compile(r"\[image\]", flags=re.I)
    ordered_uris = [uri for _, uri in sorted(image_uri_map.items(), key=lambda x: x[0])]

    def _placeholder_repl(_: re.Match[str]) -> str:
        if ordered_uris:
            uri = ordered_uris.pop(0)
            return f"![image]({uri})"
        return "[image]"

    rewritten = placeholder_pat.sub(_placeholder_repl, rewritten)

    # Convert HTML img tags to markdown image syntax so markdown renderer can handle them safely.
    img_pat = re.compile(r"<img\b[^>]*\bsrc=['\"]([^'\"]+)['\"][^>]*>", flags=re.I)

    def _img_repl(match: re.Match[str]) -> str:
        src = match.group(1).strip()
        final_src = image_uri_map.get(src, src)
        return f"![image]({final_src})"

    rewritten = img_pat.sub(_img_repl, rewritten)
    # Drop wrapper divs commonly emitted by OCR markdown to avoid noisy inline styles.
    rewritten = re.sub(r"</?div\b[^>]*>", "", rewritten, flags=re.I)
    return rewritten


def _collect_markdown_images(md_obj: Dict[str, object], item: Dict[str, object]) -> Dict[str, str]:
    images: Dict[str, str] = {}
    for raw in (md_obj.get("images"), item.get("outputImages"), item.get("images")):
        if not isinstance(raw, dict):
            continue
        for key, url in raw.items():
            k = str(key or "").strip()
            u = str(url or "").strip()
            if k and u:
                images[k] = u
    return images


def _submit_job(path: Path, session: requests.Session, headers: Dict[str, str], model: str, job_url: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Paddle OCR input file not found: {path}")
    data = {"model": model, "optionalPayload": json.dumps(_optional_payload(), ensure_ascii=False)}
    _log(f"submit start file={path.name} size={path.stat().st_size} model={model}")
    with path.open("rb") as fp:
        response = session.post(job_url, headers=headers, data=data, files={"file": fp}, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Paddle OCR submit failed {response.status_code}: {response.text[:500]}")
    try:
        payload = response.json()
        job_id = str(payload["data"]["jobId"])
        _log(f"submit done job_id={job_id}")
        return job_id
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Paddle OCR submit response missing jobId: {response.text[:500]}") from exc


def _poll_job(
    job_id: str,
    session: requests.Session,
    headers: Dict[str, str],
    job_url: str,
    timeout_sec: int,
    poll_interval_sec: float,
) -> Dict[str, object]:
    started = time.time()
    last_payload: Dict[str, object] = {}
    _log(f"poll start job_id={job_id} timeout={timeout_sec}s interval={poll_interval_sec}s")
    last_state = ""
    while True:
        response = session.get(f"{job_url}/{job_id}", headers=headers, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Paddle OCR polling failed {response.status_code}: {response.text[:500]}")
        payload = response.json().get("data", {})
        if isinstance(payload, dict):
            last_payload = payload
        state = str(last_payload.get("state", "")).strip().lower()
        progress = last_payload.get("extractProgress", {}) if isinstance(last_payload, dict) else {}
        total_pages = progress.get("totalPages") if isinstance(progress, dict) else None
        extracted_pages = progress.get("extractedPages") if isinstance(progress, dict) else None
        elapsed = time.time() - started
        _log(
            f"poll state={state or 'unknown'} elapsed={elapsed:.1f}s "
            f"extracted_pages={extracted_pages} total_pages={total_pages}"
        )
        if state != last_state:
            last_state = state
        if state == "done":
            _log(f"poll done elapsed={time.time() - started:.1f}s")
            return last_payload
        if state == "failed":
            reason = str(last_payload.get("errorMsg", "")).strip() or "unknown_error"
            raise RuntimeError(f"Paddle OCR job failed: {reason}")
        if time.time() - started > timeout_sec:
            raise RuntimeError(f"Paddle OCR job timeout after {timeout_sec}s, last_state={state or 'unknown'}")
        time.sleep(poll_interval_sec)


def _fetch_jsonl_lines(jsonl_url: str, session: requests.Session) -> List[str]:
    timeout_sec = _result_timeout_sec()
    retries = _result_retries()
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        downloaded = 0
        chunks: List[bytes] = []
        _log(f"download result jsonl start attempt={attempt}/{retries} timeout={timeout_sec}s")
        try:
            with session.get(jsonl_url, timeout=(20, timeout_sec), stream=True) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"result download failed {response.status_code}: {response.text[:500]}")
                content_length = response.headers.get("Content-Length")
                _log(f"download response ok content_length={content_length or 'unknown'}")
                for idx, chunk in enumerate(response.iter_content(chunk_size=256 * 1024), start=1):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    _log(f"download progress chunks={idx} bytes={downloaded}")

            text = b"".join(chunks).decode("utf-8", errors="replace")
            lines = [line for line in text.splitlines() if line.strip()]
            _log(f"download result jsonl done lines={len(lines)} bytes={downloaded}")
            return lines
        except Exception as exc:
            last_err = exc
            _log(f"download attempt failed attempt={attempt}/{retries} error={exc}")
            if attempt < retries:
                time.sleep(min(6, attempt * 2))

    raise RuntimeError(f"Paddle OCR result download failed after retries: {last_err}")


def parse_pdf_to_markdown(file_path: str | Path) -> PaddleParseResult:
    path = Path(file_path).expanduser().resolve()
    token = _api_token()
    job_url = _job_url()
    model = _model_name()
    timeout_sec = _timeout_sec()
    poll_interval = _poll_interval_sec()
    headers = {"Authorization": f"bearer {token}"}
    doc_key = f"{path.stem}_{path.stat().st_mtime_ns}"
    asset_dir = (OCR_ASSET_ROOT / doc_key).resolve()

    _log(f"parse start file={path}")
    with requests.Session() as session:
        job_id = _submit_job(path, session, headers, model, job_url)
        job_payload = _poll_job(job_id, session, headers, job_url, timeout_sec, poll_interval)
        result_url = job_payload.get("resultUrl", {}) if isinstance(job_payload, dict) else {}
        json_url = ""
        if isinstance(result_url, dict):
            json_url = str(result_url.get("jsonUrl", "")).strip()
        if not json_url:
            raise RuntimeError("Paddle OCR job finished but resultUrl.jsonUrl is missing.")
        _log(f"result json url={json_url}")

        lines = _fetch_jsonl_lines(json_url, session)

        page_chunks: List[PaddlePageMarkdown] = []
        merged: List[str] = []
        page_counter = 0
        image_uri_map: Dict[str, str] = {}
        for line in lines:
            try:
                line_obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            result_obj = line_obj.get("result", {})
            if not isinstance(result_obj, dict):
                continue
            parsing_results = result_obj.get("layoutParsingResults", [])
            if not isinstance(parsing_results, list):
                continue
            for layout_idx, item in enumerate(parsing_results):
                if not isinstance(item, dict):
                    continue
                md_obj = item.get("markdown", {})
                if not isinstance(md_obj, dict):
                    continue
                md_images = _collect_markdown_images(md_obj, item)
                page_image_uri_map: Dict[str, str] = {}
                if md_images:
                    page_image_uri_map = _download_markdown_images(md_images, session=session, asset_dir=asset_dir)
                    image_uri_map.update(page_image_uri_map)
                md_text = str(md_obj.get("text", "")).strip()
                if not md_text:
                    continue
                page_counter += 1
                if page_image_uri_map or image_uri_map:
                    md_text = _rewrite_markdown_image_refs(md_text, page_image_uri_map or image_uri_map)
                page_chunks.append(PaddlePageMarkdown(page_no=page_counter, markdown_text=md_text, layout_index=layout_idx))
                merged.append(md_text)

    markdown_text = "\n\n".join(merged).strip()
    if not markdown_text:
        raise RuntimeError("Paddle OCR returned empty markdown content.")
    _log(f"parse done markdown_chars={len(markdown_text)} chunks={len(page_chunks)}")

    extract_progress = job_payload.get("extractProgress", {}) if isinstance(job_payload, dict) else {}
    meta = {
        "job_id": job_id,
        "model": model,
        "state": str(job_payload.get("state", "")) if isinstance(job_payload, dict) else "",
        "result_json_url": json_url,
        "total_pages": extract_progress.get("totalPages") if isinstance(extract_progress, dict) else None,
        "extracted_pages": extract_progress.get("extractedPages") if isinstance(extract_progress, dict) else None,
        "start_time": extract_progress.get("startTime") if isinstance(extract_progress, dict) else None,
        "end_time": extract_progress.get("endTime") if isinstance(extract_progress, dict) else None,
        "image_assets_dir": str(asset_dir),
        "image_asset_count": len(image_uri_map),
    }
    return PaddleParseResult(markdown_text=markdown_text, pages=page_chunks, meta=meta)
