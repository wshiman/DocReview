from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


DEFAULT_LOCAL_OFFICE_ROOT = Path("/data/data5/shiman/libreoffice")


def _is_usable_soffice(path: str) -> bool:
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=8)
    except Exception:
        return False
    return result.returncode == 0


def _usable_or_none(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if _is_usable_soffice(path):
        return path
    return None


def find_soffice() -> Optional[str]:
    configured = os.getenv("DOC_REVIEW_SOFFICE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.exists() and os.access(str(path), os.X_OK):
            return _usable_or_none(str(path.resolve()))

    for root in [DEFAULT_LOCAL_OFFICE_ROOT, Path(os.getenv("DOC_REVIEW_LIBREOFFICE_ROOT", ""))]:
        if not str(root):
            continue
        try:
            candidates = sorted(root.expanduser().glob("opt/libreoffice*/program/soffice"))
        except Exception:
            candidates = []
        for candidate in candidates:
            if candidate.exists() and os.access(str(candidate), os.X_OK):
                usable = _usable_or_none(str(candidate.resolve()))
                if usable:
                    return usable

    return _usable_or_none(shutil.which("soffice")) or _usable_or_none(shutil.which("libreoffice"))
