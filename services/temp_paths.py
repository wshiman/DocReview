from __future__ import annotations

import os
import tempfile
from pathlib import Path


DEFAULT_TEMP_ROOT = Path("/data/data5/shiman/temp")


def workspace_temp_root() -> Path:
    root = Path(os.getenv("DOC_REVIEW_TEMP_ROOT", str(DEFAULT_TEMP_ROOT))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_workspace_temp() -> Path:
    root = workspace_temp_root()
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(root)
    tempfile.tempdir = str(root)
    return root


def mkdtemp(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(workspace_temp_root())))
