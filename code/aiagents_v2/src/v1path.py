"""Make the unmodified v1 pipeline (../../aiagents_v1/src) importable.

v1 reads its folders from environment variables when `common` is imported, so
`use(work_dir=...)` must run before any v1 module is imported.  Override the v1
location with the environment variable AIAGENTS_V1_SRC.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

V1_SRC = Path(os.environ.get("AIAGENTS_V1_SRC",
                             Path(__file__).resolve().parents[2] / "aiagents_v1" / "src")).resolve()


def use(work_dir=None, data_dir=None, out_dir=None):
    for var, val in (("ER_WORK_DIR", work_dir), ("ER_DATA_DIR", data_dir), ("ER_OUT_DIR", out_dir)):
        if val is not None:
            os.environ[var] = str(Path(val).resolve())
    if "common" in sys.modules:
        raise RuntimeError("v1path.use() must be called before importing v1 modules")
    if not (V1_SRC / "common.py").exists():
        raise SystemExit(f"v1 source not found at {V1_SRC} (set AIAGENTS_V1_SRC)")
    sys.path.insert(0, str(V1_SRC))
