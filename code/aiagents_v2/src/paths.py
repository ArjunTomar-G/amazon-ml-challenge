"""Point the pipeline modules (common.py reads env vars at import) at a work / data / out dir.

Call `use(...)` before importing any pipeline module.  Everything lives in this
folder; nothing is imported from another pipeline copy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def use(work_dir=None, data_dir=None, out_dir=None):
    for var, val in (("ER_WORK_DIR", work_dir), ("ER_DATA_DIR", data_dir), ("ER_OUT_DIR", out_dir)):
        if val is not None:
            os.environ[var] = str(Path(val).resolve())
    if "common" in sys.modules:
        raise RuntimeError("paths.use() must be called before importing pipeline modules")
