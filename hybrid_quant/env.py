from __future__ import annotations

import os
from pathlib import Path


def configure_cache_env() -> Path:
    cache_root = Path(os.environ.get("HYBRID_QUANT_CACHE_DIR", Path.cwd() / ".cache")).resolve()
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_root / "hf-xet"))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache_root / "hf-modules"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "hf" / "datasets"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    return cache_root
