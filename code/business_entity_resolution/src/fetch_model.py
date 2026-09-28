"""Download a pretrained transformer used by stage 3 (once) into <work>/hf/.

    python fetch_model.py [small|base|large]      (default small)

intfloat/multilingual-e5-small / -base / -large: MIT licence, 118M / 278M / 560M parameters (far below the 8B
limit).  Only the weights and tokenizer files are fetched; nothing is called at inference time.
"""
from __future__ import annotations

import sys

from common import log, wpath

SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
REPO = f"intfloat/multilingual-e5-{SIZE}"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "sentencepiece.bpe.model"]

if __name__ == "__main__":
    out = wpath("hf", f"multilingual-e5-{SIZE}")
    if all((out / f).exists() for f in FILES):
        log("model already present:", out)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, local_dir=str(out), allow_patterns=FILES)
        log("downloaded", REPO, "->", out)
