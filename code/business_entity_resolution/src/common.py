"""Shared paths, IO helpers and small text utilities.

Paths are resolved from environment variables so that the whole pipeline can be
pointed at any data / scratch location:

    ER_DATA_DIR  folder containing train/ and test/ sub-folders with the TSVs
    ER_WORK_DIR  scratch folder for parquet caches, candidates, features, models
    ER_OUT_DIR   folder where matching_results.tsv / candidate_pairs.tsv are written
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", _HERE.parent / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", _HERE.parent / "work"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", _HERE.parent / "output"))

SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")


def wpath(*parts) -> Path:
    p = WORK_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Timer:
    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        self.t = time.time()
        print(f"[..] {self.msg}", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[ok] {self.msg}  ({time.time() - self.t:.1f}s)", flush=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------
# raw data access
# ----------------------------------------------------------------------------
def tsv_path(split: str, source: str) -> Path:
    return DATA_DIR / split / f"{split}_{source}.tsv"


def raw_parquet(split: str, source: str) -> Path:
    return wpath("raw", f"{split}_{source}.parquet")


def ensure_raw_parquet():
    """Convert the challenge TSVs into parquet once (quote_char=None: names
    contain stray quote characters, the files are plain TSV)."""
    for split in SPLITS:
        for src in SOURCES:
            out = raw_parquet(split, src)
            if out.exists():
                continue
            df = pl.read_csv(tsv_path(split, src), separator="\t", quote_char=None,
                             infer_schema=False, encoding="utf8")
            df.write_parquet(out)
            log("wrote", out, df.shape)
    gt_out = wpath("raw", "train_ground_truth.parquet")
    if not gt_out.exists():
        gt = pl.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", separator="\t",
                         quote_char=None, infer_schema=False)
        gt.write_parquet(gt_out)
        pairs = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
                 .explode("matched_entity_ids")
                 .filter(pl.col("matched_entity_ids") != "")
                 .rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"}))
        pairs.write_parquet(wpath("raw", "train_pairs.parquet"))


def load_raw(split: str, source: str) -> pl.DataFrame:
    return pl.read_parquet(raw_parquet(split, source))


def load_pairs() -> pl.DataFrame:
    return pl.read_parquet(wpath("raw", "train_pairs.parquet"))


# ----------------------------------------------------------------------------
# text helpers
# ----------------------------------------------------------------------------
# Indic blocks: Devanagari .. Malayalam (U+0900-U+0D7F) + Sinhala just in case
INDIC_RE = re.compile(r"[ऀ-෿]")
ZW_RE = re.compile(r"[​-‏⁠﻿]")
# a "raw token" = maximal run of non-space, non-separator characters
RAW_TOKEN_RE = re.compile(r"[^\s,;:!?()\[\]{}\"|/\\#@*+&<>=~^`$%_]+")


def has_indic(s: str) -> bool:
    return bool(INDIC_RE.search(s))


def strip_accents(s: str) -> str:
    # NFKD then drop combining marks. ONLY call on latin text (Indic scripts
    # use combining marks for vowels).
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
