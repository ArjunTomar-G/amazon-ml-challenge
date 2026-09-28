"""Stage 1: raw TSV -> parquet -> normalised records (+ word segmentation).

Outputs  WORK_DIR/norm/{split}_{source}.parquet  with one row per record.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from multiprocessing import Pool

import polars as pl

from common import SOURCES, SPLITS, Timer, ensure_raw_parquet, load_raw, log, wpath
from normalize import NF_DOMAIN, NF_HANDLE, normalize_address, normalize_name
from translit import get_dict, learn_dictionary

NF_SEG = 1024  # name token was produced by word segmentation

N_PROC = max(1, min(12, (os.cpu_count() or 4) - 2))


def _work(args):
    names, addrs, countries = args
    d = get_dict()
    out = []
    for n, a, c in zip(names, addrs, countries):
        r = normalize_name(n, d)
        r.update(normalize_address(a, c, d))
        out.append(r)
    return out


def norm_path(split, src):
    return wpath("norm", f"{split}_{src}.parquet")


def normalize_file(split: str, src: str, pool: Pool):
    df = load_raw(split, src)
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    ctry = df["country"].to_list()
    n = len(names)
    step = 50_000
    tasks = [(names[i:i + step], addrs[i:i + step], ctry[i:i + step]) for i in range(0, n, step)]
    rows = []
    for part in pool.imap(_work, tasks, chunksize=1):
        rows.extend(part)
    nd = pl.DataFrame(rows)
    out = pl.concat([df.select("entity_id", "country"), nd], how="horizontal")
    out = out.with_columns(pl.lit(int(src[-1])).cast(pl.Int8).alias("src"))
    return out


# ----------------------------------------------------------------------------
# word segmentation for glued tokens ("healthoncology", "coalitionharborcom")
# ----------------------------------------------------------------------------
class Segmenter:
    """Unigram-LM word segmentation (dynamic programming, min total cost)."""

    def __init__(self, counts: Counter, min_count: int, max_len: int = 20):
        self.max_len = max_len
        tot = sum(counts.values())
        self.cost = {w: math.log(tot / c) for w, c in counts.items()
                     if c >= min_count and len(w) >= 2 and w.isalpha()}
        self.unk = 1e6   # unknown chunks are never allowed

    def segment(self, s: str):
        n = len(s)
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                c = self.cost.get(s[j:i])
                if c is None:
                    continue
                v = best[j] + c + 2.0            # +2: prefer fewer words
                if v < best[i]:
                    best[i], back[i] = v, j
        if best[n] == math.inf:
            return [s]
        out, i = [], n
        while i > 0:
            out.append(s[back[i]:i])
            i = back[i]
        return out[::-1]


def segment_names(frames: dict, split: str):
    """Split glued name tokens.  Only applied where gluing is known to happen:
    domains / handles / trailing 'com' (lenient: parts must be words seen >= 6
    times), or very long unknown tokens that split into 2-3 frequent words of
    length >= 4 (strict).  Typo'd words are left untouched so that fuzzy
    matching can still recover them."""
    cnt = Counter()
    cnt_s1 = Counter()
    for src, df in frames.items():
        w = 3 if src == "source1" else 1
        for s in df["n_all"].to_list():
            for t in s.split():
                cnt[t] += w
                if src == "source1":
                    cnt_s1[t] += 1
    seg_lenient = Segmenter(cnt, min_count=6)
    seg_strict = Segmenter(cnt, min_count=30)

    def fix(s, flags):
        # a token is a "real word" if the clean reference source (S1) uses it
        domainish = bool(flags & (NF_DOMAIN | NF_HANDLE))
        out, changed = [], False
        for t in s.split():
            cand, com = t, False
            if t.endswith("com") and len(t) > 6 and cnt_s1.get(t, 0) < 2:
                cand, com = t[:-3], True
            if cnt_s1.get(cand, 0) < 2 and cand.isalpha() and len(cand) >= 6:
                if domainish or com:
                    parts = seg_lenient.segment(cand)
                    ok = len(parts) >= 2
                elif len(cand) >= 12:
                    parts = seg_strict.segment(cand)
                    ok = 2 <= len(parts) <= 3 and all(len(p) >= 4 for p in parts)
                else:
                    ok = False
                if ok:
                    out.extend(parts)
                    changed = True
                    continue
            if cand != t:
                changed = True
            out.append(cand)
        return " ".join(out), changed

    for src, df in frames.items():
        nall, ncore, nalt, flg = [], [], [], []
        for a, c, al, f in zip(df["n_all"].to_list(), df["n_core"].to_list(),
                               df["n_alt"].to_list(), df["n_flags"].to_list()):
            a2, ch1 = fix(a, f)
            c2, ch2 = fix(c, f)
            al2, _ = fix(al, f) if al else (al, False)
            nall.append(a2)
            ncore.append(c2)
            nalt.append(al2)
            flg.append(f | NF_SEG if (ch1 or ch2) else f)
        frames[src] = df.with_columns(pl.Series("n_all", nall), pl.Series("n_core", ncore),
                                      pl.Series("n_alt", nalt), pl.Series("n_flags", flg))
        log(f"{split}/{src}: segmented names in", sum(1 for f in flg if f & NF_SEG), "records")
    return frames


def norm1_path(split, src):
    return wpath("norm1", f"{split}_{src}.parquet")


def norm2_path(split, src):
    return wpath("norm2", f"{split}_{src}.parquet")


def _pairs_rows_from_frames(frames) -> pl.DataFrame:
    from common import load_pairs
    s1 = frames["source1"].select("entity_id").with_row_index("e_row")
    q = pl.concat([frames["source2"].select("entity_id"), frames["source3"].select("entity_id")])         .with_row_index("q_row")
    p = load_pairs().join(q, left_on="mid", right_on="entity_id").join(s1, left_on="s1", right_on="entity_id")
    return p.select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))


def run():
    import canon
    with Timer("raw parquet"):
        ensure_raw_parquet()
    if not wpath("models", "translit_dict.pkl").exists():
        with Timer("learn transliteration dictionary"):
            learn_dictionary()
    # pass 1: rule-based normalisation (multiprocessing)
    with Pool(N_PROC) as pool:
        for split in SPLITS:
            for src in SOURCES:
                if norm1_path(split, src).exists():
                    continue
                with Timer(f"normalise {split}/{src}"):
                    normalize_file(split, src, pool).write_parquet(norm1_path(split, src))
    # pass 2: glued-name segmentation (per universe vocabulary)
    for split in SPLITS:
        if all(norm2_path(split, s).exists() for s in SOURCES):
            continue
        frames = {src: pl.read_parquet(norm1_path(split, src)) for src in SOURCES}
        with Timer(f"segment names {split}"):
            frames = segment_names(frames, split)
        for src, df in frames.items():
            df.write_parquet(norm2_path(split, src))
    # pass 3: learned canonicalisation (maps learned on train, applied everywhere)
    if not wpath("models", "canon_maps.pkl").exists():
        tr = {src: pl.read_parquet(norm2_path("train", src)) for src in SOURCES}
        with Timer("learn canonical token maps"):
            canon.learn(tr, _pairs_rows_from_frames(tr))
        del tr
    import pickle
    maps = pickle.load(open(wpath("models", "canon_maps.pkl"), "rb"))
    for split in SPLITS:
        if all(norm_path(split, s).exists() for s in SOURCES):
            continue
        frames = {src: pl.read_parquet(norm2_path(split, src)) for src in SOURCES}
        with Timer(f"canonicalise {split}"):
            frames = canon.apply(frames, maps)
        for src, df in frames.items():
            df.write_parquet(norm_path(split, src))


if __name__ == "__main__":
    run()
