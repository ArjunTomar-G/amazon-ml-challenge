"""Stage 2: candidate generation (blocking).

Every Source-2/3 record can match at most ONE Source-1 entity (the ground
truth is a many-to-one mapping), so blocking is formulated as retrieval: each
S2/S3 record ("query") retrieves its top-K Source-1 entities from an inverted
index built over Source-1, within the same country label.

Retrieval score = sum of IDF weights of shared blocking keys:

  n:<tok>           name core token                     (typo-free names)
  g:<trigram>       char-3-grams of the glued core name (typos, glued domains)
  a:<tok>           address token (words and numbers)
  h:<hn>|<tok>      house number x street token         (very selective)
  l:<tok>|<loc>     name token x locality token          (very selective)
  x:<tok>|<hn>      name token x house number            (very selective)

The retrieval kernel is a parallel Numba implementation of accumulate-and-
select over CSR posting lists (df-capped).  Output is a long table
(qid, s1id, score, rank) with K candidates per query.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit, prange

from common import SOURCES, Timer, log, wpath
from prep import norm_path

DF_CAP = 3000          # composite keys more frequent than this in S1 (per country) are ignored
GRAM_DF_CAP = 1500
UNI_DF_CAP = 12000     # single name / address tokens (only for "starved" queries)
LOW_DF_CAP = 1500      # unigram cap for queries that already have enough selective keys
MIN_SELECTIVE = 8      # a query with fewer usable keys of df <= LOW_DF_CAP is "starved"
K_RETRIEVE = 40
STOP_ADDR = {"st", "ave", "rd", "dr", "ct", "pl", "ln", "blvd", "n", "s", "e", "w", "rue",
             "allee", "chem", "floor", "no", "road", "near", "opp", "nagar", "colony",
             "sector", "marg", "cross", "main", "block", "th"}

FAMILIES = ["n", "g", "a", "h", "l", "x", "f", "b", "s"]
FAMILY_W = np.array([1.0, 0.35, 0.8, 1.2, 1.0, 1.0, 1.5, 1.0, 1.0], np.float32)
FAMILY_CAP = np.array([UNI_DF_CAP, GRAM_DF_CAP, UNI_DF_CAP, DF_CAP, DF_CAP, DF_CAP, DF_CAP,
                       DF_CAP, DF_CAP], np.int64)


def _explode_tokens(df: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    return (df.select("idx", pl.col(col).str.split(" ").alias(out))
            .explode(out)
            .filter(pl.col(out).str.len_chars() >= 2))


def build_keys(df: pl.DataFrame) -> pl.DataFrame:
    """df: idx + normalised columns -> long frame (idx, key:str, fam:str)."""
    name = pl.concat([
        _explode_tokens(df, "n_core", "t"),
        _explode_tokens(df.filter(pl.col("n_alt") != ""), "n_alt", "t"),
    ]).unique()
    addr = _explode_tokens(df, "a_tok", "t").filter(~pl.col("t").is_in(list(STOP_ADDR))).unique()
    parts = [
        name.select("idx", ("n:" + pl.col("t")).alias("key"), pl.lit("n").alias("fam")),
        addr.select("idx", ("a:" + pl.col("t")).alias("key"), pl.lit("a").alias("fam")),
    ]
    # glued-name trigrams
    glued = df.select("idx", pl.col("n_core").str.replace_all(" ", "").alias("g"))
    glued = glued.filter(pl.col("g").str.len_chars() >= 3)
    grams = (glued.with_columns(
        pl.int_ranges(0, pl.col("g").str.len_chars() - 2).alias("p"))
        .explode("p")
        .select("idx", ("g:" + pl.col("g").str.slice(pl.col("p"), 3)).alias("key"))
        .unique()
        .with_columns(pl.lit("g").alias("fam")))
    parts.append(grams)
    # whole name (sorted core tokens) and glued name: exact-name keys
    parts.append(df.filter(pl.col("n_core") != "").select(
        "idx", ("f:" + pl.col("n_core").str.split(" ").list.sort().list.join(" ")).alias("key"),
        pl.lit("f").alias("fam")))
    parts.append(df.filter(pl.col("n_core").str.len_chars() >= 4).select(
        "idx", ("f:" + pl.col("n_core").str.replace_all(" ", "")).alias("key"),
        pl.lit("f").alias("fam")))
    # unordered name-token pairs
    pairs_ = (name.join(name, on="idx", suffix="2")
              .filter(pl.col("t") < pl.col("t2")))
    parts.append(pairs_.select("idx", ("b:" + pl.col("t") + "|" + pl.col("t2")).alias("key"),
                               pl.lit("b").alias("fam")))
    # house number (+ single-digit-deletion variants) x street tokens
    hn = df.filter(pl.col("a_hn") != "").select("idx", "a_hn", pl.col("a_street").str.split(" ").alias("t"))
    hn_var = pl.concat([
        hn.select("idx", pl.col("a_hn").alias("v"), "t"),
        hn.filter(pl.col("a_hn").str.contains(r"^\d{3,}$")).select(
            "idx", pl.col("a_hn").str.slice(0, pl.col("a_hn").str.len_chars() - 1).alias("v"), "t"),
        hn.filter(pl.col("a_hn").str.contains(r"^\d{3,}$")).select(
            "idx", pl.col("a_hn").str.slice(1).alias("v"), "t"),
    ])
    hs = hn_var.explode("t").filter((pl.col("t").str.len_chars() >= 2) & ~pl.col("t").is_in(list(STOP_ADDR)))
    parts.append(hs.select("idx", ("h:" + pl.col("v") + "|" + pl.col("t")).alias("key"),
                           pl.lit("h").alias("fam")).unique())
    # street token x locality token
    st = (df.select("idx", pl.col("a_street").str.split(" ").alias("st")).explode("st")
          .filter((pl.col("st").str.len_chars() >= 3) & ~pl.col("st").is_in(list(STOP_ADDR))
                  & ~pl.col("st").str.contains(r"^\d+$")).unique())
    # name token x locality token
    loc = (df.select("idx", pl.col("a_loc").str.replace_all(r" \| ", " ").str.split(" ").alias("lt"))
           .explode("lt").filter(pl.col("lt").str.len_chars() >= 3)
           .filter(~pl.col("lt").is_in(list(STOP_ADDR))).unique())
    sl = st.join(loc, on="idx")
    parts.append(sl.select("idx", ("s:" + pl.col("st") + "|" + pl.col("lt")).alias("key"),
                           pl.lit("s").alias("fam")))
    nl = name.join(loc, on="idx")
    parts.append(nl.select("idx", ("l:" + pl.col("t") + "|" + pl.col("lt")).alias("key"),
                           pl.lit("l").alias("fam")))
    # name token x house number
    nh = name.join(df.filter(pl.col("a_hn") != "").select("idx", "a_hn"), on="idx")
    parts.append(nh.select("idx", ("x:" + pl.col("t") + "|" + pl.col("a_hn")).alias("key"),
                           pl.lit("x").alias("fam")))
    out = pl.concat(parts)
    fam_code = pl.col("fam").replace_strict({f: i for i, f in enumerate(FAMILIES)}, return_dtype=pl.UInt8)
    return out.select("idx", pl.col("key").hash(seed=7).alias("h"), fam_code.alias("fam"))


@njit(parallel=True, cache=True)
def _search(q_ptr, q_off, q_cnt, q_w, post, n_index, K, n_blocks):
    nq = q_ptr.shape[0] - 1
    out_i = np.full((nq, K), -1, np.int32)
    out_s = np.zeros((nq, K), np.float32)
    bs = (nq + n_blocks - 1) // n_blocks
    for b in prange(n_blocks):
        acc = np.zeros(n_index, np.float32)
        touched = np.empty(n_index, np.int32)
        topv = np.zeros(K, np.float32)
        topi = np.full(K, -1, np.int32)
        q0 = b * bs
        q1 = min(nq, q0 + bs)
        for q in range(q0, q1):
            nt = 0
            for j in range(q_ptr[q], q_ptr[q + 1]):
                w = q_w[j]
                o = q_off[j]
                for p in range(o, o + q_cnt[j]):
                    s = post[p]
                    if acc[s] == 0.0:
                        touched[nt] = s
                        nt += 1
                    acc[s] += w
            for k in range(K):
                topv[k] = 0.0
                topi[k] = -1
            for t in range(nt):
                s = touched[t]
                v = acc[s]
                acc[s] = 0.0
                if v > topv[K - 1]:
                    pos = K - 1
                    while pos > 0 and topv[pos - 1] < v:
                        topv[pos] = topv[pos - 1]
                        topi[pos] = topi[pos - 1]
                        pos -= 1
                    topv[pos] = v
                    topi[pos] = s
            for k in range(K):
                out_i[q, k] = topi[k]
                out_s[q, k] = topv[k]
    return out_i, out_s


def load_norm(split: str, src: str, cols=None) -> pl.DataFrame:
    df = pl.read_parquet(norm_path(split, src))
    return df if cols is None else df.select(cols)


KEY_COLS = ["entity_id", "country", "n_core", "n_alt", "a_tok", "a_hn", "a_street", "a_loc"]


def load_universe(split: str, cols=None):
    """S1 table and query table (S2 then S3), each with a global row index."""
    s1 = load_norm(split, "source1", cols).with_row_index("row")
    qs = pl.concat([load_norm(split, s, cols) for s in ("source2", "source3")]).with_row_index("row")
    return s1, qs


ADDR_FAMS = np.array([FAMILIES.index(f) for f in ("a", "h", "s")], np.uint8)
K_ALL = 30
K_ADDR = 10


class Index:
    """Inverted index over the Source-1 records of one country."""

    def __init__(self, s1c: pl.DataFrame):
        self.n = s1c.height
        k1 = build_keys(s1c.with_row_index("idx"))
        k1 = k1.unique(["idx", "h"]).sort("h")
        h_sorted = k1["h"].to_numpy()
        self.post = k1["idx"].to_numpy().astype(np.int32)
        self.uniq, self.first, self.cnt = np.unique(h_sorted, return_index=True, return_counts=True)
        fam = k1["fam"].to_numpy()[self.first]
        # normalised by log(1+N) so retrieval scores are comparable across universes
        self.idf = (np.log1p(self.n / self.cnt) / np.log1p(self.n)).astype(np.float32) * FAMILY_W[fam]
        self.ok = self.cnt <= FAMILY_CAP[fam]
        self.e_rows = s1c["row"].to_numpy().astype(np.int32)

    def lookup(self, kq: pl.DataFrame, nq: int, fams=None):
        """Query keys -> CSR (q_ptr, offsets, counts, weights) of usable keys."""
        if fams is not None:
            kq = kq.filter(pl.col("fam").is_in(fams.tolist()))
        hq = kq["h"].to_numpy()
        qi = kq["idx"].to_numpy().astype(np.int64)
        pos = np.minimum(np.searchsorted(self.uniq, hq), len(self.uniq) - 1)
        hit = (self.uniq[pos] == hq) & self.ok[pos]
        qi, pos = qi[hit], pos[hit]
        # adaptive capping: frequent unigrams only for queries lacking selective keys
        c = self.cnt[pos]
        selective = c <= LOW_DF_CAP
        n_sel = np.bincount(qi[selective], minlength=nq)
        keep = selective | (n_sel[qi] < MIN_SELECTIVE)
        qi, pos = qi[keep], pos[keep]
        order = np.argsort(qi, kind="stable")
        qi, pos = qi[order], pos[order]
        q_ptr = np.zeros(nq + 1, np.int64)
        np.cumsum(np.bincount(qi, minlength=nq), out=q_ptr[1:])
        return q_ptr, self.first[pos].astype(np.int64), self.cnt[pos].astype(np.int64), self.idf[pos]

    def search(self, kq, nq, K, fams=None):
        q_ptr, off, cnt, w = self.lookup(kq, nq, fams)
        oi, os_ = _search(q_ptr, off, cnt, w, self.post, self.n, K, 256)
        return oi, os_


def _long(oi, os_, q_rows, e_rows, K, prefix):
    nq = oi.shape[0]
    flat = oi.ravel()
    valid = flat >= 0
    return pl.DataFrame({
        "q_row": np.repeat(q_rows, K)[valid],
        "e_row": e_rows[flat[valid]],
        f"s_{prefix}": os_.ravel()[valid],
        f"r_{prefix}": np.tile(np.arange(K, dtype=np.int16), nq)[valid],
    })


def retrieve_chunk(index: Index, part: pl.DataFrame) -> pl.DataFrame:
    """Two-channel retrieval for a chunk of queries (same country)."""
    kq = build_keys(part.with_row_index("idx")).unique(["idx", "h"])
    nq = part.height
    q_rows = part["row"].to_numpy().astype(np.int32)
    oi, os_ = index.search(kq, nq, K_ALL)
    a = _long(oi, os_, q_rows, index.e_rows, K_ALL, "all")
    oi, os_ = index.search(kq, nq, K_ADDR, ADDR_FAMS)
    b = _long(oi, os_, q_rows, index.e_rows, K_ADDR, "addr")
    c = a.join(b, on=["q_row", "e_row"], how="full", coalesce=True)
    # query-level context of the two channels
    c = c.with_columns(
        pl.col("s_all").max().over("q_row").alias("top_all"),
        pl.col("s_addr").max().over("q_row").alias("top_addr"),
    )
    return c


def retrieve(split: str, q_chunk: int = 500_000, q_sample: float | None = None, seed: int = 0,
             chunk_fn=None, frames=None) -> pl.DataFrame:
    """Two-channel retrieval over a universe.  `chunk_fn(cands)` may
    post-process (e.g. prune) every chunk before it is collected."""
    s1, qs = frames if frames is not None else load_universe(split, KEY_COLS)
    if q_sample is not None:
        qs = qs.sample(fraction=q_sample, seed=seed)
    results = []
    for country in sorted(qs["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        qc = qs.filter(pl.col("country") == country)
        if qc.height == 0 or s1c.height == 0:
            continue
        with Timer(f"[{split}/{country}] index {s1c.height} S1 records"):
            index = Index(s1c)
        for c0 in range(0, qc.height, q_chunk):
            part = qc.slice(c0, q_chunk)
            with Timer(f"[{split}/{country}] retrieve {c0}..{c0 + part.height}"):
                c = retrieve_chunk(index, part)
                if chunk_fn is not None:
                    c = chunk_fn(c)
                results.append(c)
        del index
    return pl.concat(results, how="diagonal_relaxed")


def true_pairs_rows(split: str = "train") -> pl.DataFrame:
    """Ground-truth pairs expressed as (q_row, e_row)."""
    from common import load_pairs
    s1, qs = load_universe(split, ["entity_id"])
    p = load_pairs()
    p = p.join(qs.rename({"row": "q_row"}), left_on="mid", right_on="entity_id")
    p = p.join(s1.rename({"row": "e_row"}), left_on="s1", right_on="entity_id")
    return p.select(pl.col("q_row").cast(pl.Int32), pl.col("e_row").cast(pl.Int32))


def candidate_recall(cands: pl.DataFrame, tp: pl.DataFrame, rank_col: str = "rank",
                     ks=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40)):
    """Recall of true pairs among candidates, restricted to the queried q_rows."""
    qset = cands.select("q_row").unique()
    tp = tp.join(qset, on="q_row")
    j = tp.join(cands.select("q_row", "e_row", rank_col), on=["q_row", "e_row"], how="left")
    tot = max(1, j.height)
    out = {k: round(float((j[rank_col].is_not_null() & (j[rank_col] < k)).sum()) / tot, 5) for k in ks}
    out["any"] = round(float(j[rank_col].is_not_null().sum()) / tot, 5)
    return out


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else None
    with Timer(f"retrieve {split}"):
        c = retrieve(split, q_sample=frac)
    name = f"{split}_retrieval.parquet" if frac is None else f"{split}_retrieval_sample.parquet"
    c.write_parquet(wpath("cand", name))
    log("candidates:", c.height)
    if split == "train":
        tp = true_pairs_rows()
        print("all-channel :", candidate_recall(c, tp, "r_all"))
        print("addr-channel:", candidate_recall(c, tp, "r_addr"))
