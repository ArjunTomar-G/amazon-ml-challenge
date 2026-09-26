"""Candidate generation (blocking), run per country.

Three channels, unioned per Source-1 entity ("U8" in the EDA report):
  1. joint retrieval: IDF-weighted overlap of name tokens (with the learned Indic
     dictionary, hand stop list removed) and canonical address tokens; top-K;
  2. fuzzy-name channel: char-3-gram MinHash bands of the compact core name,
     ranked by number of colliding bands; top-K (min collisions);
  3. no-address channel: joint-retrieval scores restricted to S2/S3 records
     without an address; top-K.
Keys whose document frequency exceeds a cap are purged (block purging).
"""

import math

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

_GOLD = np.uint64(0x9E3779B97F4A7C15)


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------

def _mix(a, b):
    with np.errstate(over="ignore"):
        return a ^ (b + _GOLD + (a << np.uint64(6)) + (a >> np.uint64(2)))


def _hash_strings(arr):
    d = pc.dictionary_encode(arr)
    h = pd.util.hash_array(d.dictionary.to_numpy(zero_copy_only=False), categorize=False)
    return h[d.indices.to_numpy(zero_copy_only=False)]


_TAG = {t: pd.util.hash_array(np.array([t], dtype=object), categorize=False)[0] for t in ("n", "a")}


def token_keys(col, tag):
    """(row, key) for every whitespace token of an Arrow string array."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    lists = pc.utf8_split_whitespace(col)
    toks = pc.list_flatten(lists)
    rows = pc.list_parent_indices(lists).to_numpy().astype(np.int64)
    if len(toks) == 0:
        return rows, np.zeros(0, np.uint64)
    keys = _mix(_hash_strings(toks), np.uint64(_TAG[tag]))
    return rows, keys


def arrow_buffers(arr):
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if arr.type != pa.large_string():
        arr = arr.cast(pa.large_string())
    bufs = arr.buffers()
    offs = np.frombuffer(bufs[1], dtype=np.int64)[arr.offset: arr.offset + len(arr) + 1]
    dat = np.frombuffer(bufs[2], dtype=np.uint8) if bufs[2] is not None else np.zeros(1, np.uint8)
    return offs, dat


@nb.njit(parallel=True, cache=True)
def _minhash_bands(offs, dat, q, a, b, n_bands, rows_per_band):
    n = len(offs) - 1
    n_perm = n_bands * rows_per_band
    out = np.zeros((n, n_bands), np.uint64)
    P = np.uint64(4294967311)
    for i in nb.prange(n):
        s, e = offs[i], offs[i + 1]
        buf = np.empty(e - s, np.uint8)
        L = 0
        for j in range(s, e):
            if dat[j] != 32:
                buf[L] = dat[j]
                L += 1
        if L == 0:
            continue
        mins = np.full(n_perm, np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64)
        nq = L - q + 1 if L >= q else 1
        for j in range(nq):
            h = np.uint64(2166136261)
            for t in range(j, min(j + q, L)):
                h = ((h ^ np.uint64(buf[t])) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)
            for p in range(n_perm):
                v = (a[p] * h + b[p]) % P
                if v < mins[p]:
                    mins[p] = v
        for bb in range(n_bands):
            hk = np.uint64(1469598103934665603) + np.uint64(bb)
            for p in range(bb * rows_per_band, (bb + 1) * rows_per_band):
                hk = (hk ^ mins[p]) * np.uint64(1099511628211)
            out[i, bb] = hk if hk != 0 else np.uint64(1)
    return out


def minhash_keys(col, n_bands=16, rows_per_band=2, q=3, seed=1):
    rng = np.random.default_rng(seed)
    n_perm = n_bands * rows_per_band
    a = rng.integers(1, 4294967311, size=n_perm, dtype=np.uint64)
    b = rng.integers(0, 4294967311, size=n_perm, dtype=np.uint64)
    offs, dat = arrow_buffers(col)
    bands = _minhash_bands(offs, dat, q, a, b, n_bands, rows_per_band)
    rows = np.repeat(np.arange(bands.shape[0], dtype=np.int64), n_bands)
    keys = bands.reshape(-1)
    ok = keys != 0
    return rows[ok], keys[ok]


# ---------------------------------------------------------------------------
# postings index
# ---------------------------------------------------------------------------

class Index:
    """Sorted unique (truncated) keys -> postings of record positions."""

    def __init__(self, rows, keys, n_records, cap, query_keys=None):
        """query_keys: if given, only keys that some query uses are kept (document
        frequencies are still counted over the whole corpus)."""
        self.bits = max(1, int(n_records).bit_length())
        self.n_records = n_records
        sh = np.uint64(self.bits)
        packed = ((keys >> sh) << sh) | rows.astype(np.uint64)
        del rows, keys
        packed.sort()
        if len(packed):
            dup = np.empty(len(packed), bool)
            dup[0] = False
            dup[1:] = packed[1:] == packed[:-1]
            packed = packed[~dup]
        kt = packed >> sh
        recs = (packed & np.uint64((1 << self.bits) - 1)).astype(np.int32)
        del packed
        if len(kt):
            brk = np.flatnonzero(kt[1:] != kt[:-1]) + 1
            starts = np.concatenate([[0], brk])
            df = np.diff(np.concatenate([starts, [len(kt)]]))
        else:
            starts = np.zeros(0, np.int64)
            df = np.zeros(0, np.int64)
        keep = df <= cap
        if query_keys is not None and len(starts):
            qk = np.unique(np.asarray(query_keys, np.uint64) >> sh)
            uk = kt[starts]
            pos = np.minimum(np.searchsorted(qk, uk), max(len(qk) - 1, 0))
            keep &= (qk[pos] == uk) if len(qk) else False
        self.ukeys = kt[starts[keep]]
        self.df = df[keep].astype(np.int64)
        self.p_recs = recs[np.repeat(keep, df)]
        self.p_indptr = np.zeros(len(self.ukeys) + 1, np.int64)
        np.cumsum(self.df, out=self.p_indptr[1:])
        self.n_purged = int((~keep).sum())

    def lookup(self, keys):
        kt = keys >> np.uint64(self.bits)
        pos = np.searchsorted(self.ukeys, kt)
        pos = np.minimum(pos, max(len(self.ukeys) - 1, 0))
        ok = (self.ukeys[pos] == kt) if len(self.ukeys) else np.zeros(len(kt), bool)
        return np.where(ok, pos, -1)

    def restricted(self, allowed):
        """Same keys, postings limited to records where allowed[rec] is True."""
        sub = Index.__new__(Index)
        sub.bits, sub.n_records, sub.ukeys, sub.n_purged = self.bits, self.n_records, self.ukeys, 0
        keep = allowed[self.p_recs]
        sub.p_recs = self.p_recs[keep]
        seg = np.repeat(np.arange(len(self.ukeys)), self.df)[keep]
        sub.df = np.bincount(seg, minlength=len(self.ukeys)).astype(np.int64)
        sub.p_indptr = np.zeros(len(self.ukeys) + 1, np.int64)
        np.cumsum(sub.df, out=sub.p_indptr[1:])
        return sub


def group_queries(rows, keys, n_q, index, weight_fn):
    """CSR (indptr, key positions, weights) of deduplicated query keys."""
    if len(rows):
        order = np.lexsort((keys, rows))
        rows, keys = rows[order], keys[order]
        keep = np.ones(len(rows), bool)
        keep[1:] = (rows[1:] != rows[:-1]) | (keys[1:] != keys[:-1])
        rows, keys = rows[keep], keys[keep]
    kidx = index.lookup(keys) if len(keys) else np.zeros(0, np.int64)
    found = kidx >= 0
    rows, kidx = rows[found], kidx[found]
    indptr = np.zeros(n_q + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n_q), out=indptr[1:])
    w = weight_fn(index.df[kidx]) if len(kidx) else np.zeros(0)
    return indptr, kidx.astype(np.int64), np.asarray(w, np.float32)


def idf(n_docs):
    return lambda df: np.log((n_docs + 1.0) / (df + 1.0))


# ---------------------------------------------------------------------------
# query kernel
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _max_touch(q_indptr, q_kidx, p_indptr):
    best = 0
    for i in range(len(q_indptr) - 1):
        tot = 0
        for j in range(q_indptr[i], q_indptr[i + 1]):
            k = q_kidx[j]
            tot += p_indptr[k + 1] - p_indptr[k]
        best = max(best, tot)
    return best


@nb.njit(parallel=True, cache=True)
def _query_topk(q_indptr, q_kidx, q_w, p_indptr, p_recs, n_corpus, K, x_indptr, x_recs,
                n_threads, max_touch, min_score):
    """Per query: top-K records by summed key weight, followed by 'extra'
    records (other channels) with their weight too. Rank 0 = extra only."""
    n_q = len(q_indptr) - 1
    X = 0
    for i in range(n_q):
        X = max(X, x_indptr[i + 1] - x_indptr[i])
    W = K + X
    out_rec = np.full((n_q, W), -1, np.int32)
    out_score = np.zeros((n_q, W), np.float32)
    out_rank = np.zeros((n_q, W), np.int16)
    top1 = np.zeros(n_q, np.float32)
    touched_n = np.zeros(n_q, np.int32)
    step = (n_q + n_threads - 1) // n_threads
    for t in nb.prange(n_threads):
        score = np.zeros(n_corpus, np.float32)
        mark = np.zeros(n_corpus, np.uint8)
        touched = np.empty(max(max_touch, 1), np.int32)
        for i in range(t * step, min(n_q, (t + 1) * step)):
            nt = 0
            for j in range(q_indptr[i], q_indptr[i + 1]):
                k = q_kidx[j]
                w = q_w[j]
                for p in range(p_indptr[k], p_indptr[k + 1]):
                    r = p_recs[p]
                    if score[r] == 0.0:
                        touched[nt] = r
                        nt += 1
                    score[r] += w
            touched_n[i] = nt
            m = 0
            if nt > 0:
                cs = np.empty(nt, np.float32)
                for q in range(nt):
                    cs[q] = score[touched[q]]
                kk = min(K, nt)
                thr = -np.partition(-cs, kk - 1)[kk - 1]
                sel = np.empty(kk, np.int32)
                sel_s = np.empty(kk, np.float32)
                for q in range(nt):
                    if cs[q] > thr and m < kk:
                        sel[m] = touched[q]
                        sel_s[m] = cs[q]
                        m += 1
                for q in range(nt):
                    if m < kk and cs[q] == thr:
                        sel[m] = touched[q]
                        sel_s[m] = cs[q]
                        m += 1
                order = np.argsort(-sel_s[:m], kind="mergesort")
                w_ = 0
                for q in range(m):
                    if sel_s[order[q]] < min_score:
                        continue
                    r = sel[order[q]]
                    out_rec[i, w_] = r
                    out_score[i, w_] = sel_s[order[q]]
                    out_rank[i, w_] = q + 1
                    mark[r] = 1
                    w_ += 1
                m = w_
                if m > 0:
                    top1[i] = out_score[i, 0]
            for p in range(x_indptr[i], x_indptr[i + 1]):
                r = x_recs[p]
                if mark[r] == 0:
                    out_rec[i, m] = r
                    out_score[i, m] = score[r]
                    out_rank[i, m] = 0
                    mark[r] = 1
                    m += 1
            for q in range(nt):
                score[touched[q]] = 0.0
            for q in range(m):
                mark[out_rec[i, q]] = 0
    return out_rec, out_score, out_rank, top1, touched_n


def query(index, q_indptr, q_kidx, q_w, K, extras=None, n_threads=8, min_score=0.0):
    n_q = len(q_indptr) - 1
    if extras is None:
        x_indptr, x_recs = np.zeros(n_q + 1, np.int64), np.zeros(0, np.int32)
    else:
        x_indptr, x_recs = extras
    max_touch = int(_max_touch(q_indptr, q_kidx, index.p_indptr)) if len(q_kidx) else 0
    return _query_topk(q_indptr, q_kidx, q_w, index.p_indptr, index.p_recs, index.n_records, K,
                       x_indptr, x_recs.astype(np.int32), n_threads, max_touch, np.float32(min_score))


def rows_to_csr(mat):
    valid = mat >= 0
    indptr = np.zeros(mat.shape[0] + 1, np.int64)
    np.cumsum(valid.sum(1), out=indptr[1:])
    return indptr, mat[valid].astype(np.int32)


def union_csr(*csrs):
    """Row-wise union (order kept, duplicates allowed; the kernel dedupes)."""
    n = len(csrs[0][0]) - 1
    lens = sum(np.diff(c[0]) for c in csrs)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=indptr[1:])
    rows = np.concatenate([np.repeat(np.arange(n), np.diff(c[0])) for c in csrs])
    vals = np.concatenate([c[1] for c in csrs])
    order = np.argsort(rows, kind="stable")
    return indptr, vals[order].astype(np.int32)


@nb.njit(parallel=True, cache=True)
def _lookup_in_rows(cand, cand_q, ref_rec, ref_val):
    out = np.zeros(len(cand), np.float32)
    for i in nb.prange(len(cand)):
        q = cand_q[i]
        for j in range(ref_rec.shape[1]):
            if ref_rec[q, j] == cand[i]:
                out[i] = ref_val[q, j]
                break
    return out


# ---------------------------------------------------------------------------
# per-country candidate generation
# ---------------------------------------------------------------------------

DEFAULTS = {"k_joint": 20, "k_mh": 10, "k_empty": 10, "cap_joint": 50_000, "cap_mh": 20_000,
            "mh_bands": 16, "mh_rows": 2, "mh_min": 2.0}


def _slice(csr, b0, b1):
    indptr, *vals = csr
    lo, hi = indptr[b0], indptr[b1]
    return (indptr[b0:b1 + 1] - lo, *[v[lo:hi] for v in vals])


def candidates(q_names, q_addr, c_names, c_addr, c_empty, cfg, n_threads=8, log=print, block=200_000):
    """q_* / c_*: Arrow arrays (translit core names, canonical address tokens) of the
    Source-1 queries and the corpus of ONE country. Yields dicts of flat arrays,
    one row per (query, candidate) pair, ordered by query, for blocks of queries
    (q = query index within the country)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    n_q, n_c = len(q_names), len(c_names)
    rn, kn = token_keys(q_names, "n")
    ra, ka = token_keys(q_addr, "a")
    q_rows, q_keys = np.concatenate([rn, ra]), np.concatenate([kn, ka])
    # joint index: name + address tokens
    rn, kn = token_keys(c_names, "n")
    ra, ka = token_keys(c_addr, "a")
    joint = Index(np.concatenate([rn, ra]), np.concatenate([kn, ka]), n_c, cfg["cap_joint"], q_keys)
    del rn, kn, ra, ka
    jq = group_queries(q_rows, q_keys, n_q, joint, idf(n_c))
    del q_rows, q_keys
    # fuzzy-name channel
    rq, kq = minhash_keys(q_names, cfg["mh_bands"], cfg["mh_rows"])
    r, k = minhash_keys(c_names, cfg["mh_bands"], cfg["mh_rows"])
    mh = Index(r, k, n_c, cfg["cap_mh"], kq)
    del r, k
    mq = group_queries(rq, kq, n_q, mh, lambda df: np.ones(len(df)))
    del rq, kq
    # no-address channel (same keys/weights, postings restricted to address-less records)
    empty_idx = joint.restricted(np.asarray(c_empty, bool))
    total = 0
    for b0 in range(0, n_q, block):
        b1 = min(n_q, b0 + block)
        jqb, mqb = _slice(jq, b0, b1), _slice(mq, b0, b1)
        mh_rec, mh_cnt, _, _, _ = query(mh, *mqb, K=cfg["k_mh"], n_threads=n_threads, min_score=cfg["mh_min"])
        em_rec, _, _, _, _ = query(empty_idx, *jqb, K=cfg["k_empty"], n_threads=n_threads)
        extras = union_csr(rows_to_csr(mh_rec), rows_to_csr(em_rec))
        rec, score, rank, top1, touched = query(joint, *jqb, K=cfg["k_joint"], extras=extras,
                                                n_threads=n_threads)
        valid = rec >= 0
        q_loc = np.repeat(np.arange(b1 - b0, dtype=np.int32), valid.sum(1))
        cand = rec[valid]
        out = {
            "q": q_loc + b0, "c": cand,
            "joint_score": score[valid], "joint_rank": rank[valid].astype(np.float32),
            "mh_count": _lookup_in_rows(cand, q_loc, mh_rec, mh_cnt),
            "in_empty": (_lookup_in_rows(cand, q_loc, em_rec, np.ones(em_rec.shape, np.float32)) > 0
                         ).astype(np.float32),
            "top1": top1[q_loc], "log_touched": np.log1p(touched[q_loc]).astype(np.float32),
        }
        out["rel_score"] = np.where(out["top1"] > 0, out["joint_score"] / np.maximum(out["top1"], 1e-6), 0
                                    ).astype(np.float32)
        total += len(cand)
        yield out
    log(f"    candidates: {total:,} pairs for {n_q:,} queries "
        f"({total / max(n_q, 1):.1f} per query), corpus {n_c:,}")


_ = math
