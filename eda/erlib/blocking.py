"""Blocking evaluation engine.

A blocking strategy is a *key function*: given an Arrow table of records it
returns ``(row, key)`` pairs (uint64 keys, deduplicated per row). Two records
are candidates when they share at least ``k`` keys, ignoring keys whose block
(corpus document frequency) exceeds ``cap`` (block purging).

Evaluation is exact for a sample of Source-1 entities against the *full*
Source-2 + Source-3 corpus, streamed in chunks so memory stays bounded:

1. keys of the S1 sample are computed first;
2. the corpus is streamed; only (record, key) pairs whose key occurs in the
   sample are kept, which gives exact block sizes for every sample key;
3. a numba kernel walks the postings of each sample record, counts shared keys
   per corpus record and scores the resulting candidate set against the ground
   truth.
"""

import math

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import data

_GOLD = np.uint64(0x9E3779B97F4A7C15)


# --------------------------------------------------------------------------
# hashing helpers
# --------------------------------------------------------------------------

def mix(a, b):
    """Combine two uint64 hash arrays (boost::hash_combine style)."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return a ^ (b + _GOLD + (a << np.uint64(6)) + (a >> np.uint64(2)))


def hash_str_array(arr):
    """Arrow string array -> uint64 hashes (hashing each distinct value once)."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    d = pc.dictionary_encode(arr)
    vals = d.dictionary.to_numpy(zero_copy_only=False)
    h = pd.util.hash_array(vals, categorize=False)
    return h[d.indices.to_numpy(zero_copy_only=False)]


def hash_py_strings(strings):
    return pd.util.hash_array(np.asarray(strings, dtype=object), categorize=False)


def const_hash(s):
    return hash_py_strings([s])[0]


def tokenize(col):
    """Arrow string array -> (row index int64, token Arrow array)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    lists = pc.utf8_split_whitespace(col)
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy()
    return parent, flat


@nb.njit(cache=True)
def _group_dedupe(rows, keys, n_rows):
    """Counting-sort (row, key) pairs by row, then sort + dedupe keys inside each row."""
    starts = np.zeros(n_rows + 1, np.int64)
    for i in range(len(rows)):
        starts[rows[i] + 1] += 1
    for r in range(n_rows):
        starts[r + 1] += starts[r]
    pos = starts[:-1].copy()
    grouped = np.empty(len(keys), np.uint64)
    for i in range(len(rows)):
        r = rows[i]
        grouped[pos[r]] = keys[i]
        pos[r] += 1
    out_rows = np.empty(len(rows), np.int64)
    out_keys = np.empty(len(rows), np.uint64)
    m = 0
    for r in range(n_rows):
        seg = np.sort(grouped[starts[r]:starts[r + 1]])
        for j in range(len(seg)):
            if j == 0 or seg[j] != seg[j - 1]:
                out_rows[m] = r
                out_keys[m] = seg[j]
                m += 1
    return out_rows[:m], out_keys[:m]


def dedupe_rows(rows, keys):
    """Drop duplicate (row, key) pairs; result is grouped by row."""
    rows = np.asarray(rows, dtype=np.int64)
    keys = np.asarray(keys, dtype=np.uint64)
    if len(rows) == 0:
        return rows, keys
    return _group_dedupe(rows, keys, int(rows.max()) + 1)


def country_hash(tbl):
    return hash_str_array(tbl.column("cty"))


# --------------------------------------------------------------------------
# corpus / sample handling
# --------------------------------------------------------------------------

class Corpus:
    """Source-2 then Source-3 rows of a split, addressed by a global index."""

    def __init__(self, split="train"):
        self.split = split
        self.files = [data.prepared_path(split, 2), data.prepared_path(split, 3)]
        self.sizes = [pq.ParquetFile(f).metadata.num_rows for f in self.files]
        self.offsets = [0, self.sizes[0]]
        self.n = sum(self.sizes)
        ids = []
        for f in self.files:
            ids.append(pq.read_table(f, columns=["id"]).column("id").to_numpy())
        # composite key src*1e10 + id, sorted for lookups
        comp = np.concatenate([ids[0] + 2 * 10**10, ids[1] + 3 * 10**10])
        assert ids[0].max() < 10**10 and ids[1].max() < 10**10
        self._order = np.argsort(comp, kind="stable")
        self._sorted = comp[self._order]

    def index_of(self, src, ids):
        comp = np.asarray(ids, dtype=np.int64) + np.asarray(src, dtype=np.int64) * 10**10
        pos = np.searchsorted(self._sorted, comp)
        assert np.all(self._sorted[pos] == comp), "unknown corpus id"
        return self._order[pos]

    def iter_chunks(self, columns, batch_size=1_000_000):
        """Yield (global_offset, Arrow table) chunks."""
        for f, off in zip(self.files, self.offsets):
            pf = pq.ParquetFile(f)
            start = off
            for rb in pf.iter_batches(batch_size=batch_size, columns=columns):
                t = pa.Table.from_batches([rb])
                yield start, t
                start += t.num_rows

    def column(self, name):
        parts = [pq.read_table(f, columns=[name]).column(name) for f in self.files]
        return pa.chunked_array(parts[0].chunks + parts[1].chunks)

    def take(self, indices, columns):
        """Rows at global indices (any order) as an Arrow table in that order."""
        indices = np.asarray(indices, dtype=np.int64)
        out = {}
        src_is3 = indices >= self.sizes[0]
        local = np.where(src_is3, indices - self.sizes[0], indices)
        for col in columns:
            vals = [None, None]
            for s, f in enumerate(self.files):
                m = src_is3 if s == 1 else ~src_is3
                if m.any():
                    c = pq.read_table(f, columns=[col]).column(col).combine_chunks()
                    vals[s] = (np.nonzero(m)[0], c.take(pa.array(local[m])))
                    del c
            # stitch back in the requested order
            order = np.concatenate([v[0] for v in vals if v is not None])
            arr = pa.concat_arrays([v[1] for v in vals if v is not None])
            inv = np.empty(len(order), np.int64)
            inv[order] = np.arange(len(order))
            out[col] = arr.take(pa.array(inv))
        out["src"] = pa.array(np.where(src_is3, 3, 2).astype(np.int8))
        return pa.table(out)


# --------------------------------------------------------------------------
# numba kernels
# --------------------------------------------------------------------------

@nb.njit(cache=True)
def _match_keys(sorted_u, keys, rows, row_offset, out_rec, out_kpos, n_out):
    """Append (global row, position-in-sorted_u) for keys present in sorted_u."""
    m = len(sorted_u)
    for i in range(len(keys)):
        kk = keys[i]
        lo, hi = 0, m
        while lo < hi:
            mid = (lo + hi) >> 1
            if sorted_u[mid] < kk:
                lo = mid + 1
            else:
                hi = mid
        if lo < m and sorted_u[lo] == kk:
            out_rec[n_out] = row_offset + rows[i]
            out_kpos[n_out] = lo
            n_out += 1
    return n_out


@nb.njit(parallel=True, cache=True)
def _candidates(s_indptr, s_kidx, s_w, p_indptr, p_recs, n_corpus, k, gt_owner,
                n_threads, max_touch, topk_max, want_rank):
    """Per sample record: number of candidates (records sharing >= k kept keys),
    number of true matches among them, and (optionally) the rank of each true
    match under the summed key weights."""
    n_s = len(s_indptr) - 1
    ncand = np.zeros(n_s, np.int64)
    ntrue = np.zeros(n_s, np.int64)
    found = np.zeros(n_corpus, np.bool_)
    rank = np.full(n_corpus, -1, np.int32)
    topk = np.full((n_s, topk_max), -1, np.int32)
    step = (n_s + n_threads - 1) // n_threads
    for t in nb.prange(n_threads):
        cnt = np.zeros(n_corpus, np.uint8)
        score = np.zeros(n_corpus if want_rank else 1, np.float32)
        touched = np.empty(max_touch, np.int32)
        lo = t * step
        hi = min(n_s, lo + step)
        for i in range(lo, hi):
            nt = 0
            for j in range(s_indptr[i], s_indptr[i + 1]):
                ki = s_kidx[j]
                if ki < 0:
                    continue
                w = s_w[j]
                for p in range(p_indptr[ki], p_indptr[ki + 1]):
                    r = p_recs[p]
                    if cnt[r] == 0:
                        touched[nt] = r
                        nt += 1
                    if cnt[r] < 255:
                        cnt[r] += 1
                    if want_rank:
                        score[r] += w
            nc = 0
            ntr = 0
            if want_rank:
                # compact candidate list with scores
                cand = np.empty(nt, np.int32)
                cs = np.empty(nt, np.float32)
                for q in range(nt):
                    r = touched[q]
                    if cnt[r] >= k:
                        cand[nc] = r
                        cs[nc] = score[r]
                        nc += 1
                if nc > 0:
                    kk = min(topk_max, nc)
                    # kk-th largest score via partition (O(nc)), then sort only the head
                    thr = -np.partition(-cs[:nc], kk - 1)[kk - 1]
                    sel = np.empty(kk, np.int32)
                    sel_s = np.empty(kk, np.float32)
                    m = 0
                    for q in range(nc):
                        if cs[q] > thr and m < kk:
                            sel[m] = cand[q]
                            sel_s[m] = cs[q]
                            m += 1
                    for q in range(nc):
                        if m < kk and cs[q] == thr:
                            sel[m] = cand[q]
                            sel_s[m] = cs[q]
                            m += 1
                    order = np.argsort(-sel_s[:m], kind="mergesort")
                    for q in range(m):
                        topk[i, q] = sel[order[q]]
                    # rank of true matches: exact inside the head, topk_max+1 beyond it
                    for q in range(m):
                        r = sel[order[q]]
                        if gt_owner[r] == i:
                            rank[r] = q + 1
                    for q in range(nc):
                        r = cand[q]
                        if gt_owner[r] == i:
                            ntr += 1
                            found[r] = True
                            if rank[r] < 0:
                                rank[r] = topk_max + 1
                for q in range(nt):
                    score[touched[q]] = 0.0
            else:
                for q in range(nt):
                    r = touched[q]
                    if cnt[r] >= k:
                        nc += 1
                        if gt_owner[r] == i:
                            ntr += 1
                            found[r] = True
            for q in range(nt):
                cnt[touched[q]] = 0
            ncand[i] = nc
            ntrue[i] = ntr
    return ncand, ntrue, found, rank, topk


@nb.njit(parallel=True, cache=True)
def _candidate_lists(s_indptr, s_kidx, p_indptr, p_recs, n_corpus, k, out_indptr,
                     n_threads, max_touch):
    """Explicit candidate lists (CSR, rows given by out_indptr from a counting run)."""
    n_s = len(s_indptr) - 1
    out = np.empty(out_indptr[-1], np.int32)
    step = (n_s + n_threads - 1) // n_threads
    for t in nb.prange(n_threads):
        cnt = np.zeros(n_corpus, np.uint8)
        touched = np.empty(max_touch, np.int32)
        lo = t * step
        hi = min(n_s, lo + step)
        for i in range(lo, hi):
            nt = 0
            for j in range(s_indptr[i], s_indptr[i + 1]):
                ki = s_kidx[j]
                if ki < 0:
                    continue
                for p in range(p_indptr[ki], p_indptr[ki + 1]):
                    r = p_recs[p]
                    if cnt[r] == 0:
                        touched[nt] = r
                        nt += 1
                    if cnt[r] < 255:
                        cnt[r] += 1
            w = out_indptr[i]
            for q in range(nt):
                r = touched[q]
                if cnt[r] >= k:
                    out[w] = r
                    w += 1
                cnt[r] = 0
    return out


@nb.njit(parallel=True, cache=True)
def _union_eval(indptrs, recs, n_rows, n_lists, n_corpus, gt_owner, n_threads):
    """Union of several candidate CSRs (stacked: list l, row i -> indptrs[l, i:i+2]
    into recs). Returns union size and true matches per row, and found flags."""
    ncand = np.zeros(n_rows, np.int64)
    ntrue = np.zeros(n_rows, np.int64)
    found = np.zeros(n_corpus, np.bool_)
    step = (n_rows + n_threads - 1) // n_threads
    for t in nb.prange(n_threads):
        mark = np.zeros(n_corpus, np.bool_)
        lo = t * step
        hi = min(n_rows, lo + step)
        for i in range(lo, hi):
            nc = 0
            ntr = 0
            for l in range(n_lists):
                for p in range(indptrs[l, i], indptrs[l, i + 1]):
                    r = recs[p]
                    if not mark[r]:
                        mark[r] = True
                        nc += 1
                        if gt_owner[r] == i:
                            ntr += 1
                            found[r] = True
            for l in range(n_lists):
                for p in range(indptrs[l, i], indptrs[l, i + 1]):
                    mark[recs[p]] = False
            ncand[i] = nc
            ntrue[i] = ntr
    return ncand, ntrue, found


@nb.njit(cache=True)
def _build_postings(recs, kpos, n_keys):
    """Counting sort of (record, key position) pairs into CSR postings."""
    indptr = np.zeros(n_keys + 1, np.int64)
    for i in range(len(kpos)):
        indptr[kpos[i] + 1] += 1
    for j in range(n_keys):
        indptr[j + 1] += indptr[j]
    pos = indptr[:-1].copy()
    out = np.empty(len(recs), np.int32)
    for i in range(len(kpos)):
        out[pos[kpos[i]]] = recs[i]
        pos[kpos[i]] += 1
    return indptr, out


@nb.njit(cache=True)
def _max_touch(s_indptr, s_kidx, p_indptr):
    best = 0
    for i in range(len(s_indptr) - 1):
        tot = 0
        for j in range(s_indptr[i], s_indptr[i + 1]):
            ki = s_kidx[j]
            if ki >= 0:
                tot += p_indptr[ki + 1] - p_indptr[ki]
        if tot > best:
            best = tot
    return best


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

class Sample:
    """S1 sample + ground truth expressed in corpus indices."""

    def __init__(self, table, corpus, pair_s1, pair_src, pair_id):
        self.table = table
        self.n = table.num_rows
        ids = table.column("id").to_numpy()
        order = np.argsort(ids)
        sel = np.isin(pair_s1, ids)
        self.pair_sample = order[np.searchsorted(ids[order], pair_s1[sel])].astype(np.int64)
        self.pair_rec = corpus.index_of(pair_src[sel], pair_id[sel])
        self.pair_src = pair_src[sel]
        self.gt_count = np.bincount(self.pair_sample, minlength=self.n)
        self.gt_owner = np.full(corpus.n, -1, np.int32)
        self.gt_owner[self.pair_rec] = self.pair_sample.astype(np.int32)
        self.n_corpus = corpus.n


def f05_ceiling(ntrue, gt_count):
    r = np.divide(ntrue, gt_count, out=np.zeros(len(gt_count)), where=gt_count > 0)
    f = np.where(gt_count == 0, 1.0, 1.25 * r / (0.25 + r + 1e-12))
    return f


class Index:
    """Postings of the sample's keys over the full corpus (exact block sizes)."""

    def __init__(self, keyfn, sample, corpus, columns, hard_cap=None, batch_size=1_000_000):
        s_rows, s_keys = keyfn(sample.table)
        s_rows, s_keys = dedupe_rows(np.asarray(s_rows), np.asarray(s_keys, dtype=np.uint64))
        u = np.unique(s_keys)
        self.sample = sample
        self.u = u
        self.s_rows, self.s_keys = s_rows, s_keys
        self.s_pos = np.searchsorted(u, s_keys)

        def stream(keep_mask):
            recs, kpos = [], []
            df = np.zeros(len(u), np.int64)
            for off, tbl in corpus.iter_chunks(columns, batch_size=batch_size):
                c_rows, c_keys = keyfn(tbl)
                c_rows, c_keys = dedupe_rows(np.asarray(c_rows), np.asarray(c_keys, dtype=np.uint64))
                out_rec = np.empty(len(c_keys), np.int32)
                out_kpos = np.empty(len(c_keys), np.int32)
                n = _match_keys(u, c_keys, c_rows, off, out_rec, out_kpos, 0)
                out_rec, out_kpos = out_rec[:n], out_kpos[:n]
                df += np.bincount(out_kpos, minlength=len(u))
                if keep_mask is not None:
                    m = keep_mask[out_kpos]
                    out_rec, out_kpos = out_rec[m], out_kpos[m]
                recs.append(out_rec.copy())
                kpos.append(out_kpos.copy())
                del c_rows, c_keys, tbl
            return np.concatenate(recs), np.concatenate(kpos), df

        if hard_cap is None:
            recs, kpos, df = stream(None)
        else:  # pass 1: block sizes only; pass 2: keep postings of blocks <= hard_cap
            recs, kpos, df = stream(np.zeros(len(u), bool))
            recs, kpos, _ = stream(df <= hard_cap)
        self.df = df
        self.hard_cap = hard_cap
        self.p_indptr, self.p_recs = _build_postings(recs, kpos, len(u))
        del recs, kpos
        self.s_indptr = np.zeros(sample.n + 1, np.int64)
        np.cumsum(np.bincount(s_rows, minlength=sample.n), out=self.s_indptr[1:])

    def evaluate(self, k=1, cap=None, weight_fn=None, n_threads=8, topk_max=1, row_mask=None):
        """row_mask (bool over sample-key occurrences) lets a caller drop keys
        of some sample rows (e.g. to combine strategies)."""
        df = self.df
        keep_key = df > 0
        if cap is not None:
            keep_key &= df <= cap
        if self.hard_cap is not None:
            keep_key &= df <= self.hard_cap
        s_keep = keep_key[self.s_pos]
        if row_mask is not None:
            s_keep &= row_mask
        s_kidx = np.where(s_keep, self.s_pos, -1).astype(np.int64)
        if weight_fn is not None:
            s_w = weight_fn(df[self.s_pos], self.s_keys).astype(np.float32)
        else:
            s_w = np.ones(len(self.s_keys), np.float32)
        max_touch = max(1, int(_max_touch(self.s_indptr, s_kidx, self.p_indptr)))
        sample = self.sample
        ncand, ntrue, found, rank, topk = _candidates(
            self.s_indptr, s_kidx, s_w, self.p_indptr, self.p_recs, sample.n_corpus, k,
            sample.gt_owner, n_threads, max_touch, topk_max, weight_fn is not None,
        )
        kept_df = df[keep_key]
        n_keys_per_row = np.diff(self.s_indptr)
        kept_per_row = np.bincount(self.s_rows[s_kidx >= 0], minlength=sample.n)
        return {
            "ncand": ncand, "ntrue": ntrue,
            "pair_found": found[sample.pair_rec],
            "pair_rank": rank[sample.pair_rec] if weight_fn is not None else None,
            "topk": topk if weight_fn is not None else None,
            "max_block": int(kept_df.max()) if len(kept_df) else 0,
            "keys_per_s1": float(n_keys_per_row.mean()),
            "s1_no_key": float(np.mean(kept_per_row == 0)),
        }


    def candidate_lists(self, k=1, cap=None, n_threads=8):
        """Explicit candidate CSR (indptr, recs) for the sample."""
        res = self.evaluate(k=k, cap=cap, n_threads=n_threads)
        df = self.df
        keep_key = df > 0
        if cap is not None:
            keep_key &= df <= cap
        if self.hard_cap is not None:
            keep_key &= df <= self.hard_cap
        s_kidx = np.where(keep_key[self.s_pos], self.s_pos, -1).astype(np.int64)
        indptr = np.zeros(self.sample.n + 1, np.int64)
        np.cumsum(res["ncand"], out=indptr[1:])
        max_touch = max(1, int(_max_touch(self.s_indptr, s_kidx, self.p_indptr)))
        recs = _candidate_lists(self.s_indptr, s_kidx, self.p_indptr, self.p_recs,
                                self.sample.n_corpus, k, indptr, n_threads, max_touch)
        return indptr, recs


def topk_lists(res):
    """CSR candidate lists from the top-K matrix of a weighted evaluation."""
    topk = res["topk"]
    valid = topk >= 0
    indptr = np.zeros(topk.shape[0] + 1, np.int64)
    np.cumsum(valid.sum(1), out=indptr[1:])
    return indptr, topk[valid].astype(np.int32)


def union(lists, sample, n_threads=8):
    """Evaluate the union of several candidate CSRs. Returns an evaluate()-like dict."""
    n = sample.n
    indptrs = np.zeros((len(lists), n + 1), np.int64)
    recs = []
    base = 0
    for l, (ip, rc) in enumerate(lists):
        indptrs[l] = ip + base
        recs.append(rc)
        base += len(rc)
    recs = np.concatenate(recs).astype(np.int32)
    ncand, ntrue, found = _union_eval(indptrs, recs, n, len(lists), sample.n_corpus,
                                      sample.gt_owner, n_threads)
    return {"ncand": ncand, "ntrue": ntrue, "pair_found": found[sample.pair_rec],
            "pair_rank": None, "topk": None, "max_block": -1,
            "keys_per_s1": float("nan"), "s1_no_key": float(np.mean(ncand == 0))}


def summarize(res, sample, n_corpus, n_s1_full):
    ncand, ntrue, gt = res["ncand"], res["ntrue"], sample.gt_count
    tot_true = gt.sum()
    matched = gt > 0
    f05 = f05_ceiling(ntrue, gt)
    mean_c = float(ncand.mean())
    return {
        "recall_pairs": ntrue.sum() / tot_true,
        "recall_entity_all": float(np.mean(ntrue[matched] == gt[matched])),
        "f05_ceiling": float(f05.mean()),
        "cand_mean": mean_c,
        "cand_median": float(np.median(ncand)),
        "cand_p99": float(np.percentile(ncand, 99)),
        "cand_zero": float(np.mean(ncand == 0)),
        "reduction_ratio": 1.0 - mean_c / n_corpus,
        "false_cand_rate": 1.0 - (ntrue.sum() / max(1, ncand.sum())),
        "pairs_full_train": mean_c * n_s1_full,
        "max_block": res["max_block"],
    }


def recall_at_k(res, sample, ks):
    """Recall / F0.5 ceiling / candidates if only the top-K ranked candidates are kept."""
    rank = res["pair_rank"]
    out = []
    gt = sample.gt_count
    for kk in ks:
        hit = (rank > 0) & (rank <= kk)
        ntrue = np.bincount(sample.pair_sample[hit], minlength=sample.n)
        cands = np.minimum(res["ncand"], kk)
        out.append({
            "K": kk,
            "recall_pairs": hit.mean(),
            "recall_entity_all": float(np.mean(ntrue[gt > 0] == gt[gt > 0])),
            "f05_ceiling": float(f05_ceiling(ntrue, gt).mean()),
            "cand_mean": float(cands.mean()),
            "false_cand_rate": 1.0 - ntrue.sum() / max(1, cands.sum()),
            "reduction_ratio": 1.0 - cands.mean() / sample.n_corpus,
        })
    return out


# --------------------------------------------------------------------------
# MinHash (numba) over Arrow string buffers
# --------------------------------------------------------------------------

def arrow_buffers(arr):
    """(offsets int64[n+1], data uint8) for an Arrow string array (any offset)."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    arr = arr.cast(pa.large_string()) if arr.type != pa.large_string() else arr
    bufs = arr.buffers()
    offs = np.frombuffer(bufs[1], dtype=np.int64)[arr.offset: arr.offset + len(arr) + 1]
    dat = np.frombuffer(bufs[2], dtype=np.uint8) if bufs[2] is not None else np.zeros(1, np.uint8)
    return offs, dat


@nb.njit(parallel=True, cache=True)
def _minhash_bands(offs, dat, q, a, b, n_bands, rows_per_band, skip_space):
    """Band keys (uint64[n, n_bands]) of char q-gram MinHash signatures.
    Rows with an empty string get key 0 (treated as 'no key' by the caller)."""
    n = len(offs) - 1
    n_perm = n_bands * rows_per_band
    out = np.zeros((n, n_bands), np.uint64)
    P = np.uint64(4294967311)
    for i in nb.prange(n):
        s = offs[i]
        e = offs[i + 1]
        buf = np.empty(e - s, np.uint8)
        L = 0
        for j in range(s, e):
            c = dat[j]
            if skip_space and c == 32:
                continue
            buf[L] = c
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
            if hk == 0:
                hk = np.uint64(1)
            out[i, bb] = hk
    return out


def make_perms(n_perm, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 4294967311, size=n_perm, dtype=np.uint64)
    b = rng.integers(0, 4294967311, size=n_perm, dtype=np.uint64)
    return a, b


def minhash_keyfn(column, n_bands, rows_per_band, q=3, skip_space=True, country=True, seed=1):
    a, b = make_perms(n_bands * rows_per_band, seed)

    def keyfn(tbl):
        offs, dat = arrow_buffers(tbl.column(column))
        bands = _minhash_bands(offs, dat, q, a, b, n_bands, rows_per_band, skip_space)
        n = bands.shape[0]
        rows = np.repeat(np.arange(n, dtype=np.int64), n_bands)
        keys = bands.reshape(-1)
        if country:
            keys = mix(keys, np.repeat(country_hash(tbl), n_bands))
        ok = bands.reshape(-1) != 0
        return rows[ok], keys[ok]

    return keyfn


def lsh_threshold(n_bands, rows_per_band):
    return (1.0 / n_bands) ** (1.0 / rows_per_band)


def idf_weight(n_docs):
    def fn(df, _keys):
        return np.log((n_docs + 1.0) / (df + 1.0))
    return fn


_ = math  # keep import for callers doing threshold maths
