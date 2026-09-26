"""Pair features for (Source-1 entity, candidate) pairs.

String similarities (rapidfuzz), numba kernels for token / char-3-gram Jaccard and
IDF-weighted cosines, legal-form agreement, retrieval scores, and entity-relative
features (gap to the best candidate of the same Source-1 entity).
"""

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from .retrieval import arrow_buffers

FNV_OFF = np.uint64(14695981039346656037)
FNV_PRIME = np.uint64(1099511628211)


# ---------------------------------------------------------------------------
# numba helpers
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _units(dat, s, e, q, grams, pad, out):
    """Write hashes of tokens (grams=False) or char q-grams (grams=True) of dat[s:e]
    into out; returns count. pad=True: grams over ' ' + text + ' ' (word
    boundaries kept); pad=False: grams over the text with spaces removed."""
    n = 0
    if not grams:
        h = FNV_OFF
        in_tok = False
        for j in range(s, e):
            c = dat[j]
            if c == 32:
                if in_tok:
                    out[n] = h
                    n += 1
                    in_tok = False
                    h = FNV_OFF
            else:
                h = (h ^ np.uint64(c)) * FNV_PRIME
                in_tok = True
        if in_tok:
            out[n] = h
            n += 1
        return n
    L = e - s
    buf = np.empty(L + 2, np.uint8)
    m = 0
    if pad:
        buf[0] = 32
        m = 1
        for j in range(s, e):
            buf[m] = dat[j]
            m += 1
        buf[m] = 32
        m += 1
    else:
        for j in range(s, e):
            if dat[j] != 32:
                buf[m] = dat[j]
                m += 1
    if m == 0 or (pad and m == 2):
        return 0
    if m < q:
        h = FNV_OFF
        for t in range(m):
            h = (h ^ np.uint64(buf[t])) * FNV_PRIME
        out[0] = h
        return 1
    for j in range(m - q + 1):
        h = FNV_OFF
        for t in range(j, j + q):
            h = (h ^ np.uint64(buf[t])) * FNV_PRIME
        out[n] = h
        n += 1
    return n


@nb.njit(cache=True)
def _sorted_unique(a, n):
    if n == 0:
        return a[:0]
    b = np.sort(a[:n])
    m = 1
    for i in range(1, n):
        if b[i] != b[m - 1]:
            b[m] = b[i]
            m += 1
    return b[:m]


@nb.njit(cache=True)
def _lookup(keys, vals, default, h):
    lo, hi = 0, len(keys)
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < h:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(keys) and keys[lo] == h:
        return vals[lo]
    return default


@nb.njit(parallel=True, cache=True)
def _pair_sim(oa, da, ob, db, q, grams, pad, keys, vals, default, weighted):
    """Jaccard (weighted=False) or IDF-weighted cosine (weighted=True) of the
    unit sets of aligned string pairs; NaN if either side has no units."""
    n = len(oa) - 1
    out = np.empty(n, np.float32)
    for i in nb.prange(n):
        la = oa[i + 1] - oa[i]
        lb = ob[i + 1] - ob[i]
        ua = np.empty(la + 3, np.uint64)
        ub = np.empty(lb + 3, np.uint64)
        na = _units(da, oa[i], oa[i + 1], q, grams, pad, ua)
        nb_ = _units(db, ob[i], ob[i + 1], q, grams, pad, ub)
        if na == 0 or nb_ == 0:
            out[i] = np.nan
            continue
        sa = _sorted_unique(ua, na)
        sb = _sorted_unique(ub, nb_)
        x, y = 0, 0
        if not weighted:
            inter = 0
            while x < len(sa) and y < len(sb):
                if sa[x] == sb[y]:
                    inter += 1
                    x += 1
                    y += 1
                elif sa[x] < sb[y]:
                    x += 1
                else:
                    y += 1
            out[i] = inter / (len(sa) + len(sb) - inter)
        else:
            wa = 0.0
            for t in range(len(sa)):
                v = _lookup(keys, vals, default, sa[t])
                wa += v * v
            wb = 0.0
            for t in range(len(sb)):
                v = _lookup(keys, vals, default, sb[t])
                wb += v * v
            dot = 0.0
            while x < len(sa) and y < len(sb):
                if sa[x] == sb[y]:
                    v = _lookup(keys, vals, default, sa[x])
                    dot += v * v
                    x += 1
                    y += 1
                elif sa[x] < sb[y]:
                    x += 1
                else:
                    y += 1
            out[i] = dot / np.sqrt(wa * wb) if wa > 0 and wb > 0 else np.nan
    return out


@nb.njit(parallel=True, cache=True)
def _unit_hashes(offs, dat, q, grams, pad):
    """All unique unit hashes per string, concatenated (for document frequencies)."""
    n = len(offs) - 1
    counts = np.zeros(n, np.int64)
    for i in nb.prange(n):
        u = np.empty(offs[i + 1] - offs[i] + 3, np.uint64)
        counts[i] = len(_sorted_unique(u, _units(dat, offs[i], offs[i + 1], q, grams, pad, u)))
    start = np.zeros(n + 1, np.int64)
    for i in range(n):
        start[i + 1] = start[i] + counts[i]
    out = np.empty(start[n], np.uint64)
    for i in nb.prange(n):
        u = np.empty(offs[i + 1] - offs[i] + 3, np.uint64)
        s = _sorted_unique(u, _units(dat, offs[i], offs[i + 1], q, grams, pad, u))
        for t in range(len(s)):
            out[start[i] + t] = s[t]
    return out


@nb.njit(parallel=True, cache=True)
def _first_number_equal(oa, da, ob, db):
    n = len(oa) - 1
    out = np.empty(n, np.float32)
    for i in nb.prange(n):
        sa, ea = _first_num(da, oa[i], oa[i + 1])
        sb, eb = _first_num(db, ob[i], ob[i + 1])
        if sa < 0 or sb < 0:
            out[i] = np.nan
        elif ea - sa != eb - sb:
            out[i] = 0.0
        else:
            eq = 1.0
            for t in range(ea - sa):
                if da[sa + t] != db[sb + t]:
                    eq = 0.0
                    break
            out[i] = eq
    return out


@nb.njit(cache=True)
def _first_num(d, s, e):
    j = s
    while j < e:
        while j < e and d[j] == 32:
            j += 1
        k = j
        has = False
        while k < e and d[k] != 32:
            if 48 <= d[k] <= 57:
                has = True
            k += 1
        if has:
            return j, k
        j = k
    return -1, -1


# ---------------------------------------------------------------------------
# IDF tables
# ---------------------------------------------------------------------------

class IdfTable:
    def __init__(self, texts, q=3, grams=True, pad=True):
        self.q, self.grams, self.pad = q, grams, pad
        offs, dat = arrow_buffers(texts)
        h = _unit_hashes(offs, dat, q, grams, pad)
        n_docs = len(texts)
        keys, df = np.unique(h, return_counts=True)
        self.keys = keys
        self.vals = (np.log((n_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float64)
        self.default = float(np.log(n_docs + 1.0) + 1.0)

    def cosine(self, a, b):
        oa, da = arrow_buffers(a)
        ob, db = arrow_buffers(b)
        return _pair_sim(oa, da, ob, db, self.q, self.grams, self.pad, self.keys, self.vals,
                         self.default, True)


def jaccard(a, b, grams, q=3):
    oa, da = arrow_buffers(a)
    ob, db = arrow_buffers(b)
    empty = np.zeros(0, np.uint64)
    return _pair_sim(oa, da, ob, db, q, grams, False, empty, np.zeros(0), 0.0, False)


def build_tables(c_core, c_addr, sample=600_000, seed=0):
    """Per-country IDF tables from (a sample of) the corpus."""
    n = len(c_core)
    if n > sample:
        idx = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
        c_core, c_addr = c_core.take(pa.array(idx)), c_addr.take(pa.array(idx))
    comb = pc.binary_join_element_wise(c_core, c_addr, " ")
    return {
        "name3": IdfTable(c_core), "addr3": IdfTable(c_addr), "comb3": IdfTable(comb),
        "name_tok": IdfTable(c_core, grams=False), "addr_tok": IdfTable(c_addr, grams=False),
    }


# ---------------------------------------------------------------------------
# feature assembly
# ---------------------------------------------------------------------------

RETRIEVAL = ["joint_score", "joint_rank", "rel_score", "mh_count", "in_empty", "top1", "log_touched"]


def _rf(scorer, a, b, scale, empty):
    s = cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32) / scale
    s[empty] = np.nan
    return s


def _empty(*arrs):
    m = np.zeros(len(arrs[0]), bool)
    for a in arrs:
        m |= pc.equal(pc.utf8_length(a), 0).to_numpy(zero_copy_only=False)
    return m


def _group_max(q, v):
    """Max of v per query id q (q sorted), broadcast back to pairs."""
    v = np.nan_to_num(v, nan=-1.0)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    mx = np.maximum.reduceat(v, starts)
    return np.repeat(mx, np.diff(np.r_[starts, len(q)])), v


def pair_features(P, Q, C, tables):
    """P: dict of pair arrays (q, c + retrieval features) sorted by q.
    Q / C: dicts of Arrow arrays for the queries / corpus of one country.
    Returns (float32 matrix, feature names)."""
    qi, ci = pa.array(P["q"]), pa.array(P["c"])
    cols = ("name_nt", "core_tr", "canon", "comb", "addr_n", "addr_core", "legal")
    qa = {k: Q[k].take(qi) for k in cols}
    cb = {k: C[k].take(ci) for k in cols}
    L = {k: qa[k].to_pylist() for k in cols[:-1]}
    R = {k: cb[k].to_pylist() for k in cols[:-1]}
    E = {k: _empty(qa[k], cb[k]) for k in cols[:-1]}
    F = {}
    # rapidfuzz
    for key, col in (("nm_core", "core_tr"), ("nm_full", "name_nt"), ("nm_canon", "canon"), ("ad_core", "addr_core")):
        a, b, e = L[col], R[col], E[col]
        F[f"{key}_jw"] = _rf(JaroWinkler.similarity, a, b, 1.0, e)
        F[f"{key}_tsort"] = _rf(fuzz.token_sort_ratio, a, b, 100.0, e)
        if key in ("nm_core", "ad_core", "nm_canon"):
            F[f"{key}_tset"] = _rf(fuzz.token_set_ratio, a, b, 100.0, e)
        if key in ("nm_core", "ad_core"):
            F[f"{key}_lev"] = _rf(Levenshtein.normalized_similarity, a, b, 1.0, e)
    F["ad_full_tset"] = _rf(fuzz.token_set_ratio, L["addr_n"], R["addr_n"], 100.0, E["addr_n"])
    F["comb_tset"] = _rf(fuzz.token_set_ratio, L["comb"], R["comb"], 100.0, E["comb"])
    del L, R
    # numba set similarities
    F["nm_core_jac"] = jaccard(qa["core_tr"], cb["core_tr"], grams=False)
    F["nm_core_jac3"] = jaccard(qa["core_tr"], cb["core_tr"], grams=True)
    F["nm_canon_jac"] = jaccard(qa["canon"], cb["canon"], grams=False)
    F["ad_core_jac"] = jaccard(qa["addr_core"], cb["addr_core"], grams=False)
    F["ad_core_jac3"] = jaccard(qa["addr_core"], cb["addr_core"], grams=True)
    F["nm_tfidf3"] = tables["name3"].cosine(qa["core_tr"], cb["core_tr"])
    F["ad_tfidf3"] = tables["addr3"].cosine(qa["addr_core"], cb["addr_core"])
    F["comb_tfidf3"] = tables["comb3"].cosine(qa["comb"], cb["comb"])
    F["nm_tokidf"] = tables["name_tok"].cosine(qa["core_tr"], cb["core_tr"])
    F["ad_tokidf"] = tables["addr_tok"].cosine(qa["addr_core"], cb["addr_core"])
    oa, da = arrow_buffers(qa["addr_core"])
    ob, db = arrow_buffers(cb["addr_core"])
    F["housenum_eq"] = _first_number_equal(oa, da, ob, db)
    # legal form: token Jaccard of canonical legal forms (NaN when either has none)
    F["legal_jac"] = jaccard(qa["legal"], cb["legal"], grams=False)
    F["legal_missing"] = _empty(qa["legal"], cb["legal"]).astype(np.float32)
    # record-level
    F["c_addr_empty"] = C["a_empty"].take(ci).to_numpy(zero_copy_only=False).astype(np.float32)
    F["c_indic"] = C["indic"].take(ci).to_numpy(zero_copy_only=False).astype(np.float32)
    F["q_ntok"] = pc.list_value_length(pc.utf8_split_whitespace(qa["core_tr"])).to_numpy().astype(np.float32)
    F["c_ntok"] = pc.list_value_length(pc.utf8_split_whitespace(cb["core_tr"])).to_numpy().astype(np.float32)
    for k in RETRIEVAL:
        F[k] = np.asarray(P[k], np.float32)
    # entity-relative: gap to the best candidate of the same Source-1 entity
    q = np.asarray(P["q"])
    for k in ("comb_tfidf3", "ad_tfidf3", "nm_core_jw", "ad_core_tset", "joint_score"):
        mx, v = _group_max(q, F[k])
        F[f"{k}_gap"] = (v - mx).astype(np.float32)
    names = list(F)
    X = np.column_stack([F[k] for k in names]).astype(np.float32)
    return X, names
