"""Learned Indic-script -> Latin token dictionary.

Source-2/3 names written in Indic scripts are ASCII-folded with unidecode, which
gives distorted spellings (`phyuucr investtmentts praaivett limittedd`). Aligning
those names token-by-token with their matched Source-1 names (training ground
truth only) yields a small, clean dictionary (`innnvesttmenntts -> investments`).
"""

import json
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from . import textnorm as tn


def learn(s1_ids, s1_names, corpus_names, corpus_indic, pair_s1, pair_rec, min_count=3, min_share=0.5):
    """s1_ids/s1_names: Source-1 ids and name_n; corpus_*: name_n and Indic flag per
    corpus row; pair_s1 (S1 id) / pair_rec (corpus row) list the true pairs."""
    m = corpus_indic[pair_rec]
    pair_s1, pair_rec = pair_s1[m], pair_rec[m]
    order = np.argsort(s1_ids)
    s1_pos = order[np.searchsorted(s1_ids[order], pair_s1)]
    a_names = corpus_names.take(pa.array(pair_rec)).to_pylist()
    b_names = s1_names.take(pa.array(s1_pos)).to_pylist()
    counts = defaultdict(Counter)
    aligned = 0
    for a, b in zip(a_names, b_names):
        ta, tb = a.split(), b.split()
        if ta and len(ta) == len(tb):
            aligned += 1
            for x, y in zip(ta, tb):
                if x != y:
                    counts[x][y] += 1
    mapping = {}
    for x, c in counts.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            mapping[x] = y
    return mapping, {"indic_pairs": int(len(pair_rec)), "aligned_pairs": aligned, "tokens": len(mapping)}


def save(mapping, path):
    with open(path, "w") as f:
        json.dump(mapping, f)


def load(path):
    with open(path) as f:
        return json.load(f)


class Mapper:
    """Vectorised token mapping on Arrow string arrays."""

    def __init__(self, mapping):
        self.keys = pa.array(list(mapping.keys()), pa.string())
        self.vals = pa.array(list(mapping.values()), pa.string())
        self.stop = pa.array(sorted(tn.NAME_HAND_STOP), pa.string())

    def map_tokens(self, toks):
        if len(self.keys) == 0:
            return toks
        idx = pc.index_in(toks, value_set=self.keys)
        mapped = self.vals.take(pc.fill_null(idx, 0))
        return pc.if_else(pc.is_null(idx), toks, mapped)

    def names(self, name_n):
        """(translit name, translit core name) Arrow arrays for an Arrow array of name_n."""
        if isinstance(name_n, pa.ChunkedArray):
            name_n = name_n.combine_chunks()
        lists = pc.utf8_split_whitespace(name_n)
        flat = self.map_tokens(pc.list_flatten(lists))
        parent = pc.list_parent_indices(lists).to_numpy()
        full = _join(flat, parent, len(name_n))
        keep = pc.invert(pc.is_in(flat, value_set=self.stop))
        core = _join(flat.filter(keep), parent[keep.to_numpy(zero_copy_only=False)], len(name_n))
        return full, core


def map_join(col, mapping):
    """Replace whitespace tokens via `mapping` and re-join (vectorised)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    lists = pc.utf8_split_whitespace(col)
    flat = pc.list_flatten(lists)
    keys = pa.array(list(mapping.keys()), pa.string())
    vals = pa.array(list(mapping.values()), pa.string())
    idx = pc.index_in(flat, value_set=keys)
    flat = pc.if_else(pc.is_null(idx), flat, vals.take(pc.fill_null(idx, 0)))
    return _join(flat, pc.list_parent_indices(lists).to_numpy(), len(col))


def _join(tokens, parent, n):
    counts = np.bincount(parent, minlength=n) if len(parent) else np.zeros(n, np.int64)
    offsets = np.zeros(n + 1, np.int32)
    np.cumsum(counts, out=offsets[1:])
    lists = pa.ListArray.from_arrays(pa.array(offsets), tokens)
    return pc.binary_join(lists, " ")
