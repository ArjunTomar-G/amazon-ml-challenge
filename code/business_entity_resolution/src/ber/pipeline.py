"""Per-country driver shared by train.py and predict.py."""

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import data, features, retrieval
from . import textnorm as tn
from .translit import map_join

COLS = ["id", "cty", "name_n", "addr_n", "addr_core", "legal", "a_empty", "indic"]


def countries(work_dir, split):
    t = pq.read_table(data.prepared_path(work_dir, split, 1), columns=["cty"])
    return sorted(set(pc.unique(t.column("cty")).to_pylist()))


def load_country(work_dir, split, country, s1_ids=None):
    """Queries (Source 1) and corpus (Source 2 + 3) of one country."""
    flt = [("cty", "=", country)]
    q = pq.read_table(data.prepared_path(work_dir, split, 1), columns=COLS, filters=flt)
    if s1_ids is not None:
        q = q.filter(pa.array(np.isin(q.column("id").to_numpy(), s1_ids)))
    parts, src = [], []
    for s in (2, 3):
        t = pq.read_table(data.prepared_path(work_dir, split, s), columns=COLS, filters=flt)
        parts.append(t)
        src.append(np.full(t.num_rows, s, np.int8))
    c = pa.concat_tables(parts).combine_chunks()
    return q.combine_chunks(), c, np.concatenate(src)


def arrays(tab, mapper):
    name_nt, core_tr = mapper.names(tab.column("name_n"))
    addr_core = tab.column("addr_core").combine_chunks()
    return {"name_nt": name_nt, "core_tr": core_tr,
            "canon": map_join(name_nt, tn.LEGAL_CANON),
            "comb": pc.binary_join_element_wise(core_tr, addr_core, " "),
            "addr_n": tab.column("addr_n").combine_chunks(),
            "addr_core": addr_core,
            "legal": tab.column("legal").combine_chunks(),
            "a_empty": tab.column("a_empty").combine_chunks(),
            "indic": tab.column("indic").combine_chunks()}


def run_country(Q, C, cfg, n_threads, chunk_queries, on_chunk, log=print):
    """Generate candidates block by block, compute features chunk by chunk and hand
    each chunk to on_chunk(pairs_dict, X, feature_names)."""
    t0 = time.time()
    tables = features.build_tables(C["core_tr"], C["addr_core"])
    t_feat = 0.0
    names = None
    blocks = retrieval.candidates(Q["core_tr"], Q["addr_core"], C["core_tr"], C["addr_core"],
                                  C["a_empty"].to_numpy(zero_copy_only=False), cfg, n_threads, log)
    for P in blocks:
        q = P["q"]
        bounds = np.searchsorted(q, np.arange(q[0] if len(q) else 0, (q[-1] if len(q) else 0) + chunk_queries + 1,
                                              chunk_queries))
        tf = time.time()
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            if b1 <= b0:
                continue
            sl = {k: v[b0:b1] for k, v in P.items()}
            X, names = features.pair_features(sl, Q, C, tables)
            on_chunk(sl, X, names)
        t_feat += time.time() - tf
    log(f"    country done in {time.time() - t0:.0f}s (features {t_feat:.0f}s)")
    return names
