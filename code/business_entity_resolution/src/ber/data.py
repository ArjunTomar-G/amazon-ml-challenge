"""Paths, raw-TSV streaming, record normalisation to Parquet, ground truth."""

import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import textnorm as tn

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.dirname(SRC_DIR)                      # code/business_entity_resolution
REPO_DIR = os.path.dirname(os.path.dirname(PKG_DIR))    # submission / repo root


def default_dirs():
    return {
        "data": os.environ.get("BER_DATA", os.path.join(REPO_DIR, "dataset")),
        "work": os.environ.get("BER_WORK", os.path.join(REPO_DIR, "work")),
        "out": os.environ.get("BER_OUT", os.path.join(REPO_DIR, "output")),
    }


def raw_path(data_dir, split, src):
    return os.path.join(data_dir, split, f"{split}_source{src}.tsv")


def prepared_path(work_dir, split, src):
    return os.path.join(work_dir, f"{split}_s{src}.parquet")


def open_tsv(path, block_size=64 << 20):
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block_size),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in
                          ("entity_id", "business_name", "business_address", "country")},
            strings_can_be_null=False, quoted_strings_can_be_null=False),
    )


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

SCHEMA = pa.schema([
    ("id", pa.int64()), ("cty", pa.string()),
    ("name_n", pa.string()), ("name_core", pa.string()),
    ("addr_n", pa.string()), ("addr_core", pa.string()),
    ("legal", pa.string()), ("a_empty", pa.bool_()), ("indic", pa.bool_()),
])


def legal_form(name_n):
    """Canonical legal-form tokens of a normalised name ('' when none)."""
    forms = sorted({tn.LEGAL_CANON[t] for t in name_n.split() if t in tn.LEGAL_CANON})
    return " ".join(forms)


def normalise_rows(chunk):
    ids, names, addrs, ctys = chunk
    out = {f.name: [] for f in SCHEMA}
    for eid, name, addr, cty in zip(ids, names, addrs, ctys):
        nn = tn.norm_name(name)
        comps = tn.addr_components(addr)
        out["id"].append(int(eid[3:]))
        out["cty"].append(cty.strip())
        out["name_n"].append(nn)
        out["name_core"].append(" ".join(tn.name_core_tokens(nn)))
        out["addr_n"].append(" ".join(comps))
        out["addr_core"].append(" ".join(tn.addr_core_tokens(comps)))
        out["legal"].append(legal_form(nn))
        out["a_empty"].append(not comps)
        out["indic"].append(tn.script_of(name) not in ("ASCII", "LATIN_EXT", "OTHER"))
    return pa.table(out, schema=SCHEMA)


def _batches(path, sub=40_000):
    for rb in open_tsv(path):
        cols = [rb.column(i).to_pylist() for i in range(4)]
        for i in range(0, len(cols[0]), sub):
            yield tuple(c[i:i + sub] for c in cols)


def prepare_split(data_dir, work_dir, split, workers=None, log=print):
    """Normalise the three source files of a split into Parquet (skips existing)."""
    os.makedirs(work_dir, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    for src in (1, 2, 3):
        dst = prepared_path(work_dir, split, src)
        if os.path.exists(dst):
            log(f"[prepare] {dst} exists, skipping")
            continue
        t0, n, tmp = time.time(), 0, dst + ".tmp"
        with Pool(workers) as pool, pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as w:
            for tbl in pool.imap(normalise_rows, _batches(raw_path(data_dir, split, src))):
                w.write_table(tbl)
                n += tbl.num_rows
        os.replace(tmp, dst)
        log(f"[prepare] {split} source {src}: {n:,} rows in {time.time() - t0:.0f}s")


def read_source(work_dir, split, src, columns=None):
    return pq.read_table(prepared_path(work_dir, split, src), columns=columns)


def load_corpus(work_dir, split, columns):
    """Source 2 rows followed by Source 3 rows (global corpus index), plus src array."""
    t2 = read_source(work_dir, split, 2, columns)
    t3 = read_source(work_dir, split, 3, columns)
    src = np.concatenate([np.full(t2.num_rows, 2, np.int8), np.full(t3.num_rows, 3, np.int8)])
    return pa.concat_tables([t2, t3]).combine_chunks(), src


def load_ground_truth(data_dir):
    """(pair_s1_id, pair_src, pair_id) int arrays for every true pair."""
    t = pacsv.read_csv(
        os.path.join(data_dir, "train", "train_ground_truth.tsv"),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={"source1_entity_id": pa.string(), "matched_entity_ids": pa.string()},
            strings_can_be_null=False))
    s1 = pc.cast(pc.utf8_slice_codeunits(t.column("source1_entity_id").combine_chunks(), 3, 32),
                 pa.int64()).to_numpy()
    lists = pc.split_pattern(t.column("matched_entity_ids").combine_chunks(), ",")
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy()
    keep = pc.not_equal(flat, "")
    flat = flat.filter(keep)
    parent = parent[keep.to_numpy(zero_copy_only=False)]
    src = pc.cast(pc.utf8_slice_codeunits(flat, 1, 2), pa.int8()).to_numpy()
    ids = pc.cast(pc.utf8_slice_codeunits(flat, 3, 32), pa.int64()).to_numpy()
    return s1[parent], src, ids


def global_index(corpus_src, corpus_ids, src, ids):
    """Map (src, id) pairs to global corpus positions (-1 when absent)."""
    comp = corpus_ids + corpus_src.astype(np.int64) * 10**12
    order = np.argsort(comp, kind="stable")
    sorted_comp = comp[order]
    q = np.asarray(ids, np.int64) + np.asarray(src, np.int64) * 10**12
    pos = np.searchsorted(sorted_comp, q)
    pos = np.minimum(pos, len(sorted_comp) - 1)
    ok = sorted_comp[pos] == q
    return np.where(ok, order[pos], -1)


def entity_ids(prefix_src, ids):
    """['S2-123', ...] from source numbers and integer ids."""
    return [f"S{s}-{i}" for s, i in zip(np.asarray(prefix_src).tolist(), np.asarray(ids).tolist())]
