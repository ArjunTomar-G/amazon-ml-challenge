"""Paths and loaders shared by the EDA scripts."""

import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

EDA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("ER_DATA", os.path.join(os.path.dirname(EDA_DIR), "dataset"))
CACHE_DIR = os.environ.get("ER_CACHE", os.path.join(EDA_DIR, ".cache"))
RESULTS_DIR = os.path.join(EDA_DIR, "results")

COUNTRIES = ("US", "India")  # training countries; code must not assume this set


def raw_path(split, src):
    return os.path.join(DATA_DIR, split, f"{split}_source{src}.tsv")


def gt_path():
    return os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")


def prepared_path(split, src):
    return os.path.join(CACHE_DIR, f"{split}_s{src}.parquet")


def open_tsv(path, block_size=64 << 20):
    """Streaming reader: all columns as strings, no quote handling, '' kept as ''."""
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block_size),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={
                "entity_id": pa.string(), "business_name": pa.string(),
                "business_address": pa.string(), "country": pa.string(),
                "source1_entity_id": pa.string(), "matched_entity_ids": pa.string(),
            },
            strings_can_be_null=False,
            quoted_strings_can_be_null=False,
        ),
    )


def read_prepared(split, src, columns=None):
    return pq.read_table(prepared_path(split, src), columns=columns)


def id_num(ids):
    """'S2-681193310' -> 681193310 for an Arrow string array (vectorised)."""
    return pc.cast(pc.utf8_slice_codeunits(ids, 3, 32), pa.int64()).to_numpy()


def load_ground_truth():
    """Return (s1_ids, pair_s1, pair_src, pair_id) as numpy arrays.

    s1_ids: int64 ids of every S1 row in the GT file (incl. singletons).
    pair_*: one row per true pair; pair_src is 2 or 3.
    """
    t = pacsv.read_csv(
        gt_path(),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={"source1_entity_id": pa.string(), "matched_entity_ids": pa.string()},
            strings_can_be_null=False,
        ),
    )
    s1_ids = id_num(t.column("source1_entity_id").combine_chunks())
    lists = pc.split_pattern(t.column("matched_entity_ids").combine_chunks(), ",")
    del t
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists)
    keep = pc.not_equal(flat, "")
    flat, parent = flat.filter(keep), parent.filter(keep)
    pair_src = pc.cast(pc.utf8_slice_codeunits(flat, 1, 2), pa.int8()).to_numpy()
    pair_id = id_num(flat)
    pair_s1 = s1_ids[parent.to_numpy()]
    return s1_ids, pair_s1, pair_src, pair_id
