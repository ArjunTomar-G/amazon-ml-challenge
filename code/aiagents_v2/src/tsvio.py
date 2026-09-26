"""Plain-TSV IO for the challenge files (no quoting: names contain stray quotes)."""
from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

_RO = pcsv.ReadOptions(block_size=1 << 26)
_PO = pcsv.ParseOptions(delimiter="\t", quote_char=False)


def read_tsv(path: str, columns=None) -> pa.Table:
    """Every column as string; empty fields stay empty strings."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    co = pcsv.ConvertOptions(column_types={c: pa.string() for c in header}, include_columns=columns,
                             strings_can_be_null=False, quoted_strings_can_be_null=False)
    return pcsv.read_csv(path, read_options=_RO, parse_options=_PO, convert_options=co)


def write_tsv(t: pa.Table, path: str):
    """Write in the original format: tab separated, header, no quoting, '\n' line ends."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = [pc.fill_null(t[c].cast(pa.string()), "") for c in t.column_names]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(t.column_names) + "\n")
        step = 1_000_000
        for a in range(0, t.num_rows, step):
            parts = [c.slice(a, step).to_pylist() for c in cols]
            f.write("".join("\t".join(r) + "\n" for r in zip(*parts)))


def norm_text(arr) -> pa.ChunkedArray:
    """Lower-case ASCII-folded text with single spaces (for keys / statistics only)."""
    a = pc.utf8_normalize(pc.fill_null(arr, ""), "NFKD")
    a = pc.replace_substring_regex(a, r"[^\x00-\x7f]", "")
    a = pc.utf8_lower(a)
    a = pc.replace_substring_regex(a, r"[^a-z0-9]+", " ")
    return pc.utf8_trim_whitespace(a)


def street_key(addr) -> pa.ChunkedArray:
    """First address component without numbers / punctuation: a coarse street key."""
    first = pc.list_element(pc.split_pattern(pc.fill_null(addr, ""), ",", max_splits=1), 0)
    k = norm_text(first)
    k = pc.replace_substring_regex(k, r"\b(no|n|nr)\b|\d+[a-z]?\b", "")
    return pc.utf8_trim_whitespace(pc.replace_substring_regex(k, r"\s+", " "))
