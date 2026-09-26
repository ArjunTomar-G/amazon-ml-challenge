"""Stream the raw TSVs once, normalise every record and cache the result as Parquet.

Output (per source): ER_CACHE/<split>_s<src>.parquet with raw + normalised
strings and the per-record flags used by the profiling step.

    python 01_prepare.py [--split train] [--workers 7]
"""

import argparse
import os
import re
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from erlib import data
from erlib import textnorm as tn

_RE_DOMAIN = re.compile(r"www\.|\.(?:com|net|org|in|co|biz|info)\b")
_RE_HONOR = re.compile(r"\s*(?:m/s|mr|mrs|ms|dr|shri|sri|shree|smt|the)\b")
_RE_LANDMARK = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to)\b")
_RE_UNIT = re.compile(r"\b(?:unit|apt|apartment|suite|ste|fl|floor|flat)\b")
_RE_POBOX = re.compile(r"\bp\.?\s?o\.?\s?box\b")
_RE_CO = re.compile(r"\bc/o\b")

SCHEMA = pa.schema([
    ("id", pa.int64()), ("cty", pa.string()),
    ("name_raw", pa.string()), ("addr_raw", pa.string()),
    ("name_n", pa.string()), ("name_core", pa.string()),
    ("addr_n", pa.string()), ("addr_core", pa.string()),
    ("name_script", pa.string()), ("addr_script", pa.string()),
    ("state", pa.string()), ("zip5", pa.string()), ("pin6", pa.string()),
    ("a_pattern", pa.string()),
    ("name_len", pa.int16()), ("name_ntok", pa.int16()), ("name_ncore", pa.int16()),
    ("addr_len", pa.int16()), ("addr_ncomp", pa.int16()),
    ("f_alias", pa.bool_()), ("f_domain", pa.bool_()), ("f_phone", pa.bool_()),
    ("f_hashat", pa.bool_()), ("f_allcaps", pa.bool_()), ("f_brackets", pa.bool_()),
    ("f_honor", pa.bool_()), ("f_legal", pa.bool_()), ("f_dash", pa.bool_()),
    ("f_concat", pa.bool_()),
    ("a_empty", pa.bool_()), ("a_null", pa.bool_()), ("a_landmark", pa.bool_()),
    ("a_unit", pa.bool_()), ("a_pobox", pa.bool_()), ("a_co", pa.bool_()),
    ("a_hash", pa.bool_()), ("a_startnum", pa.bool_()),
])
LEGAL_NO_LETTERS = tn.LEGAL_SUFFIXES - {"l", "c", "p"}


def comp_type(comp, raw_comp):
    if not raw_comp.isascii() and tn.script_of(raw_comp) not in ("ASCII", "LATIN_EXT"):
        return "S"  # Indic-script component (in this data: always the state)
    if tn.state_of_component(comp):
        return "S"
    return "N" if any(ch.isdigit() for ch in comp) else "A"


def process(chunk):
    ids, names, addrs, ctys = chunk
    out = {f.name: [] for f in SCHEMA}
    for eid, name, addr, cty in zip(ids, names, addrs, ctys):
        low = name.lower()
        nn = tn.norm_name(name)
        ntoks = nn.split()
        core = [t for t in ntoks if t not in tn.NAME_HAND_STOP]
        comps = tn.addr_components(addr)
        acore = tn.addr_core_tokens(comps)
        raw_comps = [c for c in addr.split(",") if c.strip()]
        # component pattern (only when raw/cleaned components line up)
        if len(raw_comps) == len(comps):
            pattern = "".join(comp_type(c, r) for c, r in zip(comps, raw_comps))
        else:
            pattern = "".join(comp_type(c, c) for c in comps)
        state = ""
        for c in comps:
            state = tn.state_of_component(c)
            if state:
                break
        alow = addr.lower()
        out["id"].append(int(eid[3:]))
        out["cty"].append(cty)
        out["name_raw"].append(name)
        out["addr_raw"].append(addr)
        out["name_n"].append(nn)
        out["name_core"].append(" ".join(core))
        out["addr_n"].append(",".join(comps))
        out["addr_core"].append(" ".join(acore))
        out["name_script"].append(tn.script_of(name))
        out["addr_script"].append(tn.script_of(addr))
        out["state"].append(state)
        out["zip5"].append(tn.zip5(comps))
        out["pin6"].append(tn.pin6(comps))
        out["a_pattern"].append(pattern[:12])
        out["name_len"].append(min(len(name), 32767))
        out["name_ntok"].append(len(ntoks))
        out["name_ncore"].append(len(core))
        out["addr_len"].append(min(len(addr), 32767))
        out["addr_ncomp"].append(len(comps))
        out["f_alias"].append(bool(tn._ALIAS.search(low)) or "|" in name)
        out["f_domain"].append(bool(_RE_DOMAIN.search(low)))
        out["f_phone"].append(bool(tn._PHONE.search(low)))
        out["f_hashat"].append(name.lstrip()[:1] in ("#", "@"))
        out["f_allcaps"].append(name.isupper())
        out["f_brackets"].append(any(ch in name for ch in "[({"))
        out["f_honor"].append(bool(_RE_HONOR.match(low)))
        out["f_legal"].append(any(t in LEGAL_NO_LETTERS for t in ntoks))
        out["f_dash"].append("--" in name)
        out["f_concat"].append(len(core) == 1 and len(core[0]) >= 12)
        out["a_empty"].append(not addr.strip())
        out["a_null"].append(bool(tn._NULLS.search(alow)))
        out["a_landmark"].append(bool(_RE_LANDMARK.search(alow)))
        out["a_unit"].append(bool(_RE_UNIT.search(alow)))
        out["a_pobox"].append(bool(_RE_POBOX.search(alow)))
        out["a_co"].append(bool(_RE_CO.search(alow)))
        out["a_hash"].append("#" in addr)
        out["a_startnum"].append(bool(comps) and comps[0][:1].isdigit())
    return pa.table(out, schema=SCHEMA)


def batches(path, sub=40_000):
    reader = data.open_tsv(path)
    for rb in reader:
        cols = [rb.column(i).to_pylist() for i in range(4)]
        n = len(cols[0])
        for i in range(0, n, sub):
            yield tuple(c[i:i + sub] for c in cols)


def prepare(split, src, workers):
    src_path = data.raw_path(split, src)
    dst = data.prepared_path(split, src)
    if os.path.exists(dst):
        print(f"[skip] {dst} exists")
        return
    tmp = dst + ".tmp"
    t0 = time.time()
    n = 0
    with Pool(workers) as pool, pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as w:
        for tbl in pool.imap(process, batches(src_path), chunksize=1):
            w.write_table(tbl)
            n += tbl.num_rows
            if n % 1_000_000 < 40_000:
                print(f"  {split} s{src}: {n:,} rows  {time.time() - t0:.0f}s", flush=True)
    os.replace(tmp, dst)
    print(f"[done] {dst}: {n:,} rows in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--sources", default="1,2,3")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    os.makedirs(data.CACHE_DIR, exist_ok=True)
    for src in map(int, args.sources.split(",")):
        prepare(args.split, src, args.workers)


if __name__ == "__main__":
    sys.exit(main())
