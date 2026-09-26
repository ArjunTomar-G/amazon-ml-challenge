"""Test-like training data: add synthetic sibling groups to the training universe.

Why: the test universe differs from the training universe in two ways that the
v1 models never see (measured on the data, see ../README.md):

* ~37-40 % of test Source-2/3 records match nothing, against 25 % in train
  (about twice as many hard negatives per Source-1 entity);
* 40-52 % of those unmatched test records come in groups of 2-3 records at the
  same address (a sibling business with its own noisy records), against 5-13 %
  in train.

This script writes a copy of the data directory whose *training* Source-2/3
files contain extra sibling records, so that the v1 pipeline - run unchanged on
the new directory - trains on test-like data and its validation fold measures
test-like conditions.  Source 1, the ground truth and the test files are
copied unchanged (the new records are unmatched, so the labels do not change).

How a sibling group is generated (everything is learned from the training data):

1. Exemplars: unmatched training records on the same street as a Source-1
   entity, with a house number 0..30 above it and a similar name, are taken as
   observed siblings.  From them we learn the house-number shift distribution
   (mass on +1..+5, +7, +9, +11, +13, +21) and the name edits (added word,
   replaced word, legal-form change, dropped word) with their word vocabularies.
2. A base Source-1 entity is drawn at random; 1-3 of its *true* records are
   used as templates, so the synthetic records inherit the real generator's
   noise (casing, abbreviations, reordering, typos, scripts).
3. One sibling edit and one shift are drawn and applied identically to every
   template: the result is a group of noisy records of one sibling business.

    python augment.py --data-dir <dataset> --out-dir <dataset_aug>
           [--target-unmatched 0.37] [--group-sizes 0.35 0.55 0.10] [--seed 7]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from tsvio import norm_text, read_tsv, street_key, write_tsv

LEGAL = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt",
    "private", "llp", "lp", "plc", "pllc", "pc", "pa", "psc", "opc", "sarl", "sas", "sasu", "sa",
    "eurl", "sci", "snc", "scp", "selarl", "selas", "cie", "gmbh", "ag",
}
MAX_SHIFT = 30
# generator noise that looks like an added word (honorifics, domains, handles, ids)
NOISE_WORDS = {"the", "mr", "mrs", "ms", "dr", "shri", "sri", "smt", "messrs", "ms", "mx", "m", "s",
               "and", "of", "de", "la", "le", "les", "des", "du", "et", "d", "l", "en", "a", "dba", "aka"}
_TOK_RE = re.compile(r"\S+")
_NUM_RE = re.compile(r"\d+")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _key(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def _lev_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b), 1)


def first_number(addr: str) -> int | None:
    m = _NUM_RE.search(addr.split(",")[0]) if addr else None
    return int(m.group()) if m else None


# ----------------------------------------------------------------------------
# 1. learn sibling edits from observed siblings
# ----------------------------------------------------------------------------
def classify_edit(s1_name: str, rec_name: str):
    """Token diff between an entity and its sibling, typo pairs removed."""
    A = [_key(t) for t in s1_name.split()]
    B_raw = rec_name.split()
    B = [_key(t) for t in B_raw]
    A_set, B_set = {t for t in A if t}, {t for t in B if t}
    added = [(k, raw) for k, raw in zip(B, B_raw) if k and k not in A_set]
    removed = [t for t in A if t and t not in B_set]
    # a removed/added pair that is a near-typo is generator noise, not a sibling edit
    for r in list(removed):
        for a in list(added):
            if _lev_sim(r, a[0]) >= 0.7:
                removed.remove(r)
                added.remove(a)
                break
    la = [raw for k, raw in added if k in LEGAL]
    lr = [t for t in removed if t in LEGAL]
    na = [raw for k, raw in added
          if k not in LEGAL and k not in NOISE_WORDS and len(k) > 1 and not any(c.isdigit() for c in k)
          and re.fullmatch(r"[^\W\d_][\w'-]*", raw.strip("()[]"))]
    nr = [t for t in removed if t not in LEGAL]
    if na and not nr:
        op = "add"
    elif na and nr:
        op = "replace"
    elif nr:
        op = "drop"
    else:
        op = "none"
    return op, na, (lr[0] if lr else None), (_key(la[0]) if la else None)


def learn_edits(s1: pd.DataFrame, unm: pd.DataFrame, max_records: int = 600_000, seed: int = 0):
    """s1 / unm: id, country, name, addr, sk, hn.  Returns per-country edit statistics.

    Exemplar pairs are found with an exact integer join on (country, street key,
    house number - d) for d = 0..MAX_SHIFT, so memory stays bounded on the full data."""
    unm = unm[unm.hn.notna()]
    if len(unm) > max_records:
        unm = unm.sample(max_records, random_state=seed)
    s1 = s1[s1.hn.notna()]
    codes, _ = pd.factorize(pd.concat([s1.country + "|" + s1.sk, unm.country + "|" + unm.sk]), sort=False)
    c1, cu = codes[:len(s1)].astype(np.int64), codes[len(s1):].astype(np.int64)
    k1 = c1 * 10_000_000 + np.minimum(s1.hn.to_numpy(np.int64), 9_999_999)
    hu = np.minimum(unm.hn.to_numpy(np.int64), 9_999_999)
    ku = np.concatenate([cu * 10_000_000 + np.maximum(hu - d, 0) for d in range(MAX_SHIFT + 1)])
    du = np.repeat(np.arange(MAX_SHIFT + 1), len(unm))
    iu = np.tile(np.arange(len(unm)), MAX_SHIFT + 1)
    left = pd.DataFrame({"k": ku, "d": du, "iu": iu})
    left = left[np.isin(ku, k1)]
    right = pd.DataFrame({"k": k1, "i1": np.arange(len(s1))})
    j = left.merge(right, on="k").sort_values("d").drop_duplicates("iu")
    j = pd.DataFrame({
        "id": unm.id.to_numpy()[j.iu], "name": unm.name.to_numpy()[j.iu], "country": unm.country.to_numpy()[j.iu],
        "name_e": s1.name.to_numpy()[j.i1], "d": j.d.to_numpy()})
    stats = {}
    for country, g in j.groupby("country"):
        ops, add_w, rep_w, legal_sw = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
        shifts, legal_changed, n = collections.Counter(), 0, 0
        for a, b, d in zip(g.name_e, g.name, g.d):
            op, na, lr, la = classify_edit(a, b)
            # keep exemplars that look derived from the entity (share most name tokens)
            A = {_key(t) for t in a.split()} - LEGAL - {""}
            B = {_key(t) for t in b.split()} - LEGAL - {""}
            if not A or len(A & B) / len(A) < 0.5:
                continue
            n += 1
            ops[op] += 1
            shifts[int(d)] += 1
            if op == "add":
                add_w.update(na)
            elif op == "replace":
                rep_w.update(na)
            if lr or la:
                legal_changed += 1
                if lr and la:
                    legal_sw[f"{lr}>{la}"] += 1
        stats[country] = {
            "n_exemplars": n,
            "ops": dict(ops),
            "shift": {str(k): v for k, v in sorted(shifts.items())},
            "p_legal_change": legal_changed / max(1, n),
            "add_words": dict(add_w.most_common(3000)),
            "replace_words": dict(rep_w.most_common(3000)),
            "legal_swaps": dict(legal_sw.most_common(200)),
        }
        log(f"[{country}] {n} sibling exemplars; ops {dict(ops)}; "
            f"legal change {stats[country]['p_legal_change']:.2f}; top added {list(add_w)[:12]}")
    return stats


# ----------------------------------------------------------------------------
# 2. apply one sibling edit to a noisy template record
# ----------------------------------------------------------------------------
def _cased(word: str, like: str) -> str:
    if like.isupper():
        return word.upper()
    if like.islower():
        return word.lower()
    return word[:1].upper() + word[1:].lower()


class Edit:
    """One sampled sibling edit, applied identically to every record of a group."""

    def __init__(self, rng, st: dict, base_name: str, min_shift: int = 1):
        ops = st["ops"]
        names = [o for o in ("add", "replace", "drop", "none") if ops.get(o)]
        w = np.array([ops[o] for o in names], float)
        self.op = names[rng.choice(len(names), p=w / w.sum())]
        core = [t for t in base_name.split() if _key(t) and _key(t) not in LEGAL]
        vocab = st["add_words"] if self.op == "add" else st["replace_words"]
        if not vocab:
            vocab = st["add_words"] or {"Group": 1}
        words = list(vocab)
        p = np.array([vocab[x] for x in words], float)
        self.word = words[rng.choice(len(words), p=p / p.sum())]
        self.target = _key(core[rng.integers(len(core))]) if core else None
        if self.op in ("replace", "drop") and (self.target is None or len(core) < 2):
            self.op = "add"
        self.legal = None
        if rng.random() < st["p_legal_change"] and st["legal_swaps"]:
            sw = list(st["legal_swaps"])
            q = np.array([st["legal_swaps"][x] for x in sw], float)
            self.legal = sw[rng.choice(len(sw), p=q / q.sum())].split(">")
        sh = st["shift"]
        ks = [int(k) for k in sh if int(k) >= min_shift]
        v = np.array([sh[str(k)] for k in ks], float)
        self.shift = ks[rng.choice(len(ks), p=v / v.sum())]

    def name(self, rec: str) -> str:
        toks = rec.split()
        if not toks:
            return rec
        keys = [_key(t) for t in toks]
        like = next((t for t in toks if any(c.isalpha() for c in t)), toks[0])
        if self.op in ("replace", "drop") and self.target in keys:
            i = keys.index(self.target)
            if self.op == "replace":
                toks[i] = _cased(self.word, toks[i])
            else:
                del toks[i]
        else:
            # add (or fallback): before a trailing legal form, else at the end
            pos = len(toks) - 1 if keys[-1] in LEGAL and len(toks) > 1 else len(toks)
            toks.insert(pos, _cased(self.word, like))
        if self.legal:
            src, dst = self.legal
            keys = [_key(t) for t in toks]
            if src in keys:
                i = keys.index(src)
                toks[i] = _cased(dst, toks[i])
        return " ".join(toks)

    def address(self, rec_addr: str, base_hn: int | None) -> str | None:
        """Shift the entity's house number inside the noisy address (keeps the record's
        formatting and zero padding).  None if the number cannot be located."""
        if base_hn is None or self.shift == 0:
            return rec_addr
        for m in _NUM_RE.finditer(rec_addr):
            if int(m.group()) == base_hn:
                digits = m.group()
                new = str(base_hn + self.shift)
                if digits.startswith("0"):
                    new = new.zfill(len(digits))
                return rec_addr[:m.start()] + new + rec_addr[m.end():]
        return None


# ----------------------------------------------------------------------------
# 3. driver
# ----------------------------------------------------------------------------
def _frame(t: pa.Table, src: int | None = None) -> pd.DataFrame:
    addr = t["business_address"]
    d = pd.DataFrame({
        "id": t["entity_id"].to_numpy(zero_copy_only=False),
        "name": t["business_name"].to_numpy(zero_copy_only=False),
        "addr": addr.to_numpy(zero_copy_only=False),
        "country": t["country"].to_numpy(zero_copy_only=False),
        "sk": street_key(addr).to_numpy(zero_copy_only=False),
    })
    first = norm_text(pc.list_element(pc.split_pattern(pc.fill_null(addr, ""), ",", max_splits=1), 0))
    hn = pc.extract_regex(first, r"(?P<h>\d+)")
    d["hn"] = pd.to_numeric(pd.Series(pc.struct_field(hn, [0]).to_numpy(zero_copy_only=False)), errors="coerce")
    if src is not None:
        d["src"] = src
    return d


def _link_or_copy(a: str, b: str):
    if os.path.exists(b):
        os.remove(b)
    try:
        os.link(a, b)
    except OSError:
        shutil.copyfile(a, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="original dataset folder (train/, test/)")
    ap.add_argument("--out-dir", required=True, help="augmented copy is written here")
    ap.add_argument("--target-unmatched", type=float, default=0.37,
                    help="share of unmatched Source-2/3 records per country after augmentation "
                         "(test: ~0.37-0.40, train: 0.25)")
    ap.add_argument("--group-sizes", type=float, nargs=3, default=[0.35, 0.55, 0.10],
                    help="probabilities of synthetic sibling groups with 1 / 2 / 3 records")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-shift", type=int, default=1,
                    help="smallest house-number shift of a synthetic sibling (default 1). 0 reproduces the "
                         "first version, which also made siblings at the entity's own address - they look "
                         "exactly like true records with generator noise and taught the model to reject them")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tr_in, tr_out = os.path.join(a.data_dir, "train"), os.path.join(a.out_dir, "train")
    os.makedirs(tr_out, exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "test"), exist_ok=True)

    log("reading training data")
    s1 = _frame(read_tsv(os.path.join(tr_in, "train_source1.tsv")))
    tabs = {s: read_tsv(os.path.join(tr_in, f"train_source{s}.tsv")) for s in (2, 3)}
    q = pd.concat([_frame(tabs[s], s) for s in (2, 3)], ignore_index=True)
    gt = read_tsv(os.path.join(tr_in, "train_ground_truth.tsv")).to_pandas()
    gt["mid"] = gt.matched_entity_ids.str.split(",")
    gt = gt.explode("mid")
    gt = gt[gt.mid.notna() & (gt.mid != "")][["source1_entity_id", "mid"]]
    matched = q.id.isin(gt.mid)
    log(f"S1 {len(s1)}, S2+S3 {len(q)}, unmatched share {1 - matched.mean():.3f}")

    stats = learn_edits(s1[s1.sk.str.len() >= 4], q[~matched & (q.sk.str.len() >= 4)], seed=a.seed)

    # templates: true records of every entity
    rec = q.set_index("id")
    gt = gt[gt.mid.isin(rec.index)]
    by_entity = gt.groupby("source1_entity_id").mid.apply(list)
    base_info = s1.set_index("id")
    gs = np.array(a.group_sizes, float)
    gs /= gs.sum()
    mean_size = float((gs * np.arange(1, 4)).sum())

    new_rows, summary = [], {"target_unmatched": a.target_unmatched, "group_sizes": gs.tolist(), "countries": {}}
    counter = 0
    for country, st in stats.items():
        in_c = (q.country == country).to_numpy()
        n_c, u_c = int(in_c.sum()), int((in_c & ~matched.to_numpy()).sum())
        need = max(0, int((a.target_unmatched * n_c - u_c) / (1 - a.target_unmatched)))
        bases = by_entity.index[base_info.loc[by_entity.index, "country"].to_numpy() == country]
        if a.min_shift > 0:
            has_hn = np.array([first_number(x) is not None for x in base_info.loc[bases, "addr"]])
            bases = bases[has_hn]
        if need == 0 or len(bases) == 0 or st["n_exemplars"] < 100:
            log(f"[{country}] nothing to add (need {need}, exemplars {st['n_exemplars']})")
            continue
        n_groups = int(round(need / mean_size))
        pick = bases[rng.integers(0, len(bases), n_groups)]
        sizes = rng.choice([1, 2, 3], size=n_groups, p=gs)
        made, groups, failed = 0, collections.Counter(), 0
        for e, k in zip(pick, sizes):
            b = base_info.loc[e]
            base_hn = first_number(b["addr"])
            if base_hn is None and a.min_shift > 0:
                continue          # no house number to shift: the sibling would sit at the true address
            edit = Edit(rng, st, b["name"], a.min_shift)
            tmpl = by_entity[e]
            tmpl = [tmpl[i] for i in rng.permutation(len(tmpl))[:k]]
            got = 0
            for rid in tmpl:
                r = rec.loc[rid]
                addr = edit.address(r["addr"], base_hn)
                if addr is None:
                    failed += 1
                    continue
                src = int(rng.integers(2, 4))
                counter += 1
                new_rows.append((f"S{src}-X{counter:09d}", edit.name(r["name"]), addr, country, src))
                got += 1
            if got:
                groups[got] += 1
                made += got
        summary["countries"][country] = {
            "records": n_c, "unmatched_before": u_c, "added": made,
            "unmatched_share_after": round((u_c + made) / (n_c + made), 4),
            "groups_by_size": dict(groups), "templates_without_locatable_house_number": failed,
            "sibling_exemplars": st["n_exemplars"], "ops": st["ops"], "shift": st["shift"],
        }
        log(f"[{country}] added {made} sibling records in {sum(groups.values())} groups {dict(groups)} "
            f"-> unmatched share {(u_c + made) / (n_c + made):.3f}")
    del rec, q

    new = pd.DataFrame(new_rows, columns=["entity_id", "business_name", "business_address", "country", "src"])
    for s in (2, 3):
        add = new[new.src == s].drop(columns="src")
        t = pa.concat_tables([tabs[s], pa.Table.from_pandas(add, preserve_index=False).cast(tabs[s].schema)])
        t = t.take(pa.array(rng.permutation(t.num_rows)))      # no information in row order
        write_tsv(t, os.path.join(tr_out, f"train_source{s}.tsv"))
        log(f"wrote train_source{s}.tsv: {t.num_rows} records (+{len(add)})")
    for f in ("train_source1.tsv", "train_ground_truth.tsv"):
        _link_or_copy(os.path.join(tr_in, f), os.path.join(tr_out, f))
    for f in os.listdir(os.path.join(a.data_dir, "test")):
        _link_or_copy(os.path.join(a.data_dir, "test", f), os.path.join(a.out_dir, "test", f))
    with open(os.path.join(a.out_dir, "augment_stats.json"), "w") as f:
        json.dump({"summary": summary, "learned": stats}, f, indent=1)
    log("done; statistics in", os.path.join(a.out_dir, "augment_stats.json"))


if __name__ == "__main__":
    main()
