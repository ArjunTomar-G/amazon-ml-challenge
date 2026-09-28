"""France (no training labels): link-level rules on the record's best S1 (blueprint section 5).

"Signature" = identical sorted address numbers (with a digit), >= 1 core name word shared, exactly
one word added (stop words and legal forms excluded).  On the record's best S1 (pa = p / max(1, sum p)):

    A   one word swapped in, the new word in {groupe, developpement, france, fils}   -> link (pa > 0.001)
    B   one word inserted, nothing dropped, in {groupe, developpement, france}      -> link (pa > 0.001)
    C   cie -> compagnie (S1 legal form "cie", record adds "compagnie")             -> link (pa > 0.001)
    D   linked with 0.8 < pa < 0.999 but the record names a different street        -> unlink
    T2  one word swapped to another real word (>= 30 records of the country; not a
        typo / accent variant / abbreviation of the dropped word, not a generic
        replacement word; "service" / "compagnie" swaps are look-alikes too)         -> unlink
    A0  as A / B, record address without a number, best S1 clearly ahead (p2 < p1 / 2) -> link
    T3  linked, one "category" word (a word the T2 population swaps in >= CAT_MIN times: club, ecole,
        comite ...) replaced by another category word, whatever else changed               -> unlink
    A-C skip records on another street.

The data generator's replacement operator swaps a name word for a *generic* word; in France the
generic words are groupe / developpement / france / fils / associes / services (+ cie -> compagnie).
Swaps to other descriptor words at the same address ("Cosmic Maison" -> "Cosmic Sportif") are
co-located look-alike businesses.  On our v4 test predictions the label-free singleton test
(share of links whose S1 has no other confident link: 1.8 % for true links, 6-9 % for false links)
gives 10.3 % for such links at 0.8 <= p < 0.9 and 1.2 % for accent variants at p >= 0.999.
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from common import log, wpath
from normalize import STREET_TYPES, _STREET_TYPE_VALUES

A_WORDS = {"groupe", "developpement", "france", "fils"}
B_WORDS = {"groupe", "developpement", "france"}
GENERIC = A_WORDS | {"associes", "services", "service", "compagnie"}
# words whose same-address swaps are true copies.  Copy-twin test: the replacement operator picks its word
# independently for every copy of an S1, so two copies of one S1 share the word (fils 5.8 %, groupe 11.8 %,
# developpement 12.0 %, france 13.0 %, services 3.7 %, associes 2.9 %); look-alike businesses do not
# (every other swap word 0-1.3 %, "service" 0.0 %, "compagnie" 0.2 %).
T2_KEEP = A_WORDS | {"associes", "services"}
STREET_STOP = set(STREET_TYPES) | set(STREET_TYPES.values()) | _STREET_TYPE_VALUES | {
    "de", "du", "des", "la", "le", "les", "et", "sur", "sous", "aux", "the", "and"}
COUNTRY = "France"
RULES_ALL = ("A", "A0", "B", "C", "D", "T2", "T3")
CAT_MIN = 50          # T3: swap-in count that makes a word a category word


def _norm(split: str):
    cols = ["country", "n_core", "n_legal", "a_nums", "a_street", "a_tok"]
    W = wpath(os.environ.get("ER_FR_NORM", "norm"))   # norm_fr: France street fix (france_street.py)
    s1 = pl.read_parquet(W / f"{split}_source1.parquet", columns=cols).with_row_index("e_row")
    q = pl.concat([pl.read_parquet(W / f"{split}_source{i}.parquet", columns=cols) for i in (2, 3)]).with_row_index("q_row")
    return s1.filter(pl.col("country") == COUNTRY), q.filter(pl.col("country") == COUNTRY)


def _typo_like(a: str, b: str) -> bool:
    """Accent drop / typo / OCR variant / abbreviation rather than a different word."""
    if not a or not b:
        return False
    if fuzz.ratio(a, b) >= 75:
        return True
    s, t = (a, b) if len(a) <= len(b) else (b, a)
    it = iter(t)
    sub = all(ch in it for ch in s)
    # accent dropped entirely ("freres" -> "frres"), or an abbreviation with the same first letter
    # ("freres" -> "frs", "saint" -> "st", "services" -> "svcs": the transformer calls 97-100 % of these true)
    return sub and (len(t) - len(s) <= 2 or (len(s) >= 2 and s[0] == t[0]))


def signatures(pairs: pl.DataFrame, p: np.ndarray, split: str = "test") -> pl.DataFrame:
    """Best pair of every France record with its signature and rule flags."""
    s1, q = _norm(split)
    d = pairs.select(pl.col("q_row").cast(pl.UInt32), pl.col("e_row").cast(pl.UInt32)).with_columns(pl.Series("p", p))
    d = d.join(q.select("q_row"), on="q_row")
    g = (d.sort("p", descending=True).group_by("q_row", maintain_order=True)
         .agg(pl.col("e_row").first(), pl.col("p").first().alias("pbest"), pl.col("p").sum().alias("psum"),
              pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2nd")))
    g = g.with_columns((pl.col("pbest") / pl.max_horizontal(pl.lit(1.0), pl.col("psum"))).alias("pa"))
    allw = pl.concat([s1["n_core"], q["n_core"]]).str.split(" ").list.unique().explode()
    wdf = allw.value_counts()
    real = set(wdf.filter(pl.col("count") >= 30)[wdf.columns[0]].to_list()) - {""}
    adf = s1["a_tok"].str.split(" ").list.unique().explode().value_counts()
    rare = set(adf.filter(pl.col("count") < 0.005 * s1.height)[adf.columns[0]].to_list())
    g = (g.join(q.select("q_row", pl.col("n_core").alias("qn"), pl.col("a_nums").alias("qnum"), pl.col("a_street").alias("qst")), on="q_row")
         .join(s1.select("e_row", pl.col("n_core").alias("en"), pl.col("n_legal").alias("el"), pl.col("a_nums").alias("enum"),
                         pl.col("a_street").alias("est")), on="e_row"))
    sw = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    g = g.with_columns(sw("qn").alias("qw"), sw("en").alias("ew"),
                       pl.col("qnum").str.split(" ").list.sort().list.join(" ").alias("qns"),
                       pl.col("enum").str.split(" ").list.sort().list.join(" ").alias("ens"))
    g = g.with_columns(pl.col("qw").list.set_difference(pl.col("ew")).alias("added"),
                       pl.col("ew").list.set_difference(pl.col("qw")).alias("dropped"),
                       pl.col("qw").list.set_intersection(pl.col("ew")).list.len().alias("shared"))
    g = g.with_columns(((pl.col("qns") == pl.col("ens")) & pl.col("qns").str.contains(r"\d")).alias("samenum"),
                       pl.col("added").list.len().alias("nadd"), pl.col("dropped").list.len().alias("ndrop"),
                       pl.col("added").list.first().alias("aw"), pl.col("dropped").list.first().alias("dw"),
                       pl.col("el").fill_null("").str.contains(r"\bcie\b").alias("e_cie"))
    g = g.with_columns((pl.col("samenum") & (pl.col("shared") >= 1) & (pl.col("nadd") == 1)).alias("sig"))
    diff = np.zeros(g.height, bool)
    typo = np.zeros(g.height, bool)
    for i, (a, b, aw, dw) in enumerate(zip(g["qst"].to_list(), g["est"].to_list(), g["aw"].to_list(), g["dw"].to_list())):
        if a and b:
            ta = [t for t in a.split() if t.isalpha() and len(t) >= 3 and t not in STREET_STOP and t in rare]
            tb = [t for t in b.split() if t.isalpha() and len(t) >= 3 and t not in STREET_STOP and t in rare]
            if ta and tb and max(fuzz.ratio(x, y) for x in ta for y in tb) < 80:
                diff[i] = True
        if aw and dw:
            typo[i] = _typo_like(aw, dw)
    aw = g["aw"].to_list()
    g = g.with_columns(pl.Series("diffst", diff), pl.Series("typo", typo),
                       pl.Series("aw_real", [w in real if w else False for w in aw]))
    # Category vocabulary (label-free, per universe): the words that the T2 population (same-address
    # one-word swaps to a real, non-generic word) swaps in at least CAT_MIN times - club, ecole, comite ...
    t2pop = g.filter(pl.col("sig") & (pl.col("ndrop") == 1) & pl.col("aw_real") & ~pl.col("typo")
                     & ~pl.col("aw").is_in(list(T2_KEEP)))
    vc = t2pop["aw"].value_counts()
    cat = set(vc.filter(pl.col("count") >= CAT_MIN)["aw"].to_list())
    cs = [bool(D) and any(d in cat for d in D) and any(a in cat and not any(_typo_like(a, d) for d in D) for a in A)
          for A, D in zip(g["added"].to_list(), g["dropped"].to_list())]
    log(f"France category vocabulary: {len(cat)} words; category swaps on best pairs: {sum(cs)}")
    return g.with_columns(pl.Series("catswap", cs))


def rule_masks(g: pl.DataFrame, thr: float) -> dict:
    S = pl.col("sig") & ~pl.col("diffst")
    swap = pl.col("sig") & (pl.col("ndrop") == 1)
    ins = pl.col("sig") & (pl.col("ndrop") == 0)
    linked = pl.col("pbest") >= thr
    return {
        "A": S & (pl.col("ndrop") == 1) & pl.col("aw").is_in(list(A_WORDS)) & (pl.col("pa") > 0.001),
        # insertions of the generic words share the true-copy twin rate (16-18 %) whatever the model's p
        "B": S & (pl.col("ndrop") == 0) & pl.col("aw").is_in(list(B_WORDS)) & (pl.col("pa") > 0.001),
        "C": S & (pl.col("aw") == "compagnie") & pl.col("e_cie") & (pl.col("pa") > 0.001),
        # A0: the same generic swap / insertion on a record whose address has no number (number dropped or
        # address empty): same twin fingerprint (10-13 %); only when the best S1 is clearly ahead
        "A0": ((pl.col("nadd") == 1) & (pl.col("shared") >= 1) & (pl.col("qns") == "") & ~pl.col("diffst")
               & (pl.col("pa") > 0.001) & (pl.col("p2nd") < 0.5 * pl.col("pbest"))
               & (((pl.col("ndrop") == 1) & pl.col("aw").is_in(list(T2_KEEP)))
                  | ((pl.col("ndrop") == 0) & pl.col("aw").is_in(list(B_WORDS))))),
        "D": linked & (pl.col("pa") < 0.999) & pl.col("diffst"),
        "T2": swap & linked & pl.col("aw_real") & ~pl.col("typo") & ~pl.col("aw").is_in(list(T2_KEEP))
              & ~((pl.col("aw") == "compagnie") & pl.col("e_cie")),
        # T3: the T2 look-alikes that escape T2's strict signature (a second dropped word, extra address
        # numbers, a shifted house number): one category word swapped for another, whatever else changed
        "T3": linked & pl.col("catswap") & (pl.col("shared") >= 1)
              & ~(pl.col("added").list.contains("compagnie") & pl.col("e_cie")),
    }


def adjust(pairs: pl.DataFrame, p: np.ndarray, thr: float, rules=RULES_ALL, split: str = "test",
           t2_max_p: float = 2.0) -> tuple[np.ndarray, dict]:
    """Returns (p', counts): France best pairs forced to 1.0 (link rules) or 0.0 (unlink rules)."""
    g = signatures(pairs, p, split)
    m = rule_masks(g, thr)
    if t2_max_p < 2.0:
        m["T2"] = m["T2"] & (pl.col("pbest") < t2_max_p)
    key = pairs.select(pl.col("q_row").cast(pl.UInt32), pl.col("e_row").cast(pl.UInt32)).with_row_index("i")
    out = p.copy()
    counts = {}
    for r in rules:
        sel = g.filter(m[r]).select("q_row", "e_row", "pbest")
        if r in ("A", "A0", "B", "C"):    # link the record to its best S1
            idx = key.join(sel, on=["q_row", "e_row"])["i"].to_numpy()
            changed = int((out[idx] < thr).sum())
            out[idx] = 1.0
        else:                             # unlink: no pair of the record may be linked
            best = key.join(sel, on=["q_row", "e_row"])["i"].to_numpy()
            changed = int((out[best] >= thr).sum())
            idx = key.join(sel.select("q_row"), on="q_row")["i"].to_numpy()
            out[idx] = np.minimum(out[idx], 0.0)
        counts[r] = {"records": int(sel.height), "decisions_changed": changed}
    log("France rules:", counts)
    return out, counts
