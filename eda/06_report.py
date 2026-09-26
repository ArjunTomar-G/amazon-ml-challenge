"""Render results/*.json into results/REPORT_TABLES.md (all comparison tables).

    python 06_report.py
"""

import json
import os
import sys

from erlib import data

R = data.RESULTS_DIR


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=1):
    if x is None:
        return "–"
    if abs(x) >= 1e9:
        return f"{x / 1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e4:
        return f"{x / 1e3:.0f}k"
    return f"{x:,.{d}f}"


def rr(x):
    return f"{100 * x:.4f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------
# blocking
# --------------------------------------------------------------------------

def by(e, key):
    v = e.get("by", {}).get(key)
    return v[0] if v else None


def blocking_rows(res, labels):
    rows = []
    for lab in labels:
        e = res.get(lab)
        if not e:
            continue
        cfg = lab.split(" ", 1)[1] if " " in lab else ""
        rows.append([
            f"**{e['id']}** {e['desc'].replace('  U  ', ' ∪ ')}", cfg,
            pct(e["recall_pairs"]), pct(e["recall_entity_all"]), f"{e['f05_ceiling']:.3f}",
            num(e["cand_mean"]), rr(e["reduction_ratio"]), pct(e["false_cand_rate"], 2),
            num(e["max_block"], 0) if e.get("max_block", -1) >= 0 else "–",
            pct(by(e, "country=US")), pct(by(e, "country=India")),
            pct(by(e, "script=indic")), pct(by(e, "addr=empty")),
        ])
    return rows


BLOCK_HEADER = ["Strategy", "Config", "Pair recall", "Entities fully covered", "F0.5 ceiling",
                "Cand / S1", "Reduction ratio", "False-cand rate", "Largest block",
                "Recall US", "Recall India", "Recall: Indic-script match", "Recall: match w/o address"]


def blocking_section(res):
    meta = res.get("_meta", {})
    labels = [k for k in res if not k.startswith("_")]
    groups = [
        ("Exact / normalised name keys", "name-key"),
        ("Address-based keys", "address"),
        ("Token inverted index (significant tokens, block purging by DF cap)", "token-index"),
        ("MinHash LSH (char 3-grams)", "minhash-lsh"),
        ("Country-aware vs country-agnostic keys", "country"),
    ]
    md = []
    md.append(f"Sample: {meta.get('sample', '?'):,} random Source-1 entities "
              f"({meta.get('true_pairs_sample', 0):,} true pairs, {meta.get('singletons_sample', 0):,} singletons) "
              f"against the full Source-2+3 corpus ({meta.get('corpus', 0):,} records). "
              "Reduction ratio = 1 − candidates / (|S1 sample| × |S2 ∪ S3|). "
              "False-candidate rate = share of candidate pairs that are not true matches. "
              "F0.5 ceiling = macro F0.5 of a perfect matcher restricted to the candidate set "
              "(singletons score 1 by predicting nothing).\n")
    for title, grp in groups:
        labs = [lab for lab in labels if res[lab].get("group") == grp]
        if not labs:
            continue
        md.append(f"#### {title}\n")
        md.append(table(BLOCK_HEADER, blocking_rows(res, labs)))
        md.append("")
    ranked = [lab for lab in labels if res[lab].get("group") == "ranked"]
    if ranked:
        md.append("#### Ranked retrieval: keep the top-K candidates per Source-1 entity\n")
        md.append("The full candidate set (any shared key within the DF cap) is shown first; "
                  "then recall and cost when only the top-K by score are kept.\n")
        md.append(table(BLOCK_HEADER, blocking_rows(res, ranked)))
        md.append("")
        hdr = ["Strategy", "K", "Pair recall", "Entities fully covered", "F0.5 ceiling", "Cand / S1",
               "Reduction ratio", "False-cand rate"]
        rows = []
        for lab in ranked:
            e = res[lab]
            for r in e.get("recall_at_k", []):
                rows.append([f"**{e['id']}**", r["K"], pct(r["recall_pairs"]), pct(r["recall_entity_all"]),
                             f"{r['f05_ceiling']:.3f}", num(r["cand_mean"]), rr(r["reduction_ratio"]),
                             pct(r["false_cand_rate"], 2)])
        md.append(table(hdr, rows))
        md.append("")
    unions = [lab for lab in labels if res[lab].get("group") == "union"]
    if unions:
        md.append("#### Unions of complementary strategies (recommended family)\n")
        md.append(table(BLOCK_HEADER, blocking_rows(res, unions)))
        md.append("")
    return "\n".join(md)


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------

def profile_section(p):
    md = []
    srcs = ["S1", "S2", "S3"]
    ctys = sorted({c for s in srcs for c in p[s]})
    md.append("#### Volume, missing fields, scripts\n")
    rows = []
    for s in srcs:
        for c in ctys:
            r = p[s].get(c)
            if not r:
                continue
            ns = r["name_script"]
            indic = sum(v for k, v in ns.items() if k not in ("ASCII", "LATIN_EXT"))
            ascr = r["addr_script"]
            aindic = sum(v for k, v in ascr.items() if k not in ("ASCII", "LATIN_EXT"))
            rows.append([s, c, f"{r['n']:,}", pct(r.get("name_n_empty"), 2), pct(r["a_empty"], 2),
                         pct(r["a_null"], 2), pct(r.get("addr_core_empty"), 2),
                         pct(ns.get("LATIN_EXT", 0)), pct(indic), pct(aindic)])
    md.append(table(["Source", "Country", "Records", "Empty name", "Empty address",
                     "NULL/N/A placeholder", "Address empty after cleaning", "Accented Latin name",
                     "Indic-script name", "Indic-script address part"], rows))
    md.append("\n#### Field lengths (median [p5–p95])\n")
    rows = []
    for s in srcs:
        for c in ctys:
            r = p[s].get(c)
            if not r:
                continue
            f = lambda k: f"{r[k]['p50']:.0f} [{r[k]['p5']:.0f}–{r[k]['p95']:.0f}]"  # noqa: E731
            rows.append([s, c, f("name_len"), f("name_ntok"), f("name_ncore"), f("addr_len"), f("addr_ncomp")])
    md.append(table(["Source", "Country", "Name chars", "Name tokens", "Core-name tokens",
                     "Address chars", "Address components"], rows))
    md.append("\n#### Name noise patterns (share of records)\n")
    flags = [("f_legal", "legal suffix"), ("f_honor", "honorific/prefix"), ("f_domain", "domain/.com"),
             ("f_concat", "single long concatenated token"), ("f_alias", "alias (aka/dba/pipe)"),
             ("f_phone", "phone number"), ("f_hashat", "#/@ prefix"), ("f_brackets", "brackets"),
             ("f_allcaps", "ALL CAPS"), ("f_dash", "'--'")]
    rows = []
    for s in srcs:
        for c in ctys:
            r = p[s].get(c)
            if r:
                rows.append([s, c] + [pct(r[k]) for k, _ in flags])
    md.append(table(["Source", "Country"] + [lab for _, lab in flags], rows))
    md.append("\n#### Address format\n")
    aflags = [("a_startnum", "starts with number"), ("state_found", "US/India state found"),
              ("zip5_found", "5-digit ZIP"), ("pin6_found", "6-digit PIN"), ("a_landmark", "landmark (near/opp)"),
              ("a_unit", "unit/apt/floor"), ("a_pobox", "PO box"), ("a_hash", "'#'")]
    rows = []
    for s in srcs:
        for c in ctys:
            r = p[s].get(c)
            if r:
                pats = ", ".join(f"`{k}` {100 * v:.0f}%" for k, v in r["a_pattern_top"][:4])
                rows.append([s, c] + [pct(r[k]) for k, _ in aflags] + [pats])
    md.append(table(["Source", "Country"] + [lab for _, lab in aflags] +
                    ["Top component patterns (N=number, A=alpha, S=state)"], rows))
    md.append("\n#### Within-source duplicates (share of records in a duplicate group)\n")
    rows = []
    for s in srcs:
        for c in ctys:
            r = p[s].get(c)
            if r:
                rows.append([s, c, pct(r["dup_exact_raw"]["in_dup_group"], 2),
                             pct(r["dup_norm_name_addr"]["in_dup_group"], 2),
                             pct(r["dup_core_name"]["in_dup_group"], 2),
                             pct(r["dup_core_addr"]["in_dup_group"], 2)])
    md.append(table(["Source", "Country", "Exact raw name+address", "Normalised name+address",
                     "Core name only", "Core address only (non-empty)"], rows))
    g = p["gt"]
    md.append("\n#### Ground truth: match-set size per Source-1 entity\n")
    keys = ["0", "1", "2", "3", "4", "5", "6", "7", "8+"]
    rows = []
    for c in ["ALL"] + ctys:
        r = g.get(c)
        if r:
            rows.append([c, f"{r['n']:,}"] + [pct(r["size_dist"].get(k, 0)) for k in keys] +
                        [f"{r['mean_size_nonzero']:.2f}", pct(r["has_s2"]), pct(r["has_s3"])])
    md.append(table(["S1 country", "Entities"] + [f"size {k}" for k in keys] +
                    ["Mean size (matched)", "Has ≥1 S2 match", "Has ≥1 S3 match"], rows))
    pr = p["pairs"]
    md.append(f"\nTrue pairs: {g['n_pairs']:,}. Each S2/S3 record belongs to at most one S1 entity.\n")
    md.append("#### How true pairs differ (exact agreement after normalisation)\n")
    rows = []
    for s in ("S2", "S3"):
        r = pr[s]
        rows.append([s, f"{r['n_pairs']:,}", pct(r["eq_name_n"]), pct(r["eq_name_core"]), pct(r["eq_addr_n"]),
                     pct(r["eq_addr_core"]), pct(r["eq_cty"]), pct(r["match_name_indic"]), pct(r["match_addr_empty"])])
    md.append(table(["Source", "True pairs", "Name equal (normalised)", "Core name equal",
                     "Address equal (normalised)", "Core address equal", "Country equal",
                     "Match name in Indic script", "Match address empty"], rows))
    ca = pr.get("corpus_share_matched", {})
    if ca:
        md.append("\nShare of S2/S3 records that match some S1 entity (the rest are distractors): " +
                  "; ".join(f"{s} " + ", ".join(f"{c} {pct(v)}" for c, v in d.items()) for s, d in ca.items()) + "\n")
    tf = p.get("test_france")
    if tf:
        md.append("#### Unseen country check (test files, no labels): France\n")
        cc = tf.get("country_counts", {})
        md.append("Test country mix: " + "; ".join(
            f"{s}: " + ", ".join(f"{k} {v:,}" for k, v in sorted(d.items())) for s, d in cc.items()) + "\n")
        rows = []
        for s in srcs:
            r = tf.get(s)
            if not r:
                continue
            rows.append([s, f"{r['n_profiled']:,}", pct(r.get("name_script", {}).get("LATIN_EXT", 0) / r["n_profiled"]
                                                      if isinstance(r.get("name_script", {}).get("LATIN_EXT", 0), int) else None),
                         pct(r["a_empty"], 2), pct(r["state_found_with_US_India_lexicon"]), pct(r["zip5"]),
                         pct(r["f_legal"]), pct(r["name_token_share_removed_by_hand_list"]),
                         pct(r["addr_token_share_in_US_street_lexicon"]),
                         ", ".join(t for t, _ in r["top_name_tokens"][:8])])
        md.append(table(["Source", "Profiled", "Accented name", "Empty address", "State found (US/IN lexicon)",
                         "ZIP-like code", "US/IN legal suffix found", "Name tokens hit by hand stop list",
                         "Addr tokens in US street lexicon", "Most frequent name tokens"], rows))
    return "\n".join(md)


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------

FEATURE_LABEL = {
    "lev": "Levenshtein (normalised)", "jw": "Jaro-Winkler", "tsort": "token sort ratio",
    "tset": "token set ratio", "jac": "Jaccard (tokens)", "jac3": "Jaccard (char 3-grams)",
    "partial": "partial ratio", "tfidf_char": "TF-IDF cosine (char 2-4)", "tfidf_word": "TF-IDF cosine (word)",
}


def similarity_section(s):
    md = []
    n = s["n_pairs"]
    md.append("Pairs: " + ", ".join(f"{k} {v:,}" for k, v in n.items()) +
              f". Macro F0.5 uses {s['n_base_s1']:,} random S1 entities; each score is thresholded "
              "globally and every candidate at or above the threshold is predicted as a match.\n")
    md.append("#### Single features (true match vs same-block wrong match)\n")
    rows = []
    for c, r in sorted(s["features"].items(), key=lambda kv: -(kv[1].get("auc_true_vs_hardneg") or 0)):
        if c.startswith("emb_"):
            continue
        rows.append([f"`{c}`", f"{r['auc_true_vs_hardneg']:.3f}", f"{r['ap_true_vs_hardneg']:.3f}",
                     f"{r['corr']:.3f}", f"{r['auc_true_vs_all_cands']:.3f}",
                     f"{r.get('best_macro_f05', float('nan')):.3f}", pct(r.get("nan_rate_true"))])
    md.append(table(["Feature", "AUC", "AP", "Point-biserial r", "AUC incl. singleton cands",
                     "Best macro F0.5 (1 threshold)", "Missing on true pairs"], rows[:30]))
    md.append("\n#### Legal-suffix normalisation: before vs after (name features)\n")
    rows = []
    for metric, d in s["suffix_study"].items():
        for v in ("raw", "norm", "canon", "core", "core_tr"):
            r = d[v]
            rows.append([FEATURE_LABEL.get(metric, metric), v, f"{r['auc']:.3f}", f"{r['ap']:.3f}",
                         f"{r['best_f05']:.3f}", f"{r['mean_true']:.3f}", f"{r['mean_hardneg']:.3f}",
                         f"{r['mean_true'] - r['mean_hardneg']:.3f}"])
    md.append(table(["Metric", "Normalisation", "AUC", "AP", "Best macro F0.5", "Mean (true)",
                     "Mean (wrong)", "Gap"], rows))
    ss = s["suffix_subsets"]
    md.append(f"\nTargeted subsets (token-set ratio): true pairs whose legal suffixes differ "
              f"(n={ss['n_true_suffix_differs']:,}) and wrong pairs that share a legal suffix "
              f"(n={ss['n_hardneg_suffix_shared']:,}).\n")
    md.append(table(["Normalisation", "True, suffix differs", "Wrong, suffix shared"],
                    [[v, f"{d['true_suffix_differs']:.3f}", f"{d['hardneg_suffix_shared']:.3f}"]
                     for v, d in ss["mean_tset"].items()]))
    md.append("\n#### Name-only vs address-only vs combined\n")
    rows = [[k, f"{v['auc_true_vs_hardneg']:.3f}", f"{v['ap_true_vs_hardneg']:.3f}", f"{v['corr']:.3f}",
             f"{v['auc_all']:.3f}", f"{v['best_macro_f05']:.4f}"] for k, v in s["combos"].items()]
    md.append(table(["Scorer", "AUC", "AP", "Point-biserial r", "AUC incl. singleton cands",
                     "Best macro F0.5"], rows))
    if "transfer" in s:
        md.append("\n#### Cross-country transfer (proxy for unseen France)\n")
        rows = []
        for k, d in s["transfer"].items():
            tr = [v for kk, v in d.items() if kk.startswith("trained_on")][0]
            ic = d["in_country_cv"]
            rows.append([k, f"{tr['auc']:.3f}", f"{ic['auc']:.3f}", f"{tr['best_macro_f05']:.4f}",
                         f"{tr.get('macro_f05_with_source_threshold', float('nan')):.4f}",
                         f"{ic['best_macro_f05']:.4f}", f"{d['source_threshold']:.3f}",
                         f"{tr['best_thr']:.3f}"])
        md.append(table(["Train → test", "AUC (transfer)", "AUC (in-country CV)", "F0.5 transfer, re-tuned thr",
                         "F0.5 transfer, source thr", "F0.5 in-country", "Source thr", "Target best thr"], rows))
    if "embeddings" in s:
        md.append("\n#### Embeddings vs classical (same pairs; true vs same-block wrong)\n")
        rows = []
        for c, d in s["embeddings"].items():
            if c.startswith("_"):
                continue
            rows.append([f"`{c}`", f"{d['all']['auc']:.3f}", f"{d['all']['ap']:.3f}",
                         f"{d['latin_cand']['auc']:.3f}", f"{d['indic_cand']['auc']:.3f}"])
        md.append(table(["Feature", "AUC all", "AP all", "AUC Latin-script cands", "AUC Indic-script cands"], rows))
        x = s["embeddings"].get("_lgbm_classical_vs_plus_emb")
        if x:
            md.append(f"\nLightGBM on the embedding subset: classical AUC {x['classical'][0]:.4f} / AP "
                      f"{x['classical'][1]:.4f}; classical + embeddings AUC {x['classical+emb'][0]:.4f} / AP "
                      f"{x['classical+emb'][1]:.4f}.\n")
    mg = s.get("merge")
    if mg:
        md.append(f"\n#### Merging insight by match-set size (LightGBM score, global threshold {mg['threshold']:.3f})\n")
        rows = []
        for gname, r in mg["by_group"].items():
            rows.append([gname, f"{r['n_entities']:,}", f"{r['cand_per_entity']:.0f}",
                         f"{r['true_score'].get('p50', float('nan')):.3f}" if r["true_score"] else "–",
                         f"{r['true_score'].get('p05', float('nan')):.3f}" if r["true_score"] else "–",
                         f"{r['wrong_score'].get('p95', float('nan')):.3f}",
                         f"{r['top_candidate_score'].get('p50', float('nan')):.3f}",
                         pct(r["separable_share"]) if r["separable_share"] is not None else "–",
                         pct(r["pred_nonempty_share"]), f"{r['macro_f05_at_thr']:.3f}"])
        md.append(table(["Match-set size", "Entities", "Cand / entity", "True score p50", "True score p5",
                         "Wrong score p95", "Top-candidate score p50", "Perfectly separable",
                         "Predicted non-empty", "Macro F0.5"], rows))
    tr = s.get("transitive")
    if tr:
        md.append("\n#### Transitive (sibling) evidence for weak candidates\n")
        rows = [[k, f"{v['n']:,}", pct(v["has_strong_anchor_in_same_group"]), pct(v["sibling_sim>=0.85"]),
                 pct(v["sibling_sim>=0.95"])] for k, v in tr.items()]
        md.append(table(["Weak candidates (below threshold)", "n", "Entity has a strong match",
                         "Sibling similarity ≥ 0.85", "Sibling similarity ≥ 0.95"], rows))
    cd = s.get("class_dist")
    if cd:
        md.append("\n#### Score distributions by pair class (median [p5–p95])\n")
        labels = ["true", "true_missed", "hard_neg", "single_cand", "random"]
        rows = []
        for f, d in cd.items():
            rows.append([f"`{f}`"] + [
                (f"{d[l]['p50']:.2f} [{d[l]['p05']:.2f}–{d[l]['p95']:.2f}]" if d.get(l) else "–") for l in labels])
        md.append(table(["Feature"] + labels, rows))
    return "\n".join(md)


def main():
    parts = ["# EDA results tables\n", "Generated by `06_report.py` from `results/*.json`.\n"]
    p = load("profile.json")
    if p:
        parts += ["## 1. Data profiling\n", profile_section(p), ""]
    b = load("blocking_results.json")
    if b:
        parts += ["## 2. Blocking / candidate generation\n", blocking_section(b), ""]
    s = load("similarity.json")
    if s:
        parts += ["## 3. Similarity features and merging insight\n", similarity_section(s), ""]
    with open(os.path.join(R, "REPORT_TABLES.md"), "w") as f:
        f.write("\n".join(parts))
    print("wrote", os.path.join(R, "REPORT_TABLES.md"))


if __name__ == "__main__":
    sys.exit(main())
