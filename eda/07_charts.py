"""Static charts (PNG) for the EDA report -> results/figures/.

Palette: reference data-viz palette (validated: first 3 slots all-pairs, 4 slots
adjacent; light-mode figures with direct labels, values also in the tables).

    python 07_charts.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from erlib import data  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CONTEXT = "#c3c2b7"
FIG_DIR = os.path.join(data.RESULTS_DIR, "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round", "legend.frameon": False,
})


def style(ax, title, subtitle=None):
    ax.set_title(title, loc="left", color=INK, fontweight="semibold", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
    ax.tick_params(length=0)


def fig_tradeoff(b):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    pts = []
    for lab, e in b.items():
        if lab.startswith("_") or e.get("cand_mean", 0) < 1:
            continue  # postal codes (A1) retrieve ~0 candidates: see table
        pts.append((lab, e))
        for r in e.get("recall_at_k", []):
            if r["K"] in (10, 50, 200):
                pts.append((f"{e['id']} top-{r['K']}", {**e, **r}))
    rec = [(lab, e) for lab, e in pts if lab.startswith("U8@")]
    for lab, e in pts:
        if lab.startswith("U8@"):
            continue
        ax.scatter(e["cand_mean"], 100 * e["recall_pairs"], s=36, color=CONTEXT, edgecolor=SURFACE,
                   linewidth=1.5, zorder=2)
    rec.sort(key=lambda x: x[1]["cand_mean"])
    ax.plot([e["cand_mean"] for _, e in rec], [100 * e["recall_pairs"] for _, e in rec], color=SERIES[0],
            zorder=3)
    ax.scatter([e["cand_mean"] for _, e in rec], [100 * e["recall_pairs"] for _, e in rec], s=64,
               color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=4)
    for j, (lab, e) in enumerate(rec):
        if j not in (0, len(rec) - 1):
            continue  # middle points are in the table; labelling them collides
        ax.annotate(f"{lab}: {100 * e['recall_pairs']:.1f}% at {e['cand_mean']:.0f} cand.",
                    (e["cand_mean"], 100 * e["recall_pairs"]),
                    xytext=(0, -15) if j == 0 else (6, 6), textcoords="offset points", fontsize=8,
                    color=INK, ha="center" if j == 0 else "left")
    show = {"N3 k>=1": "N3 sorted token set", "A3 k>=1": "A3 street key", "L1.16x4 k>=1": "L1 name MinHash 16x4",
            "T1 k>=1 cap=10,000": "T1 name tokens", "T3 k>=2 cap=20,000": "T3 name+addr tokens k>=2",
            "N4 k>=1": "N4 Soundex", "A2 k>=1 cap=200,000": "A2 locality", "L2.16x4 k>=1": "L2 addr MinHash 16x4",
            "R1 k>=1 cap=100,000": "R1 all IDF candidates", "T2 k>=2 cap=20,000": "T2 addr tokens k>=2"}
    show.pop("L2.16x4 k>=1")  # sits next to L1; see the tables
    offsets = {"L2.16x4 k>=1": (5, -11), "T2 k>=2 cap=20,000": (-4, 5)}
    for lab, e in pts:
        if lab in show:
            dx, dy = offsets.get(lab, (5, 3))
            ax.annotate(show[lab], (e["cand_mean"], 100 * e["recall_pairs"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8, color=INK2, ha="right" if dx < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(3, 3e5)
    ax.set_xlabel("Candidates per Source-1 entity (log scale)")
    ax.set_ylabel("Pair recall (%)")
    ax.set_ylim(0, 102)
    style(ax, "Blocking trade-off: recall vs candidate volume",
          "Blue line = recommended union U8 at K = 10 / 25 / 50 / 100; gray = every other strategy / config")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "blocking_tradeoff.png"), dpi=160)
    plt.close(fig)


def fig_recall_at_k(b):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    series = [("R2", "Joint name+address IDF, top-K (R2)"), ("R3", "Name-only IDF, top-K (R3)"),
              ("R4", "Address-only IDF, top-K (R4)")]
    handles = []
    for i, (sid, name) in enumerate(series):
        e = next((v for k, v in b.items() if not k.startswith("_") and v.get("id") == sid
                  and v.get("recall_at_k")), None)
        if not e:
            continue
        xs = [r["cand_mean"] for r in e["recall_at_k"]]
        ys = [100 * r["recall_pairs"] for r in e["recall_at_k"]]
        h, = ax.plot(xs, ys, color=SERIES[i], marker="o", markersize=7, markeredgecolor=SURFACE,
                     markeredgewidth=2, label=name)
        handles.append(h)
    ux = [b[f"U8@{kk}"]["cand_mean"] for kk in (10, 25, 50, 100) if f"U8@{kk}" in b]
    uy = [100 * b[f"U8@{kk}"]["recall_pairs"] for kk in (10, 25, 50, 100) if f"U8@{kk}" in b]
    if ux:
        h, = ax.plot(ux, uy, color=SERIES[3], marker="o", markersize=7, markeredgecolor=SURFACE,
                     markeredgewidth=2, label="U8: joint top-K + name-MinHash top-K + no-address channel")
        handles.append(h)
    ax.set_xscale("log")
    ax.set_xlabel("Candidates per Source-1 entity (log scale)")
    ax.set_ylabel("Pair recall (%)")
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    style(ax, "Ranked retrieval: recall as K grows",
          "Top-K per entity, K = 1, 5, 10, 20, 50, 100, 200 (U8 at K = 10, 25, 50, 100 per list)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "recall_at_k.png"), dpi=160)
    plt.close(fig)


def fig_score_dists():
    p = os.path.join(data.CACHE_DIR, "pair_scores.npz")
    if not os.path.exists(p):
        return
    z = np.load(p, allow_pickle=True)
    classes = [("true", "True match"), ("hard_neg", "Same block, wrong entity"),
               ("single_cand", "Candidate of a no-match entity")]
    panels = [("Name token-set ratio (core + Indic dictionary)", z["label_all"], z["name_tset"]),
              ("Address token-set ratio (canonical tokens)", z["label_all"], z["addr_tset"]),
              ("LightGBM match probability (out-of-fold)", z["cand_label"], z["cand_score"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)
    bins = np.linspace(0, 1, 41)
    for pi, (ax, (title, lab, sc)) in enumerate(zip(axes, panels)):
        lab = lab.astype(str)
        for i, (c, name) in enumerate(classes):
            v = sc[(lab == c) & ~np.isnan(sc)]
            if len(v) == 0:
                continue
            h, _ = np.histogram(v, bins=bins)
            h = h / h.sum() * 100
            ax.step(bins[:-1], h, where="post", color=SERIES[i], linewidth=2, label=name)
        ax.set_xlim(0, 1)
        ax.set_xlabel("score")
        ax.set_ylabel("% of pairs in class" + (" (log scale)" if pi == 2 else ""))
        if pi == 2:
            ax.set_yscale("log")
        style(ax, title)
        ax.title.set_fontsize(10)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Wrong candidates of matched entities and candidates of no-match entities are indistinguishable "
                 "(the two lines overlap)", x=0.01, ha="left", fontsize=9, color=INK2, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "score_distributions.png"), dpi=160)
    plt.close(fig)


def fig_match_sizes(p):
    g = p["gt"]["ALL"]["size_dist"]
    keys = ["0", "1", "2", "3", "4", "5", "6", "7", "8+"]
    vals = [100 * g[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(keys, vals, width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for k, v in zip(keys, vals):
        if k in ("0", "1", "3"):
            ax.text(k, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("Number of matched S2/S3 records per Source-1 entity")
    ax.set_ylabel("% of Source-1 entities")
    ax.grid(axis="x", visible=False)
    style(ax, "Ground-truth match-set sizes", f"{p['gt']['n_s1']:,} Source-1 entities, {p['gt']['n_pairs']:,} true pairs")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "match_set_sizes.png"), dpi=160)
    plt.close(fig)


def fig_suffix(s):
    st = s["suffix_study"]
    stages = [("raw", "raw"), ("norm", "normalised"), ("canon", "suffix\ncanonicalised"),
              ("core", "suffix +\nhonorific removed"), ("core_tr", "removed +\nIndic dictionary")]
    metrics = [("jw", "Jaro-Winkler"), ("tsort", "Token sort ratio"), ("tset", "Token set ratio")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (key, ylab) in zip(axes, (("auc", "AUC, true vs same-block wrong"), ("best_f05", "Best macro F0.5 (one threshold)"))):
        for i, (m, name) in enumerate(metrics):
            ys = [st[m][v][key] for v, _ in stages]
            ax.plot(range(len(stages)), ys, color=SERIES[i], marker="o", markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=2, label=name)
        ax.set_xticks(range(len(stages)), [lab for _, lab in stages], fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        style(ax, "Name-only separability by normalisation step" if key == "auc" else "Name-only macro F0.5 by normalisation step")
        ax.title.set_fontsize(10)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "suffix_normalisation.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    R = data.RESULTS_DIR
    b = json.load(open(os.path.join(R, "blocking_results.json"))) if os.path.exists(os.path.join(R, "blocking_results.json")) else None
    p = json.load(open(os.path.join(R, "profile.json"))) if os.path.exists(os.path.join(R, "profile.json")) else None
    s = json.load(open(os.path.join(R, "similarity.json"))) if os.path.exists(os.path.join(R, "similarity.json")) else None
    if b:
        fig_tradeoff(b)
        fig_recall_at_k(b)
    if p:
        fig_match_sizes(p)
    if s:
        fig_suffix(s)
    fig_score_dists()
    print("figures ->", FIG_DIR, os.listdir(FIG_DIR))


if __name__ == "__main__":
    sys.exit(main())
