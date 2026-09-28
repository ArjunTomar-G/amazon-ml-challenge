"""France through the cross-encoder stacker (experiment; France has no labels).

    python france_stack.py score      CE logits (mean of the fold models, both input orders) for the France
                                      band pairs of test (0.005 < p < 0.995)  -> ce/test_fr_band.parquet,
                                      ce/test_fr_band_logit.npy
    python france_stack.py apply      stacker (models/stacker.txt, trained on US / India validation band pairs)
                                      on the France band -> feat/test_fr_pstack.npy (full pair vector: France
                                      band pairs re-scored, everything else = the current test_pfinal)

The France rules (france.py) are applied afterwards in final.py exactly as before, so generic-word swaps /
insertions are still linked and category swaps still unlinked whatever the stacker says.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

from common import Timer, log, wpath
from decide import final_prob

BAND = (0.005, 0.995)
import os
SUF = os.environ.get("ER_FR_SUFFIX", "")          # "_fr": inputs re-scored by france_street.py


def step_score():
    import torch
    import crossenc as ce
    from transformers import AutoTokenizer
    kt = np.load(wpath("feat", "test_ctxkeys.npz"))
    pr = pl.read_parquet(wpath("feat", "test_pairs.parquet"))
    pf = final_prob(np.load(wpath("feat", f"test_p1{SUF}.npy")), np.load(wpath("feat", f"test_p2{SUF}.npy")))
    fr = kt["q_country"][pr["q_row"].to_numpy()] == "France"
    rows = np.flatnonzero(fr & (pf > BAND[0]) & (pf < BAND[1]))
    band = pr[rows].select("q_row", "e_row").with_columns(pl.Series("row", rows.astype(np.int64)))
    band.write_parquet(wpath("ce", f"test_fr_band{SUF}.parquet"))
    log(f"France band: {len(rows)} pairs")
    tok = AutoTokenizer.from_pretrained(ce.MODEL_DIR)
    s1 = ce._texts("test", "source1")
    qt = ce._texts("test", "source2") + ce._texts("test", "source3")
    for tag, col, texts in (("e", "e_row", s1), ("q", "q_row", qt)):
        r = np.unique(band[col].to_numpy())
        flat, off = ce._tokenize(tok, [texts[i] for i in r])
        np.savez(wpath("ce", f"test_frtok_{tag}.npz"), rows=r.astype(np.int64), flat=flat, off=off)
    del s1, qt
    T = ce.Tokens.__new__(ce.Tokens)
    T.side = {}
    for tag in ("e", "q"):
        z = np.load(wpath("ce", f"test_frtok_{tag}.npz"))
        T.side[tag] = (z["rows"], z["flat"], z["off"])
    T.cls, T.sep, T.pad = np.load(wpath("ce", "special.npy")).tolist()
    ei, qi = T.index("e", band["e_row"].to_numpy()), T.index("q", band["q_row"].to_numpy())
    dev = ce._device()
    out = {}
    for variant in ("", "r"):
        ce.VARIANT = variant          # Tokens.batch reads the module-level input order
        ls = []
        for tag in ("A", "B"):
            m = ce._load_model(str(wpath("models", f"ce_{tag}{variant}")), dev).eval()
            with Timer(f"France band: CE {tag}{variant} on {len(ei)} pairs"):
                ls.append(ce._score(m, T, ei, qi, dev))
            del m
            torch.cuda.empty_cache()
        out[variant] = 0.5 * (ls[0] + ls[1])
    np.save(wpath("ce", f"test_fr_band_logit{SUF}.npy"), out[""].astype(np.float32))
    np.save(wpath("ce", f"test_fr_band_logitr{SUF}.npy"), out["r"].astype(np.float32))


def step_apply():
    import lightgbm as lgb
    import stack as S
    m = lgb.Booster(model_file=str(wpath("models", "stacker.txt")))
    keys = np.load(wpath("feat", "test_ctxkeys.npz"))
    pr = pl.read_parquet(wpath("feat", "test_pairs.parquet"))
    p1 = np.load(wpath("feat", f"test_p1{SUF}.npy"))
    pf = final_prob(p1, np.load(wpath("feat", f"test_p2{SUF}.npy")))
    band = pl.read_parquet(wpath("ce", f"test_fr_band{SUF}.parquet"))
    ce_ = np.mean([np.load(wpath("ce", f"test_fr_band_logit{SUF}.npy")), np.load(wpath("ce", f"test_fr_band_logitr{SUF}.npy"))], axis=0)
    ctx = np.load(wpath("feat", f"test_ctx{SUF}.npy"), mmap_mode="r")
    india = np.zeros(pr.height, bool)
    pos = band["row"].to_numpy()
    with Timer(f"France stacker: {len(pos)} band pairs"):
        X = S.features(pr, pf, p1, S._Rows(ctx, np.arange(pr.height)), pos, ce_.astype(np.float32), india)
        PB = S.pair_block("test", pos)
        if SUF:                        # street / locality features patched by france_street.py
            z = np.load(wpath("feat", "test_fr_patch.npz"))
            names = S.pair_names()
            j = np.searchsorted(z["rows"], pos)
            assert (z["rows"][j] == pos).all()
            for k in z.files:
                if k in names:
                    PB[:, names.index(k)] = z[k][j]
        X = np.hstack([X, PB])
        P = m.predict(X, num_threads=16)
    out = np.load(wpath("feat", f"test_pfinal{SUF}.npy")).copy()
    log(f"France band: mean p {out[pos].mean():.4f} -> {P.mean():.4f}; "
        f">=0.8: {(out[pos] >= 0.8).sum()} -> {(P >= 0.8).sum()}")
    out[pos] = P
    np.save(wpath("feat", f"test_fr_pstack{SUF}.npy"), out.astype(np.float32))


if __name__ == "__main__":
    {"score": step_score, "apply": step_apply}[sys.argv[1]]()
