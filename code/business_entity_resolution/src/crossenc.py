"""Stage 3: transformer cross-encoder on the uncertain band (blueprint system A, 4.4).

    python crossenc.py select         band pairs: CE training pairs (folds A/B), validation records'
                                      band pairs, test band pairs (US / India)
    python crossenc.py tokenize       pre-tokenise every text used by a selected pair
    python crossenc.py train A|B      fine-tune multilingual-e5-small (MIT, 118M) on one fold's pairs
    python crossenc.py score          CE logits: every validation pair by the model(s) that never saw
                                      its S1 (A pairs <- M_B, B pairs <- M_A, V pairs <- mean), test
                                      pairs by the mean of both models

Model: multilingual-e5-small with a one-logit head, binary cross-entropy, input
"s1 name | s1 address" </s></s> "record name | record address" (raw text, lower-cased,
<= 62 tokens per side).  1 epoch, lr 1e-4 (3 % warmup, linear decay), AdamW (wd 0.01),
bf16 autocast, gradient clip 1.0, word-embedding matrix frozen.  Batches are sorted by
length and capped by a token budget (4 GB GPU).

Band: 0.005 < p < 0.995.  Training pairs use the out-of-fold stage-1 probability (stage 2
exists only for validation / test pairs); validation and test pairs use the final stage-2
probability, exactly as the stacker (stack.py) sees them.  France is not scored: it has no
labels, and the blueprint's singleton test found the CE adds doubtful links there.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import polars as pl
from numba import njit

from common import Timer, log, wpath
from decide import final_prob
from model import FOLD_A, FOLD_B, VAL_FOLD

BAND = (0.005, 0.995)
CE_COUNTRIES = ("US", "India")
VARIANT = os.environ.get("ER_CE_VARIANT", "")   # "": e5-small; "r": e5-small, record text first, other seed;
                                                # "b": multilingual-e5-base (278M, MIT), own tokenizer files
                                                # "l": multilingual-e5-large (560M, MIT), own tokenizer files
                                                # "c": e5-base, record text first, other seed (tokens of "b")
MODEL_DIR = os.environ.get("ER_CE_BASE", str(wpath("hf", {"b": "multilingual-e5-base", "c": "multilingual-e5-base",
                                                           "l": "multilingual-e5-large"}.get(VARIANT, "multilingual-e5-small"))))
TOKSUF = {"b": "b", "c": "b", "l": "l"}.get(VARIANT, "")
MAX_SIDE = 62                  # tokens per side -> pair <= 128
TRAIN_TOKENS = int(os.environ.get("ER_CE_TOKENS", 8192))     # token budget of a training batch
MAX_ROWS = 128
INFER_TOKENS = int(os.environ.get("ER_CE_INFER_TOKENS", 49152 if VARIANT != "b" else 12288))   # e5-base: 4x wider FFN
LR = float(os.environ.get("ER_CE_LR", 1e-4))
# training set: ce/train_<FIT>.parquet ("fit" = the stage-2 band; "fitb" = select_fit's wider band + confident sample)
FIT = os.environ.get("ER_CE_FIT", "fit")
WARMUP = 0.03
SEED = 2026 + {"": 0, "r": 7, "b": 13, "l": 19, "c": 29}.get(VARIANT, 0)


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------
def _validation_rows(pairs: pl.DataFrame) -> np.ndarray:
    vq = pairs.filter(pl.col("fold") == VAL_FOLD)["q_row"].unique()
    return np.flatnonzero(pairs["q_row"].is_in(vq.implode()).to_numpy())


def step_select():
    lo, hi = BAND
    kt = np.load(wpath("feat", "train_ctxkeys.npz"))
    pr = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    p1 = np.load(wpath("feat", "train_p1.npy"))
    fold = pr["fold"].to_numpy()
    q = pr["q_row"].to_numpy()
    ok_c = np.isin(kt["q_country"][q], CE_COUNTRIES)
    fit = np.flatnonzero(ok_c & np.isin(fold, FOLD_A + FOLD_B) & (p1 > lo) & (p1 < hi))
    vr = _validation_rows(pr)
    pf = final_prob(p1[vr], np.load(wpath("feat", "train_p2.npy"))[vr])
    val = vr[ok_c[vr] & (pf > lo) & (pf < hi)]
    y = pr["y"].to_numpy()
    for name, rows in (("fit", fit), ("val", val)):
        d = pr[rows].select("q_row", "e_row", "fold", "y").with_columns(pl.Series("row", rows.astype(np.int64)))
        d.write_parquet(wpath("ce", f"train_{name}.parquet"))
        log(f"train {name}: {len(rows)} pairs, {y[rows].mean():.3f} true, "
            f"folds {dict(zip(*np.unique(fold[rows], return_counts=True)))}")
    kt = np.load(wpath("feat", "test_ctxkeys.npz"))
    pr = pl.read_parquet(wpath("feat", "test_pairs.parquet"))
    pf = final_prob(np.load(wpath("feat", "test_p1.npy")), np.load(wpath("feat", "test_p2.npy")))
    ok_c = np.isin(kt["q_country"][pr["q_row"].to_numpy()], CE_COUNTRIES)
    rows = np.flatnonzero(ok_c & (pf > lo) & (pf < hi))
    pr[rows].select("q_row", "e_row").with_columns(pl.Series("row", rows.astype(np.int64))) \
        .write_parquet(wpath("ce", "test_band.parquet"))
    log(f"test band: {len(rows)} pairs")


def step_select_fit():
    """Larger training set for a bigger cross-encoder (reference recipe: the uncertain band plus a sample of the
    confident pairs) -> ce/train_<FIT>.parquet.  ER_CE_FIT_BAND = "lo,hi", ER_CE_FIT_EXTRA = confident pairs."""
    lo, hi = (float(x) for x in os.environ.get("ER_CE_FIT_BAND", "0.002,0.998").split(","))
    extra = int(os.environ.get("ER_CE_FIT_EXTRA", 600_000))
    kt = np.load(wpath("feat", "train_ctxkeys.npz"))
    pr = pl.read_parquet(wpath("feat", "train_pairs.parquet"))
    p1 = np.load(wpath("feat", "train_p1.npy"))
    ok = np.isin(kt["q_country"][pr["q_row"].to_numpy()], CE_COUNTRIES) & np.isin(pr["fold"].to_numpy(), FOLD_A + FOLD_B)
    band = np.flatnonzero(ok & (p1 > lo) & (p1 < hi))
    conf = np.flatnonzero(ok & ((p1 <= lo) | (p1 >= hi)))
    rng = np.random.default_rng(SEED)
    rows = np.sort(np.concatenate([band, rng.choice(conf, size=min(extra, len(conf)), replace=False)]))
    d = pr[rows].select("q_row", "e_row", "fold", "y").with_columns(pl.Series("row", rows.astype(np.int64)))
    d.write_parquet(wpath("ce", f"train_{FIT}.parquet"))
    log(f"train {FIT}: {len(band)} band pairs ({lo} < p1 < {hi}) + {min(extra, len(conf))} confident pairs, "
        f"{d['y'].mean():.3f} true")


# ----------------------------------------------------------------------------
# tokenisation
# ----------------------------------------------------------------------------
def _texts(split: str, source: str) -> list[str]:
    d = pl.read_parquet(wpath("raw", f"{split}_{source}.parquet"), columns=["business_name", "business_address"])
    return (d.select((pl.col("business_name").fill_null("").str.strip_chars() + " | "
                      + pl.col("business_address").fill_null("").str.strip_chars()).str.to_lowercase())
            .to_series().to_list())


def _tokenize(tok, texts: list[str], chunk=200_000):
    """Token ids without special tokens, <= MAX_SIDE per text -> (flat int32, offsets)."""
    from itertools import chain
    flats, lens = [], []
    for a in range(0, len(texts), chunk):
        enc = tok(texts[a:a + chunk], add_special_tokens=False, truncation=True, max_length=MAX_SIDE)["input_ids"]
        n = np.fromiter((len(x) for x in enc), np.int64, count=len(enc))
        flats.append(np.fromiter(chain.from_iterable(enc), np.int32, count=int(n.sum())))
        lens.append(n)
    n = np.concatenate(lens) if lens else np.zeros(0, np.int64)
    off = np.zeros(len(n) + 1, np.int64)
    off[1:] = np.cumsum(n)
    flat = np.concatenate(flats) if flats else np.zeros(0, np.int32)
    return flat, off


def step_tokenize():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    np.save(wpath("ce", f"special{TOKSUF}.npy"), np.array([tok.cls_token_id, tok.sep_token_id, tok.pad_token_id], np.int64))
    for split, sets in (("train", (FIT, "val")), ("test", ("band",))):
        prs = [pl.read_parquet(wpath("ce", f"{split}_{s}.parquet")) for s in sets]
        es = np.unique(np.concatenate([p["e_row"].to_numpy() for p in prs]))
        qs = np.unique(np.concatenate([p["q_row"].to_numpy() for p in prs]))
        with Timer(f"tokenise {split}: {len(es)} S1 texts, {len(qs)} record texts"):
            s1 = _texts(split, "source1")
            qt = _texts(split, "source2") + _texts(split, "source3")
            for tag, rows, texts in (("e", es, s1), ("q", qs, qt)):
                flat, off = _tokenize(tok, [texts[i] for i in rows])
                np.savez(wpath("ce", f"{split}_tok_{tag}{TOKSUF}.npz"), rows=rows.astype(np.int64), flat=flat, off=off)
                log(f"  {tag}: {len(rows)} texts, mean {np.diff(off).mean():.1f} tokens, "
                    f"truncated {(np.diff(off) >= MAX_SIDE).mean():.4f}")


class Tokens:
    """Row -> token ids for one split (S1 side and record side)."""

    def __init__(self, split: str):
        self.side = {}
        for tag in ("e", "q"):
            z = np.load(wpath("ce", f"{split}_tok_{tag}{TOKSUF}.npz"))
            self.side[tag] = (z["rows"], z["flat"], z["off"])
        self.cls, self.sep, self.pad = np.load(wpath("ce", f"special{TOKSUF}.npy")).tolist()

    def index(self, tag: str, rows: np.ndarray) -> np.ndarray:
        r = self.side[tag][0]
        i = np.searchsorted(r, rows)
        assert (r[np.minimum(i, len(r) - 1)] == rows).all(), "row without tokens"
        return i

    def lengths(self, ei, qi):
        return (np.diff(self.side["e"][2])[ei] + np.diff(self.side["q"][2])[qi] + 4).astype(np.int64)

    def batch(self, ei, qi):
        n = len(ei)
        L = int(self.lengths(ei, qi).max())
        out = np.full((n, L), self.pad, np.int64)
        if VARIANT in ("r", "c"):  # record first, S1 second
            _fill(out, qi, ei, self.side["q"][1], self.side["q"][2], self.side["e"][1], self.side["e"][2],
                  self.cls, self.sep)
        else:
            _fill(out, ei, qi, self.side["e"][1], self.side["e"][2], self.side["q"][1], self.side["q"][2],
                  self.cls, self.sep)
        return out


@njit(cache=True)
def _fill(out, ei, qi, ef, eo, qf, qo, cls, sep):
    for k in range(len(ei)):
        j = 0
        out[k, j] = cls
        j += 1
        for t in range(eo[ei[k]], eo[ei[k] + 1]):
            out[k, j] = ef[t]
            j += 1
        out[k, j] = sep
        out[k, j + 1] = sep
        j += 2
        for t in range(qo[qi[k]], qo[qi[k] + 1]):
            out[k, j] = qf[t]
            j += 1
        out[k, j] = sep


def _batches(lengths: np.ndarray, budget: int, max_rows: int, rng=None, chunk=64):
    """Length-sorted batches under a token budget.  With rng: shuffled order, sorted inside chunks
    of `chunk` full batches, batch order shuffled (training).  Without: globally sorted (inference)."""
    if rng is None:
        order = np.argsort(lengths, kind="stable")
        groups = [order]
    else:
        order = rng.permutation(len(lengths))
        size = chunk * max_rows
        groups = [g[np.argsort(lengths[g], kind="stable")] for g in np.array_split(order, max(1, len(order) // size))]
    out = []
    for g in groups:
        a = 0
        while a < len(g):
            b = a + 1
            mx = lengths[g[a]]
            while b < len(g) and b - a < max_rows:
                mx2 = max(mx, lengths[g[b]])
                if mx2 * (b - a + 1) > budget:
                    break
                mx = mx2
                b += 1
            out.append(g[a:b])
            a = b
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def _device():
    import torch
    assert torch.cuda.is_available(), "CUDA GPU required for the cross-encoder"
    free, total = torch.cuda.mem_get_info()
    # cap our share: WDDM drivers silently spill into system RAM instead of raising OOM
    frac = max(0.3, min(0.9, (free - 350 * 2 ** 20) / total))
    torch.cuda.set_per_process_memory_fraction(frac)
    log(f"GPU {torch.cuda.get_device_name(0)}: free {free / 2**30:.2f} / {total / 2**30:.2f} GB, cap {frac:.2f}")
    return torch.device("cuda")


def _load_model(path, device):
    import torch
    from transformers import AutoModelForSequenceClassification
    try:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, attn_implementation="sdpa")
    except (ValueError, ImportError):
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
    log("attention implementation:", getattr(m.config, "_attn_implementation", "?"))
    return m.to(device)


def step_train(tag: str):
    import torch
    torch.manual_seed(SEED)
    dev = _device()
    folds = FOLD_A if tag == "A" else FOLD_B
    d = pl.read_parquet(wpath("ce", f"train_{FIT}.parquet")).filter(pl.col("fold").is_in(list(folds)))
    T = Tokens("train")
    ei = T.index("e", d["e_row"].to_numpy())
    qi = T.index("q", d["q_row"].to_numpy())
    y = d["y"].to_numpy().astype(np.float32)
    lens = T.lengths(ei, qi)
    rng = np.random.default_rng(SEED + (tag == "B"))
    torch.manual_seed(SEED + (tag == "B"))
    batches = _batches(lens, TRAIN_TOKENS, MAX_ROWS, rng)
    n_steps = len(batches)
    log(f"CE {tag}: {len(y)} pairs ({y.mean():.3f} true), {n_steps} steps, mean length {lens.mean():.1f}")
    m = _load_model(MODEL_DIR, dev)
    m.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
    if VARIANT in ("b", "c", "l"):   # 278M / 560M model; on a small GPU also gradient checkpointing (ER_CE_GC=0 on a large GPU)
        m.base_model.embeddings.word_embeddings.to(torch.bfloat16)
        if os.environ.get("ER_CE_GC", "1") == "1":
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in m.parameters() if p.requires_grad]
    decay = [p for n, p in m.named_parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for n, p in m.named_parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.999), eps=1e-6, fused=True)
    warm = max(1, int(WARMUP * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    m.train()
    t0 = time.time()
    run = 0.0
    yt = torch.from_numpy(y)
    for s, b in enumerate(batches):
        x = torch.from_numpy(T.batch(ei[b], qi[b])).to(dev, non_blocking=True)
        att = (x != T.pad).long()
        yb = yt[b].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = m(input_ids=x, attention_mask=att).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * float(loss) if s else float(loss)
        if s % 250 == 0 or s == n_steps - 1:
            el = time.time() - t0
            log(f"  step {s}/{n_steps} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{el / 60:.1f} min, eta {el / (s + 1) * (n_steps - s - 1) / 60:.1f} min, "
                f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    out = wpath("models", f"ce_{tag}{VARIANT}")
    m.save_pretrained(str(out))
    log(f"saved {out} ({(time.time() - t0) / 60:.1f} min)")


def _score(model, T: Tokens, ei, qi, dev):
    import torch
    lens = T.lengths(ei, qi)
    out = np.zeros(len(ei), np.float32)
    bs = _batches(lens, INFER_TOKENS, 1024)
    t0 = time.time()
    with torch.inference_mode():
        for k, b in enumerate(bs):
            x = torch.from_numpy(T.batch(ei[b], qi[b])).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=x, attention_mask=(x != T.pad).long()).logits.squeeze(-1)
            out[b] = lg.float().cpu().numpy()
            if k % 500 == 0:
                log(f"    batch {k}/{len(bs)} {(time.time() - t0) / 60:.1f} min")
    return out


def step_score(which=("val", "test")):
    import torch
    dev = _device()
    models = {}
    for tag in ("A", "B"):
        models[tag] = _load_model(str(wpath("models", f"ce_{tag}{VARIANT}")), dev).eval()
    if "val" in which:
        d = pl.read_parquet(wpath("ce", "train_val.parquet"))
        T = Tokens("train")
        ei, qi = T.index("e", d["e_row"].to_numpy()), T.index("q", d["q_row"].to_numpy())
        f = d["fold"].to_numpy()
        la = np.full(len(f), np.nan, np.float32)
        lb = np.full(len(f), np.nan, np.float32)
        need_a = ~np.isin(f, FOLD_A)          # M_A may score everything it never trained on
        need_b = ~np.isin(f, FOLD_B)
        with Timer(f"score validation pairs: {need_a.sum()} by M_A, {need_b.sum()} by M_B"):
            la[need_a] = _score(models["A"], T, ei[need_a], qi[need_a], dev)
            lb[need_b] = _score(models["B"], T, ei[need_b], qi[need_b], dev)
        ce = np.where(np.isin(f, FOLD_A), lb, np.where(np.isin(f, FOLD_B), la, 0.5 * (la + lb)))
        np.save(wpath("ce", f"train_val_logit{VARIANT}.npy"), ce.astype(np.float32))
    if "test" in which:
        d = pl.read_parquet(wpath("ce", "test_band.parquet"))
        T = Tokens("test")
        ei, qi = T.index("e", d["e_row"].to_numpy()), T.index("q", d["q_row"].to_numpy())
        with Timer(f"score test band: {len(ei)} pairs x 2 models"):
            la = _score(models["A"], T, ei, qi, dev)
            lb = _score(models["B"], T, ei, qi, dev)
        np.save(wpath("ce", f"test_band_logit{VARIANT}.npy"), (0.5 * (la + lb)).astype(np.float32))
        np.save(wpath("ce", f"test_band_logit_ab{VARIANT}.npy"), np.stack([la, lb], 1).astype(np.float32))


if __name__ == "__main__":
    s = sys.argv[1]
    if s == "select":
        step_select()
    elif s == "select_fit":
        step_select_fit()
    elif s == "tokenize":
        step_tokenize()
    elif s == "train":
        step_train(sys.argv[2])
    elif s == "score":
        step_score(tuple(sys.argv[2:]) or ("val", "test"))
    else:
        raise SystemExit(f"unknown step {s}")
