"""Rescue retriever v2: contrastive fine-tuning of multilingual-e5 as a bi-encoder, for records WITH and
WITHOUT an address (-> models/bienc2).

    python retriever2.py [small|base]

Differences to retriever.py (v10c, records with an address only):
  * positives also include records without an address (78 % of all blocking misses are such records;
    for 20 % of them the true entity's name is unique in the country, i.e. resolvable by name alone)
  * hard negative = the record's most probable WRONG blocking candidate by the pruner probability
    (available right after blocking, so the retriever does not wait for stage 1)
Trained on train folds A/B only (US / India).  Gate: recall@K on the validation fold's blocking misses
(never seen), with and without an address, against all train S1 of the country.
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from common import log, wpath
from blocking import true_pairs_rows, load_universe
from model import entity_folds, FOLD_A, FOLD_B, VAL_FOLD
import crossenc as ce
import rescue as R

SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
BASE = str(wpath("hf", f"multilingual-e5-{SIZE}"))
OUT = "bienc2" if SIZE == "small" else f"bienc2_{SIZE}"
B = 128
LR = 2e-5 if SIZE == "small" else 1.5e-5
TAU = 0.05
torch.manual_seed(11)
rng = np.random.default_rng(11)

s1, qall = load_universe("train", ["entity_id", "country", "a_all"])
efold = entity_folds(s1["entity_id"])
ec = s1["country"].to_numpy().astype("U")
qc = qall["country"].to_numpy().astype("U")
q_has_addr = (qall["a_all"].fill_null("") != "").to_numpy()
truth = true_pairs_rows().select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))
truth = truth.filter(pl.Series(np.isin(ec[truth["e_row"].to_numpy()], ["US", "India"])))
cand = pl.read_parquet(wpath("cand", "train_cands.parquet"), columns=["q_row", "e_row", "p_prune"]).with_columns(
    pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))
tr = truth.filter(pl.Series(np.isin(efold[truth["e_row"].to_numpy()], FOLD_A + FOLD_B)))
miss = tr.join(cand, on=["q_row", "e_row"], how="anti")
hit = tr.join(cand, on=["q_row", "e_row"], how="semi")
ha = q_has_addr[miss["q_row"].to_numpy()]
hh = q_has_addr[hit["q_row"].to_numpy()]
log(f"A/B true pairs {tr.height}: blocking misses {miss.height} ({ha.sum()} with address, {(~ha).sum()} without)")
hit_a = hit.filter(pl.Series(hh))
hit_n = hit.filter(pl.Series(~hh))
sample = pl.concat([miss] * 4 + [hit_a[rng.choice(hit_a.height, 300_000, replace=False)],
                                 hit_n[rng.choice(hit_n.height, min(hit_n.height, 120_000), replace=False)]])
# hard negative: the record's most probable wrong blocking candidate (pruner probability)
tp_flag = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))
neg = (cand.join(tp_flag, on=["q_row", "e_row"], how="left").filter(pl.col("y").is_null())
       .sort("p_prune", descending=True).unique("q_row", keep="first").select("q_row", pl.col("e_row").alias("neg")))
sample = sample.join(neg, on="q_row", how="left")
nr = sample["neg"].fill_null(-1).to_numpy().astype(np.int64)
er_np = sample["e_row"].to_numpy()
e_by_c = {c: np.flatnonzero(ec == c) for c in ("US", "India")}
for i in np.flatnonzero(nr < 0):
    nr[i] = rng.choice(e_by_c[ec[er_np[i]]])
sample = sample.with_columns(pl.Series("neg", nr))
sample = sample[rng.permutation(sample.height)]
log(f"training pairs: {sample.height}")

tok = AutoTokenizer.from_pretrained(BASE)
s1_txt = R._raw_texts("train", "source1")
q_txt = R._raw_texts("train", "source2") + R._raw_texts("train", "source3")


def enc_rows(texts, rows):
    return ce._tokenize(tok, [texts[i] for i in rows])


dev = ce._device()
m = AutoModel.from_pretrained(BASE, attn_implementation="sdpa").to(dev)
m.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
if SIZE != "small":
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01, fused=True)
n_steps = sample.height // B
warm = int(0.05 * n_steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (n_steps - s) / (n_steps - warm)))
pad, cls, sep = tok.pad_token_id, tok.cls_token_id, tok.sep_token_id


def embed_batch(flat, off, idx):
    lens = np.diff(off)[idx] + 2
    x = np.full((len(idx), int(lens.max())), pad, np.int64)
    R._fill1(x, idx.astype(np.int64), flat, off, cls, sep)
    x = torch.from_numpy(x).to(dev)
    att = x != pad
    h = m(input_ids=x, attention_mask=att.long()).last_hidden_state
    h = (h * att.unsqueeze(-1)).sum(1) / att.sum(1, keepdim=True)
    return torch.nn.functional.normalize(h.float(), dim=-1)


qf, qo = enc_rows(q_txt, sample["q_row"].to_numpy())
pf_, po = enc_rows(s1_txt, sample["e_row"].to_numpy())
nf, no = enc_rows(s1_txt, sample["neg"].to_numpy())
m.train()
t0 = time.time()
for s in range(n_steps):
    idx = np.arange(s * B, (s + 1) * B)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = embed_batch(qf, qo, idx)
        p = embed_batch(pf_, po, idx)
        n = embed_batch(nf, no, idx)
    sim = q @ torch.cat([p, n]).T / TAU
    loss = torch.nn.functional.cross_entropy(sim, torch.arange(B, device=dev))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if s % 500 == 0:
        el = time.time() - t0
        log(f"  step {s}/{n_steps} loss {float(loss):.4f} {el / 60:.1f} min eta {el / (s + 1) * (n_steps - s - 1) / 60:.1f} min "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
m.save_pretrained(str(wpath("models", OUT)))
tok.save_pretrained(str(wpath("models", OUT)))
del opt
torch.cuda.empty_cache()
m.eval()

# gate: recall@K on the validation fold's blocking misses against ALL train S1 of the country
tV = truth.filter(pl.Series(efold[truth["e_row"].to_numpy()] == VAL_FOLD))
vm = tV.join(cand, on=["q_row", "e_row"], how="anti")
log(f"gate: {vm.height} validation blocking misses ({q_has_addr[vm['q_row'].to_numpy()].sum()} with address)")
ranks, has = [], []
with torch.inference_mode():
    for c in ("US", "India"):
        erows = e_by_c[c]
        ef, eo = enc_rows(s1_txt, erows)
        E = torch.zeros((len(erows), m.config.hidden_size), dtype=torch.float16, device=dev)
        lens = np.diff(eo) + 2
        for b in ce._batches(lens, 64 * 1024, 2048):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E[torch.from_numpy(b).to(dev)] = embed_batch(ef, eo, b).half()
        sub = vm.filter(pl.Series(ec[vm["e_row"].to_numpy()] == c))
        sf, so = enc_rows(q_txt, sub["q_row"].to_numpy())
        tpos = torch.tensor(np.searchsorted(erows, sub["e_row"].to_numpy()), device=dev)
        for a in range(0, sub.height, 64):
            idx = np.arange(a, min(a + 64, sub.height))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Q = embed_batch(sf, so, idx).half()
            sims = Q @ E.T
            tsim = sims.gather(1, tpos[a:a + 64][:, None])
            ranks.append(((sims > tsim).sum(1) + 1).cpu().numpy())
        has.append(q_has_addr[sub["q_row"].to_numpy()])
        del E
        torch.cuda.empty_cache()
r = np.concatenate(ranks)
h = np.concatenate(has)
for nm_, msk in (("all", np.ones(len(r), bool)), ("with address", h), ("without address", ~h)):
    log(f"{OUT} recall on {msk.sum()} validation blocking misses ({nm_}): " +
        "  ".join(f"@{k} {np.mean(r[msk] <= k):.3f}" for k in (1, 3, 5, 10, 20)))
