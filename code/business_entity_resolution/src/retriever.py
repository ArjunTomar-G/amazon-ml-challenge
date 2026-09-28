"""Rescue retriever: contrastive fine-tuning of multilingual-e5-small as a bi-encoder (-> models/bienc).

    python retriever.py

Trained on train folds A/B only: true pairs of records with an address (US / India), the A/B blocking
misses oversampled x4, in-batch negatives + the record's most probable WRONG blocking candidate as hard
negative (InfoNCE, temperature 0.05, lr 2e-5, 1 epoch, gradient checkpointing for a 4 GB GPU).
Check on the validation fold's blocking misses with an address (never seen): recall@3 0.488 (untuned)
-> 0.816, recall@10 0.573 -> 0.889.  rescue.py uses it to retrieve the top-5 S1 entities."""
import sys, time
import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, ".")
from common import wpath, log
from blocking import true_pairs_rows, load_universe
from model import entity_folds, FOLD_A, FOLD_B
import crossenc as ce
import rescue as R

torch.manual_seed(7)
rng = np.random.default_rng(7)
keys = np.load(wpath("feat", "train_ctxkeys.npz"))
efold = keys["e_fold"]
truth = true_pairs_rows().select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))
qa = pl.concat([pl.read_parquet(wpath("norm", f"train_{s}.parquet"), columns=["a_all", "country"]) for s in ("source2", "source3")]).with_row_index("q_row").with_columns(pl.col("q_row").cast(pl.Int64))
has_addr = qa.filter((pl.col("a_all").fill_null("") != "") & pl.col("country").is_in(["US", "India"])).select("q_row")
pr = pl.read_parquet(wpath("feat", "train_pairs.parquet")).select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64), "y")
p1 = np.load(wpath("feat", "train_p1.npy"))
cand = pr.select("q_row", "e_row")
tr = truth.join(has_addr, on="q_row").filter(pl.Series(np.isin(efold[truth.join(has_addr, on="q_row")["e_row"].to_numpy()], FOLD_A + FOLD_B)))
miss = tr.join(cand, on=["q_row", "e_row"], how="anti")
hit = tr.join(cand, on=["q_row", "e_row"], how="semi")
# hard negative: the record's most probable wrong blocking candidate
neg = (pr.with_columns(pl.Series("p1", p1)).filter(pl.col("y") == 0).sort("p1", descending=True)
       .unique("q_row", keep="first").select("q_row", pl.col("e_row").alias("neg")))
sample = pl.concat([miss, miss, miss, miss, hit[rng.choice(hit.height, 300_000, replace=False)]])
sample = sample.join(neg, on="q_row", how="left")
# records without a wrong candidate: a random S1 of the same country as negative
nulls = sample["neg"].is_null().to_numpy()
ec = keys["e_country"]
e_by_c = {c: np.flatnonzero(ec == c) for c in ("US", "India")}
nr = sample["neg"].fill_null(-1).to_numpy().astype(np.int64)
er_np = sample["e_row"].to_numpy()
for i in np.flatnonzero(nulls):
    nr[i] = rng.choice(e_by_c[ec[er_np[i]]])
sample = sample.with_columns(pl.Series("neg", nr.astype(np.int64)))
sample = sample[rng.permutation(sample.height)]
log(f"pilot training pairs: {sample.height} ({miss.height} A/B blocking misses x4 + 300k retrieved true pairs)")

tok = AutoTokenizer.from_pretrained(ce.MODEL_DIR)
s1_txt = R._raw_texts("train", "source1")
q_txt = R._raw_texts("train", "source2") + R._raw_texts("train", "source3")


def enc_rows(texts, rows):
    flat, off = ce._tokenize(tok, [texts[i] for i in rows])
    return flat, off


dev = ce._device()
m = AutoModel.from_pretrained(ce.MODEL_DIR, attn_implementation="sdpa").to(dev)
m.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})   # 3 x 128 texts per step
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01, fused=True)
B = 128
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
    sim = q @ torch.cat([p, n]).T / 0.05
    loss = torch.nn.functional.cross_entropy(sim, torch.arange(B, device=dev))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if s % 500 == 0:
        el = time.time() - t0
        log(f"  step {s}/{n_steps} loss {float(loss):.4f} {el / 60:.1f} min eta {el / (s + 1) * (n_steps - s - 1) / 60:.1f} min peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
m.save_pretrained(str(wpath("models", "bienc")))
del opt
torch.cuda.empty_cache()
m.eval()

# gate: recall@K on the validation blocking misses with an address (unlinked queries), against ALL train S1
Vm = pl.read_parquet(wpath("feat", "train_pairs.parquet")).select(pl.col("q_row").cast(pl.Int64), pl.col("e_row").cast(pl.Int64))
tV = truth.filter(pl.col("e_row").is_in(np.flatnonzero(efold == 4)))
vm = tV.join(Vm, on=["q_row", "e_row"], how="anti").join(has_addr, on="q_row")
log(f"gate: {vm.height} validation blocking misses with an address (all records, as retriever2.py)")
ranks = []
with torch.inference_mode():
    for c in ("US", "India"):
        erows = e_by_c[c]
        ef, eo = enc_rows(s1_txt, erows)
        E = torch.zeros((len(erows), 384), dtype=torch.float16, device=dev)
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
        del E
        torch.cuda.empty_cache()
r = np.concatenate(ranks)
log(f"tuned retriever on {len(r)} validation blocking misses (untuned: @1 0.404 @3 0.488 @5 0.528 @10 0.573 @20 0.621):")
for k in (1, 3, 5, 10, 20):
    log(f"  recall@{k:<3d} {np.mean(r <= k):.3f}")
