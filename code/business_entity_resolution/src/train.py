"""Train the matcher on the training split.

1. normalise the training TSVs (cached in the work dir);
2. learn the Indic->Latin token dictionary from the training ground truth;
3. draw disjoint train / holdout samples of Source-1 entities, generate their
   candidates against the FULL training corpus and compute pair features;
4. fit LightGBM (early stopping on the holdout), pick the probability threshold
   that maximises macro F0.5 on the holdout, and save model + config.

    python src/train.py [--n-train 150000] [--n-holdout 50000] [--threads 8]
"""

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber import data, metrics, pipeline, retrieval, translit  # noqa: E402


def main():
    d = data.default_dirs()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=d["data"])
    ap.add_argument("--work-dir", default=d["work"])
    ap.add_argument("--n-train", type=int, default=150_000, help="Source-1 entities used for training")
    ap.add_argument("--n-holdout", type=int, default=50_000, help="Source-1 entities for early stopping / threshold")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--chunk", type=int, default=20_000, help="Source-1 entities per feature chunk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=2000, help="max boosting rounds (early stopping on the holdout)")
    ap.add_argument("--lr", type=float, default=0.05, help="LightGBM learning rate")
    for k, v in retrieval.DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in retrieval.DEFAULTS}
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731

    data.prepare_split(args.data_dir, args.work_dir, "train", log=log)

    # ground truth + learned transliteration dictionary
    pair_s1, pair_src, pair_id = data.load_ground_truth(args.data_dir)
    tpath = os.path.join(args.work_dir, "translit.json")
    if not os.path.exists(tpath):
        s1 = pq.read_table(data.prepared_path(args.work_dir, "train", 1), columns=["id", "name_n"])
        corpus, csrc = data.load_corpus(args.work_dir, "train", ["id", "name_n", "indic"])
        rec = data.global_index(csrc, corpus.column("id").to_numpy(), pair_src, pair_id)
        mapping, info = translit.learn(
            s1.column("id").to_numpy(), s1.column("name_n").combine_chunks(),
            corpus.column("name_n"), corpus.column("indic").to_numpy(zero_copy_only=False),
            pair_s1, rec)
        translit.save(mapping, tpath)
        log(f"transliteration dictionary: {info}")
        del s1, corpus, csrc, rec
    mapper = translit.Mapper(translit.load(tpath))

    # entity samples
    all_ids = pq.read_table(data.prepared_path(args.work_dir, "train", 1), columns=["id"]).column("id").to_numpy()
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(all_ids), size=min(len(all_ids), args.n_train + args.n_holdout), replace=False)
    train_ids = np.sort(all_ids[pick[:args.n_train]])
    hold_ids = np.sort(all_ids[pick[args.n_train:]])
    sample_ids = np.sort(np.concatenate([train_ids, hold_ids]))
    gt_count_by_id = dict(zip(*np.unique(pair_s1, return_counts=True)))

    Xs, ys, qs, is_hold, gts = [], [], [], [], []
    names = None
    q_offset = 0
    for country in pipeline.countries(args.work_dir, "train"):
        log(f"country {country}")
        qtab, ctab, csrc = pipeline.load_country(args.work_dir, "train", country, sample_ids)
        if qtab.num_rows == 0:
            continue
        Q, C = pipeline.arrays(qtab, mapper), pipeline.arrays(ctab, mapper)
        q_ids = qtab.column("id").to_numpy()
        # labels: owner (local query index) of every corpus record in the ground truth
        in_q = np.isin(pair_s1, q_ids)
        rec = data.global_index(csrc, ctab.column("id").to_numpy(), pair_src[in_q], pair_id[in_q])
        q_order = np.argsort(q_ids)
        q_pos = q_order[np.searchsorted(q_ids[q_order], pair_s1[in_q])]
        owner = np.full(ctab.num_rows, -1, np.int64)
        ok = rec >= 0
        owner[rec[ok]] = q_pos[ok]
        q_gt = np.array([gt_count_by_id.get(i, 0) for i in q_ids.tolist()], np.int64)
        q_hold = np.isin(q_ids, hold_ids)

        def on_chunk(P, X, nm):
            nonlocal names
            names = nm
            Xs.append(X)
            ys.append(owner[P["c"]] == P["q"])
            qs.append(P["q"].astype(np.int64) + q_offset)
            is_hold.append(q_hold[P["q"]])

        pipeline.run_country(Q, C, cfg, args.threads, args.chunk, on_chunk, log)
        gts.append((q_gt, q_hold))
        q_offset += len(q_ids)
        del Q, C, qtab, ctab

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    q = np.concatenate(qs)
    hold = np.concatenate(is_hold)
    gt_count = np.concatenate([g for g, _ in gts])
    q_is_hold = np.concatenate([h for _, h in gts])
    del Xs
    log(f"pairs: {len(y):,} ({y.mean():.3%} positive), features: {len(names)}")

    # blocking quality on the holdout
    hold_q = np.flatnonzero(q_is_hold)
    found = np.bincount(q[hold & y], minlength=len(gt_count))
    rec_pairs = found[hold_q].sum() / max(1, gt_count[hold_q].sum())
    ceiling = metrics.macro_f05(q, y, y & hold, gt_count)[hold_q].mean()
    log(f"holdout blocking: pair recall {rec_pairs:.4f}, F0.5 ceiling {ceiling:.4f}, "
        f"candidates per entity {hold.sum() / len(hold_q):.1f}")

    params = dict(objective="binary", learning_rate=args.lr, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1, seed=args.seed)
    dtr = lgb.Dataset(X[~hold], y[~hold], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[hold], y[hold], reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    prob = booster.predict(X[hold], num_iteration=booster.best_iteration)
    qh = q[hold]
    f05, thr = metrics.best_threshold(qh, y[hold], prob, gt_count)
    f_all = metrics.macro_f05(qh, y[hold], prob >= thr, gt_count)[hold_q]
    single = gt_count[hold_q] == 0
    log(f"holdout macro F0.5 = {f_all.mean():.4f} at threshold {thr:.2f} "
        f"(singletons {f_all[single].mean():.4f}, matched {f_all[~single].mean():.4f}); "
        f"best iteration {booster.best_iteration}")

    os.makedirs(args.work_dir, exist_ok=True)
    booster.save_model(os.path.join(args.work_dir, "model.txt"), num_iteration=booster.best_iteration)
    imp = sorted(zip(names, booster.feature_importance("gain").tolist()), key=lambda x: -x[1])
    config = {"threshold": thr, "features": names, "retrieval": cfg,
              "holdout": {"macro_f05": float(f_all.mean()), "pair_recall": float(rec_pairs),
                          "f05_ceiling": float(ceiling), "n_entities": int(len(hold_q))},
              "n_train_entities": int(args.n_train), "best_iteration": int(booster.best_iteration),
              "top_features": imp[:15]}
    with open(os.path.join(args.work_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    log(f"saved {os.path.join(args.work_dir, 'model.txt')} and model_config.json")


if __name__ == "__main__":
    sys.exit(main())
