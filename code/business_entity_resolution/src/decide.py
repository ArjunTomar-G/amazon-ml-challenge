"""Decision layer: from calibrated pair probabilities to per-entity match sets.

1. Exclusivity: every S2/S3 record belongs to at most one S1 entity, so a
   record can only be predicted for its highest-probability entity.
2. Expected-F0.5 maximisation (per S1 entity).  Given the entity's candidate
   probabilities p_1 >= p_2 >= ... (treated as independent) and a Poisson(lam)
   number of true matches that blocking missed, the expected score of
   predicting the top-k set is

      E[F(k)] = sum_a sum_b P(TP=a) P(FN=b) * 1.25 a / (0.25 (a + b) + k)
      E[F(0)] = P(no true match at all)          (empty prediction scores 1)

   with TP ~ PoissonBinomial(p_1..p_k), FN ~ PoissonBinomial(p_k+1..p_n) *
   Poisson(lam).  We pick argmax_k E[F(k)] - this handles singletons (predict
   empty when the entity probably has no match) and the precision-heavy F0.5
   trade-off exactly, instead of a single global threshold.
3. The official metric: macro-averaged F0.5 over all Source-1 entities.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from numba import njit, prange

BETA2 = 0.25
MAXN = 40


RAISE_GUARD = 0.1


def final_prob(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Final pair probability.  The context-based stage-2 model may lower any
    probability, but it may not RAISE a pair that the pairwise stage-1 model
    rejects (p1 < RAISE_GUARD).  Upward overrides of that kind are rare and
    only 80 % precise on validation, and on the test universe - where sibling
    businesses come as groups of records that "support" each other - they are
    35x more frequent; the guard costs 0.00005 F0.5 on validation."""
    return np.where(p1 < RAISE_GUARD, np.minimum(p1, p2), p2).astype(np.float32)


def exclusive(c: pl.DataFrame, pcol: str = "p") -> pl.DataFrame:
    """Zero the probability of every pair that is not its record's argmax."""
    return c.with_columns(
        pl.when(pl.col(pcol) >= pl.col(pcol).max().over("q_row"))
        .then(pl.col(pcol)).otherwise(0.0).alias(pcol))


@njit(cache=True)
def _poisson_binomial(p, out):
    n = p.shape[0]
    out[:] = 0.0
    out[0] = 1.0
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, 0, -1):
            out[j] = out[j] * (1.0 - pi) + out[j - 1] * pi
        out[0] *= (1.0 - pi)


@njit(parallel=True, cache=True)
def _expected_f_choose(ptr, probs, lam, fn_max):
    """probs sorted descending inside every group.  Returns k* per group and its E[F]."""
    G = ptr.shape[0] - 1
    kbest = np.zeros(G, np.int32)
    fbest = np.zeros(G, np.float64)
    # Poisson pmf for missed matches
    pois = np.zeros(fn_max + 1)
    for b in range(fn_max + 1):
        pois[b] = math.exp(-lam) * lam ** b / math.gamma(b + 1)
    for g in prange(G):
        a0 = ptr[g]
        n = min(ptr[g + 1] - a0, MAXN)
        p = probs[a0:a0 + n]
        tp = np.zeros(n + 2)
        rest = np.zeros(n + 2)
        fn = np.zeros(n + fn_max + 2)
        # k = 0
        pz = 1.0
        for i in range(n):
            pz *= (1.0 - p[i])
        best_f = pz * pois[0]
        best_k = 0
        for k in range(1, n + 1):
            _poisson_binomial(p[:k], tp)
            _poisson_binomial(p[k:], rest)
            # convolve rest with poisson
            m = n - k
            for x in range(m + fn_max + 1):
                fn[x] = 0.0
            for x in range(m + 1):
                if rest[x] == 0.0:
                    continue
                for y in range(fn_max + 1):
                    fn[x + y] += rest[x] * pois[y]
            ef = 0.0
            for a in range(1, k + 1):
                if tp[a] == 0.0:
                    continue
                s = 0.0
                for b in range(m + fn_max + 1):
                    if fn[b] == 0.0:
                        continue
                    s += fn[b] * (1.0 + BETA2) * a / (BETA2 * (a + b) + k)
                ef += tp[a] * s
            if ef > best_f:
                best_f = ef
                best_k = k
        kbest[g] = best_k
        fbest[g] = best_f
    return kbest, fbest


def expected_f_select(c: pl.DataFrame, pcol: str = "p", lam: float = 0.03,
                      min_p: float = 1e-3) -> pl.DataFrame:
    """c: e_row, q_row, pcol (already exclusive).  Returns selected (e_row, q_row)."""
    d = (c.filter(pl.col(pcol) >= min_p)
         .sort(["e_row", pcol], descending=[False, True]))
    if d.height == 0:
        return d.select("e_row", "q_row")
    e = d["e_row"].to_numpy()
    starts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    ptr = np.r_[starts, len(e)].astype(np.int64)
    kb, fb = _expected_f_choose(ptr, d[pcol].to_numpy().astype(np.float64), lam, 6)
    rank = np.arange(len(e)) - np.repeat(starts, np.diff(ptr))
    keep = rank < np.repeat(kb, np.diff(ptr))
    return d.filter(pl.Series(keep)).select("e_row", "q_row")


def threshold_select(c: pl.DataFrame, t: float, pcol: str = "p") -> pl.DataFrame:
    return c.filter(pl.col(pcol) >= t).select("e_row", "q_row")


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, entities: pl.DataFrame) -> float:
    """pred/truth: (e_row, q_row); entities: e_row of every S1 entity evaluated."""
    ents = entities.select(pl.col("e_row").cast(pl.Int64)).unique()
    pred = pred.select(pl.col("e_row").cast(pl.Int64), pl.col("q_row").cast(pl.Int64)).join(ents, on="e_row")
    truth = truth.select(pl.col("e_row").cast(pl.Int64), pl.col("q_row").cast(pl.Int64)).join(ents, on="e_row")
    tp = pred.join(truth, on=["e_row", "q_row"]).group_by("e_row").len().rename({"len": "tp"})
    npred = pred.group_by("e_row").len().rename({"len": "np"})
    ntrue = truth.group_by("e_row").len().rename({"len": "nt"})
    d = (ents.join(tp, on="e_row", how="left").join(npred, on="e_row", how="left")
         .join(ntrue, on="e_row", how="left").fill_null(0))
    d = d.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
        .when((pl.col("tp") == 0)).then(0.0)
        .otherwise((1 + BETA2) * pl.col("tp") / (BETA2 * pl.col("nt") + pl.col("np")))
        .alias("f"))
    return float(d["f"].mean())


# ----------------------------------------------------------------------------
# label-shift (prior) correction
# ----------------------------------------------------------------------------
def adjust_prior(p: np.ndarray, pi_train: float, pi_new: float) -> np.ndarray:
    """Bayes-rule posterior correction for a change of class prior."""
    a = (pi_new / pi_train) * p
    b = ((1.0 - pi_new) / (1.0 - pi_train)) * (1.0 - p)
    return a / np.maximum(a + b, 1e-12)


def em_prior(p: np.ndarray, pi_train: float, iters: int = 200, tol: float = 1e-7):
    """Saerens-Latinne-Decaestecker EM estimate of the positive prior of a new
    sample, given posteriors p calibrated under prior pi_train."""
    pi = pi_train
    for _ in range(iters):
        w = adjust_prior(p, pi_train, pi)
        new = float(w.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi


# ----------------------------------------------------------------------------
# label-free prior anchoring per country
# ----------------------------------------------------------------------------
def anchor_kappa(p: np.ndarray, n_entities: int, target: float) -> float:
    """Odds multiplier k such that sum(adjusted p) / n_entities == target, where p are the
    probabilities of a country's exclusive (record-argmax) candidate pairs.

    The data generator is symmetric across countries (train US and India have the same
    3.46 true matches per entity; test US and India reproduce the density-augmented
    validation value), so the expected number of matches per entity is known for an
    unseen country too.  If the model's probabilities imply more (or fewer) matches
    than that, they are over- (under-) confident there and every pair's odds are
    scaled by the same factor (a Bayes prior-shift correction)."""
    p = p.astype(np.float64)
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = (lo * hi) ** 0.5
        if adjust_prior(p, 0.5, mid / (1.0 + mid)).sum() / n_entities > target:
            hi = mid
        else:
            lo = mid
    return float((lo * hi) ** 0.5)


def apply_kappa(p: np.ndarray, kappa: float) -> np.ndarray:
    return adjust_prior(p.astype(np.float64), 0.5, kappa / (1.0 + kappa)).astype(np.float32)
