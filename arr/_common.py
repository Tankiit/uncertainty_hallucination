"""Shared statistics helpers for the ARR disagreement analyses.

Everything here is deliberately small and dependency-light (numpy + scipy).
No Modal decorators: these run on whatever host already has the per-item
score dump. See PATCHES_analyse_hf.md for how that dump is produced.

Conventions used across this package, all deliberate:

  * SIGNED Spearman by default. The sign is a finding in this paper, and
    analyse_hf.pooled_gain_ci's use of abs() discards it. Pass signed=False
    only where magnitude genuinely is the question.
  * Bootstrap resamples ITEMS, never signals or seeds. Paired comparisons
    recompute both sides on the same resample.
  * Never average across tau. size_only is exactly tau-invariant, so a
    tau-averaged quantity compared against it compares different kinds of
    thing (RESULTS_ALL.md defect 6).
"""
import numpy as np
from scipy.stats import spearmanr, rankdata


def rho(x, y, signed=True):
    """Spearman correlation, NaN-safe, signed by default."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    r = spearmanr(x[ok], y[ok])[0]
    return float(r if signed else abs(r))


def partial_rho(x, y, z):
    """Partial Spearman rho(x, y | z).

    Rank-transform all three, then take the Pearson partial correlation on
    ranks. This is the standard construction and it is tie-tolerant, which
    matters here: size_only is discrete and heavily tied, and annotator
    entropy has a large mass at exactly zero on the binary dataset.
    """
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 30:
        return np.nan
    rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))

    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta

    ex, ey = resid(rx, rz), resid(ry, rz)
    denom = np.linalg.norm(ex) * np.linalg.norm(ey)
    return float(ex @ ey / denom) if denom > 0 else np.nan


def item_bootstrap(fn, n, B=2000, seed=0):
    """Bootstrap `fn(idx) -> float` over item indices.

    Returns (lo, hi, p_two_sided). The p-value is the standard
    bootstrap-percentile inversion: twice the smaller tail mass about zero.

    NOTE: ARR_run_summary.md section 8 correctly refused to infer a p-value
    from whether a CI crossed zero. This is a real bootstrap p, not that.
    It is still an approximation; if a reviewer presses, a permutation test
    on the target is the defensible fallback.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        out[b] = fn(rng.integers(0, n, n))
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    frac_pos = float(np.nanmean(out > 0))
    p = 2.0 * min(frac_pos, 1.0 - frac_pos)
    return float(lo), float(hi), float(min(p, 1.0))


def benjamini_hochberg(pvals, alpha=0.05):
    """BH step-up. Returns (reject_mask, qvalues).

    Apply across the rows you pre-registered as informative, not across
    everything you happened to compute. The ARR panel excludes uninformative
    rows (max-mass > 0.90 or r(W, size_only) > 0.98) before correcting --
    keep doing that, and record which rows were excluded.
    """
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    q = np.full_like(p, np.nan)
    if m == 0:
        return np.zeros_like(p, bool), q
    idx = np.where(ok)[0][np.argsort(p[ok])]
    ranked = p[idx] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[idx] = np.clip(ranked, 0, 1)
    return (q <= alpha) & ok, q


def load_scores(path):
    """Load a per-item score dump written by dump_scores (see PATCHES).

    Expected columns: item_id, signal, value, plus one row per item for
    the target under signal == 'annot_entropy'.

    TODO: settle whether the dump is long or wide before wiring the callers.
          Long is easier to extend with new signals; wide is easier to join.
          Pick one and make every module in this package agree.
    """
    import pandas as pd
    df = pd.read_parquet(path)
    raise NotImplementedError(
        "Reshape to (n_items, n_signals) + target once the dump format is "
        "fixed. Do not guess -- inspect one file first."
    )
