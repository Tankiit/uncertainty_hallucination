"""Selective-prediction screening for the representation-space deferral task.

RECONSTRUCTED. The original screen.py was missing from the repo; this restores
the five names run_repspace_deferral.py imports, to their definitions used
elsewhere in the project (STC = rsuq.diagnostics.redundancy_screen; AURC =
selective risk; the gap is AURC_conf - AURC_conf+W with a positive value meaning
W helps). Contracts match the call sites in run_repspace_deferral.run_cell.

  stc(W, y)                                   -> point-biserial(y, W>median)
  aurc(unc, y)                                -> area under the risk-coverage curve
  compose_aurc_fixed(sig_tr, y_tr, sig_te, y_te) -> (aurc_on_test, unc_test|None)
  aurc_gap_bootstrap(unc_a, unc_b, y, B, seed)   -> (mean, lo, hi) of aurc(a)-aurc(b)
  verdict_from_ci(mean, lo, hi)               -> "HELPS"/"HURTS"/"TIE" string
"""
from __future__ import annotations
import numpy as np


def stc(W, y) -> float:
    """Signal-threshold correlation: point-biserial of the error label against
    the above-median indicator of the signal. Positive = higher W on errors."""
    from scipy.stats import pointbiserialr
    W = np.asarray(W, float); y = np.asarray(y, float)
    med = np.median(W)
    ind = (W > med).astype(float)
    if len(np.unique(ind)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(pointbiserialr(y, ind)[0])


def aurc(unc, y) -> float:
    """Area under the risk-coverage curve. `unc` is an uncertainty score (high =
    defer). Order most-confident first, accumulate error rate, average over
    coverage. Lower is better; 0.5 ~ chance on a balanced contrastive split."""
    unc = np.asarray(unc, float); y = np.asarray(y, float)
    ok = np.isfinite(unc) & np.isfinite(y)
    unc, y = unc[ok], y[ok]
    if len(y) == 0:
        return float("nan")
    order = np.argsort(unc, kind="mergesort")          # ascending uncertainty
    ys = y[order]
    risk = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(risk.mean())


def _standardize(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def compose_aurc_fixed(sig_tr, y_tr, sig_te, y_te):
    """Combine one or more signals into an uncertainty score and score its AURC
    on the test split. `sig_tr`/`sig_te` are lists of per-item arrays.

    A single signal is used directly (monotone, no fit). Multiple signals are
    combined by a logistic model of the error label fit on train and applied to
    test. If the train split is single-class the model is undefined -> returns
    (aurc_from_first_signal, None); the None flags run_cell's NULL branch so no
    gap is bootstrapped.

    Returns (aurc_on_test, uncertainty_on_test | None).
    """
    Xtr = np.column_stack([np.asarray(s, float) for s in sig_tr])
    Xte = np.column_stack([np.asarray(s, float) for s in sig_te])
    y_tr = np.asarray(y_tr, int); y_te = np.asarray(y_te, int)
    if Xtr.shape[1] == 1:
        return aurc(Xte[:, 0], y_te), Xte[:, 0]
    if len(np.unique(y_tr)) < 2:
        return aurc(Xte[:, 0], y_te), None
    from sklearn.linear_model import LogisticRegression
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-12] = 1.0
    clf = LogisticRegression(max_iter=500).fit((Xtr - mu) / sd, y_tr)
    unc = clf.predict_proba((Xte - mu) / sd)[:, 1]     # P(error) on test
    return aurc(unc, y_te), unc


def aurc_gap_bootstrap(unc_a, unc_b, y, B: int = 2000, seed: int = 0):
    """Item bootstrap of the AURC gap = aurc(unc_a) - aurc(unc_b). Positive means
    b (the richer signal set) achieves lower risk, i.e. it HELPS. Returns
    (mean, 2.5pct, 97.5pct)."""
    unc_a = np.asarray(unc_a, float); unc_b = np.asarray(unc_b, float)
    y = np.asarray(y, int); n = len(y)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        out[b] = aurc(unc_a[i], y[i]) - aurc(unc_b[i], y[i])
    obs = aurc(unc_a, y) - aurc(unc_b, y)
    return float(obs), float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def verdict_from_ci(mean, lo, hi) -> str:
    """HELPS if the whole gap CI is above 0, HURTS if below, else TIE.
    (verdict_aggregate keys off the 'HELPS' prefix.)"""
    if lo is None or hi is None or not np.isfinite(mean):
        return "NULL (no CI)"
    if lo > 0:
        return f"HELPS (gap CI [{lo:+.4f},{hi:+.4f}] > 0)"
    if hi < 0:
        return f"HURTS (gap CI [{lo:+.4f},{hi:+.4f}] < 0)"
    return f"TIE (gap CI [{lo:+.4f},{hi:+.4f}] spans 0)"
