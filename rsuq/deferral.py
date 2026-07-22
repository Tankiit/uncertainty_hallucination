
from __future__ import annotations
import numpy as np


def risk_coverage(uncertainty, error, n_points: int = 50):
    u = np.asarray(uncertainty, float)
    e = np.asarray(error, float)
    order = np.argsort(u)                                                           
    e_sorted = e[order]
    n = len(e_sorted)
    covs = np.linspace(1.0 / n_points, 1.0, n_points)
    risks = np.array([e_sorted[:max(1, int(c * n))].mean() for c in covs])
    return covs, risks, float(np.trapezoid(risks, covs))


def _aurc(uncertainty, error, n_points=50):
    return risk_coverage(uncertainty, error, n_points)[2]


def compose_aurc(signals: list, error, seed: int = 0, n_points: int = 50):
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([np.asarray(s, float) for s in signals])
    err = np.asarray(error, int)
    if len(set(err.tolist())) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(err))
    tr, te = idx[:len(err) // 2], idx[len(err) // 2:]
    lr = LogisticRegression(max_iter=500).fit(X[tr], err[tr])
    score = lr.predict_proba(X[te])[:, 1]
    return _aurc(score, err[te], n_points)


def aurc_delta_ci(unc_a, unc_b, error, B: int = 2000, seed: int = 0):
    a, b, e = map(lambda v: np.asarray(v, float), (unc_a, unc_b, error))
    if len(set(e.astype(int).tolist())) < 2:
        return {"delta": None, "significant": False, "note": "degenerate"}
    rng = np.random.default_rng(seed)
    base = _aurc(a, e) - _aurc(b, e)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(e), len(e))
        if len(set(e[i].astype(int).tolist())) < 2:
            continue
        d.append(_aurc(a[i], e[i]) - _aurc(b[i], e[i]))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"delta": float(base), "ci": [float(lo), float(hi)],
            "b_better": bool(lo > 0)}


def deferral_battery(conf, W_fixed, W_ctx, error, seed: int = 0) -> dict:
    conf = np.asarray(conf, float)
    u_conf = -conf                                            
    out = {
        "n": len(conf),
        "aurc_confidence": _aurc(u_conf, error),
        "aurc_W_fixed": _aurc(W_fixed, error),
        "aurc_W_ctx": _aurc(W_ctx, error),
        "compose_conf": compose_aurc([u_conf], error, seed),
        "compose_conf_W_ctx": compose_aurc([u_conf, W_ctx], error, seed),
        "compose_conf_W_fixed": compose_aurc([u_conf, W_fixed], error, seed),
                                                                         
        "delta_Wfixed_vs_Wctx": aurc_delta_ci(W_fixed, W_ctx, error, seed=seed),
    }
    cc = out["compose_conf"]
    out["ctx_adds_to_conf"] = (out["compose_conf_W_ctx"] is not None and cc is not None
                               and out["compose_conf_W_ctx"] < cc)
    out["fixed_adds_to_conf"] = (out["compose_conf_W_fixed"] is not None and cc is not None
                                 and out["compose_conf_W_fixed"] < cc - 1e-4)
    out["deferral_claim"] = (
        "PASS: ctx-width composes with confidence; fixed-width does not "
        "(positive mirror of the deferral-negative result)"
        if out["ctx_adds_to_conf"] and not out["fixed_adds_to_conf"]
        else "INSPECT: composition pattern not as predicted")
    return out
