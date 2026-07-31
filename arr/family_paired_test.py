"""Does the representation-vs-output family split survive inference?

Currently the split is selection-on-max with no inferential machinery: each
cell's "best metric" is an argmax over nine signals, and comparing
best-of-family against best-of-family per cell inflates both sides by an
unknown amount.

Family membership is NOT chosen here. It is a collapse of the GROUP dict in
analyse_hf.py, which was fixed before any result on these datasets was
inspected. Cite that dict as the paper's pre-registration; do not re-derive
membership, and do not resolve an ambiguous signal by looking at its rho.
"""
import numpy as np
from _common import rho, item_bootstrap, benjamini_hochberg

# Collapsed from analyse_hf.py GROUP: mass + density -> REPR, token -> OUTPUT.
REPR = {"credal_width", "mass_entropy", "one_minus_max",
        "size_only", "knn_dist"}
OUTPUT = {"pred_entropy", "pred_maxprob", "pred_margin", "neg_mean_logprob"}
EXCLUDED = set()   # a signal reading BOTH goes here, not into a family.

# --- PRE-REGISTRATION. Freeze BEFORE running. ----------------------------
STAT = "mean"          # FROZEN 2026-07-31. mean primary: no selection
                       #  problem. max is the robustness row (it matches the
                       #  existing 'best metric' column but IS the selection
                       #  problem, needing a max-statistic null to be legit).
                       #  If mean and max disagree, the split is a selection
                       #  artefact. REPR has five members, OUTPUT four -- mean
                       #  is unbalanced, so watch whether that bites.
MIN_CELLS = 5          # FROZEN 2026-07-31. 5 of 9 must clear BH for the
                       #       split to be a claim.
# -------------------------------------------------------------------------


def family_stat(scores, ent, family, stat):
    vals = [abs(rho(scores[s], ent)) for s in family if s in scores]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals) if stat == "mean" else np.max(vals))


def paired_family_gain(scores, ent, B=2000, seed=0):
    """Delta = family_stat(REPR) - family_stat(OUTPUT) for one cell.

    Resamples ITEMS, not signals. Both families are recomputed on the same
    resample -- that is what makes it paired, and what makes the interval
    mean anything.

    NOTE this uses |rho| deliberately: here the question is which family is
    more predictive, not which direction it points. That is the one place in
    this package where abs() is correct. Everywhere else, keep the sign.
    """
    assert STAT is not None, "freeze STAT first"

    def delta(idx=None):
        sub = ({k: v[idx] for k, v in scores.items()} if idx is not None
               else scores)
        e = ent[idx] if idx is not None else ent
        return (family_stat(sub, e, REPR, STAT)
                - family_stat(sub, e, OUTPUT, STAT))

    obs = delta()
    lo, hi, p = item_bootstrap(delta, len(ent), B=B, seed=seed)
    return dict(delta=obs, ci_lo=lo, ci_hi=hi, p=p)


def panel(cells, alpha=0.05):
    """Nine cells -> nine deltas -> BH across the nine.

    DECISION RULE, pre-registered:
      split SURVIVES if sign(delta) is constant within dataset, differs
      across dataset, and at least MIN_CELLS clear BH.
      Otherwise: the split is reported as an observation in Sec. Robustness
      and CUT from the abstract. Do not soften it -- cut it.

    Note this test says nothing about the fold confound. Even if the split
    clears here it is still confounded with target support size until
    fold_check.py clears too. Both gates, not either.
    """
    assert MIN_CELLS is not None, "freeze MIN_CELLS first"
    rows = []
    for name, (scores, ent) in cells.items():
        r = paired_family_gain(scores, ent)
        r["cell"] = name
        rows.append(r)
    rej, q = benjamini_hochberg([r["p"] for r in rows], alpha=alpha)
    for r, rj, qq in zip(rows, rej, q):
        r["reject"], r["q"] = bool(rj), float(qq)
    return rows
