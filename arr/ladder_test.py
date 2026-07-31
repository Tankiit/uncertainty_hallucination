"""Does elaboration add information over the rung below?

The paper's spine. Three representation-space signals read the same set of
vectors with increasing machinery:

    rung 0  knn_dist      mean distance to the k nearest neighbours (no frame)
    rung 1  size_only     1 - 1/|F_argmax|                          (frame)
    rung 2  credal_width  sum_k m_k (1 - 1/|F_k|)          (frame + mass)

Claim: rho(rung1, H | rung0) and rho(rung2, H | rung1) are both
indistinguishable from zero. That converts "elaboration buys nothing" from
an assertion into a measurement.

BLOCKED on the per-item score dump (PATCHES_analyse_hf.md).
"""
import numpy as np
from _common import partial_rho, item_bootstrap, benjamini_hochberg, rho

LADDER = ["knn_dist", "size_only", "credal_width"]

# --- PRE-REGISTRATION. Fill in and freeze BEFORE running. -----------------
# "Indistinguishable from zero" needs a number, not a vibe.
NULL_THRESHOLD = 0.05      # FROZEN 2026-07-31. |partial rho| below this AND
                           #       CI containing 0  ->  rung adds nothing.
MIN_CELLS_FOR_CLAIM = 5    # FROZEN 2026-07-31. majority of 9 cells before the
                           #       ladder is a general result, not per-cell.
# -------------------------------------------------------------------------


def nested_gain(scores, ent, lower, upper, B=2000, seed=0):
    """Partial rank correlation of `upper` with the target, given `lower`."""
    x, z, y = scores[upper], scores[lower], ent
    obs = partial_rho(x, y, z)
    lo, hi, p = item_bootstrap(
        lambda i: partial_rho(x[i], y[i], z[i]), len(y), B=B, seed=seed)
    return dict(lower=lower, upper=upper, partial_rho=obs,
                ci_lo=lo, ci_hi=hi, p=p)


def sign_stability(cells):
    """Is the credal_width sign a property of the construction or of tau=10?

    The whole reframe rests on this. Report sign(rho) across
    tau x seed x cell as a grid.

    DO NOT AVERAGE OVER TAU. size_only is exactly tau-invariant (argmax is
    tau-free), so a tau-averaged width compared against it compares
    quantities of different kinds -- that is RESULTS_ALL.md defect 6.

    TODO: reading rule, pre-registered:
      - sign constant outside the binary-target cells  -> mechanism claim holds
      - any flip                                       -> weaken to a
        per-dataset claim and report the flip explicitly. Do not smooth it.
    """
    raise NotImplementedError("needs the per-item dump across the tau grid")


def panel(cells, alpha=0.05):
    """Run both nested gains on every cell, then BH across the panel.

    TODO: BH over which rows? There are 9 cells x 2 nested gains = 18.
          Correcting over 18 is the honest default. Record the choice --
          the ARR dispersion panel corrects over 22 informative rows of 27
          and the exclusion rule is documented; match that discipline.
    """
    assert NULL_THRESHOLD is not None, "freeze NULL_THRESHOLD first"
    rows = []
    for name, (scores, ent) in cells.items():
        for lo_, up_ in zip(LADDER[:-1], LADDER[1:]):
            r = nested_gain(scores, ent, lo_, up_)
            r["cell"] = name
            rows.append(r)
    rej, q = benjamini_hochberg([r["p"] for r in rows], alpha=alpha)
    for r, rj, qq in zip(rows, rej, q):
        r["reject"], r["q"] = bool(rj), float(qq)
        r["adds_nothing"] = (abs(r["partial_rho"]) < NULL_THRESHOLD
                             and r["ci_lo"] <= 0 <= r["ci_hi"])
    return rows
