"""Does the binary target explain the alpha-NLI result?

The problem. On chaosnli_alpha the label set has two elements, so annotator
entropy H(p) is symmetric about p = 0.5 and discards WHICH option annotators
favoured. Two items with opposite majority preferences and equal margins get
identical targets.

Why it matters. alpha-NLI is the only binary cell in the panel AND the only
cell where the output-space family leads. Task type and target geometry are
perfectly confounded. "Output space wins on abductive reasoning" and "output
space wins when the target is folded" are currently indistinguishable.

This is the cheapest experiment in the package and it can invalidate a
headline. Run it before the family claim goes anywhere near the abstract.
"""
import numpy as np
from _common import rho

# --- PRE-REGISTRATION. Freeze BEFORE running. ----------------------------
MATERIAL_RISE = 0.05   # FROZEN 2026-07-31. rep-space |rho| must rise to
                       #       within 0.05 of the token family's rho on the
                       #       same cell to call the split a target artefact.
# -------------------------------------------------------------------------


def unfolded_targets(p_binary):
    """Alternative targets for the binary cell.

    lean  : p[1]. Signed and directional. This is the real variant -- it
            retains what H throws away.
    split : |p[1] - 0.5|. This is a MONOTONE reparameterisation of H, so
            Spearman against it must be IDENTICAL (up to sign) to Spearman
            against H. It is a NULL CONTROL, not a variant. If it differs,
            something upstream is wrong and that is worth knowing.
    """
    p1 = np.asarray(p_binary, float)[:, 1]
    return dict(lean=p1, split=np.abs(p1 - 0.5))


def run(scores, p_binary, ent):
    """Recompute the nine-signal panel on the binary cell against `lean`.

    READING, pre-registered:
      (a) rep-space |rho| rises materially against lean
          -> the family split is an artefact of the target summary.
             CUT the split from the abstract. Report the fold as the finding
             instead -- it is a genuine contribution to how HLV targets are
             constructed, and a better one than the split was.
      (b) rep-space stays flat
          -> fold ruled out. The split survives this objection and proceeds
             to family_paired_test.py.

    TODO: also report the null control. `split` must reproduce the H result.
    """
    assert MATERIAL_RISE is not None, "freeze MATERIAL_RISE first"
    tgt = unfolded_targets(p_binary)
    rows = []
    for sig, v in scores.items():
        rows.append(dict(signal=sig,
                         rho_entropy=rho(v, ent),
                         rho_lean=rho(v, tgt["lean"]),
                         rho_split_control=rho(v, tgt["split"])))
    return rows
