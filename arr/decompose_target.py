"""The proper-scoring decomposition the correlational panel skips.

Every number in the ARR panel is a rank correlation against H(p_human) -- the
aleatoric term alone. The exact identity is

    CE(p_human, p_model) = H(p_human) + KL(p_human || p_model)
        total                 AU (irreducible)   EU (reducible, never examined)

so the model's forecast quality of the human distribution decomposes into a
term we already target and a term nobody has looked at. This module correlates
every signal against BOTH terms.

WHAT IS ACTUALLY ON THE VOLUME (checked). The extraction is a two-arm
(correct / wrong) contrast: per item it stores realised_logprob, ans_ids and
top-k for a correct answer sequence and one wrong answer sequence. It is NOT a
per-label scan. Therefore:

  * chaosnli_alpha (binary): correct+wrong ARE the label set -> CE/AU/EU are
    clean and computable now. `binary_decomposition` does this.
  * chaosnli (3-way): the decision is at the first token
    (degenerate_first_token=False), so p_model over labels is recoverable from
    the first-position top-k -- BUT only with a label->token map, which is not
    stored. `label_distribution` is the stub for that (needs the label words +
    tokenizer, i.e. a small extraction, not "no extraction").
  * pavlick_nli (5-way, multi-token, ORDINAL): needs a multi-token pooling
    decision AND an ordinal metric (Wasserstein/CRPS, not KL). Out of scope
    until the extraction above exists.

CIRCULARITY (read before trusting the panel). The output-family signals
(pred_entropy, neg_mean_logprob, pred_maxprob, pred_margin) are functions of
p_model. EU = KL(p_human || p_model) is ALSO a function of p_model. So their EU
correlation is partly mechanical -- neg_mean_logprob and EU(len-norm) are both
built from mean realised_logprob and share it directly. The CLEAN test is the
representation family (credal_width, size_only, knn_dist, ...), which is read
off hidden states and never sees p_model. Report the two families separately;
the representation column is the interpretable one.
"""
import numpy as np

# --- PRE-REGISTRATION. Frozen 2026-07-31 (reviewer left the margin blank). ---
DENSITY_NULL = 0.05  # mechanism test rho(W, EU | knn_dist), frozen BEFORE the
                     #   number existed. THE RULE FIRED AND WAS THE WRONG SHAPE:
                     #   an absolute bound on a residual passes any small-effect
                     #   cell regardless of mechanism (the alpha-inflation trap,
                     #   cf. the ladder). Report PROPORTIONAL mediation instead
                     #   (residual/total ~ const, and the product decomposition
                     #   rho(W,EU) ~ rho(W,knn)*rho(knn,EU)). And control LENGTH:
                     #   on chaosnli rho(W,n_ans_tok)=+0.85, so W/knn/length are
                     #   collinear and no residual survives controlling both.
                     #   Kept as the pre-registered artefact, not a live gate.
EU_MARGIN = 0.05   # W "tracks EU" if |rho(W,EU)| > |rho(W,AU)| + EU_MARGIN.
                   #   Sign is a finding: rho(W,EU) > 0 -> width predicts a
                   #   worse forecast (the construction works, wrong target).
                   #   rho(W,EU) < 0 -> width is INVERTED against epistemic
                   #   uncertainty (dense clusters = low EU); a sharper version
                   #   of the current negative-sign claim. Both are reportable.
# ----------------------------------------------------------------------------


def label_distribution(topk_ids, topk_logprobs, label_token_ids):
    """p_model over the label set, restricted to the label words' tokens.

    Recover the model's next-token distribution over the label first-tokens at
    the decision position (chaosnli: the first answer token) and renormalise.

    TODO: restriction changes the entropy. State it in the paper -- a
          distribution renormalised over 3 tokens is not the model's predictive
          distribution and must not be called one.
    TODO: multi-token label words need a pooling decision (first token, sum
          logprob, length-normalised). NOT neutral -- this is where the AAAI
          length confound lives. Pick one, justify it, report the alternative.
    TODO: `label_token_ids` (label -> first-token id, in the SAME order as
          soft_labels) is NOT stored on the volume. It needs the label words +
          the model tokenizer. Until then chaosnli/pavlick cannot run.
    """
    raise NotImplementedError(
        "needs label_token_ids aligned to soft_labels -- a small extraction, "
        "not on the volume. binary_decomposition covers chaosnli_alpha now.")


def binary_decomposition(p_human, logprob_correct, logprob_wrong, pool="mean"):
    """CE = H(p_human) + KL(p_human || p_model) on a two-arm (correct/wrong) cell.

    p_human      : [p_correct, p_wrong], the annotator distribution.
    logprob_*    : per-token realised logprobs of each answer sequence.
    pool         : 'mean' (length-normalised, defuses the answer-length
                   confound) or 'sum'. corr(AU,EU) flips sign between the two on
                   alpha, so report both.

    Returns (AU, EU, TOT) scalars.
    """
    lc = np.mean(logprob_correct) if pool == "mean" else np.sum(logprob_correct)
    lw = np.mean(logprob_wrong) if pool == "mean" else np.sum(logprob_wrong)
    pm = np.array([lc, lw], float); pm = np.exp(pm - pm.max()); pm /= pm.sum()
    ph = np.asarray(p_human, float); ph = ph / max(ph.sum(), 1e-12)
    au = float(-(ph * np.log(np.clip(ph, 1e-12, None))).sum())
    eu = float((ph * np.log(np.clip(ph, 1e-12, None) / np.clip(pm, 1e-12, None))).sum())
    return au, eu, au + eu


def two_target_panel(scores, AU, EU, repr_signals, output_signals):
    """Nine signals x {AU, EU}, signed Spearman. One row per signal, tagged by
    family. The cross-cell verdict (below) is applied on the REPRESENTATION
    family only -- the output family's EU correlation is mechanically inflated
    (see module docstring).

    PRE-REGISTER:
      GATE FIRST -- corr(AU, EU). If |corr| is high the split separates nothing
      and the panel is uninterpretable; compute it before anything else.
      Then, for W (credal_width): it "tracks EU" if
        |rho(W, EU)| > |rho(W, AU)| + EU_MARGIN.
      Sign of rho(W, EU) is the finding (see EU_MARGIN).
    """
    from scipy.stats import spearmanr

    def rho(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 30 else np.nan

    rows = []
    for s, v in scores.items():
        fam = ("repr" if s in repr_signals else
               "output" if s in output_signals else "other")
        rows.append(dict(signal=s, family=fam,
                         rho_AU=rho(v, AU), rho_EU=rho(v, EU),
                         circular=(fam == "output")))  # EU shares p_model
    return rows
