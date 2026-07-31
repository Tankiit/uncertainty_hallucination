# What's different from the existing rsuq results

This folder is the ARR **disagreement** analysis run on 2026-07-31. It sits on
top of the same data and cells as the prior rsuq results but adds inferential
machinery and seven experiments that did not exist before. Prior results:

- `../RESULTS_ALL.md` — consolidated RSUQ (dispersion/UQ panels, correctness,
  density/saturation). §8 lists open issues.
- `../ARR_run_summary.md` — the prior ARR panel: "3 win / 1 lose / 5 tie" for
  `credal_width (W)` vs the `size_only` density null, plus a baseline
  best-metric UQ panel. Its §8 was **blocked**.

## Same substrate (unchanged)

- Same volume `rsuq-latent`, same 9 cells, same `K=200 / seeds{0,1,2} /
  scheme=span_mean / final layer / arm=correct`.
- Same τ grid `{5,10,25}` (prior panel used it too; primary ARR analyses here
  are at τ=10, sign_stability spans all three).
- Same target: human annotator entropy. Prior work used plug-in Shannon;
  `PRIMARY_TARGET=h_plugin` keeps that (zero drift), and `audit_target` now
  also emits `h_norm` and `h_mm` as robustness.

## Blocker resolved

| prior status | now |
|---|---|
| `ARR_run_summary.md` §8: `pooled_gain_ci()` returns a CI but **no p-value**, so BH could not correct the paired gain. Reported, not faked. | **Patch 2** adds a bootstrap-percentile two-sided p (6th return element). BH is now applied within the ladder and family panels (`data/ladder_bh.csv`, `data/family_bh.csv`). |
| `RESULTS_ALL.md` open issue #1: multiplicity uncorrected across 27 nested rows. | ARR panels correct **within** a panel (ladder 18 rows, family 9 rows), never pooled across panels testing different things. |

## New infrastructure

- **Patch 1** (`analyse_hf.uq_panel`): persists per-item score dumps
  (`peritem_*.parquet`). These did not exist before — the prior panel kept
  only collapsed correlation rows. Every item-level analysis below depends on
  this.

## New experiments (seven, not in rsuq at all)

| experiment | what it replaces / adds | headline |
|---|---|---|
| **fold_gold** | Nothing prior questioned the α target. This session found ChaosNLI-α is canonicalised **by crowd majority**, recovered the gold-oriented target from the original ChaosNLI release, and tested the fold with a powered matched-dispersion statistic. | **Not detected, low power** — no direction read, but α's oracle ceiling is ≈0, so not a clearance. |
| **family** | Replaces the prior "**best metric**" column (an argmax over 9 signals — selection-on-max, no inference) with a paired REPR-vs-OUTPUT test (item bootstrap + BH). | Split is significant but **not constant within Pavlick** → cut from abstract. |
| **ladder** | Deepens the prior flat "W vs size_only" gain into nested partial correlations `knn→size→credal`. | "Elaboration buys nothing" holds only in low-signal cells; negative partials elsewhere. |
| **sign_stability** | The prior panel had per-τ rows but never tested sign constancy. | credal_width's negative sign is stable across τ wherever it carries signal. |
| **AU/EU decomposition** | New. Splits CE(p_human,p_model)=H(p_human)+KL into the aleatoric term the panel already targets and the epistemic term (KL) nobody examined; correlates every signal with both. Answers Baan et al. directly. | Alive gate; raw density (knn_dist) tracks epistemic with the right (+) sign while credal_width is INVERTED (−, up to −0.55) — credal_width is an inverted epistemic predictor, not just a weak aleatoric one. |
| **components** | New. Decomposes the target `H = log k · balance` and asks whether credal_width tracks support size or balance — turns the prior "W ≈ size_only" observation into a falsifiable test. | Inconclusive: the deciding Pavlick cells are confounded by a varying annotator count that drives k; only 3 clean cells (< MIN_CELLS). |
| **neighbourhood** | New. Prior work had `knn_dist` as a baseline but no recoverability ceiling. | Oracle ceiling ρ 0.41–0.70 vs built signals ~0.2–0.4; binary α ceiling ≈ 0. |

## How the numbers line up with the prior panel

The prior `ARR_run_summary.md` best-metric panel (τ=10) and the "W vs knn tie"
column are **consistent** with the new decomposition:

- Prior: `W vs knn = tie` in most cells; `W vs size_only` mostly tie/uninformative.
  New: the ladder shows the **mass** rung (credal over size) adds nothing in
  5/9 cells, and sign_stability shows the strong cells all carry the same
  **negative** sign — i.e. credal_width ≈ a negatively-signed size readout.
  Same conclusion, now with partial correlations and a stability test behind it.
- Prior: per-cell "best metric" was `size_only`/`credal_width` on chaosnli and
  `neg_mean_logprob` (an OUTPUT signal) on chaosnli_alpha. New: the **family**
  test formalises exactly that contrast (REPR leads chaosnli, OUTPUT leads α)
  — and then shows it does not generalise to Pavlick.
- Prior dispersion verdict "3 win / 1 lose / 5 tie" (W beats the density null)
  is a different question (W vs size_only paired gain) and is **not**
  re-litigated here; the ARR panel it came from is untouched.

## Caveats surfaced this session (new)

1. The naive fold target on α is degenerate (majority-canonicalised). Even the
   gold-oriented `lean_gold` is monotone in H on 89.4% of items, so the naive
   gain is tautological; the powered matched-dispersion test reads no direction
   but α's ceiling is ≈0, so it is "not detected, low power," not "cleared."
2. The ladder "adds nothing" count is inflated by the low-signal α cells.
   r(W, size_only) is 0.85–0.96 at τ=10 (0.9996 is a τ→0 artifact), so the
   nested partials are legitimate.
3. The family split is model-dependent on Pavlick — not a general task law.
4. Binary α has a near-zero oracle ceiling: the flat panel there is the target,
   not a failure of the signals.
5. `components`: `MIN_FRAC` as a fraction is not a constant operation across
   datasets (2% is ≥2 votes at N=100 but <1 vote at Pavlick's ~28); and k is
   driven by the varying annotator count on Pavlick, confounding the deciding
   cells. The size-readout question is left inconclusive, not decided.

## Reproduce

Every table here is regenerated from the volume by `../arr_modal.py`
(`audit_target`, `fold_gold`, `ladder`, `family`, `sign_stability`,
`neighbourhood`, `collect`) after Patch 1's `peritem_*.parquet` dumps exist.
`data/*.csv` were pulled with the export step in the session scratchpad;
`lean_gold` is rebuilt by `../arr_build_leangold.py`.
