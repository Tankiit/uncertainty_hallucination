# arr_results/

Results of the ARR **human-disagreement** pipeline, run 2026-07-31 over
3 models × 3 datasets (9 cells). Produced by `../arr_modal.py` on the
`rsuq-latent` Modal volume.

## Contents

| file | what |
|---|---|
| `RESULTS_ARR.md` | Compiled results with all tables and per-experiment verdicts. **Start here.** |
| `DIFF_vs_rsuq.md` | What is new / changed vs `../RESULTS_ALL.md` and `../ARR_run_summary.md`. |
| `data/audit_target.csv` | Per-dataset target audit (n, support, annotator counts, h_plugin/h_norm/h_mm). |
| `data/fold_gold.csv` | Fold test on the gold-oriented un-canonicalised target (the real test). |
| `data/fold_naive.csv` | Fold test on the naive majority-canonicalised target (degenerate; kept for the record). |
| `data/ladder_bh.csv` | Nested partial correlations `knn→size→credal`, BH-corrected (18 rows). |
| `data/family_bh.csv` | Paired REPR-vs-OUTPUT family test, BH-corrected (9 rows). |
| `data/sign_stability.csv` | ρ(signal, H) across τ∈{5,10,25} for the ladder signals (81 rows). |
| `data/components.csv` | Target decomposed into log k (support) and balance; ρ of every signal with each (81 rows). |
| `data/ce_decomp.csv` | AU/EU (aleatoric/epistemic) split on α + chaosnli: ρ(signal, AU) vs ρ(signal, EU) + h_mm, by family. |
| `data/mechanism.csv` | Mechanism test: ρ(W,EU\|knn_dist) and the AU/EU partials — density explains the double inversion (4/5 cells). |
| `data/neighbourhood.csv` | Neighbourhood reads + cross-fitted oracle ceiling (ρ and R²) per cell × k. |
| `data/deferral_truthfulqa.csv/.json` | STC + full AURC-gap deferral verdict on truthfulqa (3 models × 3 arms), via the reconstructed `../screen.py`. |

## Headline

| # | experiment | verdict |
|---|---|---|
| 01 | audit_target | passed |
| 02 | **fold (gold-oriented)** | **not detected, low power** — powered test reads no direction, but α's ceiling is ≈0, so not a clearance |
| 03 | ladder | mixed — elaboration redundant only in low-signal cells; r(W,size)=0.85–0.96 at τ=10, so partials are legitimate |
| 04 | **family split** | **cut from abstract** — significant but flips across models within Pavlick |
| 05 | sign_stability | stable — credal_width's negative sign holds across τ where it carries signal |
| 06 | neighbourhood | oracle ceiling (0.41–0.70) exceeds the built signals (~0.2–0.4); binary α ceiling ≈ 0 |
| 07 | **components** | **inconclusive** — deciding Pavlick cells confounded by annotator count; only 3 clean cells < MIN_CELLS |
| 08 | **AU/EU decomposition** | gate alive (α+chaosnli); raw density (knn_dist) tracks epistemic **right sign**, credal_width **inverted** (up to −0.55) — clean-signal finding |

**Through-line:** the representation-vs-output framing is clean on ChaosNLI vs α
and honestly broken on Pavlick (model-dependent → cut). The α side is **not
certified** — the fold is *not detected but under-powered* (α's oracle ceiling
is ≈ 0, so nothing is recoverable there to begin with). The oracle ceiling shows
the constructed signals leave recoverable disagreement unread on ChaosNLI and
Pavlick — most on Qwen (`qwen > mistral > llama`). The "credal_width is a size
readout" reframe is **not settled**: the decomposition that would test it is
confounded by annotator count on the deciding cells.

**Solid vs pending:** sign_stability and the oracle-ceiling gap are solid and
paper-ready. §Elaboration (ladder) is unblocked now that r(W,size)=0.85–0.96 at
τ=10 is confirmed. The **α narrative** (fold) and the **size-readout claim**
(components) are both under-powered / confounded — write them as such, not as
settled results.

## Conventions (frozen pre-registration)

Signed Spearman; bootstrap resamples items; never average across τ; BH within
panel, never pooled across panels. Thresholds frozen 2026-07-31 before results
were seen: `NULL_THRESHOLD=0.05 · MIN_CELLS=5 · MATERIAL_RISE=0.05 ·
FAMILY_STAT=mean · PRIMARY_TARGET=h_plugin`. `nbhd_oracle_CEILING` observes the
target and is never reported as a method.

## Reproduce

```
modal run arr_build_leangold.py                              # recover lean_gold (fold)
modal run arr_modal.py::audit_target   --model=… --dataset=… # per cell
modal run arr_modal.py::fold_gold      --model=… --dataset=chaosnli_alpha
modal run arr_modal.py::ladder         --model=… --dataset=…
modal run arr_modal.py::family         --model=… --dataset=…
modal run arr_modal.py::sign_stability --model=… --dataset=…
modal run arr_modal.py::components     --model=… --dataset=…  # H = log k · balance
modal run arr_modal.py::neighbourhood  --model=… --dataset=…
modal run arr_modal.py::collect                              # BH + component verdict
```

All of the above depend on Patch 1 (`analyse_hf.uq_panel` writing
`peritem_*.parquet`) having been run for the cell first.
