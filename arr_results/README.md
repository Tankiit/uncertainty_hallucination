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
| `data/neighbourhood.csv` | Neighbourhood reads + cross-fitted oracle ceiling (ρ and R²) per cell × k. |

## Headline

| # | experiment | verdict |
|---|---|---|
| 01 | audit_target | passed |
| 02 | **fold (gold-oriented)** | **cleared** — the REPR-vs-OUTPUT split is not a target-folding artefact |
| 03 | ladder | mixed — elaboration is redundant only in low-signal cells |
| 04 | **family split** | **cut from abstract** — significant but flips across models within Pavlick |
| 05 | sign_stability | stable — credal_width's negative sign holds across τ where it carries signal |
| 06 | neighbourhood | oracle ceiling (0.41–0.70) exceeds the built signals (~0.2–0.4); binary α ceiling ≈ 0 |

**Through-line:** the representation-vs-output framing is real where it's clean
(the ChaosNLI / α task contrast, certified by the fold) and honestly broken
where it isn't (Pavlick is model-dependent). The oracle ceiling shows the
constructed signals leave recoverable disagreement unread — most on Qwen
(`qwen > mistral > llama` throughout) — except on binary α-NLI, where nothing
is recoverable at all.

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
modal run arr_modal.py::neighbourhood  --model=… --dataset=…
modal run arr_modal.py::collect                              # BH within panel
```

All of the above depend on Patch 1 (`analyse_hf.uq_panel` writing
`peritem_*.parquet`) having been run for the cell first.
