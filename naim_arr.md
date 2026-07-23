# Brief: ARR Disagreement Panel — Verify, Correct, Reconcile

**Paper:** ARR submission, *"Does Credal Width Track Human Disagreement, or
Just Density?"* (title placeholder, content locked per
`AAAI_and_ARR_abstracts.md`)
**Repo:** `github.com/Tankiit/uncertainty_hallucination` — clone this, it's
the same repo as the AAAI work.
**Reports to:** Tanmoy
**Time estimate:** depends almost entirely on Modal queue time — the
functions below have 2–4 hour timeouts each. The analysis once outputs
exist is fast (under an hour).

---

## The claim being checked

ARR's core result is that credal width $W$'s apparent correlation with
human annotator disagreement (ChaosNLI, Pavlick-NLI, AmbigQA) is
substantially explained by local density rather than by anything
Dempster-Shafer-specific — a panel comparing $W$ against a density baseline
across 9 (model, dataset) cells. **You are not re-deriving this claim or
deciding what it means.** You're doing three concrete, mechanical things
that the codebase's own inline notes already flag as unresolved:

1. Confirming the panel can actually be regenerated (right now it can't,
from a plain clone — see below).
2. Applying a multiple-comparisons correction the code itself says is
missing.
3. Reconciling a small but real discrepancy between two different counts
of the same result.

## Input

You'll need **Modal access** — the disagreement-target datasets (ChaosNLI,
Pavlick-NLI) live only on the Modal volume, not in the repo, per
`stratify_by_cardinality.py`'s own header note. Confirm you can auth
(`modal token new` or whatever the project's existing auth setup is) before
anything else — if you can't, that's the first thing to report back, not a
blocker to quietly work around.

## Your task

**Step 0 — orient yourself before running anything expensive.**

```
modal run analyse_hf.py::cells
```

This is a cheap, read-only diagnostic ("what has actually been extracted,
and is it usable?") — run it first so you know what's actually on the
volume before kicking off 2–4 hour jobs against cells that might not exist
or might be malformed.

**Step 1 — regenerate the three missing inputs.** `compile_all.py` (the
script that produces `RESULTS_ALL.md`) reads three files that don't exist
in a fresh clone: `a1_pooled.log`, `uq_panel.log`, `correctness_panel.log`.
The functions that produce the underlying data are real Modal functions in
`analyse_hf.py`:

```
modal run analyse_hf.py::dispersion --model=<model> --dataset=<dataset> \
  --arm=correct --n-clusters=200 --taus=5,10,25 --seeds=0,1,2
modal run analyse_hf.py::uq_panel --model=<model> --dataset=<dataset>
modal run analyse_hf.py::uq_panel_correctness --model=<model> --dataset=<dataset>
```

Use the current checked-in `analyse_hf.py`. Two handoff-blocking wiring defects
have been corrected:

- `dispersion` now defaults to `arm=correct`, matching the stored paired
  extraction. The previous `arm=main` default crashed with
  `no states for arm='main'`. You may pass `--arm=correct` explicitly for
  auditability.
- All three functions now print `##### model / dataset` before their report.
  This is the block delimiter consumed by `compile_all.py`; the older
  `uq_panel` stdout lacked it and was silently ignored by the parser.

across the 3 models (`llama3_8b`, `mistral_7b`, `qwen2_5_7b`) × the
disagreement datasets (`chaosnli`, `chaosnli_alpha`, `pavlick_nli`) for the
first two. Run the correctness panel across the 3 models × 5 accuracy datasets
(`halueval_qa`, `truthfulqa`, `ambigqa_kge2`, `medqa`, `pubmedqa`), for 15
cells total.

**Output bridge (resolved):** the three analysis functions write
machine-readable `.parquet` files to `/vol/derived/` and now emit textual
blocks beginning with `##### model / dataset`. That delimiter is the format
`compile_all.py` reads for the panel logs. Concatenate the nine disagreement
blocks into `a1_pooled.log` and `uq_panel.log`, and the 15 accuracy blocks into
`correctness_panel.log`. Modal lifecycle lines may be discarded; do not
recompute, round differently, or otherwise change the numerical report while
assembling the logs.

**Important overwrite hazard:** the dispersion parquet filename records the
layer and pooling scheme but omits `n_clusters`, `taus`, and `seeds`. A smoke
test or ablation therefore overwrites the paper-setting parquet for that cell
without warning. Do not run reduced configurations against the shared volume.
For regeneration, retain the registered defaults (`n_clusters=200`,
`seeds=0,1,2`) and explicitly pass the A1 panel grid `--taus=5,10,25`.
Do **not** rely on the function's current `0.5,1,2` default: that is the
low-temperature structural diagnostic, not the 27-row ARR panel. This defect is
recorded in `compile_all.py::defects()`; do not silently rename or reinterpret
existing artifacts.

**Step 2 — multiplicity correction.** `a1()` in `compile_all.py` prints
its own caveat: *"Multiplicity is uncorrected across 27 rows."* Apply a
standard correction (Benjamini-Hochberg is the reasonable default for this
many comparisons; Bonferroni if you want the conservative version) to the
27-row panel and report which of the "3 cells beat null" survive.

**Step 3 — reconcile the count discrepancy.** `a1()`'s inline summary says
**3 win / 1 lose / 5 tie**. The locked ARR abstract says **2 win / 1 lose /
6 tie**. The "lose" count matches exactly; win/tie is off by one in both
directions. Find out why — likely candidate is that one cell (mistral/pavlick
is the most likely, per the specific cells named in `a1()`'s note) sits
right at whatever threshold separates win from tie, and different runs or
threshold choices land it differently. Report the actual reason, not a
guess dressed up as one.

## Output (definition of done)

1. Regenerated `a1_pooled.log`, `uq_panel.log`, `correctness_panel.log` (or
a clear report on why they can't be regenerated as specified).
2. The 27-row panel with a multiplicity correction applied, alongside the
uncorrected numbers — both, not just the corrected one, so the
difference is visible.
3. A one-paragraph explanation of the 3-vs-2 discrepancy, with the specific
cell(s) and threshold values involved.
4. Anything else that looks off while you're in there — flag it the way
the existing `defects()` list does (what it is, what it would have
silently produced if unfixed), don't fix it silently yourself if it
touches how a result should be interpreted.

## Fence — what NOT to do

- Don't decide whether the corrected panel still supports the paper's
claim — that's Tanmoy's call once he has the corrected numbers.
- Don't pick which of the 3-vs-2 counts is "right" without being able to
explain why — surfacing the mechanism is the job, not picking a winner.
- Don't build a new `.log`-generation step that changes what any existing
number means. The output bridge is now the existing `#####` textual report;
only remove Modal lifecycle lines and concatenate blocks.
- Carry the ChaosNLI licensing note forward if any ChaosNLI text ends up
in anything shared: `defects()` already flags it as recorded incorrectly
(CC-BY-SA-4.0 in the metadata, actually CC Non-Commercial 4.0 per
ChaosNLI's own README) — the NC restriction applies regardless of what
the metadata says.

## If something's unclear

Same rule as the other briefs: flag it back rather than guessing, especially
for the parquet→log gap in Step 1 — that's a real unknown, not something
with an obvious right answer I'm withholding.
