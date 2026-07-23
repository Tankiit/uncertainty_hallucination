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
modal run analyse_hf.py::dispersion --model=<model> --dataset=<dataset>
modal run analyse_hf.py::uq_panel --model=<model> --dataset=<dataset>
modal run analyse_hf.py::uq_panel_correctness --model=<model> --dataset=<dataset>
```

across the 3 models (`llama3_8b`, `mistral_7b`, `qwen2_5_7b`) × the
disagreement datasets (`chaosnli`, `chaosnli_alpha`, `pavlick_nli`) for the
first two, and whatever cells `correctness_panel` is gated to for the
third (its own docstring says it's the correctness-task analogue, 15
cells per the note about `layer_sweep`).

**What I could NOT trace, and you'll need to figure out or ask about:**
`uq_panel`/`uq_panel_correctness` write `.parquet` files to
`/vol/derived/`, not the `.log` text format `compile_all.py` actually
reads. I did not find the conversion step between the two in this pass —
either there's a formatting script I didn't locate, or the `.log` files
were produced by hand from Modal's console output, or `report()`/`cells()`
need a small extension to emit that format. Don't guess at this — report
back what you find rather than inventing a converter that might not match
whatever `compile_all.py` actually expects.

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
number means — if the parquet→log gap needs a real conversion script,
flag it as a design decision rather than inventing the format.
- Carry the ChaosNLI licensing note forward if any ChaosNLI text ends up
in anything shared: `defects()` already flags it as recorded incorrectly
(CC-BY-SA-4.0 in the metadata, actually CC Non-Commercial 4.0 per
ChaosNLI's own README) — the NC restriction applies regardless of what
the metadata says.

## If something's unclear

Same rule as the other briefs: flag it back rather than guessing, especially
for the parquet→log gap in Step 1 — that's a real unknown, not something
with an obvious right answer I'm withholding.
