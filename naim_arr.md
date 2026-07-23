# ARR disagreement panel — collaborator runbook

**Paper:** *Does Credal Width Track Human Disagreement, or Just Density?*

**Repository:** `https://github.com/Tankiit/uncertainty_hallucination`


**Reports to:** Tanmoy

## Objective

Re-run and audit the ARR disagreement experiment, apply a correction across
its 27 comparisons, and explain why one summary says 3 win / 1 lose / 5 tie
while the locked abstract says 2 win / 1 lose / 6 tie.

This is an experiment and analysis assignment. Do not download, replace, or
reconstruct the datasets. The extracted states, logits, and disagreement
targets already live on the shared Modal volume.

The experiment compares credal width

`W = sum_k m_k (1 - 1 / |F_k|)`

with the landing-cluster density baseline

`size_only = 1 - 1 / |F_argmax(m)|`

using absolute Spearman correlation with human annotator entropy.

## 1. Clone the exact code

```bash
git clone https://github.com/Tankiit/uncertainty_hallucination.git
cd uncertainty_hallucination
git fetch origin
git checkout RSUQ
git pull --ff-only origin RSUQ
```

Before running, confirm that the checked-out version contains both fixes:

```bash
grep -n 'arm: str = "correct"' analyse_hf.py
grep -n '##### {model} / {dataset}' analyse_hf.py
```

Both commands must return matches. If they do not, stop and tell Tanmoy which
commit you checked out. The older code crashes in `dispersion` because it asks
for the nonexistent `main` arm, and its panel output lacks the delimiter read
by `compile_all.py`.

Record the exact revision:

```bash
git rev-parse HEAD
git status --short
```

The worktree should initially be clean.

## 2. Prepare the local submission environment
This is for the cluster 

<!-- The heavy computation runs on Modal, not on the login node or laptop. Locally,
only Python, Git, and the Modal client are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip modal
modal --version
```

Do not install PyTorch or scikit-learn locally for these runs:
`analyse_hf.py` defines a Modal image containing the remote dependencies. -->

## 3. Authenticate to the correct Modal workspace

```bash
modal token new
modal profile list
modal profile current
```

The active profile must have access to the workspace containing the
`rsuq-latent` volume. Repository access does not grant Modal access. If the
profile or volume is unavailable, stop and report the authorization failure;
do not substitute local data.

## 4. Verify the cluster inputs before launching experiments

Run the cheap, read-only inventory:

```bash
modal run -q analyse_hf.py::cells | tee cells.log
```

Expected inventory:

- 24 cells total;
- models: `llama3_8b`, `mistral_7b`, `qwen2_5_7b`;
- nine disagreement cells: each model crossed with `chaosnli`,
  `chaosnli_alpha`, and `pavlick_nli`;
- each disagreement row reports `dispersion=True`;
- state shards are nonzero.

Do not start the expensive jobs if any of these checks fail.

## 5. Run the 27-row disagreement experiment on Modal

The experimental matrix is:

- models: `llama3_8b`, `mistral_7b`, `qwen2_5_7b`;
- datasets: `chaosnli`, `chaosnli_alpha`, `pavlick_nli`;
- `K=200`;
- `tau in {5, 10, 25}`;
- seeds `{0,1,2}`;
- pooling `span_mean`;
- final captured layer;
- arm `correct`.

Create local folders for unedited cluster logs:

```bash
mkdir -p logs/arr/dispersion logs/arr/uq
```

For every model/dataset pair, run:

```bash
modal run -q analyse_hf.py::dispersion \
  --model=<model> \
  --dataset=<dataset> \
  --arm=correct \
  --scheme=span_mean \
  --n-clusters=200 \
  --taus=5,10,25 \
  --seeds=0,1,2 \
  2>&1 | tee logs/arr/dispersion/<model>__<dataset>.log
```

Each invocation submits the function to the Modal cluster. It does not perform
KMeans on the local machine. The nine cells are independent and may be launched
concurrently in separate terminals. Do not launch reduced smoke configurations
against the shared volume.

Use `--detach` only if maintaining the terminal connection is impossible:

```bash
modal run --detach analyse_hf.py::dispersion ...
```

Detached jobs continue on Modal, but their console output must later be
retrieved from the Modal run page. For reproducible local logs, attached runs
with `tee` are preferred.

Every completed log must contain:

```text
##### model / dataset
=== per-tau verdict (never averaged over tau) ===
```

followed by exactly three tau rows.

### Shared-volume overwrite warning

`dispersion` currently writes a parquet filename that omits K, tau, and seeds.
A smoke test or ablation can therefore overwrite the paper-setting parquet for
that cell without warning. Use only the registered command above. If a reduced
run is accidentally submitted, rerun the full registered cell immediately and
report the collision.

## 6. Run the nine-cell baseline panel

For the same nine model/dataset cells, run:

```bash
modal run -q analyse_hf.py::uq_panel \
  --model=<model> \
  --dataset=<dataset> \
  --scheme=span_mean \
  --n-clusters=200 \
  --tau=10 \
  --seeds=0,1,2 \
  --arm=correct \
  --knn-k=10 \
  2>&1 | tee logs/arr/uq/<model>__<dataset>.log
```

This compares W with mass, density, and token-probability baselines. Confirm
that each log begins a report block with `##### model / dataset` and contains
the section `credal_width vs each baseline`.

The 15-cell correctness panel is not part of the ARR disagreement claim. Do not
spend cluster time regenerating it unless Tanmoy separately requests it.

## 7. Assemble the compiler inputs without changing results

Keep the per-cell raw logs. Assemble the nine textual report blocks in model
then dataset order:

```text
llama3_8b: chaosnli, chaosnli_alpha, pavlick_nli
mistral_7b: chaosnli, chaosnli_alpha, pavlick_nli
qwen2_5_7b: chaosnli, chaosnli_alpha, pavlick_nli
```

The dispersion blocks become `a1_pooled.log`; the baseline blocks become
`uq_panel.log`. `compile_all.py` recognizes blocks beginning with
`##### model / dataset`. Modal lifecycle lines may be removed, but do not
recompute values, change rounding, or hand-edit verdicts.

Run:

```bash
python compile_all.py
```

Verify that the generated `RESULTS_ALL.md` contains nine A1 blocks and nine UQ
panel rows. Preserve the raw logs alongside the assembled inputs.

## 8. Apply multiplicity correction across 27 comparisons

The family is the 3 models × 3 datasets × 3 tau values: 27 W-versus-density
comparisons.

Report, for every row:

- model, dataset, tau;
- observed gain `|rho(W,y)| - |rho(size_only,y)|`;
- uncorrected interval and uncorrected p-value;
- Benjamini-Hochberg q-value across all 27 rows;
- Bonferroni-adjusted p-value as a conservative sensitivity analysis;
- original verdict and multiplicity-adjusted verdict;
- whether the row was suppressed as uninformative by
  `median_max_mass > 0.90` or `corr(W,size_only) > 0.98`.

Do not apply BH to the standalone Spearman p-values for W or `size_only`.
The tested hypothesis is their paired gain. If the current code does not emit a
valid p-value for that paired gain, report that as a missing statistic before
implementing it. Any implementation must reuse the same item-level resampling
across all seeds and must be validated on a synthetic null. Do not infer a
p-value from whether a percentile interval crosses zero.

Keep both corrected and uncorrected results visible.

## 9. Reconcile 3/1/5 versus 2/1/6

Reconstruct the cell-level classification mechanically from the 27 rows. For
each of the nine cells, record:

- all three tau-row verdicts;
- which rows are informative;
- the rule used to collapse three tau rows to one cell verdict;
- the exact gain, interval, and thresholds for any row changing category;
- the result before and after multiplicity correction.

Then compare:

- `compile_all.py`: 3 win / 1 lose / 5 tie;
- locked ARR abstract: 2 win / 1 lose / 6 tie.

Identify the exact cell responsible and whether the discrepancy comes from
different run output, an informativeness threshold, a tau-to-cell aggregation
rule, or multiplicity handling. Do not assume Mistral/Pavlick is responsible;
demonstrate it from the regenerated rows.

## 10. Definition of done

Return:

1. Git commit, Modal profile/environment, and the `cells` inventory.
2. Nine raw dispersion logs and nine raw UQ-panel logs.
3. `a1_pooled.log` and `uq_panel.log`, plus regenerated `RESULTS_ALL.md`.
4. A 27-row table containing uncorrected and corrected statistics.
5. A one-paragraph, evidence-backed explanation of the count discrepancy.
6. A defect list in the form: defect → silent consequence → action taken.

## Fences

- Do not decide whether the corrected result supports the paper. Tanmoy makes
  that interpretation.
- Do not alter the frame, pooling, layer, K, tau grid, seeds, arm, target, or
  cell-level aggregation rule without reporting the proposed change first.
- Do not replace the Modal extraction with locally downloaded datasets.
- Do not run reduced configurations against the shared derived-output path.
- Do not select between the 3-win and 2-win summaries without identifying the
  exact mechanism.
- ChaosNLI is non-commercial. Carry the CC Non-Commercial 4.0 restriction
  forward if any source text is shared; repository metadata may state the
  licence incorrectly.

If anything is unclear or a prerequisite fails, stop and report the exact
command, error, active profile, and Git commit rather than guessing.
