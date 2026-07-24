#!/usr/bin/env bash
#
# naim.sh — ARR disagreement-panel RUNBOOK (chronological record).
#
# What I (Naim) ran, in order, to regenerate the 27-row panel per naim_arr.md,
# and where it currently stands. Reads top-to-bottom as the actual sequence.
#
#   STATUS:  §4 cells ✅   §5 dispersion ✅   §6 uq_panel ✅   §7 compile ✅
#            §8 multiplicity ⛔ BLOCKED (missing paired-gain p-value — see bottom)
#
# Locked grid (brief §5-6): K=200, seeds 0,1,2, scheme=span_mean, final layer,
# arm=correct; dispersion taus 5,10,25 / uq_panel tau=10. Modal uses NO GPU
# (CPU containers). Result reproduces a1()'s summary: 3 win / 1 lose / 5 tie.
#
# Notes / corrections made along the way:
#   - Dropped `-q` from the modal commands: in this Modal version `-q` suppresses
#     the function's own stdout, leaving an EMPTY log. (verbosity only, no param change)
#   - A calibration run at the default taus (0.5,1,2) — a smoke config — overwrote
#     mistral/pavlick's paper-setting parquet (brief §5 warns about this). Repaired
#     by re-running the registered command (taus=5,10,25) below.
#
# This is a runbook, not a batch job: running it end-to-end re-launches all 18
# Modal cells (hours). Pass --run to actually execute; otherwise it just prints.
# ---------------------------------------------------------------------------

set -u
cd "$(dirname "$0")"

if [ "${1:-}" != "--run" ]; then
  sed -n '2,40p' "$0"          # show this header
  echo "Runbook only. Copy the section you need, or pass --run to execute everything."
  exit 0
fi

# deps + modal auth live in this venv (default shell python is pyenv 3.6, no torch)
source ~/Documents/hf/bin/activate

MODELS=(llama3_8b mistral_7b qwen2_5_7b)
DATASETS=(chaosnli chaosnli_alpha pavlick_nli)   # the 3 disagreement datasets
mkdir -p logs/arr/dispersion logs/arr/uq

# === Setup: pulled Tanmoy's fix (arm=correct + '#####' headers), HEAD e818f9d ===
#   git pull --ff-only origin main
#   grep -n 'arm: str = "correct"'      analyse_hf.py   # must match
#   grep -n '##### {model} / {dataset}' analyse_hf.py   # must match

# === §4  Read-only cluster inventory ============================= ✅ DONE ===
# 24 cells; 9 disagreement cells (3 models × {chaosnli,chaosnli_alpha,pavlick_nli}),
# all dispersion=True, shards nonzero.
modal run analyse_hf.py::cells | tee cells.log

# === §5  Dispersion: 27-row experiment (9 cells × 3 taus) ======== ✅ DONE ===
# -> logs/arr/dispersion/<model>__<dataset>.log ; each has '#####', the per-tau
# verdict, and exactly 3 tau rows (5,10,25). No failures.
for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do
  modal run analyse_hf.py::dispersion \
    --model="$m" --dataset="$d" \
    --arm=correct --scheme=span_mean --n-clusters=200 \
    --taus=5,10,25 --seeds=0,1,2 \
    2>&1 | tee "logs/arr/dispersion/${m}__${d}.log"
done; done

# === §6  Baseline UQ panel (9 cells) ============================= ✅ DONE ===
# -> logs/arr/uq/<model>__<dataset>.log ; each has '#####' and the section
# 'credal_width vs each baseline'. No failures.
for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do
  modal run analyse_hf.py::uq_panel \
    --model="$m" --dataset="$d" \
    --scheme=span_mean --n-clusters=200 --tau=10 \
    --seeds=0,1,2 --arm=correct --knn-k=10 \
    2>&1 | tee "logs/arr/uq/${m}__${d}.log"
done; done

# === §7  Assemble compiler inputs + compile ====================== ✅ DONE ===
# Concatenate the per-cell blocks in model->dataset order (Modal lifecycle lines
# left in; compile_all.py splits on '##### '), then compile. RESULTS_ALL.md then
# contains 9 A1 blocks + 9 UQ-panel rows.
cat logs/arr/dispersion/{llama3_8b,mistral_7b,qwen2_5_7b}__{chaosnli,chaosnli_alpha,pavlick_nli}.log > a1_pooled.log
cat logs/arr/uq/{llama3_8b,mistral_7b,qwen2_5_7b}__{chaosnli,chaosnli_alpha,pavlick_nli}.log > uq_panel.log
python compile_all.py > RESULTS_ALL.md

# === §8  Multiplicity correction ================================= ⛔ BLOCKED ===
# The tested hypothesis is the PAIRED gain |rho(W)| - |rho(size_only)|. But the
# code emits NO p-value for it: pooled_gain_ci() returns only (gain, CI_lo, CI_hi,
# seedSD, n_pos) — a bootstrap percentile CI, not a p-value. The standalone Spearman
# `p` in the parquet is the wrong statistic (brief §8 forbids BH on it).
# BH/Bonferroni need p-values. Per brief §8, this is reported as a MISSING STATISTIC
# before implementing anything, and a p-value is NOT inferred from the CI crossing
# zero. Awaiting Tanmoy's call (report vs. implement a validated bootstrap p-value).
#
# NEXT after §8: §9 reconcile 3/1/5 vs locked 2/1/6 ; §10 deliverables.
