# RSUQ — consolidated results

Every quantitative result from this session, with the provenance and caveats attached to each. Sections are ordered by what they can establish, not by when they were run.

**Reading guide.** Two different tasks appear throughout and they are not interchangeable:

- **Correctness ranking** — separate a supplied correct answer from a supplied wrong one. Statistic: marginal Mann-Whitney `theta = P(metric_wrong > metric_correct)`, null 0.5.
- **Dispersion validation** — predict *human annotator disagreement*. Statistic: Spearman `rho(metric, annotator_entropy)`. This is the only task with a target the model cannot see, so it is the only one immune to the density confound.


---

## 1. Data and provenance

### 1.1 The original local cache

`outputs/llama3_8b/halueval_qa/hidden_states.pt` — Llama-3.1-8B-Instruct on HaluEval-QA, 6,554 paired items, d=4096.

| field | content |
|---|---|
| `h_correct / h_wrong` | (6554, 4096) fp32 — **first answer token only** (`token_mode='first'`), not a span mean |
| `lp_correct / lp_wrong` | generation logprobs; the only other signal-bearing field |
| `questions / correct_ans / wrong_ans` | full raw text, all 6,554 questions unique — the cell is re-extractable |
| `y_expert_correct / y_expert_wrong` | **all 1** — zero information (confirms the memo trap at `run_repspace_deferral.py:192`) |
| `human_accs` | **all 1.0** — zero information |
| `categories` | all `HaluEval-QA` — no stratification possible |
| `n_probe_layers=8, n_total_layers=32` | 8 layers probed at extraction, one vector kept, **no layer index recorded** |

89 rows have `h_correct` byte-identical to `h_wrong` (shared first token) — they contribute nothing to any paired comparison and are the source of the ties in the sign tests.

**Model-produced error signal:** the model assigns higher logprob to the hallucination on **33.3%** of items (2,184/6,554, 1 exact tie). This is a genuine model-generated label, unlike `y_expert_*`. Caveat: `lp_correct > -0.01` for 1,274 items (19%) vs 12 on the wrong side, so the correct-side logprobs are heavily ceiling-saturated.

### 1.2 Modal re-extraction (24 cells)

3 models x 8 datasets, bf16, 8 probe layers, spans + top-50 logits + attention. **$2.23, 26.9 GB, 0% NaN.**

| | A1-capable (has annotator distribution) | paired only |
|---|---|---|
| datasets | chaosnli (3000), chaosnli_alpha (1532), pavlick_nli (496) | halueval_qa (3000), medqa (3000), ambigqa_kge2 (1408), pubmedqa (1000), truthfulqa (817) |

Sources built from the OVA_ARR adapter registry, not re-derived, so distractor choice and label mapping match that project's decisions.

**Not available, and why:**

- *Semantic entropy* — needs multiple sampled generations per item; `extract_hf.py` defers sampling and states it cannot be retrofitted from stored states.
- *EigenScore / INSIDE* — want covariance across sampled generations' embeddings; per-token states within a span are a different object.
- *`ambigqa_kge2` as a disagreement cell* — its meta carries `n_answers`, a COUNT of interpretations, not a distribution over a fixed label set. Nothing to take an entropy of.
- *`multinli_disagreement`* — adapter points at `metaeval/nli-disagreement-tasks`, which no longer exists on the Hub.


---

## 2. Defects found, and what each would have done

Recorded because several would have silently produced publishable-looking numbers.

| # | defect | consequence if unfixed |
|---|---|---|
| 1 | `FixedFrame.members` never existed (`rsnn_compat.py:103`, `test_core.py:99`) | 2 tests failed on `AttributeError`; the RS-NN pignistic equivalence claim had **never** been verified |
| 2 | `mass_geometric` allocated an (N,K,d) broadcast | ~86 GB at N=13k/K=200/d=4096 — OOM-killed; the function **could not run** on any real cache |
| 3 | `screen.verdict_from_ci` only tests `lo > 0` | a CI entirely BELOW zero prints as `NULL (CI includes 0)` — conflates 'no effect' with 'actively hurts' |
| 4 | Sign test counted ties in the denominator | null expectation pulled to (1-tie_rate)/2, making every result look mildly below chance for free |
| 5 | `analyse.py` bootstrap re-seeded inside the comprehension | all 500 replicates identical -> **every CI zero-width**, printed as if real |
| 6 | A1 verdict averaged over tau | `size_only` is EXACTLY tau-invariant (argmax is tau-free), so this compared a tau-averaged W to a tau-constant baseline |
| 7 | `dispersion` parquet filename omits K, tau grid, and seed grid | a smoke or ablation run silently overwrites the paper-setting A1 parquet for that cell |
| 8 | `o.attentions` indexed with `keep` | hidden_states has L+1 entries, attentions has L -> IndexError; layer 0 has no attention row |
| 9 | Empty answer span for unpaired cells | `slice(p-1,p-1)` is empty; safetensors stores zero-length arrays **silently** (72-byte tensors) — A1 would have had no data |
| 10 | Qwen2.5 in fp16 | NaN in final layer for 5-100% of items, **exactly the layer A1 uses**; max finite value 148 vs fp16's 65504 ceiling, so intermediate activations overflowed, not the stored states |
| 11 | `ambigqa_kge2` marked `kind='disagreement'` | extraction aborted 3 cells on a correct guard — the metadata was wrong, not the guard |
| 12 | ChaosNLI licence recorded as CC-BY-SA-4.0 | its README says **Creative Commons NON-COMMERCIAL 4.0**; the NC restriction must travel with the cell |
| 13 | Span starts at `p-1`, which is answer-independent | for single-token answers the whole span is that one position, so `h_correct` is BIT-IDENTICAL to `h_wrong` — medqa and pubmedqa are degenerate by construction (see 7.3.1) |

---

## 3. Tier 0 / Tier 1 (HaluEval, local cache)


> **Tier 0/1** — not found (`results_tier01.json`); not run or not saved.


---

## 4. Paired contrastive width (independent re-derivation)

`check_paired_width.py` shares no code with `screen.py` or the tier scripts. All 6,554 pairs (the geometric arm trains nothing, so no split is needed).


**Findings.** The effect is real and robust for K>=100 (9 cells, all p<1e-26). It is **inverted**: width is higher on the *correct* answer. The `break-pairing` control retains 88-108% of the deviation, so it is a **marginal** difference between pools, not a paired effect — which invalidates T1.3's matched-confidence framing. Trimming clusters with |F_k|<100 (41-53% of mass, W spread down 18x) does not remove it, so it is not driven by tiny clusters. K=50 is the one unstable regime (seed 0 *above* chance at 0.5183, seeds 1-2 below). `swap-labels` stayed in [0.4906, 0.5065] across all 34 configurations, so the test is calibrated.


---

## 5. Density & saturation (Density_paper.md §1-§4)


> **Density paper** — not found (`results_density.json`); not run or not saved.


---

## 6. A1 — dispersion validation (9 cells, bf16)

Pooled CI: one item-resample index applied to **every** seed's frame, gains averaged inside the replicate. A positive verdict requires the pooled CI to exclude 0 **and** all seeds individually positive.

Rows are suppressed as UNINFORMATIVE where `max-mass > 0.90` or `r(W, size_only) > 0.98` — there W == size_only algebraically and the comparison carries no information regardless of significance.

```
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-oV217EjgtghwUR8pwLOyZt
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function layer_sweep.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function report.
└── 🔨 Created function cells.

##### llama3_8b / chaosnli
3000 items | H=(3000, 4096) | layer=32 arm=correct
  annot-entropy mean=0.6548 std=0.2281  pcts 0/5/25/50/75/95/100 = 0.000/0.224/0.517/0.685/0.807/1.004/1.098
  frac within +/-0.1 of median: 0.379  (high => concentrated target, rho is weakly identified)

=== per-tau verdict (never averaged over tau) ===
Stopping app - local entrypoint completed.
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.1836     0.9546   -0.1827    -0.1849  -0.0022 [-0.0162,+0.0121]  0.0087  tie -> W is a size readout (CI spans 0)
 10.00     0.0745     0.8247   -0.1812    -0.1849  -0.0038 [-0.0194,+0.0120]  0.0151  tie -> W is a size readout (CI spans 0)
 25.00     0.0318     0.5025   -0.1714    -0.1849  -0.0135 [-0.0336,+0.0060]  0.0260  tie -> W is a size readout (CI spans 0)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [5.0, 10.0, 25.0]
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-oV217EjgtghwUR8pwLOyZt
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-Nfwo9Q9KI3hreraB9iOUIA
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function layer_sweep.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function report.
└── 🔨 Created function cells.

##### llama3_8b / chaosnli_alpha
1532 items | H=(1532, 4096) | layer=32 arm=correct
  annot-entropy mean=0.2872 std=0.2182  pcts 0/5/25/50/75/95/100 = 0.000/-0.000/0.098/0.227/0.471/0.680/0.693
  frac within +/-0.1 of median: 0.343  (high => concentrated target, rho is weakly identified)

Stopping app - local entrypoint completed.
=== per-tau verdict (never averaged over tau) ===
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.1260     0.9190   +0.0019    +0.0137  +0.0009 [-0.0187,+0.0198]  0.0146  tie -> W is a size readout (CI spans 0)
 10.00     0.0306     0.7499   -0.0188    +0.0137  +0.0029 [-0.0249,+0.0383]  0.0267  tie -> W is a size readout (CI spans 0)
 25.00     0.0109     0.7271   -0.0255    +0.0137  +0.0074 [-0.0279,+0.0462]  0.0206  tie -> W is a size readout (CI spans 0)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [5.0, 10.0, 25.0]
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-Nfwo9Q9KI3hreraB9iOUIA
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-g44Ego9AmmDvnccfaDLol2
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function layer_sweep.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function report.
└── 🔨 Created function cells.

##### llama3_8b / pavlick_nli
496 items | H=(496, 4096) | layer=32 arm=correct
  annot-entropy mean=0.9193 std=0.4176  pcts 0/5/25/50/75/95/100 = 0.000/0.168/0.612/1.019/1.255/1.463/1.584
  frac within +/-0.1 of median: 0.153  (high => concentrated target, rho is weakly identified)

Stopping app - local entrypoint completed.
=== per-tau verdict (never averaged over tau) ===
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.1402     0.8650   -0.0062    +0.0024  -0.0127 [-0.0441,+0.0391]  0.0183  tie -> W is a size readout (CI spans 0)
 10.00     0.0494     0.5599   +0.0063    +0.0024  -0.0186 [-0.0522,+0.0444]  0.0147  tie -> W is a size readout (CI spans 0)
 25.00     0.0201     0.4209   -0.0082    +0.0024  -0.0167 [-0.0503,+0.0501]  0.0082  tie -> W is a size readout (CI spans 0)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [5.0, 10.0, 25.0]
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-g44Ego9AmmDvnccfaDLol2
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-8xPiZrA1TRtokFFyiPwL8q
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function layer_sweep.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function report.
└── 🔨 Created function cells.

##### mistral_7b / chaosnli
3000 items | H=(3000, 4096) | layer=32 arm=correct
  annot-entropy mean=0.6548 std=0.2281  pcts 0/5/25/50/75/95/100 = 0.000/0.224/0.517/0.685/0.807/1.004/1.098
  frac within +/-0.1 of median: 0.379  (high => concentrated target, rho is weakly identified)

=== per-tau verdict (never averaged over tau) ===
Stopping app - local entrypoint completed.
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.3944     0.9901   -0.2580    -0.2505  +0.0074 [-0.0034,+0.0181]  0.0062  UNINFORMATIVE (W == size_only algebraically)
 10.00     0.1590     0.9582   -0.2350    -0.2505  -0.0156 [-0.0312,-0.0001]  0.0075  W WORSE than density null (pooled CI<0, all seeds agree)
 25.00     0.0544     0.7847   -0.2057    -0.2505  -0.0449 [-0.0640,-0.0259]  0.0231  W WORSE than density null (pooled CI<0, all seeds agree)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [10.0, 25.0]
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-8xPiZrA1TRtokFFyiPwL8q
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-x0MDKXT9QN2UH2wOjwYXNQ
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function layer_sweep.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function cells.
└── 🔨 Created function report.

##### mistral_7b / chaosnli_alpha
1532 items | H=(1532, 4096) | layer=32 arm=correct
  annot-entropy mean=0.2872 std=0.2182  pcts 0/5/25/50/75/95/100 = 0.000/-0.000/0.098/0.227/0.471/0.680/0.693
  frac within +/-0.1 of median: 0.343  (high => concentrated target, rho is weakly identified)

=== per-tau verdict (never averaged over tau) ===
Stopping app - local entrypoint completed.
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.8961     0.9961   +0.0111    +0.0055  +0.0078 [-0.0067,+0.0140]  0.0035  UNINFORMATIVE (W == size_only algebraically)
 10.00     0.3174     0.9725   +0.0206    +0.0055  +0.0184 [-0.0088,+0.0317]  0.0097  tie -> W is a size readout (CI spans 0)
 25.00     0.0404     0.8214   +0.0126    +0.0055  +0.0087 [-0.0191,+0.0307]  0.0063  tie -> W is a size readout (CI spans 0)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [10.0, 25.0]
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-x0MDKXT9QN2UH2wOjwYXNQ
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-ojw57G0r0dQSvJSMXNERpK
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function layer_sweep.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function report.
└── 🔨 Created function cells.

##### mistral_7b / pavlick_nli
496 items | H=(496, 4096) | layer=32 arm=correct
  annot-entropy mean=0.9193 std=0.4176  pcts 0/5/25/50/75/95/100 = 0.000/0.168/0.612/1.019/1.255/1.463/1.584
  frac within +/-0.1 of median: 0.153  (high => concentrated target, rho is weakly identified)

=== per-tau verdict (never averaged over tau) ===
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.7195     0.9880   -0.1997    -0.1622  +0.0375 [+0.0109,+0.0660]  0.0239  UNINFORMATIVE (W == size_only algebraically)
 10.00     0.2096     0.9435   -0.2372    -0.1622  +0.0750 [+0.0268,+0.1260]  0.0322  W BEATS density null (pooled CI>0, all seeds agree)
 25.00     0.0422     0.5909   -0.2080    -0.1622  +0.0458 [-0.0279,+0.1182]  0.0346  tie -> W is a size readout (CI spans 0)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [10.0, 25.0]
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-ojw57G0r0dQSvJSMXNERpK
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-RbodE4ihrolGIEuslmH9u7
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function layer_sweep.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function report.
└── 🔨 Created function cells.

##### qwen2_5_7b / chaosnli
3000 items | H=(3000, 3584) | layer=28 arm=correct
  annot-entropy mean=0.6548 std=0.2281  pcts 0/5/25/50/75/95/100 = 0.000/0.224/0.517/0.685/0.807/1.004/1.098
  frac within +/-0.1 of median: 0.379  (high => concentrated target, rho is weakly identified)

Stopping app - local entrypoint completed.
=== per-tau verdict (never averaged over tau) ===
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.4494     0.9832   -0.2527    -0.2203  +0.0323 [+0.0252,+0.0396]  0.0083  UNINFORMATIVE (W == size_only algebraically)
 10.00     0.1971     0.9382   -0.2605    -0.2203  +0.0402 [+0.0294,+0.0509]  0.0146  W BEATS density null (pooled CI>0, all seeds agree)
 25.00     0.0655     0.7462   -0.2551    -0.2203  +0.0348 [+0.0209,+0.0478]  0.0263  W BEATS density null (pooled CI>0, all seeds agree)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [10.0, 25.0]
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-RbodE4ihrolGIEuslmH9u7
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-33yvi6iSjAfUgHuBTj5J3O
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function layer_sweep.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function report.
└── 🔨 Created function cells.

##### qwen2_5_7b / chaosnli_alpha
1532 items | H=(1532, 3584) | layer=28 arm=correct
  annot-entropy mean=0.2872 std=0.2182  pcts 0/5/25/50/75/95/100 = 0.000/-0.000/0.098/0.227/0.471/0.680/0.693
  frac within +/-0.1 of median: 0.343  (high => concentrated target, rho is weakly identified)

=== per-tau verdict (never averaged over tau) ===
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.7924     0.9919   +0.0147    +0.0172  +0.0052 [-0.0111,+0.0150]  0.0029  UNINFORMATIVE (W == size_only algebraically)
 10.00     0.2343     0.9492   +0.0114    +0.0172  +0.0013 [-0.0192,+0.0202]  0.0057  tie -> W is a size readout (CI spans 0)
 25.00     0.0336     0.7628   -0.0083    +0.0172  +0.0228 [-0.0194,+0.0456]  0.0072  tie -> W is a size readout (CI spans 0)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [10.0, 25.0]
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-33yvi6iSjAfUgHuBTj5J3O
✓ Initialized. View run at 
https://modal.com/apps/cril-nlp/main/ap-2GB506EBqmD4gUeClnCrPK
✓ Created objects.
├── 🔨 Created mount 
│   /media/naim/encrypted_drive/project/uncertainty_hallucination/analyse_hf.py
├── 🔨 Created function dispersion.
├── 🔨 Created function uq_panel.
├── 🔨 Created function uq_panel_correctness.
├── 🔨 Created function ensemble_disagreement.
├── 🔨 Created function decoder_frame.
├── 🔨 Created function report.
├── 🔨 Created function layer_sweep.
└── 🔨 Created function cells.

##### qwen2_5_7b / pavlick_nli
496 items | H=(496, 3584) | layer=28 arm=correct
  annot-entropy mean=0.9193 std=0.4176  pcts 0/5/25/50/75/95/100 = 0.000/0.168/0.612/1.019/1.255/1.463/1.584
  frac within +/-0.1 of median: 0.153  (high => concentrated target, rho is weakly identified)

=== per-tau verdict (never averaged over tau) ===
Stopping app - local entrypoint completed.
   tau  med max_m  r(W,size)    rho(W)  rho(size)     gain      pooled 95% CI  seedSD  status
  5.00     0.4634     0.9730   -0.3767    -0.3029  +0.0738 [+0.0442,+0.1021]  0.0249  W BEATS density null (pooled CI>0, all seeds agree)
 10.00     0.1251     0.8759   -0.3878    -0.3029  +0.0849 [+0.0374,+0.1290]  0.0374  W BEATS density null (pooled CI>0, all seeds agree)
 25.00     0.0332     0.5576   -0.3100    -0.3029  +0.0071 [-0.0632,+0.0794]  0.0160  tie -> W is a size readout (CI spans 0)

informative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): [5.0, 10.0, 25.0]
✓ App completed. View run at 
https://modal.com/apps/cril-nlp/main/ap-2GB506EBqmD4gUeClnCrPK
```

**3 of 9 cells beat the density null** (mistral/pavlick tau=10, qwen/chaosnli tau=10 and 25, qwen/pavlick tau=5 and 10), **1 is worse** (mistral/chaosnli tau=25), 5 tie. Model split: **llama 0/3, mistral 1/3, qwen 2/3.** Seed SD runs 0.015-0.037 against gains of 0.035-0.085 — up to 43% of the effect size. Every significant rho is **negative**: higher width, *lower* human disagreement.

**Structural result (holds regardless of the above):** at tau <= 2, `r(W, size_only)` = 0.994-0.9998. The entire low-tau regime — where all prior work in this project ran — cannot distinguish W from the density null as a matter of algebra, not evidence.

**Multiplicity is uncorrected** across 27 rows.


---

## 7. Multi-baseline UQ panels

Nine metrics in three groups: **mass** (need the frame) credal_width, mass_entropy, one_minus_max; **density** size_only, knn_dist; **token** (need only logits) pred_entropy, pred_maxprob, pred_margin, neg_mean_logprob.

### 7.1 Dispersion task (Spearman vs annotator entropy)

| cell | best metric | \|rho\| | W | knn | size | pred_ent | W vs knn |
|---|---|---|---|---|---|---|---|
| llama3_8b / chaosnli | **size_only** | 0.201 | -0.197 | +0.197 | -0.201 | +0.042 | tie |
| llama3_8b / chaosnli_alpha | **neg_mean_logprob** | 0.148 | -0.026 | +0.051 | +0.014 | +0.089 | tie |
| llama3_8b / pavlick_nli | **pred_maxprob** | 0.191 | +0.005 | +0.058 | -0.019 | +0.058 | tie |
| mistral_7b / chaosnli | **size_only** | 0.267 | -0.236 | +0.249 | -0.267 | +0.080 | W worse |
| mistral_7b / chaosnli_alpha | **neg_mean_logprob** | 0.185 | +0.024 | +0.050 | +0.009 | +0.132 | tie |
| mistral_7b / pavlick_nli | **credal_width** | 0.246 | -0.246 | +0.153 | -0.187 | +0.189 | W better |
| qwen2_5_7b / chaosnli | **credal_width** | 0.272 | -0.272 | +0.249 | -0.248 | -0.095 | W better |
| qwen2_5_7b / chaosnli_alpha | **neg_mean_logprob** | 0.149 | +0.007 | +0.021 | +0.004 | +0.111 | tie |
| qwen2_5_7b / pavlick_nli | **credal_width** | 0.396 | -0.396 | +0.375 | -0.327 | +0.201 | tie |

**The alphaNLI result is the important one.** On all three models the best predictor is `neg_mean_logprob` — plain sequence likelihood, no frame, no clustering — at rho = 0.148/0.185/0.149, while credal width is **0.026 / 0.024 / 0.007**. alphaNLI is also the *clean* cell: its answers are full sentences of comparable length across arms, so it lacks the label-word span-length confound that contaminates ChaosNLI. On the one uncontaminated cell type, W has no signal and the cheapest baseline wins.

Against kNN density W ties in 6/9, wins 2, loses 1. `knn_dist` recovers ~91% of W's correlation on qwen/chaosnli with none of the DS machinery.

`pred_entropy` is positive (correct direction) in 8/9 cells; the one inversion is qwen/chaosnli, so the label confound is narrower than a single cell suggested.

### 7.2 Cross-model ensemble disagreement

Usually filed as 'needs new extraction', but the three models saw identical `item_id`s, so they form a cross-architecture ensemble. Preference = mean logprob(correct) - mean logprob(wrong), z-scored per model before combining (tokenizers differ, raw scales are not comparable).

```
chaosnli: 3000 items shared across 3 models
  llama3_8b    prefers correct on 53.1% of items
  mistral_7b   prefers correct on 52.8% of items
  qwen2_5_7b   prefers correct on 58.2% of items
  all three agree on 92.2% of items
  ensemble_pref_std      rho=+0.0635 CI[+0.0310,+0.0970]
  ensemble_vote_entropy  rho=+0.0593 CI[+0.0203,+0.0994]
  credal_width vs ensemble_pref_std  gain=+0.1371 [+0.0835,+0.1876] W better
```

Real but weak, and **in the correct direction** (more model disagreement -> more human disagreement) — unlike W. The models prefer the majority label on only 53-58% of items, which is itself a comment on ChaosNLI's difficulty.

### 7.3 Correctness-ranking task (marginal theta)


> **Correctness panel** — not found (`correctness_panel.log`); not run or not saved.


This panel did not exist before: `layer_sweep` reported theta for `credal_width` alone, so 15 accuracy cells carried a headline number with no baseline beside it.

#### 7.3.1 Two cells are degenerate BY CONSTRUCTION

`medqa` and `pubmedqa` return `theta = 0.5000` with **zero-width CIs** for every hidden-state and top-k metric. That is not a weak result — the values are identical between arms. Verified directly:

```
llama3_8b__medqa       n_ans_tok med=1   h_correct==h_wrong on 100.0% of items
llama3_8b__pubmedqa    n_ans_tok med=1   h_correct==h_wrong on 100.0% of items
llama3_8b__halueval_qa n_ans_tok med=3   h_correct==h_wrong on   0.0% of items
```

**Mechanism.** The extraction span is `slice(p-1, p+a-1)`. With a single-token answer (a=1) that is the one position `p-1`, whose hidden state and predictive distribution are computed from the PROMPT alone — before the model has seen which answer follows. Both arms share the prompt, so the states are bit-identical and every state-derived metric is exactly tied.

The answers make this concrete: `medqa` uses option letters (`'D'` vs `'C'`), `pubmedqa` uses `'yes'`/`'no'`/`'maybe'`. Both are one token.

Only `neg_mean_logprob` escapes, because it gathers the probability of the ACTUAL answer token: theta = **0.7615** (medqa) and **0.7403** (pubmedqa). So the model does discriminate correct from wrong on these cells — credal width simply cannot see it by construction.

**This generalises beyond the two dead cells.** The first span position is answer-independent for EVERY cell, so it contributes an identical value to both arms and dilutes `span_mean` wherever it is included. `ambigqa_kge2` shows the same signature on its length-matched subset (credal_width 0.4772 -> **0.5000**).

| cell | equal-len arms | theta(W) | theta(W) len-matched | usable? |
|---|---|---|---|---|
| halueval_qa | 1.9% | 0.9096 | 0.5753 | length-confounded |
| medqa | 100% | 0.5000 | 0.5000 | **degenerate** |
| pubmedqa | 100% | 0.5000 | 0.5000 | **degenerate** |
| truthfulqa | 16.8% | 0.4444 | 0.4785 | weak |
| ambigqa_kge2 | 78.8% | 0.4772 | 0.5000 | **degenerate on matched** |

**Consequence for the correctness task.** On 3 of 5 accuracy datasets the hidden-state contrast is exactly or nearly zero; on the fourth (HaluEval) it is 0.91 unmatched and 0.575 matched. There is no accuracy cell where credal width demonstrates length-independent discrimination.

**HaluEval / llama result.** theta(credal_width) = **0.9096** [0.9029, 0.9162] — W beats every baseline. But **98.1% of pairs have arms of different token length**, and on the 56 equal-length pairs W collapses to **0.5753**. Every other metric collapses too (knn 0.150->0.418, neg_mean_logprob 0.151->0.537, pred_entropy 0.325->0.607). A uniform collapse across mass, density AND token metrics is the signature of one shared confound, not nine effects.

The cause is in the data: HaluEval `correct_ans` has median 14 characters, `wrong_ans` median 59 — correct answers are short entities, hallucinations are verbose sentences. With `span_mean` pooling, W is largely reading answer length.

Caveats: the matched subset is 56 items, so 0.5753 is imprecise — not significantly different from 0.5, but not tightly pinned either. And verbosity may be a genuine property of hallucinated text; if so the claim is about detecting verbose answers, not about credal width.


---

## 8. Open issues

| # | issue | why it matters |
|---|---|---|
| 1 | Multiplicity uncorrected | 27 A1 rows tested at 95%; tau rows within a cell are nested and dependent, so the effective count is nearer 9 cell-level tests |
| 2 | `screen.py` verdict blind spot unpatched | the file is frozen by its own header; every verdict it has emitted labels harm as null. `run_tier01.py` adds `verdict_signed` as an adjunct, but the frozen file is unchanged |
| 3 | Answer-length confound is general, not local | it drives the correctness task on HaluEval and contaminates ChaosNLI's label-word arms. `n_ans_tok_*` is now in `index.parquet`, so it can be conditioned on — but nothing except A1d does so yet |
| 4 | A1d run only on llama (5 cells); mistral/qwen not yet run | the degeneracy is structural so it should replicate, but that is untested |
| 5 | Single-token cells need a different extraction to be usable at all | the span must include positions AFTER the answer tokens, or the answer must be embedded in a longer continuation, or those cells carry no hidden-state signal by construction |
| 6 | Tier-A extraction not run | only Tier B (8 probe layers) exists; the layer sweep is therefore coarse |
| 7 | `analyse.py` still on disk with the zero-width-CI bug | nothing uses it, but it can be picked up by mistake — worth deleting |

---

*Generated by `compile_all.py` from the session's JSON dumps and run logs.*
