# ARR disagreement pipeline — results

*Runs 2026-07-31 (§01–§08), 2026-08-02 (§08c), 2026-08-03 (§08d–§11). Models ×
datasets = 9 cells. Signed Spearman throughout; item bootstrap; BH within panel;
every threshold frozen before the numbers were seen (see `../arr_modal.py`
pre-registration block).*

Grid: `K=200, tau∈{5,10,25}, seeds={0,1,2}, scheme=span_mean, final layer
(L32 llama/mistral, L28 qwen2.5-7b), arm=correct`. Target: human annotator
entropy (`h_plugin` primary; `h_norm`/`h_mm` robustness — see §11).

**Bootstrap B is not uniform.** Everything from `analyse_hf.py` (`boot_rho`,
`pooled_gain_ci`, `theta_ci`) uses **B=1000**; everything from `arr_modal.py`
(`_item_bootstrap`: ladder, family, §08d, §08e, §09) uses **B=2000**. Both are
percentile intervals, not BCa. Permutation nulls use 200 draws.

Verdicts at a glance:

| # | experiment | verdict |
|---|---|---|
| 01 | audit_target | passed |
| 02 | **fold (gold-oriented)** | **not detected, low power** — no direction read, but α has ~0 ceiling |
| 03 | ladder | mixed — 3 cells survive BH on the credal rung |
| 04 | **family split** | **cut from abstract** — not constant within Pavlick |
| 05 | sign_stability | stable — significant in 5/9 cells at all three τ |
| 06 | neighbourhood | oracle ceiling exceeds built signals |
| 07 | **components** | **inconclusive** — the deciding cells are confounded by annotator count |
| 08 | **AU/EU decomposition** | ⛔ **not robust to the length convention** — the EU column is an artefact of mean-pooling `p_model`; see §08c/§08d |
| 08e | output-family pooling check | ✅ **robust** — 1 null-to-null flip in 36 pairs; the length channel reaches `p_model`, not `H_ann` |
| 09 | label-factor decomposition | η² must be on **ranks**; llama·pavlick is a masked negative, not a null |
| 10 | paired-gain panel | 3 win / 1 lose / 5 tie — but the exclusion threshold is fragile |
| 11 | target variants | ✅ no verdict changes; `h_norm` rank-identical by construction |
| 12 | method provenance | ✅ prompts, verbalizers, subsets, frame construction recovered from the volume |

Raw tables in `data/*.csv`.

---

## 02 · fold_gold — the fold  ⚠️ not detected, low power

ChaosNLI-α was canonicalised **by crowd majority**, so the naive unfolded
target `p(option 2) ≤ 0.5` everywhere — a monotone reparam of entropy. The real
direction was recovered from the original ChaosNLI release, joined positionally
(verified 1532/1532), oriented on the α-NLI **gold** axis. `lean_gold` spans
[0,1]; the crowd majority-disagrees with gold on **10.6%** of items.

**Why the first pass was under-powered (corrected).** On the **89.2%** of items
where the crowd agrees with gold (`lean_gold > 0.5`), `lean_gold` is a monotone
function of `H`, so `rho(X, lean_gold) = −rho(X, H)` **mechanically, for every
signal**. "All magnitudes match, gain ≈ 0" is what that degenerate branch
predicts — it is consistent with anything and is *not* evidence for the null.
That earlier "cleared" was premature.

**Powered test (matched dispersion).** Isolate direction at fixed dispersion:
`dir_partial = partial rho(X, lean_gold | H)`, with a binned cross-check
(rho within deciles of `|lean − 0.5|`).

| model | max \|dir_partial\| | max \|dir_binned\| | verdict |
|---|--:|--:|---|
| llama3_8b | 0.033 | 0.033 | no direction |
| mistral_7b | 0.027 | 0.028 | no direction |
| qwen2_5_7b | 0.033 | 0.053* | no direction |

*One noisy binned value (qwen·size_only) grazes 0.05; the principled
`dir_partial` is the decider and stays ≤ 0.033 everywhere.

**Verdict: no direction detected — but low power, not "cleared."** α is exactly
the cell where the oracle ceiling is ≈ 0 (§06), so this is "no signal of any
kind here," not a demonstration that direction specifically is absent. The
family ordering on α (OUTPUT > REPR) is on the folded `H` and is unchanged by
unfolding, so it is not *created* by the fold — but the α narrative should say
"not detected, low power," never "the fold is ruled out." Data: `data/fold_gold.csv`.

---

## 04 · family — representation vs output  ⛔ cut from abstract

`delta = mean|rho|(REPR) − mean|rho|(OUTPUT)`. Positive → representation
family leads; negative → output family leads. BH over the 9 rows.

| dataset | llama3_8b | mistral_7b | qwen2_5_7b | within-dataset sign |
|---|--:|--:|--:|---|
| chaosnli | +0.124 ✓ | +0.119 ✓ | +0.061 ✓ | constant (REPR) |
| chaosnli_alpha | −0.059 ✓ | −0.092 ✓ | −0.072 ✓ | constant (OUTPUT) |
| pavlick_nli | −0.130 ✓ | +0.008 (tie, p=.90) | +0.203 ✓ | **flips** |

✓ = clears BH at 0.05 (8/9 significant). Pre-registered rule: survives only
if sign is constant within each dataset. **It isn't** — within Pavlick, Llama
says output-space and Qwen says representation-space, both significant and
opposite. The chaosnli(REPR) vs α(OUTPUT) contrast is clean and real (and the
fold check certifies the α side), but which family wins on Pavlick is
model-dependent, not task-dependent. Per pre-registration: report as
robustness, **cut from the abstract**. Data: `data/family_bh.csv`.

---

## 03 · ladder — does elaboration add anything?  ⚠️ mixed

Rungs: `knn_dist → size_only` (frame) `→ credal_width` (frame+mass).
"Adds nothing" = |partial rho| < 0.05 **and** CI contains 0.

| rung | adds-nothing (of 9) | excluding binary α (of 6) |
|---|--:|--:|
| frame · knn → size | 4 / 9 | 1 / 6 |
| mass · size → credal | 5 / 9 | 2 / 6 |

The "adds nothing" count is **carried by the low-signal α cells** (every
correlation ~0.05 there, so every partial is trivially near zero). In the
real-signal cells credal_width's partial is significantly *negative* in three
(−0.17, −0.12, −0.24). Not a clean redundancy null. Data: `data/ladder_bh.csv`.

**Cell level, both rungs, with intervals** (B=2000; BH over all 18 rows, per
`collect()`; \* = 95% CI excludes 0; **rej** = survives BH at α=0.05):

| cell | rung 1 knn→size | q | BH | rung 2 size→credal | q | BH |
|---|--:|--:|:--:|--:|--:|:--:|
| llama·chaosnli | −0.059 [−0.095,−0.024]\* | .006 | **rej** | −0.039 [−0.077,−0.002]\* | .095 | — |
| mistral·chaosnli | −0.105 [−0.140,−0.067]\* | .000 | **rej** | +0.005 [−0.031,+0.041] | .826 | — |
| qwen·chaosnli | −0.062 [−0.097,−0.027]\* | .004 | **rej** | −0.118 [−0.154,−0.081]\* | .000 | **rej** |
| llama·α | +0.030 [−0.020,+0.080] | .387 | — | −0.044 [−0.092,+0.007] | .171 | — |
| mistral·α | +0.023 [−0.025,+0.073] | .526 | — | +0.044 [−0.004,+0.094] | .138 | — |
| qwen·α | +0.011 [−0.040,+0.062] | .737 | — | +0.007 [−0.038,+0.056] | .791 | — |
| llama·pavlick | +0.057 [−0.026,+0.149] | .290 | — | +0.028 [−0.064,+0.114] | .708 | — |
| mistral·pavlick | −0.109 [−0.196,−0.020]\* | .059 | — | −0.169 [−0.257,−0.083]\* | .000 | **rej** |
| qwen·pavlick | −0.018 [−0.106,+0.064] | .737 | — | −0.237 [−0.318,−0.153]\* | .000 | **rej** |

**The "three cells" claim is right, but only after correction — say so.** On the
credal rung **four** intervals exclude zero; the fourth, llama·chaosnli (−0.039,
q=.095), does **not** survive BH. The three that do are exactly the quoted
−0.169 / −0.118 / −0.237, all at q < 0.001. The same pattern appears on rung 1:
mistral·pavlick's interval excludes zero but q=.059 fails.

**The r = 0.9996 objection, resolved — with the correct numbers (N23).** A
near-identity `corr(credal_width, size_only)` would make a partial of −0.24
(controlling for `size_only`) algebraically impossible. r(W, size_only) is
**τ-dependent**, and the full 27-row grid is now measured
(`data/w_size_collinearity.csv`):

| τ | Pearson range | Spearman range |
|---|---|---|
| 5 | 0.879 – 0.996 | 0.820 – 0.988 |
| **10** | **0.598 – 0.975** | **0.625 – 0.940** |
| 25 | 0.466 – 0.835 | 0.471 – 0.914 |

*An earlier version of this paragraph said "0.85–0.96 at τ=10". That is neither
range.* The figure the argument needs is the **Spearman** one, because
`_partial_rho` rank-transforms before residualising: at τ=10 the worst cell is
qwen·chaosnli at ρ=0.940 and the highest-collinearity ChaosNLI cell
(mistral, ρ=0.892) still leaves **20.5%** residual rank variance — enough for the
observed partials. On Pearson that same cell leaves only 4.9%, so quoting the
Pearson number would not have supported the claim. The conclusion stands; the
number and the metric were both wrong. qwen·pavlick at τ=10 is 0.894 (Pearson),
which is where the previously quoted value came from.

The 0.9996 identity is a **τ→0 artefact** — `compile_all.py::a1` records
r = 0.994–0.9998 at τ ≤ 2 — and τ=5 already reaches 0.992–0.996 on every
mistral and qwen cell.

---

## 05 · sign_stability — is credal_width's sign a property of τ?  ✅ stable

ρ(credal_width, H) across τ = 5 / 10 / 25, with 95% item-bootstrap intervals
(B=1000, `boot_rho`). **All nine cells**; an earlier version of this table
listed seven and silently dropped llama·α and mistral·α.

| cell | τ=5 | τ=10 | τ=25 | verdict |
|---|--:|--:|--:|---|
| llama·chaosnli | −0.197 [−0.232,−0.163]\* | −0.197 [−0.233,−0.164]\* | −0.172 [−0.209,−0.136]\* | stable neg, 3/3 sig |
| mistral·chaosnli | −0.264 [−0.298,−0.229]\* | −0.236 [−0.270,−0.201]\* | −0.188 [−0.224,−0.154]\* | stable neg, 3/3 sig |
| qwen·chaosnli | −0.275 [−0.309,−0.240]\* | −0.272 [−0.307,−0.238]\* | −0.264 [−0.299,−0.231]\* | stable neg, 3/3 sig |
| **llama·α** | **−0.007 [−0.056,+0.044]** | **−0.026 [−0.075,+0.027]** | **−0.029 [−0.077,+0.024]** | **null, 0/3 sig** |
| **mistral·α** | **+0.016 [−0.032,+0.067]** | **+0.024 [−0.022,+0.074]** | **+0.010 [−0.037,+0.058]** | **null, 0/3 sig** |
| qwen·α | +0.001 [−0.050,+0.056] | +0.007 [−0.043,+0.062] | −0.017 [−0.064,+0.037] | flips, 0/3 sig |
| llama·pavlick | −0.001 [−0.086,+0.087] | +0.005 [−0.079,+0.092] | −0.007 [−0.095,+0.084] | flips, 0/3 sig |
| mistral·pavlick | −0.210 [−0.292,−0.118]\* | −0.246 [−0.328,−0.158]\* | −0.209 [−0.296,−0.122]\* | stable neg, 3/3 sig |
| qwen·pavlick | −0.377 [−0.447,−0.298]\* | −0.396 [−0.464,−0.316]\* | −0.307 [−0.386,−0.220]\* | stable neg, 3/3 sig |

\* = 95% CI excludes 0. The significance decision is read off the interval, not
from a separate test.

The two restored rows are **null at every temperature** — no interval excludes
zero — so the omission did not flatter the claim, but the table should never
have been partial. The pattern is now exact: credal_width's sign is stable and
significant in **5/9 cells at all three τ**, and in the remaining 4 cells no τ
reaches significance, so "sign flips" there describes noise around zero rather
than instability. Point estimates cross-checked against
`data/sign_stability.csv` (81 rows, 9 cells × 3 signals × 3 τ): **0 mismatches
of 81**. Intervals in `data/sign_stability_ci.csv`; full nine-signal panel with
intervals at every τ in `data/a1_panel_full.csv` (243 rows).

---

## 06 · neighbourhood — the recoverability ceiling  ✅ gap confirmed

A cross-fitted kNN **oracle** (it sees the target — never a deployable
method) bounds how much annotator entropy is locally smooth in each
representation. ρ and R² at k=25:

| cell | spread | heterogeneity | oracle ρ | oracle R² |
|---|--:|--:|--:|--:|
| llama·chaosnli | 0.19 | 0.06 | 0.41 | 0.15 |
| mistral·chaosnli | 0.23 | 0.06 | 0.44 | 0.19 |
| qwen·chaosnli | 0.27 | 0.13 | **0.58** | **0.38** |
| llama·alpha | ~0 | N/A | 0.07 | <0 |
| mistral·alpha | ~0 | N/A | 0.03 | <0 |
| qwen·alpha | ~0 | N/A | 0.05 | <0 |
| llama·pavlick | 0.04 | 0.33 | 0.39 | 0.16 |
| mistral·pavlick | 0.19 | 0.45 | 0.49 | 0.25 |
| qwen·pavlick | 0.54 | **0.65** | **0.70** | **0.51** |

- **The ceiling clears the built signals** (0.41–0.70 vs ~0.2–0.4) — recoverable
  disagreement is being left unread, most on Qwen.
- **Binary α is a floor, not a ceiling**: even the oracle recovers nothing
  (R² < 0), so the near-zero panel there is the target, not the method.
- **Gold-label heterogeneity is the strongest read on Pavlick** (up to 0.65);
  N/A on α rather than silently filled. (Reason, sharpened: not "gold is
  single-valued" — the deeper problem is that α's two options are
  *item-specific hypotheses with no shared categorical label space*, unlike
  NLI's fixed entail/neutral/contradict, so a neighbourhood gold-entropy is
  undefined. The 89/11 gold-vs-majority split is a crowd *agreement rate*, not
  a shared gold category, so it cannot seed the signal either.)
- Recoverability orders `qwen > mistral > llama` throughout.

`nbhd_oracle_CEILING` is an ORACLE and is never reported as a method. Data:
`data/neighbourhood.csv` (per cell × k∈{10,25,50}).

---

## 07 · components — is credal_width a support-size readout?  ⛔ inconclusive

Decompose the target: `H = log(k) · balance`, where `k` = effective support and
`balance = H/log(k)` ∈ [0,1]. A pure support-size readout tracks `log k`, not
`balance`. Pre-registered: confirmed iff `|rho(W, log k)| > 2·|rho(W, balance)|`
and `rho(W, log k) < 0` in a majority of usable cells.

**The threshold was mis-specified, and the confound inverts the story
(corrected).** `MIN_FRAC=0.02` is a *fraction*, but annotator counts differ 3.5×:
2% is ≥2 votes at ChaosNLI's fixed 100 annotators but **<1 vote at Pavlick's
median 28.5** — no filter at all. The runs below use a **vote threshold (≥2,
identical on every dataset)**, drop unanimous `k=1` items, and control for
annotator count. The hazard is that **k is mechanically increasing in the
annotator count**, which *varies per item* on Pavlick (the `audit_target`
flag) — so `k` is partly a proxy for how many people annotated:

| cell | rho(W,logk) | rho(W,bal) | partial(W,logk\|n_annot) | rho(k,n_annot) | usable? |
|---|--:|--:|--:|--:|:--:|
| llama·chaosnli | −0.153 | −0.089 | −0.153 | — (n fixed) | ✓ |
| mistral·chaosnli | −0.173 | −0.111 | −0.173 | — (n fixed) | ✓ |
| qwen·chaosnli | −0.137 | −0.188 | −0.137 | — (n fixed) | ✓ |
| llama·pavlick | −0.009 | +0.018 | +0.030 | **0.62** | confounded |
| mistral·pavlick | −0.167 | −0.153 | −0.266 | **0.62** | confounded |
| qwen·pavlick | −0.295 | −0.127 | −0.161 | **0.62** | confounded |
| *(3× α)* | *log k has no variance after dropping k=1* | | | | excluded |

On qwen·pavlick, `rho(W, n_annot) = −0.35` directly, and controlling for it
moves `rho(W, log k)` from −0.295 to −0.161. **All three Pavlick cells are
confounded** (`rho(k, n_annot) = 0.62`, count varies), so the test cannot
separate support from annotator count there.

**Verdict: INCONCLUSIVE.** Only the 3 ChaosNLI cells are usable (fixed
annotator count), and 3 < `MIN_CELLS = 5` — underpowered. On those clean cells W
is *not* size-dominated (0/3; on qwen·chaosnli `balance` is the *stronger*
component, −0.188 vs −0.137), which leans against a pure size readout, but this
is too few cells to conclude. The stable negative sign (§05) still holds; the
support-vs-balance question is simply **not settled by this test on this data.**
Data: `data/components.csv`.

*(Correction: the previous write-up said Pavlick's k was "more reliable, which
strengthens the rejection." That was backwards — Pavlick's k is the least
reliable, driven by its varying annotator count. `h_mm` remains the follow-up
robustness target for k's plug-in downward bias.)*

---

## 08 · AU/EU decomposition — answering Baan et al.  ⛔ RETRACTED by §08c

> **Read §08c before anything in §08 or §08b.** Every number below is
> `pool=mean`. Under `pool=sum` — the sequence-probability convention —
> ρ(W, EU) holds its sign in 1 of 5 cells and the gate reverses sign on all
> three α cells. The finding as stated is not reproducible across the pooling
> convention. §08 and §08b are retained unedited as the record of what was run;
> they are **not** a current claim.

The panel correlates signals against `H(p_human)` only — the aleatoric term.
The proper score decomposes it:

    CE(p_human, p_model) = H(p_human) + KL(p_human ‖ p_model)
        total                AU (current target)   EU (never examined)

**Availability (checked with the tokenizer).** The extraction stored a two-arm
(correct/wrong) contrast, not a per-label scan, so `p_model` is computed from
the two scored answer sequences (the project's own notion; §7.3 of RESULTS_ALL).
- **chaosnli_alpha** (binary): correct+wrong *are* the 2 labels — clean.
- **chaosnli** (3-way): **binary-pair** reduction — correct + the one extracted
  distractor, `p_human` restricted to those 2 labels (lossy per item, but it
  reaches the cells where signals correlate with AU). Runs on qwen/mistral
  (llama's label tokens are gated). Computed and uploaded to the volume
  (`arr_cedecomp_*.parquet`).
- **pavlick**: **not runnable** — the extraction used a 2-way entail/contradict
  framing while the human labels are 5-way ordinal (supports don't match). A
  full multi-class `p_model` needs a GPU re-extraction with the original prompt
  (not in this repo); the pos0 top-k can't substitute (its mass is on CoT-
  preamble tokens like `' To'`, not labels).

**Gate — `corr(AU, EU)`** (if high the split separates nothing):
α −0.33/−0.23/−0.18; **chaosnli +0.21 (qwen) / +0.12 (mistral)**. |corr| < 0.4
everywhere → **the split is alive.** `corr(EU, n_annot) ≈ 0` and **h_mm leaves
every EU correlation unchanged to 3 decimals** — so EU is *not* an annotator-
count / plug-in-bias readout here (the reviewer's concern, checked and cleared).

**Two-target panel — clean (representation) signals**, ρ(·, AU) → ρ(·, EU):

| signal | α (avg) | chaosnli·qwen | chaosnli·mistral | reading |
|---|--:|--:|--:|---|
| **knn_dist** | +0.05 → **+0.14** | +0.18 → **+0.11** | +0.16 → **+0.53** | raw density → *proper* EU proxy (**right sign**) |
| **credal_width** | −0.00 → **−0.08** | −0.22 → **−0.12** | −0.16 → **−0.55** | **INVERTED vs epistemic** |
| size_only | +0.01 → −0.06 | −0.19 → −0.10 | −0.18 → −0.42 | inverted, like W |

**The finding.** Across α and chaosnli, 2–3 models: **raw density (`knn_dist`)
correlates with epistemic uncertainty in the expected positive direction, while
the credal construction (`credal_width`, `size_only`) is inverted** — negative
on EU, up to −0.55 on mistral. The frame+mass machinery *flips the sign that raw
density gets right*. This recontextualises credal_width's whole story: it is not
merely a weak aleatoric predictor (§07 sign result) — it is an **inverted
epistemic predictor**. KL is non-negative and directional, so the sign is a
statement about forecast quality, not an orientation convention.

**Set aside:** the output family (`neg_mean_logprob` up to +0.56 on EU) is
mechanically **circular** — those signals and EU are both functions of `p_model`
— so their EU column is not evidence. Flagged `circular=True` in the data.

**Limits.** Binary / binary-pair only (full multi-class needs the GPU
re-extraction above); no llama on chaosnli (gated tokens); magnitudes model-
dependent. ~~But the *sign* is consistent everywhere it is clean.~~ **That last
clause is false — see §08c.** The sign is consistent across cells only within
`pool=mean`; it is not consistent across the pooling convention, and the
convention was never disclosed. Data: `data/ce_decomp.csv` (mean),
`data/ce_decomp_sum.csv` (sum), `data/ce_decomp_ci.csv` (intervals for both);
pre-registration + caveats in `../arr/decompose_target.py`.

### 08b · Mechanism — density AND length, no residual  ⛔ `pool=mean` only

The double inversion is real. What mediates it, measured as **proportional
mediation** (not a threshold — `DENSITY_NULL=0.05` was the wrong shape: an
absolute bound on a residual passes any small-effect cell regardless of
mechanism, the §03 α-inflation trap again). Data: `data/mechanism.csv`.

**Product decomposition** (positive evidence, not a null): `ρ(W,EU) ≈ ρ(W,knn) ×
ρ(knn,EU)` — the density path recovers ~85–91% of the total:

| cell | ρ(W,EU) | ρ(W,knn)×ρ(knn,EU) | residual\|knn (95% CI) |
|---|--:|--:|--:|
| llama·α | −0.110 | −0.091 | −0.030 [−0.079, +0.020] |
| mistral·α | −0.075 | −0.064 | −0.012 [−0.064, +0.038] |
| qwen·α | −0.063 | −0.086 | +0.027 [−0.023, +0.078] |
| qwen·chaosnli | −0.117 | −0.098 | −0.045 [−0.080, −0.012] |
| mistral·chaosnli | −0.545 | −0.495 | −0.173 [−0.210, −0.134] |

The residual scales with the total at a roughly constant fraction (~30%), across
an **8× range of effect sizes** — one consistent proportion, not "4 pass / 1
fail." `ρ(W, knn_dist) = −0.76 … −0.94` (W is essentially *inverted density*);
`knn_dist` is a proper **positive** EU proxy. So W is EU-inverted *because* it is
density-inverted.

**But length is a co-driver, and on chaosnli it is huge (blocker 1, resolved).**
The AAAI answer-length confound lands: on chaosnli `ρ(W, n_ans_tok) = +0.85`,
`ρ(knn, n_ans_tok) = −0.84` — W, density and answer length are nearly
collinear (W's `span_mean` pools over the answer span; longer answer → more
positions). Consequences:

| cell | ρ(W,EU) | \|knn | \|n_ans_tok | \|**knn AND length** |
|---|--:|--:|--:|--:|
| qwen·chaosnli | −0.117 | −0.045 | **+0.099** (flips!) | **+0.031** |
| mistral·chaosnli | −0.545 | −0.173 | −0.187 | **+0.018** |
| α (avg) | −0.083 | ~−0.005 | ~−0.036 | ~−0.000 |

Controlling **both** density and length leaves **≈0 residual everywhere**. On
qwen·chaosnli the sign even *flips* under length alone. So the −0.173
"beyond-density" residual on mistral is **length, not a non-redundant component**
— the stronger Claim-C reading does *not* survive. The honest statement: W's
inversion against forecast error is jointly explained by local density and
answer length, which its own construction (`span_mean` over the answer) makes
nearly collinear with it; nothing remains after both.

**Fragility (both real).** The conditional estimates are differences of
near-equal quantities (`ρ(W,knn)=−0.94`), so a 0.02 shift in `ρ(W,knn)` moves the
partial ~0.04 — hence the bootstrap CIs above, not point estimates. And
controlling a variable correlated at −0.76…−0.94 leaves 6–42% residual variance,
so the test has least power exactly where a residual would matter: "no residual"
is weaker than it sounds.

**Distractor selection (blocker 2, resolved).** On chaosnli the arms are
deterministic: `correct` = the **top-voted** human label (rank 0, 100% of
items), `wrong` = the **second**-voted (rank 1, 100%), never the least. So the
binary-pair is always the top-2 labels and the rule is constant (no
item-dependent selection confound), but restricted AU is the entropy of the
*confusable pair* — a Setup sentence, whatever else.

**Q1 `corr(AU,EU)` ∈ [−0.33, +0.21]**, sign differs by dataset → "inverted
against both" is two facts, not one. **Q3 `ρ(W,EU|AU)`** stays negative
everywhere → the epistemic inversion is separate from the aleatoric one.

**Two-by-two — do not claim it cleanly.** `pred_entropy` is positive on both
targets on 4/5 cells but **−0.101 on AU for qwen·chaosnli**, and its EU column is
**circular** (shares `p_model` with EU). Representation (W) inverted on EU;
output positive on AU on most cells; the output-EU cell is not evidence. Not an
airtight 2×2 — if the contrast is kept, the circularity belongs in the main text.

**Net.** The paper is still not exclusion-only, but the mechanism is *density +
length*, not density alone, and there is **no residual beyond them** — so W is a
typicality/length readout inverted against forecast error, and the frame+mass
machinery adds nothing beyond distance and length (consistent with §03). `h_mm`
leaves EU unchanged, so it is not an annotator-count artefact.

*Superseded by §08c: every number in §08 and §08b is `pool=mean`, and the EU
column does not survive the other pooling convention.*

### 08c · Length convention — the EU column does not survive it  ⛔ run 2026-08-02

`ce_decomp` builds `p_model` as a two-way softmax over the arms' realised
logprobs, aggregated by `pool` (`arr_modal.py:1021`):

    lcs = lc.mean() if pool == "mean" else lc.sum()

A **mean** logprob is a per-token rate; the probability of an answer *sequence*
is the **sum**. Every persisted row in §08/§08b was `pool=mean`; the `sum`
variant had never been run. It has now been run on all five runnable cells
(`data/ce_decomp_sum.csv`, 45 rows; distinct filenames, nothing overwritten).

`rho_AU` is **identical to three decimals in all 45 rows** — AU = H(p_human)
does not involve `p_model`, so the two runs differ in exactly the place they
should. Everything below is the EU column alone.

| cell | ρ(W, EU) mean → sum | ρ(knn, EU) mean → sum | gate ρ(AU,EU) mean → sum |
|---|--:|--:|--:|
| mistral·chaosnli | −0.545 → **−0.123** | +0.527 → **+0.040** | +0.123 → +0.134 |
| qwen·chaosnli | −0.117 → **−0.362** | +0.108 → **+0.314** | +0.211 → +0.360 |
| llama·α | −0.110 → **+0.056** | +0.120 → **−0.018** | −0.328 → **+0.347** |
| mistral·α | −0.075 → **+0.044** | +0.155 → **−0.046** | −0.234 → **+0.403** |
| qwen·α | −0.063 → **+0.034** | +0.154 → **−0.033** | −0.178 → **+0.363** |

**The double inversion does not replicate.** Under `sum`, ρ(W, EU) keeps its
negative sign in **1 of 5 cells** (qwen·chaosnli, where it more than triples).
The headline "up to −0.55" on mistral·chaosnli collapses by a factor of 4.4, and
all three α cells **reverse sign**. The companion claim — raw density as a
*proper, positively-signed* EU proxy — reverses on the same three α cells and
collapses from +0.527 to +0.040 on mistral·chaosnli. `size_only`, `mass_entropy`
and `one_minus_max` flip on 2–3 cells each. Under `EU_MARGIN=0.05`, `tracks_eu`
now fires on qwen·chaosnli only.

The **gate itself changes sign** on all three α cells (−0.18…−0.33 → +0.35…+0.40),
and mistral·α at +0.403 now sits just outside the "|corr| < 0.4 everywhere"
sentence in §08 (the code's own gate is 0.7, so it still passes that).

**Why, and what is actually identified.** The ChaosNLI label words tokenise to
different lengths — `n_ans_tok_correct ∈ {1,3}` on llama/qwen, `{1,2,3}` on
mistral — so `mean` systematically inflates the short label and `sum`
systematically penalises the long one. Both conventions carry a length channel,
in *opposite* directions. And ρ(W, n_ans_tok) = +0.85 (§08b), i.e. W is close to
a readout of which label word was scored. So the shared length channel between
`p_model` and W is sufficient to manufacture ρ(W, EU) under either convention,
with the sign set by the convention rather than by the model's epistemic state.

**Verdict: EU is not identified independently of the length convention on these
cells, so §08's finding cannot be claimed.** This is not "mean was wrong and sum
is right" — neither pooling is neutral while the label verbalizers differ in
length. Closing it needs a `p_model` that is a genuine distribution over the
label set (a per-label scan, i.e. the GPU re-extraction §08 already flags as
out of scope), not a two-arm contrast under either aggregation. Both runs are
retained rather than one being rewritten away. Data: `data/ce_decomp.csv`
(mean), `data/ce_decomp_sum.csv` (sum).

### 08d · Intervals for every AU/EU quantity  ✅ run 2026-08-03

`mechanism.csv` carried a 95% interval for exactly one column
(`partial_W_EU_knn`). Everything else — the gate, ρ(W,EU), and the partials
behind the −0.06…−0.54 range — was a bare point estimate. All of it is now
item-bootstrapped (B=2000, `_item_bootstrap`, percentile-inverted two-sided p),
for **both** pooling conventions. Data: `data/ce_decomp_ci.csv` (90 rows).

**Drift guard.** `ce_decomp` does not expose its AU/EU arrays, so this is a
second implementation of the same arithmetic — the failure mode ARR.md warns
about. Every `pool=mean` point estimate was recomputed and compared against
`mechanism.csv`: **max |Δ| = 0.0005 across all 10 checked quantities, zero
flags.** The intervals therefore attach to the published numbers.

ρ(AU, EU) — the gate — and ρ(W, EU), which is the −0.06…−0.54 range:

| cell | gate ρ(AU,EU), mean | gate, sum | ρ(W,EU), mean | ρ(W,EU), sum |
|---|--:|--:|--:|--:|
| llama·α | −0.328 [−0.371,−0.282] | +0.347 [+0.301,+0.397] | −0.110 [−0.159,−0.064] | +0.056 [+0.005,+0.106] |
| mistral·α | −0.234 [−0.281,−0.185] | +0.403 [+0.356,+0.450] | −0.075 [−0.126,−0.024] | +0.044 [−0.006,+0.095] n.s. |
| qwen·α | −0.178 [−0.225,−0.130] | +0.363 [+0.313,+0.414] | −0.063 [−0.114,−0.015] | +0.034 [−0.018,+0.086] n.s. |
| mistral·chaosnli | +0.123 [+0.085,+0.165] | +0.134 [+0.094,+0.173] | −0.545 [−0.569,−0.521] | −0.123 [−0.158,−0.089] |
| qwen·chaosnli | +0.211 [+0.174,+0.245] | +0.360 [+0.327,+0.390] | −0.117 [−0.152,−0.080] | −0.362 [−0.393,−0.327] |

**The −0.06…−0.54 range is supported under `mean`** — all five intervals
exclude zero, and the two endpoints are −0.063 [−0.114,−0.015] (qwen·α) and
−0.545 [−0.569,−0.521] (mistral·chaosnli). It does **not** survive `sum`: two of
five lose significance and three change sign (§08c).

Partials, `pool=mean` (the published column) — 95% CI, n.s. = interval contains 0:

| cell | ρ(W,EU\|AU) | ρ(W,EU\|knn) | ρ(W,EU\|len) | ρ(W,EU\|knn **and** len) |
|---|--:|--:|--:|--:|
| llama·α | −0.126 [−0.174,−0.078] | −0.030 [−0.079,+0.020] n.s. | −0.046 [−0.095,+0.001] n.s. | −0.015 [−0.063,+0.033] n.s. |
| mistral·α | −0.072 [−0.122,−0.022] | −0.012 [−0.064,+0.038] n.s. | −0.045 [−0.095,+0.005] n.s. | −0.013 [−0.064,+0.036] n.s. |
| qwen·α | −0.063 [−0.114,−0.014] | +0.027 [−0.023,+0.078] n.s. | −0.018 [−0.070,+0.033] n.s. | +0.029 [−0.021,+0.079] n.s. |
| mistral·chaosnli | −0.537 [−0.559,−0.512] | −0.173 [−0.210,−0.134] | −0.187 [−0.229,−0.145] | +0.018 [−0.018,+0.056] n.s. |
| qwen·chaosnli | −0.073 [−0.109,−0.035] | −0.045 [−0.080,−0.012] | +0.099 [+0.061,+0.139] | +0.031 [−0.007,+0.069] n.s. |

**§08b's "≈0 residual after density and length" now has intervals, and under
`mean` it holds**: the joint partial contains zero in **5/5 cells**. That claim
is the one part of §08 the intervals strengthen rather than undercut.

Under `sum` it fails on both ChaosNLI cells: mistral +0.044 [+0.010,+0.080] and
qwen **−0.324 [−0.361,−0.285]**, both excluding zero. The length partial alone
swings from +0.099 [+0.061,+0.139] to −0.481 [−0.515,−0.446] on qwen·chaosnli —
a 0.58 move from one pooling convention. Sign-flipping partials of that size are
what a shared length channel between W and `p_model` looks like, and they are
the quantitative form of §08c's verdict.

**Net for §08b.** "No residual beyond density and length" is interval-supported
on the published `mean` numbers and only there. Report it as conditional on the
convention, or drop it with the rest of §08.

### 08e · The lever does not reach the output family  ✅ run 2026-08-03

**Premise correction first.** The four output signals do **not** pass through
`ce_decomp`'s two-way softmax. `uq_panel` builds them from the *correct* arm's
per-position top-k distributions and averages over answer positions
(`analyse_hf.py:366-370`), so `ce_decomp`'s `pool` cannot move them. The
circularity flagged in §08c is specific to EU.

The same lever does exist one level down, though: that per-position `mean_t`
*is* the length normalisation, and `sum_t = mean_t × T`. Both aggregations were
run for all four signals on all **nine** cells against `h_plugin`, with item
bootstrap (B=2000). Drift guard: the `mean` arm reproduces
`data/a1_panel_full.csv` on every cell, which also proves the keep-set aligns
without reloading states. Data: `data/output_pool_sensitivity.csv` (72 rows).

**Result: 1 sign change in 36 signal × cell pairs, and it is a null-to-null
move.** llama·chaosnli `pred_maxprob` +0.007 [−0.028,+0.042] → −0.032
[−0.069,+0.003]; neither interval excludes zero.

`pred_entropy`, the abstract's claim, is unmoved:

| cell | mean | sum | | cell | mean | sum |
|---|--:|--:|---|---|--:|--:|
| llama·chaosnli | +0.042 | +0.032 | | llama·α | +0.090 | +0.075 |
| mistral·chaosnli | +0.080 | +0.102 | | mistral·α | +0.132 | +0.097 |
| qwen·chaosnli | −0.095 | −0.179 | | qwen·α | +0.111 | +0.091 |
| llama·pavlick | +0.058 | +0.058 | | mistral·pavlick | +0.189 | +0.183 |
| qwen·pavlick | +0.201 | +0.201 | | | | |

Positive in **8/9 under both conventions**, the negative cell being
qwen·chaosnli either way. The two Pavlick cells with constant `n_ans_tok` are
*identical* to all digits, exactly as `sum = mean × T` predicts when T is
constant — an internal check that the lever is doing what it should.

**Why the output family is robust where EU is not.** ρ(`n_ans_tok`, `H_ann`) is
tiny in every cell: −0.024 (llama/qwen·chaosnli), −0.043 (mistral·chaosnli),
+0.021…+0.032 (α), −0.069 (mistral·pavlick), undefined on the two constant-length
Pavlick cells. **Answer length carries almost no information about annotator
entropy.** Re-weighting by T therefore barely moves a correlation against
`H_ann`, whereas EU's `p_model` is a softmax *between two arms* whose relative
weighting the convention changes outright.

That is the single statement unifying §08c with `ablations.md` §1: **the length
channel is a channel into `p_model`, and hence into EU — not into `H_ann`.**
It is why W's inversion against annotator entropy survived length adjustment
and first-token re-pooling, while its inversion against forecast divergence did
not.

---

## 09 · Label-factor decomposition (N16)  ✅ run 2026-08-03

The verbalizer control's η² was on **raw** W while the partial residualises
**ranks**. Confirmed: mistral·chaosnli η²(raw W) = **0.2598**, which is the 0.26
in the table. Recomputed on ranks, with the full identity

    partial = (rho_WH − r_between) / sqrt((1 − eta2_W)(1 − eta2_H))

verified against direct dummy residualisation to 1e-9 in every usable cell.
Label factor = `argmax(soft_labels)`. Data: `data/eta2_label.csv`.

| cell | η²(W) raw | η²(W) **ranked** | η²(H) | ρ(W,H) | r_between | partial [95% CI] |
|---|--:|--:|--:|--:|--:|--:|
| llama·chaosnli | 0.632 | **0.736** | 0.003 | −0.197 | −0.020 | −0.345 [−0.377,−0.312] |
| mistral·chaosnli | 0.260 | **0.720** | 0.003 | −0.236 | −0.027 | −0.396 [−0.428,−0.362] |
| qwen·chaosnli | 0.574 | **0.748** | 0.003 | −0.272 | −0.030 | −0.482 [−0.510,−0.454] |
| llama·pavlick | 0.625 | **0.499** | 0.271 | +0.005 | +0.089 | −0.139 [−0.217,−0.055] |
| mistral·pavlick | 0.146 | **0.242** | 0.271 | −0.246 | −0.098 | −0.199 [−0.277,−0.116] |
| qwen·pavlick | 0.155 | **0.248** | 0.271 | −0.396 | −0.083 | −0.423 [−0.490,−0.353] |
| *3× α* | *N/A — majority label is a single level (canonicalised target); η² undefined* | | | | | |

**Mistral reconciles exactly on ranks.** Naive with the raw value gives
−0.236/√(1−0.260) = −0.274; the exact identity with the ranked value gives
(−0.236 − (−0.027))/√((1−0.720)(1−0.003)) = **−0.3955** against a reported
−0.398. The gap was scale, nothing else.

**Rank, and say so in the caption.** Raw η²(W) ranges 0.15–0.63 *across models on
the same dataset*; ranked it is 0.72–0.75 on all three ChaosNLI cells. The raw
figure is tracking W's distributional shape, which differs by model; the ranked
figure is the quantity the partial actually removes.

Three further things the decomposition gives the caption for free:

- **η²(H) is identical across models within a dataset** (0.0032 ChaosNLI, 0.2714
  Pavlick) — direct confirmation that the target-side quantity is model-invariant,
  which is what the caption asserts.
- **On ChaosNLI both correction terms are negligible** (η²(H)=0.003,
  r_between≈−0.03), so ρ/√(1−η²_W) is a fine approximation *there*. On Pavlick
  neither is: η²(H)=0.271 and r_between reaches +0.089, so only the full identity
  reconciles.
- **llama·pavlick is a finding, not a null.** ρ(W,H) = +0.005 — the cell §05
  records as "flips, all ≈0" — but the within-label partial is
  **−0.139 [−0.217,−0.055]**. The label factor was masking a real negative
  association. Same direction as every other non-null cell.

The exact scored label (recoverable for mistral/qwen·chaosnli via
`LABEL_FIRST_TOKEN`) agrees with the majority label on 99.7% of items and moves
the partial by ≤0.006 (−0.3975 vs −0.3955; −0.4876 vs −0.4821), so the majority
proxy is sound wherever the scored label cannot be recovered.

---

## 10 · Paired-gain panel and its exclusion rule (N24)  ⚠️ threshold-fragile

The 27-row panel with gains and CIs is persisted in `../ARR_run_summary.md`;
it is **parsed, never recomputed** (a second implementation would drift —
ARR.md). Machine-readable copy with the collinearity column joined on:
`data/paired_gain_panel.csv`.

**The exclusion rule was never in this document.** It is in
`compile_all.py:305-307`: rows are suppressed as UNINFORMATIVE where
**`max-mass > 0.90` or `r(W, size_only) > 0.98`** — there W ≡ size_only
algebraically and the comparison carries no information regardless of
significance. That is the numeric definition of "effectively one-hot".

Which arm binds: the max-mass arm **never fires**. The highest mean max-mass in
any cell is 0.797 (mistral·α, τ=5) and the highest per-item fraction above 0.99
is 13.3% (mistral·pavlick, τ=5). Every suppression is driven by the
`r > 0.98` criterion, and all five sit at τ=5.

**All 27 rows, with the τ column and the collinearity that drives the rule.**
Gains and CIs are the persisted values (B=1000, `pooled_gain_ci`, |ρ| gain of
`credal_width` over `size_only`); `r(W, size_only)` is joined from §N23.

| cell | τ | gain | 95% CI | r(W,size_only) | verdict |
|---|--:|--:|--:|--:|---|
| llama·chaosnli | 5 | −0.0022 | [−0.0162,+0.0121] | 0.9669 | tie |
| llama·chaosnli | 10 | −0.0038 | [−0.0194,+0.0120] | 0.8521 | tie |
| llama·chaosnli | 25 | −0.0135 | [−0.0336,+0.0060] | 0.5962 | tie |
| mistral·chaosnli | 5 | +0.0074 | [−0.0034,+0.0181] | 0.9948 | *uninf.* |
| mistral·chaosnli | 10 | −0.0156 | [−0.0312,−0.0001] | 0.9752 | **LOSE** |
| mistral·chaosnli | 25 | −0.0449 | [−0.0640,−0.0259] | 0.8209 | **LOSE** |
| qwen·chaosnli | 5 | +0.0323 | [+0.0252,+0.0396] | 0.9904 | *uninf.* |
| qwen·chaosnli | 10 | +0.0402 | [+0.0294,+0.0509] | 0.9610 | **WIN** |
| qwen·chaosnli | 25 | +0.0348 | [+0.0209,+0.0478] | 0.8332 | **WIN** |
| llama·α | 5 | +0.0009 | [−0.0187,+0.0198] | 0.9275 | tie |
| llama·α | 10 | +0.0029 | [−0.0249,+0.0383] | 0.7503 | tie |
| llama·α | 25 | +0.0074 | [−0.0279,+0.0462] | 0.7290 | tie |
| mistral·α | 5 | +0.0078 | [−0.0067,+0.0140] | 0.9964 | *uninf.* |
| mistral·α | 10 | +0.0184 | [−0.0088,+0.0317] | 0.9743 | tie |
| mistral·α | 25 | +0.0087 | [−0.0191,+0.0307] | 0.8346 | tie |
| qwen·α | 5 | +0.0052 | [−0.0111,+0.0150] | 0.9929 | *uninf.* |
| qwen·α | 10 | +0.0013 | [−0.0192,+0.0202] | 0.9539 | tie |
| qwen·α | 25 | +0.0228 | [−0.0194,+0.0456] | 0.7639 | tie |
| llama·pavlick | 5 | −0.0127 | [−0.0441,+0.0391] | 0.8792 | tie |
| llama·pavlick | 10 | −0.0186 | [−0.0522,+0.0444] | 0.5977 | tie |
| llama·pavlick | 25 | −0.0167 | [−0.0503,+0.0501] | 0.4656 | tie |
| mistral·pavlick | 5 | +0.0375 | [+0.0109,+0.0660] | 0.9922 | *uninf.* |
| mistral·pavlick | 10 | +0.0750 | [+0.0268,+0.1260] | 0.9574 | **WIN** |
| mistral·pavlick | 25 | +0.0458 | [−0.0279,+0.1182] | 0.6216 | tie |
| qwen·pavlick | 5 | +0.0738 | [+0.0442,+0.1021] | 0.9804 ⚠️ | **WIN** |
| qwen·pavlick | 10 | +0.0849 | [+0.0374,+0.1290] | 0.8942 | **WIN** |
| qwen·pavlick | 25 | +0.0071 | [−0.0632,+0.0794] | 0.5990 | tie |

⚠️ marks the one row that meets the suppression rule but was kept — see below.
Note two rows have gains whose CI excludes zero yet are suppressed as
uninformative (mistral·pavlick τ=5, qwen·chaosnli τ=5): that is the rule working
as intended — at r > 0.98 a significant gain is still uninformative, because the
two signals are algebraically the same thing.

**Reconciliation, 27 rows → 3/1/5:**

| row-level | count |
|---|--:|
| WIN | 5 (mistral·pavlick τ10; qwen·chaosnli τ10, τ25; qwen·pavlick τ5, τ10) |
| LOSE | 2 (mistral·chaosnli τ10, τ25) |
| tie | 15 |
| uninformative | 5 |
| **total** | **27** |

Cell-level (win = beats the null at ≥1 informative τ): **3 WIN**
(mistral·pavlick, qwen·chaosnli, qwen·pavlick) / **1 LOSE** (mistral·chaosnli)
/ **5 tie**. The 22 informative rows are 27 − 5.

⚠️ **The 22 count is fragile, and one WIN row sits on the wrong side of the
threshold.** Recomputing r(W, size_only) from the dumps puts **qwen·pavlick
τ=5 at 0.9804** — above the 0.98 cutoff — yet it was retained and counted as
a WIN. Six rows meet the rule, five were suppressed. Consequences:

- the informative-row count is **21 or 22** depending on a margin of 0.0004;
- the **cell-level verdict is unaffected**: qwen·pavlick still WINs on τ=10
  (+0.0849 [+0.0374,+0.1290]), so 3/1/5 holds either way.

The recomputation uses seed-averaged signals from the per-item dump while `a1`
computed per-seed, so the discrepancy may be an artefact of that difference
rather than an error. Either way a threshold this tight should not be reported
bare — state the rule, the margin, and that the headline does not turn on it.

Also noted: `a1`'s prose says "1 is worse (mistral/chaosnli tau=25)" while its
own table marks mistral·chaosnli LOSE at **both** τ=10 and τ=25.

---

## 11 · Target variants (N25)  ✅ no verdict changes

**`h_norm` cannot change anything, by construction.** `h_norm = H / log(support)`
with `support` constant within a dataset, so it is a strictly monotone rescale
and every Spearman is rank-identical to `h_plugin`. Measured: max |Δρ| across
all 81 signal × cell pairs is **7.7×10⁻⁶**, and exactly **0** on six of nine
cells; the residue on the three ChaosNLI cells is floating-point tie-merging
from dividing by log 3, not a difference. Report the invariance as an argument;
do not table 81 duplicate numbers.

**`h_mm` is the Miller-Madow estimator**, computed per item in
`arr_modal.py:274-276`:

    h_mm = h_plugin + (k_obs - 1) / (2N)

with `k_obs` the number of labels receiving nonzero annotator mass and `N` the
per-item annotator count. Panel recomputed against it
(`data/target_variants_panel.csv`, 81 rows):

| dataset | max \|Δρ\| over 9 signals × 3 models | why |
|---|--:|---|
| chaosnli_alpha | **0.0000** | N=100 fixed, k_obs fixed at 2 after canonicalisation → the correction is a constant |
| chaosnli | 0.0005 – 0.0008 | N=100 fixed; only k_obs varies |
| pavlick_nli | 0.0087 – 0.0091 | **N varies per item** (median 28.5) → the correction varies per item |

`credal_width` specifically: largest moves are mistral·pavlick −0.2459 →
−0.2547 and qwen·pavlick −0.3958 → −0.3885. **No sign changes, no verdict
changes, nothing near `NULL_THRESHOLD`.**

Pavlick is the only dataset where the correction moves anything, and it moves it
by ≤0.009 — which is the quantitative answer to §01's warning that Pavlick's
varying annotator count makes the plug-in bias an item-level confound *in the
target*. The confound is real and its magnitude is under 0.01.

**Neighbourhood at every k** (`data/neighbourhood.csv`, 27 rows) is already
persisted per cell × k ∈ {10, 25, 50} with ρ for spread, heterogeneity and the
oracle plus oracle R² — formatting only, no run needed. Cross-fitting details
for the oracle are in §06 and `arr_modal.py:706-721`: 5 folds assigned by
`rng.integers(0, 5, n)` at seed 0, neighbours fit on training folds only, R²
computed on held-out predictions as 1 − SS_res/SS_tot.

---

## 01 · audit_target

Per-dataset target audit (Table 1 material). All three entropy variants
(`h_plugin`/`h_norm`/`h_mm`) emitted. Data: `data/audit_target.csv`.

| dataset | n | support | annot/item | mean H (plugin) |
|---|--:|--:|--:|--:|
| chaosnli | 3000 | 3 | 100 (fixed) | 0.655 |
| chaosnli_alpha | 1532 | 2 | 100 (fixed) | 0.287 |
| pavlick_nli | 496 | 5 | **28.5 (varies)** | 0.919 |

**Caveat (new):** Pavlick's annotator count **varies per item** (median 28.5,
not fixed). The plug-in entropy bias then varies per item too, making it an
item-level confound *in the target itself* — not just a cross-dataset
nuisance. This is exactly the case `audit_target` warns turns Setup "from a
sentence to a paragraph." chaosnli / chaosnli_alpha are fixed at 100/item and
are unaffected.

Quantitative follow-up in §11: the Miller-Madow correction moves ρ by ≤0.009 on
Pavlick and ≤0.0008 elsewhere, so the confound is real but small.

---

## 12 · Method provenance (N01–N03, N07)  ✅ recovered 2026-08-04

Recovered from `/vol/source/*.json` + `index.parquet` + `meta.json`. The OVA_ARR
registry that *built* these is missing, but its outputs — rendered prompts,
label strings, ids, categories — are all on the volume.

### 12.1 Data (N01)

| dataset | n | provenance | subsets | licence | dropped |
|---|--:|---|---|---|--:|
| chaosnli | 3000 | `OVA_ARR data.registry.load_dataset('chaosnli')` | ChaosNLI-mnli_m 1599 + ChaosNLI-snli 1401 | CC-BY-NC-4.0 | 0 |
| chaosnli_alpha | 1532 | **local file** `chaosNLI_v1.0/chaosNLI_alphanli.jsonl` | ChaosNLI-alphanli 1532 | CC-BY-NC-4.0 | 0 |
| pavlick_nli | 496 | `OVA_ARR data.registry.load_dataset('pavlick_nli')` | snli 100, rte2 100, dnc 100, mnli 99, joci 97 | CC-BY-4.0 | 0 |

- **Release version: ChaosNLI v1.0**, explicit in the α provenance path.
- **No filtering at extraction.** `n(index.parquet) == n(source)` for all three;
  `item_id` is contiguous 0…n−1. Whatever selection occurred happened *inside*
  the registry, before the JSON was written.
- **ChaosNLI is a subset and must be described as one.** ChaosNLI v1.0 ships
  1,599 MNLI-m + 1,514 SNLI; the cell has 1,599 + **1,401**, so **113 SNLI items
  are absent**. Selection rule not recoverable. Pavlick is 496 of the released
  ~500.
- α is the **complete** αNLI split, and the only cell not built through the
  registry — it came from an absolute local path on a machine that no longer
  holds it.
- `example_id` is a unique 16-hex digest per item, so every item rejoins the
  release.

### 12.2 Prompts and verbalizers (N02)

No system prompt, no few-shot exemplars, no chat wrapper — raw completion
prompts (`n_ctx` median 53 / 38 / 45 tokens).

    chaosnli        Premise: {premise}
                    Hypothesis: {hypothesis}
                    What is the NLI relation? Answer with entailment, neutral, or contradiction.

    pavlick_nli     Premise: {premise}
                    Hypothesis: {hypothesis}
                    Does the premise entail the hypothesis? Answer with entailment or contradiction.

    chaosnli_alpha  Observation 1: {obs1}
                    Observation 2: {obs2}
                    Which hypothesis better explains the observations?

| dataset | verbalizers | scored-arm distribution |
|---|---|---|
| chaosnli | entailment / neutral / contradiction | 1127 / 1336 / 537 |
| pavlick_nli | entailment / contradiction (**2-way**) | 276 / 220 |
| chaosnli_alpha | **none — free text** | 1532 unique `correct_ans`, 1532 unique `wrong_ans` |

Two consequences the paper needs:

- **Pavlick's prompt is 2-way while its human labels are 5-way ordinal.** The
  support mismatch §08 inferred from the logits is confirmed at the prompt.
- **α has no verbalizer.** Both arms are full free-text hypotheses, unique per
  item. This is the structural form of §06's "item-specific hypotheses with no
  shared categorical label space", and it is why the label-factor control (§09)
  and the fold (§02) are undefined there.

**`degenerate_first_token`: α 767/1532 = 50.07%; chaosnli 0%; pavlick 0%.** On
half of α the two arms share their first answer token — free-text sentences
starting with the same word. First-token pooling cannot separate the arms on
those items, which is worth stating wherever the pooling ablation reports α.

### 12.3 `p_model` (N03)

- **There is no decoding.** `meta.json`: `"sampling": "not captured (deferred;
  needs a second generation pass)"`. Every number is teacher-forced scoring of a
  supplied answer string — no sampling, no beam, **no decoding temperature**.
- **τ is not a decoding temperature.** It is the frame-assignment softmax
  temperature in `soft_mass`: `m ∝ exp(−(d − d_min)/τ)` over K centroids. Say
  this explicitly in §4.2 or every reader will misread the τ grid.
- **Multi-token aggregation:** `neg_mean_logprob = −(1/T) Σ_t log p(a_t | a_<t, x)`.
  The other three output signals are per-position means over the top-50
  truncated distribution with tail correction `−tail·log tail`,
  `tail = 1 − Σ_{top-50} p`. Note `pred_maxprob` and `pred_margin` are coded as
  `1 − max p` and `1 − (p₁ − p₂)`, not `−max p` / `−(p₁ − p₂)` — rank-equivalent,
  so no reported ρ changes, but the equations differ from the usual form.
- **Candidate normalisation happens only in `ce_decomp`, over two candidates:**
  `p_model = softmax([agg(lp_correct), agg(lp_wrong)])`, `agg ∈ {mean, sum}`,
  default `mean`. There is no per-label scan. **This is the A1 lever — §4.2 must
  state `agg=mean` in body text**, because §08c shows the EU result reverses
  under `agg=sum`.

### 12.4 Frames (N07)

- **One frame per (model, dataset, layer, scheme, K, seed)** —
  `KMeans(n_clusters, random_state=seed, n_init=4).fit(H)`. Three frames per cell
  (seeds 0/1/2); signals are computed per frame and **averaged across seeds**
  before any correlation.
- **Fit on the full evaluation set** — all n items of the cell, `arm="correct"`.
  Transductive, no train/test split. The only cross-fitting anywhere is the
  neighbourhood oracle (§06).
- **No scaling of any kind** before K-means: no standardisation, no whitening, no
  PCA, no L2 normalisation. Euclidean distance on raw pooled hidden states.
- ⚠️ **Do not describe the frame as the library's.**
  `rsuq/core/frame.py::FixedFrame.from_embeddings` builds a frame by PCA(64) →
  MiniBatchKMeans on the **input-embedding matrix** — a vocabulary partition.
  The paper's frames are plain K-means on **pooled hidden states**. Two different
  constructions coexist in this project; §4.3 describes the second.

---

## Data inventory

Every file under `data/` is either cited above or listed here. Nothing is
orphaned silently.

| file | status |
|---|---|
| `fold_naive.csv` | **superseded** by `fold_gold.csv` (§02: the naive unfolded target is degenerate on the majority-canonicalised release). Retained as the record of what was run; do not cite. |
| `deferral_truthfulqa.csv` | **out of scope** — selective-prediction / AURC results belong to the AAAI paper under the fence in `../README.md`. Not an ARR artefact. |

Two things this document depends on that are **not** reproducible from this
repository, both extraction-side:

- **`extract_hf.py`** — not in the working tree, in any commit on any branch, on
  the volume, or anywhere on the authoring machine. Extraction ran 2026-07-20,
  two days before this repo's first commit. Provenance survives as
  `code_sha1: 4fd06c4bc3f0` in each cell's `meta.json`; verify any recovered
  copy against it. Blocks re-extraction (pre-answer states, per-label
  `p_model`), not re-analysis.
- **The OVA_ARR adapter registry** — decides a⁻ per dataset. Also absent, but
  its *output* is on the volume: `/vol/source/*.json` carries per-item
  `correct_ans` / `wrong_ans`, so the distractor rule is recoverable by lookup.

The scored arm was verified independently of both: it matches the stored
majority label on **99.7%** of ChaosNLI items for mistral and qwen (exact, via
`LABEL_FIRST_TOKEN` on `ans_ids`), with the residual attributable to vote-tie
breaking. Llama is unverifiable this way — its label tokens are gated.
