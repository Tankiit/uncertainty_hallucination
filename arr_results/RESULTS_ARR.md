# ARR disagreement pipeline — results

*Run 2026-07-31. Models × datasets = 9 cells. Signed Spearman throughout;
item bootstrap; BH within panel; every threshold frozen before the numbers
were seen (see `../arr_modal.py` pre-registration block).*

Grid: `K=200, tau∈{5,10,25}, seeds={0,1,2}, scheme=span_mean, final layer
(L32 llama/mistral, L28 qwen2.5-7b), arm=correct`. Target: human annotator
entropy (`h_plugin` primary; `h_norm`/`h_mm` emitted as robustness).

Verdicts at a glance:

| # | experiment | verdict |
|---|---|---|
| 01 | audit_target | passed |
| 02 | **fold (gold-oriented)** | **not detected, low power** — no direction read, but α has ~0 ceiling |
| 03 | ladder | mixed |
| 04 | **family split** | **cut from abstract** — not constant within Pavlick |
| 05 | sign_stability | stable |
| 06 | neighbourhood | oracle ceiling exceeds built signals |
| 07 | **components** | **inconclusive** — the deciding cells are confounded by annotator count |
| 08 | **AU/EU decomposition** | **finding (α + chaosnli)** — raw density tracks epistemic (right sign), credal_width inverted (up to −0.55) |

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

**The r = 0.9996 objection, resolved.** A near-identity `corr(credal_width,
size_only)` would make a partial of −0.24 (controlling for `size_only`)
algebraically impossible. But r(W, size_only) is **τ-dependent**: at the paper's
**τ=10 it is 0.85–0.96** (qwen·pavlick 0.894), leaving ~20% residual variance —
plenty for the observed partials. The 0.9996 identity is a **τ→0 artifact**
(sharp softmax collapses `credal_width` onto `size_only`; at τ=5 the mistral
cells already reach 0.994). At τ=10 the two signals are distinct enough that the
nested partials are legitimate, so §Elaboration can be written honestly.

---

## 05 · sign_stability — is credal_width's sign a property of τ?  ✅ stable

ρ(credal_width, H) across τ = 5 / 10 / 25:

| cell | τ=5 | τ=10 | τ=25 | verdict |
|---|--:|--:|--:|---|
| llama·chaosnli | −0.197 | −0.197 | −0.172 | stable (neg) |
| mistral·chaosnli | −0.264 | −0.236 | −0.188 | stable (neg) |
| qwen·chaosnli | −0.275 | −0.272 | −0.264 | stable (neg) |
| mistral·pavlick | −0.210 | −0.246 | −0.209 | stable (neg) |
| qwen·pavlick | −0.377 | −0.396 | −0.307 | stable (neg) |
| llama·pavlick | −0.001 | +0.005 | −0.007 | flips (all ≈0) |
| qwen·alpha | +0.001 | +0.007 | −0.017 | flips (all ≈0) |

The negative sign is stable across temperature **everywhere credal_width
carries signal**. Both flips sit where |ρ| ≈ 0 (undefined in practice). The
consistent negative sign supports the reframe: credal_width behaves as a
negatively-signed size readout, not a disagreement detector. Data:
`data/sign_stability.csv` (all 9 cells × 3 signals × 3 τ).

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

## 08 · AU/EU decomposition — answering Baan et al.  ✅ finding (α + chaosnli)

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
dependent. But the *sign* is consistent everywhere it is clean. Data:
`data/ce_decomp.csv`; pre-registration + caveats in `../arr/decompose_target.py`.

### 08b · Mechanism — the double inversion is density AND length, no residual

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
