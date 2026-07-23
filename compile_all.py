#!/usr/bin/env python3
from __future__ import annotations
import json, os, re
import numpy as np

P = print


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def text(path):
    return open(path).read() if os.path.exists(path) else None


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and not
         (isinstance(r[key], float) and np.isnan(r[key]))]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2


def missing(label, path):
    P(f"\n> **{label}** — not found (`{path}`); not run or not saved.\n")


                                                                          
def header():
    P("# RSUQ — consolidated results\n")
    P("Every quantitative result from this session, with the provenance and "
      "caveats attached to each. Sections are ordered by what they can "
      "establish, not by when they were run.\n")
    P("**Reading guide.** Two different tasks appear throughout and they are "
      "not interchangeable:\n")
    P("- **Correctness ranking** — separate a supplied correct answer from a "
      "supplied wrong one. Statistic: marginal Mann-Whitney "
      "`theta = P(metric_wrong > metric_correct)`, null 0.5.")
    P("- **Dispersion validation** — predict *human annotator disagreement*. "
      "Statistic: Spearman `rho(metric, annotator_entropy)`. This is the only "
      "task with a target the model cannot see, so it is the only one immune "
      "to the density confound.\n")


                                                                              
def provenance():
    P("\n---\n\n## 1. Data and provenance\n")
    P("### 1.1 The original local cache\n")
    P("`outputs/llama3_8b/halueval_qa/hidden_states.pt` — Llama-3.1-8B-Instruct "
      "on HaluEval-QA, 6,554 paired items, d=4096.\n")
    P("| field | content |")
    P("|---|---|")
    for k, v in [
        ("h_correct / h_wrong", "(6554, 4096) fp32 — **first answer token only** (`token_mode='first'`), not a span mean"),
        ("lp_correct / lp_wrong", "generation logprobs; the only other signal-bearing field"),
        ("questions / correct_ans / wrong_ans", "full raw text, all 6,554 questions unique — the cell is re-extractable"),
        ("y_expert_correct / y_expert_wrong", "**all 1** — zero information (confirms the memo trap at `run_repspace_deferral.py:192`)"),
        ("human_accs", "**all 1.0** — zero information"),
        ("categories", "all `HaluEval-QA` — no stratification possible"),
        ("n_probe_layers=8, n_total_layers=32", "8 layers probed at extraction, one vector kept, **no layer index recorded**"),
    ]:
        P(f"| `{k}` | {v} |")
    P("\n89 rows have `h_correct` byte-identical to `h_wrong` (shared first "
      "token) — they contribute nothing to any paired comparison and are the "
      "source of the ties in the sign tests.\n")
    P("**Model-produced error signal:** the model assigns higher logprob to the "
      "hallucination on **33.3%** of items (2,184/6,554, 1 exact tie). This is "
      "a genuine model-generated label, unlike `y_expert_*`. Caveat: "
      "`lp_correct > -0.01` for 1,274 items (19%) vs 12 on the wrong side, so "
      "the correct-side logprobs are heavily ceiling-saturated.\n")

    P("### 1.2 Modal re-extraction (24 cells)\n")
    P("3 models x 8 datasets, bf16, 8 probe layers, spans + top-50 logits + "
      "attention. **$2.23, 26.9 GB, 0% NaN.**\n")
    P("| | A1-capable (has annotator distribution) | paired only |")
    P("|---|---|---|")
    P("| datasets | chaosnli (3000), chaosnli_alpha (1532), pavlick_nli (496) "
      "| halueval_qa (3000), medqa (3000), ambigqa_kge2 (1408), pubmedqa "
      "(1000), truthfulqa (817) |")
    P("\nSources built from the OVA_ARR adapter registry, not re-derived, so "
      "distractor choice and label mapping match that project's decisions.\n")
    P("**Not available, and why:**\n")
    P("- *Semantic entropy* — needs multiple sampled generations per item; "
      "`extract_hf.py` defers sampling and states it cannot be retrofitted "
      "from stored states.")
    P("- *EigenScore / INSIDE* — want covariance across sampled generations' "
      "embeddings; per-token states within a span are a different object.")
    P("- *`ambigqa_kge2` as a disagreement cell* — its meta carries "
      "`n_answers`, a COUNT of interpretations, not a distribution over a "
      "fixed label set. Nothing to take an entropy of.")
    P("- *`multinli_disagreement`* — adapter points at "
      "`metaeval/nli-disagreement-tasks`, which no longer exists on the Hub.\n")


                                                                           
def defects():
    P("\n---\n\n## 2. Defects found, and what each would have done\n")
    P("Recorded because several would have silently produced publishable-looking "
      "numbers.\n")
    P("| # | defect | consequence if unfixed |")
    P("|---|---|---|")
    for i, (d, c) in enumerate([
        ("`FixedFrame.members` never existed (`rsnn_compat.py:103`, `test_core.py:99`)",
         "2 tests failed on `AttributeError`; the RS-NN pignistic equivalence claim had **never** been verified"),
        ("`mass_geometric` allocated an (N,K,d) broadcast",
         "~86 GB at N=13k/K=200/d=4096 — OOM-killed; the function **could not run** on any real cache"),
        ("`screen.verdict_from_ci` only tests `lo > 0`",
         "a CI entirely BELOW zero prints as `NULL (CI includes 0)` — conflates 'no effect' with 'actively hurts'"),
        ("Sign test counted ties in the denominator",
         "null expectation pulled to (1-tie_rate)/2, making every result look mildly below chance for free"),
        ("`analyse.py` bootstrap re-seeded inside the comprehension",
         "all 500 replicates identical -> **every CI zero-width**, printed as if real"),
        ("A1 verdict averaged over tau",
         "`size_only` is EXACTLY tau-invariant (argmax is tau-free), so this compared a tau-averaged W to a tau-constant baseline"),
        ("`o.attentions` indexed with `keep`",
         "hidden_states has L+1 entries, attentions has L -> IndexError; layer 0 has no attention row"),
        ("Empty answer span for unpaired cells",
         "`slice(p-1,p-1)` is empty; safetensors stores zero-length arrays **silently** (72-byte tensors) — A1 would have had no data"),
        ("Qwen2.5 in fp16",
         "NaN in final layer for 5-100% of items, **exactly the layer A1 uses**; max finite value 148 vs fp16's 65504 ceiling, so intermediate activations overflowed, not the stored states"),
        ("`ambigqa_kge2` marked `kind='disagreement'`",
         "extraction aborted 3 cells on a correct guard — the metadata was wrong, not the guard"),
        ("ChaosNLI licence recorded as CC-BY-SA-4.0",
         "its README says **Creative Commons NON-COMMERCIAL 4.0**; the NC restriction must travel with the cell"),
        ("Span starts at `p-1`, which is answer-independent",
         "for single-token answers the whole span is that one position, so `h_correct` is BIT-IDENTICAL to `h_wrong` — medqa and pubmedqa are degenerate by construction (see 7.3.1)"),
    ], 1):
        P(f"| {i} | {d} | {c} |")


                                                                          
def tier01():
    R = load("results_tier01.json")
    P("\n---\n\n## 3. Tier 0 / Tier 1 (HaluEval, local cache)\n")
    if R is None:
        return missing("Tier 0/1", "results_tier01.json")
    c = R["cell"]
    P(f"Cell: {c['model']} / {c['dataset']}, {c['n_pairs']} pairs. "
      f"K={R['config']['headline_K']}, seeds={R['config']['seeds']}, "
      f"B={R['config']['B']}.\n")

    P("### T0.2 — focal-set size distribution\n")
    P("| K | pts/cluster | singletons | size min/med/max | width_coef range |")
    P("|---|---|---|---|---|")
    for K, rows in R["T0.2_size_distribution"].items():
        r = rows[0]
        P(f"| {K} | {r['pts_per_cluster']:.0f}"
          f"{' (underpowered)' if r['underpowered_lt50_per_cluster'] else ''} "
          f"| {np.mean([x['n_singletons'] for x in rows]):.1f} | "
          f"{r['size_min']:.0f} / {r['size_median']:.0f} / {r['size_max']:.0f} | "
          f"[{r['width_coef_min']:.4f}, {r['width_coef_max']:.4f}] |")
    P("\nSizes vary over three orders of magnitude, so the cardinality channel "
      "is not degenerate. But `1 - 1/|F_k|` saturates: the median cluster (32 "
      "members) already maps to 0.969.\n")

    P("### T0.1 — rho(W, err | conf) per arm (test split)\n")
    P("| arm | rho | per-seed | AURC gap |")
    P("|---|---|---|---|")
    gaps = {a: agg(f["credal_width (CARDINALITY)"], "gap_mean")[0]
            for a, f in R["T1.1_A7_cardinality"].items()}
    for arm, rows in R["T0.1_partial_corr_per_arm"].items():
        m, s = agg(rows, "partial_rho")
        ps = ", ".join(f"{r['partial_rho']:+.4f}" for r in rows)
        P(f"| {arm} | {m:+.4f} +/- {s:.4f} | {ps} | {gaps.get(arm, float('nan')):+.4f} |")
    P("\n**|rho| ~ 0.07 is small, not large.** The doc's rule was "
      "'moderate -> C stands; large -> B in disguise'. Neither branch applies: "
      "the supervised arm is not a probe in disguise, supervision simply bought "
      "nothing (rho = -0.074, gap = -0.0018). The sign also flips across arms "
      "(geometric -0.059, supervised -0.074, unsupervised +0.062), which on its "
      "own disqualifies it as a stable signal.\n")

    P("### T0.3 — geometric vs recon, and why the recon arm is unusable\n")
    for k, lbl in [("mean_total_variation", "mean TV(m_geo, m_recon)"),
                   ("corr_W_geo_vs_W_recon", "corr(W_geo, W_recon)"),
                   ("argmax_agreement", "argmax cluster agreement")]:
        m, s = agg(R["T0.3_geo_vs_recon"], k)
        P(f"- **{lbl}**: {m:.4f} +/- {s:.4f}")
    P("\nThe arms differ — but because the recon head is broken, not because it "
      "learned something distinct. Diagnostic (MLPRegressor, K=200, seed 0): "
      "converged at `n_iter_=16`, training `loss_=0.00103`, but **R^2 = -0.0390 "
      "against its own geometric target** vs -0.0003 for predicting the train "
      "mean. It is a near-constant predictor that is worse than the mean; the "
      "small loss is a scale artifact of 200-component mass vectors. Its "
      "T1.1 gap of -0.0238 is the only arm that actively HURTS.\n")

    P("### T1.1 — A7: cardinality vs cardinality-blind\n")
    P("| arm | functional | gap (mean +/- sd) |")
    P("|---|---|---|")
    for arm, fns in R["T1.1_A7_cardinality"].items():
        for label, rows in fns.items():
            m, s = agg(rows, "gap_mean")
            P(f"| {arm} | {label} | {m:+.4f} +/- {s:.4f} |")
    P("\n`credal_width` never helps on any arm. It is merely *less bad* than the "
      "blind functionals on the supervised arm (-0.002 vs ~-0.19). Correctly "
      "reframed: this tested a channel with almost no dynamic range.\n")

    P("### T1.3 — same-conf stratum (superseded)\n")
    P("Reported for completeness; superseded by the marginal analysis in "
      "section 4, which showed the pairing is inert.\n")


                                                                          
def paired():
    P("\n---\n\n## 4. Paired contrastive width (independent re-derivation)\n")
    P("`check_paired_width.py` shares no code with `screen.py` or the tier "
      "scripts. All 6,554 pairs (the geometric arm trains nothing, so no split "
      "is needed).\n")
    tau = load("results_tau_sweep.json")
    if tau:
        P("### tau sweep (K=200, seed 0)\n")
        P("| tau | P(W_wrong>W_correct) | 95% CI | break-pairing | marginal share | swap-labels |")
        P("|---|---|---|---|---|---|")
        for r in tau:
            b = r.get("control_break", {}).get("p_hat", float("nan"))
            s = r.get("control_swap", {}).get("p_hat", float("nan"))
            d1, d2 = 0.5 - r["p_hat"], 0.5 - b
            sh = f"{d2/d1*100:.0f}%" if r["binom_p"] < 0.05 and abs(d1) > 1e-3 else "n/a"
            P(f"| {r['tau']} | {r['p_hat']:.4f} | [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] "
              f"| {b:.4f} | {sh} | {s:.4f} |")
    ks = load("results_ksweep.json")
    if ks:
        P("\n### seed stability (K sweep, tau=2)\n")
        P("| K | min_size | mean P | sd across seeds | all below chance |")
        P("|---|---|---|---|---|")
        for K in sorted({r["K"] for r in ks}):
            for ms in sorted({r["min_size"] for r in ks}):
                sub = [r["p_hat"] for r in ks if r["K"] == K and r["min_size"] == ms]
                if sub:
                    P(f"| {K} | {ms} | {np.mean(sub):.4f} | {np.std(sub):.4f} | "
                      f"{'yes' if all(p < 0.5 for p in sub) else '**NO**'} |")
    P("\n**Findings.** The effect is real and robust for K>=100 (9 cells, all "
      "p<1e-26). It is **inverted**: width is higher on the *correct* answer. "
      "The `break-pairing` control retains 88-108% of the deviation, so it is a "
      "**marginal** difference between pools, not a paired effect — which "
      "invalidates T1.3's matched-confidence framing. Trimming clusters with "
      "|F_k|<100 (41-53% of mass, W spread down 18x) does not remove it, so it "
      "is not driven by tiny clusters. K=50 is the one unstable regime (seed 0 "
      "*above* chance at 0.5183, seeds 1-2 below). `swap-labels` stayed in "
      "[0.4906, 0.5065] across all 34 configurations, so the test is calibrated.\n")


                                                                           
def density():
    R = load("results_density.json")
    P("\n---\n\n## 5. Density & saturation (Density_paper.md §1-§4)\n")
    if R is None:
        return missing("Density paper", "results_density.json")
    rows = R["main"]; taus = R["config"]["taus"]
    P("### R1 — is W a landing-cluster-size lookup?\n")
    P("| tau | Pearson r (mean +/- sd) | mean max_k m_k | >= 0.95? |")
    P("|---|---|---|---|")
    for t in taus:
        sub = [r for r in rows if r["tau"] == t]
        p, ps = agg(sub, "R1_pearson_W_vs_coef_land")
        mm, _ = agg(sub, "mean_max_mass")
        P(f"| {t} | {p:.4f} +/- {ps:.4f} | {mm:.4f} | {'**yes**' if p >= 0.95 else 'no'} |")
    P("\n### R2 / R3 — does size explain the gap, and does it survive conditioning?\n")
    P("| tau | theta(size alone) | theta(W) | theta within size deciles |")
    P("|---|---|---|---|")
    for t in taus:
        sub = [r for r in rows if r["tau"] == t]
        sz, _ = agg(sub, "R2_auc_size_alone"); w, _ = agg(sub, "R2_auc_W")
        s3, _ = agg(sub, "R3_pooled_within_size_auc")
        P(f"| {t} | {sz:.4f} | {w:.4f} | {s3:.4f} |")
    r3 = load("results_r3_ci.json"); ex = load("results_r3_exact.json")
    if r3 and ex:
        np_ = sum("PERSISTS" in x["verdict"] for x in r3)
        ne = sum("PERSISTS" in x["verdict"] for x in ex)
        P(f"\n**R3 with CIs — the stop condition.** Under decile strata "
          f"**{np_}/{len(r3)}** cells persist; under EXACT landing-cluster "
          f"conditioning **{ne}/{len(ex)}** persist, with mean deviation "
          f"removed rising from "
          f"{np.mean([x['frac_deviation_removed'] for x in r3])*100:.0f}% to "
          f"{np.mean([x['frac_deviation_removed'] for x in ex])*100:.0f}%. "
          f"Exact conditioning covers ~73% of items (clusters need >=10 per "
          f"pool). Per Density_paper §7 this means **path (c) is not cleared "
          f"as an exclusive claim.**\n")
    P("### R4 — density directly (no clustering)\n")
    for k, v in R["R4_knn_density"].items():
        P(f"- **{k}**: theta = **{v['auc_wrong_gt_correct']:.4f}** "
          f"[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
    P("\n### §3A — saturation\n")
    P("| \\|F_k\\| | " + " | ".join(R["S3A_saturation_curve"].keys()) + " |")
    P("|---|" + "---|" * len(R["S3A_saturation_curve"]))
    P("| coefficient | " + " | ".join(f"{v:.4f}" for v in
                                      R["S3A_saturation_curve"].values()) + " |")
    s = list(R["S3A_empirical_saturation"].values())[0]
    P(f"\nEmpirically: **{s['frac_items_land_size_ge_10']*100:.1f}%** of items "
      f"land in clusters with |F_k| >= 10 and "
      f"**{s['frac_items_land_size_ge_100']*100:.1f}%** in |F_k| >= 100. The "
      f"channel is flat over almost all the mass — which is also the correct "
      f"reframing of the T1.1/A7 null.\n")


                                                                      
def a1():
    P("\n---\n\n## 6. A1 — dispersion validation (9 cells, bf16)\n")
    t = text("a1_pooled.log")
    if t is None:
        return missing("A1 pooled", "a1_pooled.log")
    P("Pooled CI: one item-resample index applied to **every** seed's frame, "
      "gains averaged inside the replicate. A positive verdict requires the "
      "pooled CI to exclude 0 **and** all seeds individually positive.\n")
    P("Rows are suppressed as UNINFORMATIVE where `max-mass > 0.90` or "
      "`r(W, size_only) > 0.98` — there W == size_only algebraically and the "
      "comparison carries no information regardless of significance.\n")
    P("```")
    P(t.strip())
    P("```")
    P("\n**3 of 9 cells beat the density null** (mistral/pavlick tau=10, "
      "qwen/chaosnli tau=10 and 25, qwen/pavlick tau=5 and 10), **1 is worse** "
      "(mistral/chaosnli tau=25), 5 tie. Model split: **llama 0/3, mistral 1/3, "
      "qwen 2/3.** Seed SD runs 0.015-0.037 against gains of 0.035-0.085 — up "
      "to 43% of the effect size. Every significant rho is **negative**: higher "
      "width, *lower* human disagreement.\n")
    P("**Structural result (holds regardless of the above):** at tau <= 2, "
      "`r(W, size_only)` = 0.994-0.9998. The entire low-tau regime — where all "
      "prior work in this project ran — cannot distinguish W from the density "
      "null as a matter of algebra, not evidence.\n")
    P("**Multiplicity is uncorrected** across 27 rows.\n")


                                                                          
def panels():
    P("\n---\n\n## 7. Multi-baseline UQ panels\n")
    P("Nine metrics in three groups: **mass** (need the frame) credal_width, "
      "mass_entropy, one_minus_max; **density** size_only, knn_dist; **token** "
      "(need only logits) pred_entropy, pred_maxprob, pred_margin, "
      "neg_mean_logprob.\n")

    P("### 7.1 Dispersion task (Spearman vs annotator entropy)\n")
    t = text("uq_panel.log")
    if t is None:
        missing("UQ panel", "uq_panel.log")
    else:
        blocks = re.split(r"##### ", t)[1:]
        P("| cell | best metric | \\|rho\\| | W | knn | size | pred_ent | W vs knn |")
        P("|---|---|---|---|---|---|---|---|")
        for b in blocks:
            lines = b.strip().split("\n"); cell = lines[0].strip()
            rows = {}
            for l in lines[1:]:
                m = re.match(r"^([a-z_]+)\s+(mass|density|token)\s+([+-][\d.]+)", l)
                if m:
                    rows[m.group(1)] = float(m.group(3))
            vs = {m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                  for m in re.finditer(r"vs (\w+)\s+gain=([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]", b)}
            if not rows:
                continue
            best = max(rows, key=lambda k: abs(rows[k]))
            g, lo, hi = vs.get("knn_dist", (float("nan"),) * 3)
            v = "W better" if lo > 0 else "W worse" if hi < 0 else "tie"
            P(f"| {cell} | **{best}** | {abs(rows[best]):.3f} | "
              f"{rows.get('credal_width', float('nan')):+.3f} | "
              f"{rows.get('knn_dist', float('nan')):+.3f} | "
              f"{rows.get('size_only', float('nan')):+.3f} | "
              f"{rows.get('pred_entropy', float('nan')):+.3f} | {v} |")
        P("\n**The alphaNLI result is the important one.** On all three models "
          "the best predictor is `neg_mean_logprob` — plain sequence "
          "likelihood, no frame, no clustering — at rho = 0.148/0.185/0.149, "
          "while credal width is **0.026 / 0.024 / 0.007**. alphaNLI is also "
          "the *clean* cell: its answers are full sentences of comparable "
          "length across arms, so it lacks the label-word span-length confound "
          "that contaminates ChaosNLI. On the one uncontaminated cell type, W "
          "has no signal and the cheapest baseline wins.\n")
        P("Against kNN density W ties in 6/9, wins 2, loses 1. `knn_dist` "
          "recovers ~91% of W's correlation on qwen/chaosnli with none of the "
          "DS machinery.\n")
        P("`pred_entropy` is positive (correct direction) in 8/9 cells; the "
          "one inversion is qwen/chaosnli, so the label confound is narrower "
          "than a single cell suggested.\n")

    P("### 7.2 Cross-model ensemble disagreement\n")
    P("Usually filed as 'needs new extraction', but the three models saw "
      "identical `item_id`s, so they form a cross-architecture ensemble. "
      "Preference = mean logprob(correct) - mean logprob(wrong), z-scored per "
      "model before combining (tokenizers differ, raw scales are not "
      "comparable).\n")
    P("```")
    P("chaosnli: 3000 items shared across 3 models")
    P("  llama3_8b    prefers correct on 53.1% of items")
    P("  mistral_7b   prefers correct on 52.8% of items")
    P("  qwen2_5_7b   prefers correct on 58.2% of items")
    P("  all three agree on 92.2% of items")
    P("  ensemble_pref_std      rho=+0.0635 CI[+0.0310,+0.0970]")
    P("  ensemble_vote_entropy  rho=+0.0593 CI[+0.0203,+0.0994]")
    P("  credal_width vs ensemble_pref_std  gain=+0.1371 [+0.0835,+0.1876] W better")
    P("```")
    P("\nReal but weak, and **in the correct direction** (more model "
      "disagreement -> more human disagreement) — unlike W. The models prefer "
      "the majority label on only 53-58% of items, which is itself a comment "
      "on ChaosNLI's difficulty.\n")

    P("### 7.3 Correctness-ranking task (marginal theta)\n")
    t = text("correctness_panel.log")
    if t is None:
        missing("Correctness panel", "correctness_panel.log")
    else:
        P("```")
        P(t.strip())
        P("```")
    P("\nThis panel did not exist before: `layer_sweep` reported theta for "
      "`credal_width` alone, so 15 accuracy cells carried a headline number "
      "with no baseline beside it.\n")

    P("#### 7.3.1 Two cells are degenerate BY CONSTRUCTION\n")
    P("`medqa` and `pubmedqa` return `theta = 0.5000` with **zero-width CIs** "
      "for every hidden-state and top-k metric. That is not a weak result — "
      "the values are identical between arms. Verified directly:\n")
    P("```")
    P("llama3_8b__medqa       n_ans_tok med=1   h_correct==h_wrong on 100.0% of items")
    P("llama3_8b__pubmedqa    n_ans_tok med=1   h_correct==h_wrong on 100.0% of items")
    P("llama3_8b__halueval_qa n_ans_tok med=3   h_correct==h_wrong on   0.0% of items")
    P("```")
    P("\n**Mechanism.** The extraction span is `slice(p-1, p+a-1)`. With a "
      "single-token answer (a=1) that is the one position `p-1`, whose hidden "
      "state and predictive distribution are computed from the PROMPT alone — "
      "before the model has seen which answer follows. Both arms share the "
      "prompt, so the states are bit-identical and every state-derived metric "
      "is exactly tied.\n")
    P("The answers make this concrete: `medqa` uses option letters (`'D'` vs "
      "`'C'`), `pubmedqa` uses `'yes'`/`'no'`/`'maybe'`. Both are one token.\n")
    P("Only `neg_mean_logprob` escapes, because it gathers the probability of "
      "the ACTUAL answer token: theta = **0.7615** (medqa) and **0.7403** "
      "(pubmedqa). So the model does discriminate correct from wrong on these "
      "cells — credal width simply cannot see it by construction.\n")
    P("**This generalises beyond the two dead cells.** The first span position "
      "is answer-independent for EVERY cell, so it contributes an identical "
      "value to both arms and dilutes `span_mean` wherever it is included. "
      "`ambigqa_kge2` shows the same signature on its length-matched subset "
      "(credal_width 0.4772 -> **0.5000**).\n")
    P("| cell | equal-len arms | theta(W) | theta(W) len-matched | usable? |")
    P("|---|---|---|---|---|")
    for c, e, t, m, u in [
        ("halueval_qa", "1.9%", "0.9096", "0.5753", "length-confounded"),
        ("medqa", "100%", "0.5000", "0.5000", "**degenerate**"),
        ("pubmedqa", "100%", "0.5000", "0.5000", "**degenerate**"),
        ("truthfulqa", "16.8%", "0.4444", "0.4785", "weak"),
        ("ambigqa_kge2", "78.8%", "0.4772", "0.5000", "**degenerate on matched**"),
    ]:
        P(f"| {c} | {e} | {t} | {m} | {u} |")
    P("\n**Consequence for the correctness task.** On 3 of 5 accuracy datasets "
      "the hidden-state contrast is exactly or nearly zero; on the fourth "
      "(HaluEval) it is 0.91 unmatched and 0.575 matched. There is no accuracy "
      "cell where credal width demonstrates length-independent discrimination.\n")
    P("**HaluEval / llama result.** theta(credal_width) = **0.9096** "
      "[0.9029, 0.9162] — W beats every baseline. But **98.1% of pairs have "
      "arms of different token length**, and on the 56 equal-length pairs W "
      "collapses to **0.5753**. Every other metric collapses too (knn "
      "0.150->0.418, neg_mean_logprob 0.151->0.537, pred_entropy "
      "0.325->0.607). A uniform collapse across mass, density AND token "
      "metrics is the signature of one shared confound, not nine effects.\n")
    P("The cause is in the data: HaluEval `correct_ans` has median 14 "
      "characters, `wrong_ans` median 59 — correct answers are short entities, "
      "hallucinations are verbose sentences. With `span_mean` pooling, W is "
      "largely reading answer length.\n")
    P("Caveats: the matched subset is 56 items, so 0.5753 is imprecise — not "
      "significantly different from 0.5, but not tightly pinned either. And "
      "verbosity may be a genuine property of hallucinated text; if so the "
      "claim is about detecting verbose answers, not about credal width.\n")


                                                                        
def open_issues():
    P("\n---\n\n## 8. Open issues\n")
    P("| # | issue | why it matters |")
    P("|---|---|---|")
    for i, (a, b) in enumerate([
        ("Multiplicity uncorrected", "27 A1 rows tested at 95%; tau rows within a cell are nested and dependent, so the effective count is nearer 9 cell-level tests"),
        ("`screen.py` verdict blind spot unpatched", "the file is frozen by its own header; every verdict it has emitted labels harm as null. `run_tier01.py` adds `verdict_signed` as an adjunct, but the frozen file is unchanged"),
        ("Answer-length confound is general, not local", "it drives the correctness task on HaluEval and contaminates ChaosNLI's label-word arms. `n_ans_tok_*` is now in `index.parquet`, so it can be conditioned on — but nothing except A1d does so yet"),
        ("A1d run only on llama (5 cells); mistral/qwen not yet run", "the degeneracy is structural so it should replicate, but that is untested"),
        ("Single-token cells need a different extraction to be usable at all", "the span must include positions AFTER the answer tokens, or the answer must be embedded in a longer continuation, or those cells carry no hidden-state signal by construction"),
        ("Tier-A extraction not run", "only Tier B (8 probe layers) exists; the layer sweep is therefore coarse"),
        ("`analyse.py` still on disk with the zero-width-CI bug", "nothing uses it, but it can be picked up by mistake — worth deleting"),
    ], 1):
        P(f"| {i} | {a} | {b} |")


if __name__ == "__main__":
    header(); provenance(); defects(); tier01(); paired(); density()
    a1(); panels(); open_issues()
    P("\n---\n\n*Generated by `compile_all.py` from the session's JSON dumps "
      "and run logs.*")
