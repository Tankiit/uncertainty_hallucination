"""Compile token-boundary findings, scope, precision, and the raw-state diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DEST=ROOT/"arr_results/rs_value"


def table(df):
    def fmt(v):
        if pd.isna(v): return "—"
        return f"{v:.4f}" if isinstance(v,(float,np.floating)) else str(v)
    return "\n".join(["| "+" | ".join(df.columns)+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]+
        ["| "+" | ".join(fmt(v) for v in row)+" |" for row in df.itertuples(index=False,name=None)])


def main():
    files=sorted((DEST/"token_index").glob("*.json"))
    rows=[]
    for p in files:
        d=json.loads(p.read_text())
        assert d["embedding_first_equal_fraction"]==1
        assert d["first_embedding_unique_correct"]==1
        assert d["different_first_answer_id_n"]==d["equal_embedding_with_different_answer_id_n"]
        assert d["second_position_token_embedding_bijection"]
        if d["same_length_n"]:
            assert d["same_length_exact_fraction"]==1
        rows.append({k:d[k] for k in ["model","dataset","n","exact_equal_fraction","same_length_n",
            "same_length_exact_fraction","different_length_exact_fraction","max_abs_difference"]})
    assert len(rows)==9
    tok=pd.DataFrame(rows);tok.to_csv(DEST/"token_index_audit.csv",index=False)
    g=pd.read_csv(DEST/"heldout_gains.csv");m=pd.read_csv(DEST/"heldout_metrics.csv")
    precision=[]
    for _,r in g[(g.baseline=="baseline")&(g.augmented=="baseline_all_rs")&(g.scheme=="first")].iterrows():
        b=m[(m.model==r.model)&(m.dataset==r.dataset)&(m.scheme==r.scheme)&(m.predictor=="baseline")].iloc[0]
        precision.append(dict(model=r.model,dataset=r.dataset,mse_reduction_pct=100*r.relative_mse_reduction,
                              normalized_gain_ci_lo_pct=100*r.mse_gain_lo/b.mse,
                              normalized_gain_ci_hi_pct=100*r.mse_gain_hi/b.mse))
    precision=pd.DataFrame(precision);precision.to_csv(DEST/"first_position_effect_precision.csv",index=False)
    lengths=pd.read_csv(DEST/"crossfit_length_label_controls.csv")
    lengths=lengths[(lengths.dataset=="chaosnli")&(lengths.scheme=="first")][["model","rho_W_length","rho_length_H","rho_W_H"]]
    ab=pd.read_parquet(ROOT/"results/arr_ablations_all.parquet")
    grid=ab[(ab.ablation=="grid")&(ab.signal=="credal_width")]
    assert len(grid)==81
    pq=grid[(grid.model=="qwen2_5_7b")&(grid.dataset=="pavlick_nli")]
    report=["# Paper scope and token-boundary audit — correction to the RS interpretation",
        "## Which paper",
        "The local repository explicitly identifies ARR.md:3 as *Do LLM Internal States Predict Human Label Variation?*. This analysis belongs to that human-disagreement paper. The length-confound paper can cite the pooling/measurement issue as a bridge, but the task-specific disagreement gains should not be relabeled as hallucination or correctness results. THREE_PAPER_SEPARATION was not found in the searched research tree or Codex/Claude file catalogs, so its exact instructions have not been checked.",
        "## Which token",
        "The historical extraction record (RESULTS_ALL.md:439) states `slice(p-1,p+a-1)`, where p is prompt length and a is answer length. analyse_hf.py:26–28 implements `first` as `span[0]`. Therefore the nominal global position is p−1: the last prompt token, whose output predicts the first answer token. It is not the state after reading that answer token. The genuine first answer embedding is at span[1], where a≥2; it is absent from the cached span for a=1.",
        "This is now checked against all nine caches, not merely inferred from the older prose. Metadata says captured layer zero is embeddings and all nine cells carry extraction SHA 4fd06c4bc3f0. In every cell, span[0]'s layer-zero embedding is a single common vector and is identical across both arms, including every pair with different first answer IDs. At span[1], where stored, the embedding vectors are in one-to-one correspondence with the first answer IDs. The original extractor remains missing, so the exact line of extraction code cannot be independently reopened.",
        "**Correction:** earlier descriptions of these results as ‘first answer-token representations’ were wrong. Use ‘first stored span position’ or ‘last-prompt-token/prompt-boundary representation’. This changes the interpretation of the first-position gains, not the computed gains themselves. The original eta-squared label-factor panel was measured on span_mean features; it cannot be transferred to the prompt-boundary frame as an established interpretability asset without recomputing that panel.",
        "## A second boundary issue: cached forward-pass differences",
        table(tok),
        "Layer-zero boundaries are unambiguous, but final-layer vectors are not universally equal across candidate arms. Every available equal-length pair is bit-identical. Unequal-length pairs often differ. Llama/Pavlick and Qwen/Pavlick have equal candidate lengths throughout and hence 496/496 equal vectors. Mistral/Pavlick has unequal candidate lengths throughout and only 4/496 equal vectors. Qwen/ChaosNLI has a maximum coordinate difference of 52.5 despite the identical boundary token.",
        "This establishes length-associated forward-cache variation, not its cause. BF16 numerical kernels, padding/position handling, or an extraction/attention error are possibilities; the missing extractor prevents deciding among them. Do not certify all first-position caches as numerically answer-independent, or assume the length channel is fully removed. The Qwen/Pavlick boundary equality gate is clean for the diagnostic below; it still does not validate arbitrary K/layer choices.",
        "## Frame and distance fitting",
        "The existing held-out experiment already refits all three K-means frames inside each of five outer training folds: arr_rs_value.py:78–94. Both training and test kNN reference only H[tr], with explicit self exclusion for training queries (lines 83–89). Each of the 18 settings records 15 frame fits, train/test sizes, and disjoint item IDs. The separate full-frame ladder is intentionally transductive; it must not be mistaken for the held-out predictor. No rerun to repair outer frame leakage is needed.",
        "## First-position length check",
        table(lengths),
        "These are cross-fitted width features. The length correlations are much smaller than span-mean's ≈0.85, but −0.111, −0.172, and +0.107 are not zero. The representation is at the prompt boundary, and its cached numerical values can still vary across unequal-length candidate arms. Neither low marginal correlation nor selecting one position establishes that length is pure noise or absent conditionally.",
        "## Effect precision and design sensitivity",
        table(precision),
        "Intervals above are absolute paired MSE-gain intervals divided by the observed baseline MSE; this is a fixed-reference normalization, not a separately bootstrapped ratio. They are exploratory, conditional on fixed OOF predictions, and unadjusted across cells. Mistral/ChaosNLI's small lower bound is particularly important; no post-hoc ‘observed power’ calculation is substituted for these intervals.",
        f"Yes: Qwen/Pavlick is the strongest new first-position cell and was also especially unstable in the earlier **span-mean** K/layer grid: rho(W,H) ranged {pq.rho.min():.3f} to {pq.rho.max():+.3f}. That grid does not directly establish first-position instability, but it is a reason to restrict the new claim to its fixed K=200/final-layer setting. The first-position K/layer robustness grid has not been run.",
        "## Forking paths and unseen caches",
        "The new predictor panel has 18 settings (3 models × 3 datasets × 2 pooling schemes), of which six have positive conditional intervals. It was preceded by 81 width correlations on a span-mean model/dataset/K/layer grid, 27 width correlations across tau, and the pooling, label, length, family, and neighbourhood analyses. These are not a fully crossed grid, and not 81 independent hypothesis tests; the exact selection history is broader than the 18 predictor comparisons. Correcting only the displayed four first-position positives would be invalid.",
        "The existing volume inventory has exactly three human-disagreement datasets, all already analyzed. The other cached datasets (HaluEval, MedQA, AmbigQA, PubMedQA, TruthfulQA) have different targets and have also appeared in RESULTS_ALL.md. No repository-evidenced unseen cached human-entropy dataset was found. Do not call a newly selected split of these already-examined items confirmatory. A clean confirmation requires genuinely unexamined human-disagreement data and a protocol frozen before access, or an explicit retrospective multiplicity/selection analysis with the full result history.",
        "## Raw-state probe: prediction and protocol",
        "Before running the probe, the recorded prediction was: a tuned ridge probe on h will be competitive with or stronger than A+RS; D−C may be uncertain. Start with Qwen/Pavlick because its prompt-boundary states are identical across answer arms and it is the strongest prior RS cell. This cell selection is explicitly exploratory.",
        "A=output scores+training-only kNN; B=A+four RS features; C=A+standardized raw h; D=C+RS. Also evaluate h alone. All use the original five outer folds. Ridge alpha is selected separately for each predictor by three-fold inner CV over {0.1,1,10,100,1000,10000}; preprocessing is fit within those inner folds. Frames are refit on outer training items only; frame/RS features are held fixed for the inner alpha selection, which uses only outer-training covariates and labels. Thus outer evaluation is inductive, while frame construction is transductive within the inner alpha-selection loop. This distinction is disclosed rather than called fully nested frame fitting.",
        "The h probe tests a linear readout class. RS is derived from h, the training frame, and its occupancies, so a positive D−C result would indicate a useful nonlinear/structured feature map for that readout, not new information absent from the hidden-state data. A null D−C comparison also does not prove general redundancy. Both comparisons need their effect intervals."]
    probe_path=DEST/"h_probe/qwen2_5_7b__pavlick_nli.json"
    if probe_path.exists():
        r=json.loads(probe_path.read_text())
        metrics=pd.DataFrame(r["metrics"]); comparisons=pd.DataFrame(r["comparisons"])
        oo=pd.DataFrame(r["oof"]); y=oo.target.to_numpy()
        from scipy.stats import spearmanr
        assert set(oo.fold)==set(range(5)) and oo.item_id.is_unique
        for row in metrics.itertuples():
            pred=oo[row.predictor].to_numpy()
            assert np.isfinite(pred).all()
            assert np.isclose(row.mse,np.mean((y-pred)**2),rtol=1e-12)
            assert np.isclose(row.rho,spearmanr(pred,y).statistic,rtol=1e-12)
        for row in comparisons.itertuples():
            diff=(y-oo[row.baseline].to_numpy())**2-(y-oo[row.augmented].to_numpy())**2
            assert np.isclose(row.mse_gain,diff.mean(),rtol=1e-12,atol=1e-12)
            assert row.mse_gain_lo<=row.mse_gain_hi
        metric_map=metrics.set_index("predictor")
        bc=comparisons[(comparisons.baseline=="B")&(comparisons.augmented=="C")].iloc[0]
        cd=comparisons[(comparisons.baseline=="C")&(comparisons.augmented=="D")].iloc[0]
        report += [f"**Interpretation in this cell:** the tuned C predictor reduces MSE by {100*bc.relative_mse_reduction:.2f}% relative to B; its paired interval is [{bc.mse_gain_lo:.5f},{bc.mse_gain_hi:.5f}] in squared nats. D−C has gain {cd.mse_gain:.6f}, with interval [{cd.mse_gain_lo:.6f},{cd.mse_gain_hi:.6f}]. This supports RS as a structured summary useful against scalar baselines, with no clear additional predictive value after this tuned raw-state readout in the strongest previously selected cell. It is not a universal redundancy result."]
        report += ["### Raw-state results",table(metrics),table(comparisons.drop(columns="fold_gains")),
            "Fixed-alpha reference predictions were checked against the earlier experiment item-by-item to 1e−9 on identical folds. All raw-probe intervals are paired item bootstraps of fixed OOF predictions, not refitting bootstraps or selection-adjusted tests."]
        metrics.to_csv(DEST/"h_probe/metrics.csv",index=False)
        comparisons.to_csv(DEST/"h_probe/comparisons.csv",index=False)
        pd.DataFrame(r["alpha_selection"]).to_csv(DEST/"h_probe/selected_alphas.csv",index=False)
    else:
        report.append("Raw-state diagnostic is running; no probe result is claimed yet.")
    (DEST/"TOKEN_BOUNDARY_AND_SCOPE.md").write_text("\n\n".join(report)+"\n")
    print("Compiled token-boundary/scope audit; nine cache-boundary checks passed")


if __name__=="__main__": main()
