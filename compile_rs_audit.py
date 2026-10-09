"""Compile the RS claim audit and held-out experiments without altering prior results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "arr_results" / "rs_value"
DEST.mkdir(parents=True, exist_ok=True)


def table(df, columns=None):
    df = df[columns] if columns else df
    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(df.columns) + " |",
                       "| " + " | ".join(["---"] * len(df.columns)) + " |"] +
                      ["| " + " | ".join(row) + " |" for row in rows])


def main():
    ab = pd.read_parquet(ROOT/"results/arr_ablations_all.parquet")
    pool = ab[(ab.ablation=="pool") & (ab.signal=="credal_width")].copy()
    assert len(pool)==18
    pool = pool[["model","dataset","scheme","rho_len","rho_H","rho_knn"]]
    pool.to_csv(DEST/"pooling_density_audit.csv",index=False)
    length = ab[(ab.ablation=="length") & (ab.signal=="credal_width")]
    nb = pd.read_csv(ROOT/"arr_results/data/neighbourhood.csv")
    ratios = length.merge(nb[nb.k==25],on=["model","dataset"])
    ratios["conditional_to_knn_correlation_ratio"] = abs(ratios.partial)/ratios.rho_nbhd_oracle_CEILING
    ratios = ratios[ratios.dataset=="chaosnli"][["model","rho_H","partial",
        "rho_nbhd_oracle_CEILING","conditional_to_knn_correlation_ratio"]]
    ratios.to_csv(DEST/"correlation_ratios_not_information_fraction.csv",index=False)
    shuffle = ab[ab.ablation=="shuffle"].copy()
    # Persisted perm_p is exceedance fraction with exactly 200 draws (source checked).
    shuffle["exceedances"] = np.rint(shuffle.perm_p*200).astype(int)
    assert np.allclose(shuffle.exceedances/200,shuffle.perm_p)
    shuffle["permutations"] = 200
    shuffle["plus_one_p"] = (shuffle.exceedances+1)/201
    shuffle[["model","dataset","signal","rho_obs","exceedances","permutations",
             "perm_p","plus_one_p"]].to_csv(DEST/"permutation_reporting_audit.csv",index=False)
    ce = pd.read_csv(ROOT/"arr_results/data/ce_decomp_ci.csv")
    ce_w = ce[ce.quantity=="rho_W_EU"].pivot(index=["model","dataset"],columns="pool",values="point").reset_index()
    ce_w["sign_changed"] = np.sign(ce_w["mean"]) != np.sign(ce_w["sum"])
    ce_w.to_csv(DEST/"pooling_divergence_audit.csv",index=False)
    eta = pd.read_csv(ROOT/"arr_results/data/eta2_label.csv")
    eta = eta[eta.usable==True].copy()
    eta["identity_error"] = abs(eta.partial_direct-eta.partial_exact)
    assert eta.identity_error.max()<1e-9
    eta.to_csv(DEST/"label_factor_audit.csv",index=False)

    # Exact construction checks, separate from empirical claims.
    distances = np.array([2.,3.,5.])
    shift_error = float(np.max(abs(softmax(-distances/10)-softmax(-(distances-distances.min())/10))))
    centers = np.array([[-1.,0.],[1.,0.]])
    near = np.array([0.,0.]); outlier = np.array([0.,100.])
    mn = softmax(-np.linalg.norm(centers-near,axis=1))
    mo = softmax(-np.linalg.norm(centers-outlier,axis=1))
    sizes = np.array([1.,100.]); c = 1-1/sizes
    proofs = dict(softmax_shift_max_error=shift_error,
        near_distances=np.linalg.norm(centers-near,axis=1).tolist(),
        outlier_distances=np.linalg.norm(centers-outlier,axis=1).tolist(),
        near_mass=mn.tolist(),outlier_mass=mo.tolist(),
        near_W=float(mn@c),outlier_W=float(mo@c),
        equal_entropy_different_W=[float(np.array([.9,.1])@c),float(np.array([.1,.9])@c)],
        equal_size_W=float(1-1/10),
        equal_arm_lengths_mean_probability=softmax(np.array([-.1,-.5])).tolist(),
        equal_arm_lengths_sum_probability=softmax(2*np.array([-.1,-.5])).tolist())
    assert shift_error<1e-14 and np.allclose(mn,mo)
    (DEST/"construction_checks.json").write_text(json.dumps(proofs,indent=2)+"\n")

    results = [json.loads(p.read_text()) for p in sorted(DEST.glob("*__*__*.json"))]
    metrics, gains, ladders, mechanisms, coefficients = [],[],[],[],[]
    corr_rows,control_rows = [],[]
    for r in results:
        for key,acc in [("metrics",metrics),("comparisons",gains),("ladder",ladders),
                        ("mechanisms",mechanisms),("coefficients",coefficients)]:
            acc.extend(dict(row,scheme=r["scheme"]) for row in r[key])
        corr_rows.append(dict(model=r["model"],dataset=r["dataset"],scheme=r["scheme"],**r["full_correlations"]))
        assert len({x["item_id"] for x in r["oof"]})==len(r["oof"])
        assert all(0<=x["fold"]<5 for x in r["oof"])
        pd.DataFrame(r["oof"]).to_csv(DEST/f"oof_{r['model']}__{r['dataset']}__{r['scheme']}.csv",index=False)
        control_path=DEST/f"controls_{r['model']}__{r['dataset']}.json"
        if control_path.exists():
            lookup={x["item_id"]:x for x in json.loads(control_path.read_text())["rows"]}
            oo=pd.DataFrame(r["oof"])
            controls=[lookup[int(i)] for i in oo.item_id]
            length=np.array([x["length"] for x in controls])
            label=np.array([x["majority_label"] for x in controls])
            w=oo.feature_credal_width.to_numpy(); y=oo.target.to_numpy()
            def corr(a,b):
                return float(spearmanr(a,b).statistic) if np.std(a)>1e-14 and np.std(b)>1e-14 else np.nan
            def partial(z):
                design=np.column_stack([np.ones(len(y)),z])
                a,b=rankdata(w),rankdata(y)
                ra=a-design@np.linalg.lstsq(design,a,rcond=None)[0]
                rb=b-design@np.linalg.lstsq(design,b,rcond=None)[0]
                return float(np.corrcoef(ra,rb)[0,1])
            fold_dummies=pd.get_dummies(oo.fold,drop_first=True).to_numpy(float)
            label_dummies=pd.get_dummies(label,drop_first=True).to_numpy(float)
            control_rows.append(dict(model=r["model"],dataset=r["dataset"],scheme=r["scheme"],
                rho_W_H=corr(w,y),rho_W_knn=corr(w,oo.feature_knn_dist),
                rho_W_length=corr(w,length),rho_length_H=corr(length,y),
                partial_W_H_length=partial(rankdata(length)[:,None]),
                partial_W_H_length_fold=partial(np.column_stack([rankdata(length),fold_dummies])),
                partial_W_H_label=partial(label_dummies) if label_dummies.shape[1] else np.nan,
                matched_supervised_knn_rho=corr(oo.supervised_knn25,y)))
    for name,rows in [("heldout_metrics",metrics),("heldout_gains",gains),
                      ("ladder_recomputed",ladders),("size_shuffle_mechanism",mechanisms),
                      ("predictor_coefficients",coefficients),("full_frame_correlations",corr_rows)]:
        if rows:
            pd.DataFrame(rows).to_csv(DEST/f"{name}.csv",index=False)
    if control_rows:
        pd.DataFrame(control_rows).to_csv(DEST/"crossfit_length_label_controls.csv",index=False)
    mechanism_summary=pd.DataFrame()
    if mechanisms:
        mm=pd.DataFrame(mechanisms)
        mechanism_summary=mm.groupby(["model","dataset","scheme"],as_index=False).agg(
            median_rho_W_knn=("rho_W_knn","median"),
            median_rho_W_mass_entropy=("rho_W_mass_entropy","median"),
            median_shuffled_size_rho=("shuffled_size_rho_median","median"),
            max_equal_size_W_range=("equal_size_W_range","max"))
        mechanism_summary.to_csv(DEST/"mechanism_summary.csv",index=False)
    ladder_df = pd.DataFrame(ladders)
    if len(ladder_df):
        # Approximate bootstrap tail probabilities; distinct from permutation p.
        # Correct original two-rung panels (18 rows per scheme) separately.
        ladder_df["q_approx"] = np.nan
        for scheme in ["span_mean","first"]:
            mask = (ladder_df.scheme==scheme)&~((ladder_df.lower=="knn_dist")&(ladder_df.upper=="credal_width"))
            ix = ladder_df.index[mask]
            if len(ix)==18:
                p = ladder_df.loc[ix,"approximate_bootstrap_tail_p"].to_numpy()
                order=np.argsort(p); q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
                ladder_df.loc[ix[order],"q_approx"] = np.minimum(q,1)
        ladder_df.to_csv(DEST/"ladder_recomputed.csv",index=False)

    report = ["# Random sets: claim audit and incremental-value experiments",
        "Compiled from the persisted ARR results, ablations, source implementation, and new CPU-only analyses. Original records are preserved. This is exploratory follow-up work after seeing the original panel, not a new preregistration.",
        "## 1. Pooling and density: the proposed hinge already has an answer",
        "First-token pooling reduces the length channel, but does not eliminate the width–distance relationship. On ChaosNLI, width–distance Spearman correlations remain −0.884 (Llama), −0.905 (Mistral), and −0.673 (Qwen). On all three α cells the absolute relationship becomes stronger. A universal ‘inverted density is only span pooling’ explanation is inconsistent with these persisted results. Conversely, high correlation alone does not establish redundancy: conditional and held-out comparisons below address added value.",
        table(pool),
        "Source (repository-relative): `results/arr_ablations_all.parquet`, ablation=pool, signal=credal_width; all 18 rows. The original table already persisted experiment 1.",
        "## 2. The inversion, length control, and the correlation ratio",
        "The original width–entropy association is negative with intervals excluding zero in 5/9 cells at every tested τ. Only ChaosNLI is negative across the full tested K/layer grid in all three models; Pavlick is not design-stable. Length adjustment strengthens the ChaosNLI association. First-token refitting strengthens it on Llama and Mistral and modestly strengthens Qwen. These interventions change different things: partial correlation residualizes both ranked width and ranked target, while re-pooling changes the representation and the fitted frame. Neither establishes causality or that the length variable is pure noise. A small marginal correlation with entropy does not rule out conditional effects.",
        table(ratios),
        "The last column is only a ratio of correlations. It is not a proportion of information, explained variance, or a ceiling achieved. Its numerator is a conditional correlation of residualized ranks, and its denominator is a marginal correlation of supervised out-of-fold predictions. Even their targets differ through residualization. First-token correlations must also be compared against a first-token supervised reference, not a span-mean reference. New matched-representation cross-fitting below avoids selecting orientation on held-out labels.",
        "## 3. What the mass and width formulas imply",
        "The implementation uses W = Σ m_k(1−1/s_k), where s_k is a hard-cluster occupancy. Thus W is an expected cardinality coefficient; it is not mass entropy. Equal occupancies make W constant for every possible mass vector. Swapping mass between small and large focal sets can change W while keeping mass entropy fixed.",
        "Subtracting d_min cancels exactly in the softmax. Common additive distance shifts leave mass unchanged. A faraway point on the perpendicular bisector of two centroids has mass (0.5,0.5), exactly like the midpoint: outliers do not universally concentrate their mass. Numerical counterexamples are in `construction_checks.json`.",
        "At τ→0, a unique nearest centroid yields W→1−1/s_nearest; at τ→∞, W→the average coefficient, independent of the item. More generally dW/dτ = Cov_m(1−1/s, d)/τ², whose sign is not fixed. This is a construction-dependent boundary, not a universal density law. Ties at τ→0 retain shared mass, so the single-centroid statement requires uniqueness.",
        "New foldwise size shuffles hold distances and mass fixed while permuting the assignment of cardinalities to centroids. Equal-size and shuffled-size checks probe how W depends on occupancy–geometry alignment; they do not establish that density causes disagreement. One hundred shuffles per frame are descriptive, not a significance panel.",
        "## 4. First-token ladder and two-term predictor",
        "The full-frame first-token ladder refits K=200 K-means with n_init=4 at seeds 0/1/2, averages scores, then repeats both original rungs and a direct width-given-distance check. Percentile item bootstrap B=2000 gives intervals. Approximate bootstrap tail probabilities use plus-one counts and BH separately within each complete original 18-row two-rung panel. The direct distance-control rung is exploratory and excluded from those original-panel corrections. Negative conditional association means potential complementary predictive information; it does not mean information was destroyed.",
        "The two-term predictor uses rank-transformed kNN distance and width with learned coefficients. The held-out comparison is `knn` versus `knn_width`, not a post-hoc sign flip evaluated on the same labels.",
        "## 5. Cross-fitted frames, combinations, and supervised reference",
        f"Completed cells × representations: {len(results)}/18. Both span_mean and first-token use the same five shuffled item folds (seed 20261007), final layer, K=200, τ=10, frame seeds 0/1/2. Each frame sees only the outer training items; training kNN excludes self and test kNN searches training items only. No PCA, whitening, or normalization is introduced before K-means. Output scores come from the original per-item dump; the target is original annotator entropy in nats.",
        "Predictors use training-fitted quantile transforms, scaling, and Ridge(alpha=1), fixed before this experiment's results. No hyperparameter selection or test-time routing is used. Comparisons include output alone, output+distance, RS alone, baseline+size, baseline+size+width, and baseline+all RS features. The supervised kNN25 reference is cross-fitted within each representation. It demonstrates what this predictor recovers; it is neither an upper bound nor a deployable label-free uncertainty score. R²<0 only establishes failure of this predictor relative to the mean on that evaluation.",
        "Primary comparison: MSE(baseline)−MSE(baseline_all_rs). Positive gain means adding RS helps. Also report the mass-only addition, the two-term addition, rank-correlation gains, and foldwise gains. Intervals resample fixed OOF predictions jointly at the item level (B=2000). They omit model-refitting uncertainty and dependence from overlapping training folds. They are exploratory conditional intervals, with no claim of familywise-corrected significance across the 18 cells or secondary comparisons.",
        "All state inputs are teacher-forced supplied answers, usually selected using crowd labels. Cross-fitting K-means removes full-evaluation-set frame fitting but does not remove that answer-selection information. This experiment is about the original measurement setup, not autonomous pre-answer deployment. The majority-label neighbourhood feature is deliberately absent from the baseline because it requires annotation access."]
    if metrics:
        m=pd.DataFrame(metrics); g=pd.DataFrame(gains)
        primary=g[(g.augmented=="baseline_all_rs")&(g.baseline=="baseline")]
        summary=["## Findings from the new experiments",
            f"{len(results)}/18 model/dataset/representation runs completed. In the primary comparison, {int((primary.mse_gain_lo>0).sum())} have positive conditional intervals, {int((primary.mse_gain_hi<0).sum())} have negative conditional intervals, and {int(((primary.mse_gain_lo<=0)&(primary.mse_gain_hi>=0)).sum())} remain uncertain. These counts are exploratory; the intervals are not adjusted across cells.",
            "Suggested empirical claim: **Random-set features can provide complementary information about human label variation beyond output uncertainty and local distance. Their value depends on model, dataset, and pooling; first-token pooling reveals conditional mass information on all three ChaosNLI models.** The negative orientation of width is retained and learned from training data, not hidden by silently reporting −W. Span-mean also produces positive primary gains on Mistral and Qwen/ChaosNLI, so ‘span-mean fails universally’ is not supported.",
            "The density-collapse hypothesis fails as a universal explanation: first-token width remains strongly related to distance on Llama and Mistral/ChaosNLI. The narrower question—whether width adds information despite that relationship—is addressed by the first-token ladder and the held-out two-term predictor."]
        positives=primary[primary.mse_gain_lo>0]
        for row in positives.itertuples():
            b=m[(m.model==row.model)&(m.dataset==row.dataset)&(m.scheme==row.scheme)&(m.predictor=="baseline")].iloc[0]
            a=m[(m.model==row.model)&(m.dataset==row.dataset)&(m.scheme==row.scheme)&(m.predictor=="baseline_all_rs")].iloc[0]
            fold_positive=sum(v>0 for v in row.fold_gains)
            summary.append(f"Promising setting: **{row.model} / {row.dataset} / {row.scheme}**. Adding all RS features reduces MSE by {100*row.relative_mse_reduction:.2f}% (absolute gain {row.mse_gain:.5f}, conditional 95% interval [{row.mse_gain_lo:.5f}, {row.mse_gain_hi:.5f}]); Spearman rises {b.rho:.3f}→{a.rho:.3f}. Gain is positive in {fold_positive}/5 folds. This is a setting-specific result under a fixed rank-transform/Ridge predictor, not a general RS guarantee.")
        if (DEST/"rs_value_overview.png").exists():
            summary.append("![Pooling association and incremental RS value](rs_value_overview.png)")
        report[2:2]=summary
        if len(mechanism_summary):
            report += ["### Occupancy–geometry diagnostic",
                table(mechanism_summary),
                "Entries summarize 15 training-fitted frames per setting (five folds × three seeds). The shuffled-size statistic is the median over 100 permutations per frame, then the median over frames. Holding masses fixed while equalizing sizes makes W numerically constant, as the formula requires. Size shuffling changes the occupancy-to-centroid alignment; this is a diagnostic of the construction, not a causal intervention on language ambiguity."]
        report += ["### Held-out metrics",table(m),"### Primary incremental RS comparison",
            table(g[(g.augmented=="baseline_all_rs") & (g.baseline=="baseline")],
                  ["model","dataset","scheme","mse_gain","mse_gain_lo","mse_gain_hi","relative_mse_reduction","rho_gain","status"]),
            "### Width added to distance",table(g[g.baseline=="knn"],
                  ["model","dataset","scheme","mse_gain","mse_gain_lo","mse_gain_hi","rho_gain","status"]),
            "### Recomputed ladder",table(ladder_df,["model","dataset","scheme","lower","upper","partial_rho","ci_lo","ci_hi","q_approx"])]
        if control_rows:
            report += ["### Length and label controls on cross-fitted width",
                table(pd.DataFrame(control_rows)),
                "These are descriptive correlations/partials of OOF feature vectors, not predictors fitted using held-out targets. Every item's width was computed using a frame fitted without that item. Rank residualization is performed afterward as an association diagnostic; the fold-adjusted column removes fold offsets as well as length. Majority-label adjustment is undefined on canonicalized α. Conditional and marginal correlations remain different estimands, so no ‘percentage of information ceiling’ is assigned."]
    report += ["## 6. Pooling sensitivity of forecast divergence",
        table(ce_w),
        "Mean-versus-sum log-probability aggregation changes width–divergence signs on all three α cells and changes magnitudes on ChaosNLI. Keep the previous EU finding retracted. CE(p_human,p_model)=H(p_human)+KL(p_human||p_model) is an exact proper-score decomposition, but KL here is forecast mismatch, not automatically Bayesian epistemic uncertainty.",
        "Mean scores are per-token rates; sum scores are sequence log-probabilities. Unequal candidate lengths can change relative weights; which label is favoured also depends on actual token probabilities, so ‘mean always inflates the short label’ is not a general rule. Even equal candidate lengths T>1 turn sum into a temperature-scaled softmax and can change probabilities and divergence. For output scalar scores summed with a fixed T across items, rankings can remain identical; this is a different operation from candidate normalization. No claim about what all other papers disclose has been verified.",
        "The output-family pooling panel reports tiny length–human-entropy correlations and 1 null-to-null sign flip in 36 pairs. This supports robustness against that particular output-score change, not universal absence of length information. Full label-support model probabilities and pre-answer states are absent from these analyses; the missing original extractor still blocks reproducing the original extraction, although persisted states enable every new analysis above.",
        "## 7. Label-factor decomposition and masked association",
        table(eta,["model","dataset","eta2_W_ranked","eta2_H_ranked","rho_WH","partial_direct","partial_ci_lo","partial_ci_hi","identity_error"]),
        "The decomposition identity reconciles direct residualization to <1e−9. Llama/Pavlick’s raw near-null becomes a negative within-label partial. This is a defensible masked-association result. η²(W_rank,label)=0.72–0.75 on ChaosNLI means width ranks strongly stratify the stored majority/scored label; it is not classifier accuracy or proof that the frame independently recovers model decisions. The extraction usually scores crowd-selected label strings, so answer identity is a direct alternative explanation. Label-based and length-based controls are related on verbalized tasks and should not be described as statistically independent evidence.",
        "## 8. Annotation, encoding, and metric checks",
        "The gold-oriented α direction test is near-null but low-power; the majority-canonicalized target makes the naive unfolded axis degenerate. Support thresholds must be specified in votes if annotator counts vary; Pavlick's support/count correlation is 0.62 and the support-versus-balance mechanism remains inconclusive. Rank-based partial correlations require ranked η² and Spearman collinearity. The original τ=10 collinearity ranges are Pearson 0.598–0.975 and Spearman 0.625–0.940; the near-identity at low τ cannot be transferred to τ=10.",
        "Pavlick remains model- and design-sensitive. The dataset-wide representation-versus-output family claim fails its own constant-sign condition. The original 3-win/1-loss/5-tie magnitude panel survives the disputed τ=5 exclusion, but does not establish a universally beneficial method. A bootstrap-tail p was added to the helper, yet corrections in the ladder/family panels do not automatically correct the separate paired-gain panel. The older diff document's ‘blocker resolved’ wording conflates these panels.",
        "## 9. Permutation reporting",
        "The persisted permutation records use raw exceedance fractions from 200 permutations. Zero exceedances are reported here as 0/200 and plus-one p=1/201≈0.00498, not p=0. This is a reporting correction, not 200 new permutations. These permutation controls were for the original score panel; they are not automatically tests of first-token or length-residualized scores. Original bootstrap-tail zeroes are a separate issue and cannot be repaired using the permutation denominator.",
        "## 10. Reproducibility and scope",
        "Sources: `arr_results/RESULTS_ARR.md` (latest record, including retractions), `results/ablations.md`, `results/arr_ablations_all.parquet`, `arr_results/data/*.csv`, `analyse_hf.py`, `arr_modal.py`. Older README/diff narratives are not authoritative where superseded. New scripts: `arr_rs_value.py` and `compile_rs_audit.py`. Shared Modal volume is mounted read-only by convention: the new function never commits or writes data there. Local per-cell JSON stores OOF predictions, fold/seed audits, coefficients, full-frame correlations, and size-shuffle diagnostics. CSV files are generated from those JSON records.",
        "Reproduce new experiments: `modal run arr_rs_value.py`; export the length/label controls: `modal run arr_rs_value.py::controls`. Recompile: `python compile_rs_audit.py`; validate: `python validate_rs_value.py`; render the figure: `python plot_rs_value.py`. Recompile again after rendering to include the figure link. The Modal commands use existing extracted states and logits-derived per-item scores; no GPU extraction or replacement data. Original unpublished results already existed in the working tree and have not been overwritten."]
    if (DEST/"TOKEN_BOUNDARY_AND_SCOPE.md").exists():
        report.insert(2,"**Token-boundary correction:** the configuration historically called `first_token` selects `span[0]`, the **last prompt token**, not the first answer token. The nine-cell cache audit confirms the shifted boundary and finds length-associated numerical differences between some candidate-arm caches. Read [the scope and boundary audit](TOKEN_BOUNDARY_AND_SCOPE.md) before interpreting the first-position gains. The numerical comparisons are preserved; their representation interpretation is corrected.")
    (DEST/"REPORT.md").write_text("\n\n".join(report)+"\n")
    print(f"Compiled {DEST/'REPORT.md'}; {len(results)}/18 held-out cells complete")


if __name__=="__main__":
    main()
