"""Held-out incremental value of RS, using the existing Modal extraction.

Run: modal run arr_rs_value.py
No extraction, no substitute datasets, no writes to the shared volume.
Exploratory protocol fixed in this file before this experiment is run.
"""
import json
from pathlib import Path
import modal

app = modal.App("rsuq-arr-rs-value")
vol = modal.Volume.from_name("rsuq-latent", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.4.6", "scipy==1.17.1", "pandas==3.0.6",
                      "pyarrow==25.0.1", "scikit-learn==1.9.1",
                      "safetensors==0.8.0", "threadpoolctl==3.7.0")
         .add_local_python_source("analyse_hf", "arr_modal"))
OUTPUT = ["pred_entropy", "pred_maxprob", "pred_margin", "neg_mean_logprob"]
GROUPS = {
    "knn": ["knn_dist"],
    "knn_width": ["knn_dist", "credal_width"],
    "output": OUTPUT,
    "baseline": OUTPUT + ["knn_dist"],
    "rs_only": ["size_only", "credal_width", "mass_entropy", "one_minus_max"],
    "baseline_size": OUTPUT + ["knn_dist", "size_only"],
    "baseline_size_width": OUTPUT + ["knn_dist", "size_only", "credal_width"],
    "baseline_all_rs": OUTPUT + ["knn_dist", "size_only", "credal_width",
                                  "mass_entropy", "one_minus_max"],
}
PROTOCOL = dict(outer_folds=5, split_seed=20261007, frame_seeds=[0, 1, 2],
                K=200, tau=10.0, k=10, scheme="span_mean", arm="correct",
                kmeans_n_init=4, ridge_alpha=1.0, bootstrap_B=2000,
                primary="MSE(baseline)-MSE(baseline_all_rs)",
                groups=GROUPS, target="annotator entropy, nats",
                inference="conditional item bootstrap of fixed OOF predictions; exploratory",
                limitation="teacher-forced crowd-selected answer states; not pre-answer UQ")


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=32768,
              timeout=60 * 60 * 2)
def evaluate(model: str, dataset: str, scheme: str = "span_mean"):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from threadpoolctl import threadpool_limits
    import analyse_hf as A
    from arr_modal import _load_peritem

    root = A._root(model, dataset)
    meta = A._meta(root)
    layer = meta["n_total_layers"]
    scores, y, ids = _load_peritem(model, dataset, layer, "span_mean", 200, 10.0, "0,1,2")
    assert scheme in {"span_mean", "first"}
    hid, H0 = A.load_states(root, meta, layer, "correct", scheme)
    pos = {int(i): j for j, i in enumerate(hid)}
    assert len(np.unique(ids)) == len(ids)
    H = H0[[pos[int(i)] for i in ids]]
    assert np.isfinite(H).all() and np.isfinite(y).all()
    outputs = {name: scores[name] for name in OUTPUT}
    assert all(np.isfinite(v).all() for v in outputs.values())
    n = len(ids)
    predictions = {name: np.full(n, np.nan) for name in GROUPS}
    predictions["mean"] = np.full(n, np.nan)
    predictions["supervised_knn25"] = np.full(n, np.nan)
    folds = np.full(n, -1)
    audits, coefficients, mechanisms = [], [], []
    saved_features = {name: np.full(n, np.nan) for name in
                      ["knn_dist", "size_only", "credal_width", "mass_entropy", "one_minus_max"]}
    def rho(a, b):
        return float(spearmanr(a, b).statistic) if np.std(a) > 1e-14 and np.std(b) > 1e-14 else None

    with threadpool_limits(limits=4):
        for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=20261007).split(H)):
            assert not set(ids[tr]).intersection(ids[te])
            folds[te] = fold
            features_tr = {name: val[tr] for name, val in outputs.items()}
            features_te = {name: val[te] for name, val in outputs.items()}
            nn = NearestNeighbors(n_neighbors=11, algorithm="brute").fit(H[tr])
            # Explicit self exclusion avoids assumptions about ordering at ties.
            dt, jt = nn.kneighbors(H[tr])
            features_tr["knn_dist"] = np.array([
                ds[js != i][:10].mean() for i, (ds, js) in enumerate(zip(dt, jt))])
            de, _ = nn.kneighbors(H[te], n_neighbors=10)
            features_te["knn_dist"] = de.mean(1)
            acc_tr, acc_te = {}, {}
            for seed in PROTOCOL["frame_seeds"]:
                km = KMeans(200, random_state=seed, n_init=4).fit(H[tr])
                sizes = np.bincount(km.labels_, minlength=200).astype(float)
                # Match original convention for any empty cluster; record it.
                empty = int((sizes == 0).sum())
                sizes = np.maximum(sizes, 1)
                mt = A.soft_mass(H[tr], km.cluster_centers_, 10.0)
                me = A.soft_mass(H[te], km.cluster_centers_, 10.0)
                for name, value in A._signals(mt, sizes).items():
                    acc_tr.setdefault(name, []).append(value)
                ss = A._signals(me, sizes)
                for name, value in ss.items():
                    acc_te.setdefault(name, []).append(value)
                # Equal cardinalities make W constant irrespective of ambiguity.
                equal_w = A.credal_width(me, np.full(200, len(tr)/200))
                rng = np.random.default_rng(1000 + 10*fold + seed)
                shuffled_rho = [rho(A.credal_width(me, rng.permutation(sizes)),
                                    features_te["knn_dist"]) for _ in range(100)]
                valid = np.array([v for v in shuffled_rho if v is not None])
                mechanisms.append(dict(model=model, dataset=dataset, fold=fold,
                    seed=seed, rho_W_knn=rho(ss["credal_width"], features_te["knn_dist"]),
                    rho_W_mass_entropy=rho(ss["credal_width"], ss["mass_entropy"]),
                    rho_W_H=rho(ss["credal_width"], y[te]),
                    equal_size_W_range=float(np.ptp(equal_w)),
                    shuffled_size_rho_median=float(np.median(valid)) if len(valid) else None,
                    shuffled_size_rho_lo=float(np.quantile(valid,.025)) if len(valid) else None,
                    shuffled_size_rho_hi=float(np.quantile(valid,.975)) if len(valid) else None))
                audits.append(dict(fold=fold, seed=seed, train_n=len(tr), test_n=len(te),
                                   K=200, empty_clusters=empty, inertia=float(km.inertia_)))
            features_tr.update({k: np.mean(v, axis=0) for k, v in acc_tr.items()})
            features_te.update({k: np.mean(v, axis=0) for k, v in acc_te.items()})
            for name in saved_features:
                saved_features[name][te] = features_te[name]
            predictions["mean"][te] = y[tr].mean()
            predictions["supervised_knn25"][te] = KNeighborsRegressor(
                n_neighbors=25, algorithm="brute").fit(H[tr], y[tr]).predict(H[te])
            for name, columns in GROUPS.items():
                xt = np.column_stack([features_tr[c] for c in columns])
                xe = np.column_stack([features_te[c] for c in columns])
                # Frozen monotone transform accommodates signs and nonlinear scales.
                estimator = make_pipeline(QuantileTransformer(
                    n_quantiles=min(1000,len(tr)), output_distribution="uniform",
                    random_state=0), StandardScaler(), Ridge(alpha=1.0))
                estimator.fit(xt, y[tr])
                predictions[name][te] = estimator.predict(xe)
                coefficients.extend(dict(model=model, dataset=dataset, fold=fold,
                    predictor=name, feature=c, coefficient=float(v))
                    for c,v in zip(columns, estimator[-1].coef_))
            print(f"{model}/{dataset} fold {fold+1}/5 complete", flush=True)
    assert (folds >= 0).all()
    assert all(np.isfinite(v).all() for v in predictions.values())
    metrics = [dict(model=model, dataset=dataset, predictor=name, n=n,
                   mse=float(np.mean((y-p)**2)), mae=float(np.mean(abs(y-p))),
                   r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)),
                   rho=rho(p,y)) for name,p in predictions.items()]
    comparisons = []
    pairs = [("baseline","baseline_all_rs"), ("baseline","baseline_size"),
             ("baseline_size","baseline_size_width"),
             ("baseline_size_width","baseline_all_rs"), ("output","baseline"),
             ("knn","knn_width")]
    for base, augmented in pairs:
        pb, pa = predictions[base], predictions[augmented]
        diff = (y-pb)**2-(y-pa)**2
        rng = np.random.default_rng(12345)
        gains, rhos = [], []
        for _ in range(2000):
            ix = rng.integers(0,n,n)
            gains.append(float(diff[ix].mean()))
            rhos.append(rho(pa[ix],y[ix])-rho(pb[ix],y[ix]))
        lo,hi = np.quantile(gains,[.025,.975])
        rlo,rhi = np.quantile(rhos,[.025,.975])
        comparisons.append(dict(model=model,dataset=dataset,baseline=base,augmented=augmented,
            n=n,mse_gain=float(diff.mean()),mse_gain_lo=float(lo),mse_gain_hi=float(hi),
            relative_mse_reduction=float(diff.mean()/np.mean((y-pb)**2)),
            rho_gain=rho(pa,y)-rho(pb,y),rho_gain_lo=float(rlo),rho_gain_hi=float(rhi),
            fold_gains=[float(diff[folds==f].mean()) for f in range(5)],
            status="positive CI" if lo>0 else "negative CI" if hi<0 else "uncertain"))
    oof = [dict(item_id=int(ids[i]),fold=int(folds[i]),target=float(y[i]),
                **{name:float(p[i]) for name,p in predictions.items()},
                **{"feature_"+name:float(v[i]) for name,v in saved_features.items()}) for i in range(n)]
    # Full-frame first-token ladder matches the original transductive protocol.
    # This is a distinct descriptive analysis from the held-out experiment.
    from arr_modal import _partial_rho, _rho
    full_acc = {}
    if scheme == "first":
        with threadpool_limits(limits=4):
            for seed in PROTOCOL["frame_seeds"]:
                km = KMeans(200, random_state=seed,n_init=4).fit(H)
                sizes = np.maximum(np.bincount(km.labels_,minlength=200),1).astype(float)
                for name,value in A._signals(A.soft_mass(H,km.cluster_centers_,10.),sizes).items():
                    full_acc.setdefault(name,[]).append(value)
        full = {name:np.mean(v,axis=0) for name,v in full_acc.items()}
        nn = NearestNeighbors(n_neighbors=11,algorithm="brute").fit(H)
        dd,jj = nn.kneighbors(H)
        full["knn_dist"] = np.array([d[j != i][:10].mean() for i,(d,j) in enumerate(zip(dd,jj))])
    else:
        full = scores
    ladder = []
    for lower,upper in [("knn_dist","size_only"),("size_only","credal_width"),
                        ("knn_dist","credal_width")]:
        obs = _partial_rho(full[upper],y,full[lower])
        rng = np.random.default_rng(24680)
        vals = []
        for _ in range(2000):
            ix = rng.integers(0,n,n)
            vals.append(_partial_rho(full[upper][ix],y[ix],full[lower][ix]))
        lo,hi = np.quantile(vals,[.025,.975])
        ladder.append(dict(model=model,dataset=dataset,lower=lower,upper=upper,
            partial_rho=float(obs),ci_lo=float(lo),ci_hi=float(hi),
            approximate_bootstrap_tail_p=float(min(1.,2*min((np.count_nonzero(np.array(vals)<=0)+1)/2001,
                                                          (np.count_nonzero(np.array(vals)>=0)+1)/2001)))))
    full_correlations = dict(rho_W_knn=_rho(full["credal_width"],full["knn_dist"]),
                            rho_W_H=_rho(full["credal_width"],y),
                            rho_knn_H=_rho(full["knn_dist"],y),
                            rho_W_mass_entropy=_rho(full["credal_width"],full["mass_entropy"]))
    import importlib.metadata
    protocol = dict(PROTOCOL,scheme=scheme,versions={name:importlib.metadata.version(name)
                    for name in ["numpy","scipy","pandas","scikit-learn","safetensors"]})
    return dict(model=model,dataset=dataset,scheme=scheme,protocol=protocol,metrics=metrics,
                comparisons=comparisons,coefficients=coefficients,mechanisms=mechanisms,
                frame_audit=audits,oof=oof,ladder=ladder,full_correlations=full_correlations)


@app.local_entrypoint()
def main(model: str = "", dataset: str = "", schemes: str = "span_mean,first"):
    dest = Path("arr_results/rs_value")
    dest.mkdir(parents=True, exist_ok=True)
    (dest/"protocol.json").write_text(json.dumps(dict(PROTOCOL,schemes=schemes.split(",")), indent=2)+"\n")
    def reusable(m,d,s):
        path=dest/f"{m}__{d}__{s}.json"
        if not path.exists():
            return False
        prior=json.loads(path.read_text())
        expected=dict(PROTOCOL,scheme=s)
        if not all(prior["protocol"].get(k)==v for k,v in expected.items()):
            raise RuntimeError(f"{path}: cached protocol differs; use a separate output directory")
        return True
    cells = [(m,d,s) for m in ["llama3_8b","mistral_7b","qwen2_5_7b"]
             for d in ["chaosnli","chaosnli_alpha","pavlick_nli"]
             for s in schemes.split(",")
             if (not model or m==model) and (not dataset or d==dataset)
             and not reusable(m,d,s)]
    for result in evaluate.starmap(cells, order_outputs=False):
        path = dest / f"{result['model']}__{result['dataset']}__{result['scheme']}.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(f"Saved {path}", flush=True)


@app.function(image=image,volumes={"/vol":vol},timeout=600)
def export_control_cell(model: str,dataset: str):
    import numpy as np
    import analyse_hf as A
    idx=A._index(A._root(model,dataset))
    col="soft_labels" if "soft_labels" in idx.columns else "annotator_labels"
    rows=[]
    for _,r in idx.iterrows():
        raw=json.loads(r[col]) if isinstance(r[col],str) else r[col]
        labels=np.asarray(raw,float).ravel()
        rows.append(dict(item_id=int(r.item_id),length=float(r.n_ans_tok_correct),
                         majority_label=int(np.argmax(labels))))
    return dict(model=model,dataset=dataset,rows=rows)


@app.local_entrypoint()
def controls():
    dest=Path("arr_results/rs_value")
    dest.mkdir(parents=True,exist_ok=True)
    cells=[(m,d) for m in ["llama3_8b","mistral_7b","qwen2_5_7b"]
           for d in ["chaosnli","chaosnli_alpha","pavlick_nli"]]
    for result in export_control_cell.starmap(cells,order_outputs=False):
        path=dest/f"controls_{result['model']}__{result['dataset']}.json"
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(f"Saved {path}")
