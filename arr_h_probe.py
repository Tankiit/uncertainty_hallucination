"""Exploratory raw-state probe on the clean-boundary Qwen/Pavlick cell."""
import json
from pathlib import Path
import modal

app=modal.App("rsuq-arr-raw-state-probe")
vol=modal.Volume.from_name("rsuq-latent",create_if_missing=False)
image=(modal.Image.debian_slim(python_version="3.11")
       .pip_install("numpy==2.4.6","scipy==1.17.1","pandas==3.0.6",
                    "pyarrow==25.0.1","scikit-learn==1.9.1","safetensors==0.8.0","threadpoolctl==3.7.0")
       .add_local_python_source("analyse_hf","arr_modal"))
CFG=dict(model="qwen2_5_7b",dataset="pavlick_nli",pooling="first stored span position / last prompt token",
    K=200,tau=10.,frame_seeds=[0,1,2],n_init=4,outer_folds=5,split_seed=20261007,
    inner_folds=3,inner_seed=42,alphas=[.1,1.,10.,100.,1000.,10000.],bootstrap_B=2000,
    prediction="Raw h probe will be competitive with or stronger than A+RS; D-C may be uncertain.",
    status="Exploratory selected strongest prior cell; not unseen confirmation",
    tuning="Transforms and ridge alpha selected by inner CV on outer training data; RS frame built once per outer training fold",
    hidden_probe_dtype="float64; original cached states remain float32 for frame fitting",
    interpretation="RS provides derived nonlinear/structured features, never new information absent from h plus fitted frame")


@app.function(image=image,volumes={"/vol":vol},cpu=8,memory=16384,timeout=3600)
def evaluate():
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.model_selection import KFold,GridSearchCV
    from sklearn.neighbors import NearestNeighbors
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler,QuantileTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    from threadpoolctl import threadpool_limits
    import analyse_hf as A
    from arr_modal import _load_peritem
    model,dataset=CFG["model"],CFG["dataset"]
    root=A._root(model,dataset); meta=A._meta(root); layer=meta["n_total_layers"]
    scores,y,ids=_load_peritem(model,dataset,layer,"span_mean",200,10.,"0,1,2")
    hi,H0=A.load_states(root,meta,layer,"correct","first")
    pos={int(i):j for j,i in enumerate(hi)}; H=H0[[pos[int(i)] for i in ids]]
    wi,W=A.load_states(root,meta,layer,"wrong","first")
    pos={int(i):j for j,i in enumerate(wi)}
    assert np.array_equal(H,W[[pos[int(i)] for i in ids]]), "clean-boundary gate failed"
    out_names=["pred_entropy","pred_maxprob","pred_margin","neg_mean_logprob"]
    rs_names=["size_only","credal_width","mass_entropy","one_minus_max"]
    preds={k:np.full(len(y),np.nan) for k in ["A","B","C","D","h_only","A_fixed1","B_fixed1"]}
    folds=np.full(len(y),-1); logs=[]
    with threadpool_limits(limits=4):
        for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=20261007).split(H)):
            folds[te]=fold
            assert not set(ids[tr]).intersection(ids[te])
            nn=NearestNeighbors(n_neighbors=11,algorithm="brute").fit(H[tr])
            dd,jj=nn.kneighbors(H[tr])
            dt=np.array([d[j!=i][:10].mean() for i,(d,j) in enumerate(zip(dd,jj))])
            de=nn.kneighbors(H[te],n_neighbors=10)[0].mean(1)
            bt=np.column_stack([scores[k][tr] for k in out_names]+[dt])
            be=np.column_stack([scores[k][te] for k in out_names]+[de])
            rt,re=[],[]
            for seed in [0,1,2]:
                km=KMeans(200,n_init=4,random_state=seed).fit(H[tr])
                sizes=np.maximum(np.bincount(km.labels_,minlength=200),1).astype(float)
                st=A._signals(A.soft_mass(H[tr],km.cluster_centers_,10.),sizes)
                se=A._signals(A.soft_mass(H[te],km.cluster_centers_,10.),sizes)
                rt.append(np.column_stack([st[k] for k in rs_names]))
                re.append(np.column_stack([se[k] for k in rs_names]))
            rt=np.mean(rt,axis=0); re=np.mean(re,axis=0)
            sets={"A":(bt,be,5),"B":(np.column_stack([bt,rt]),np.column_stack([be,re]),9),
                  "C":(np.column_stack([bt,H[tr]]),np.column_stack([be,H[te]]),5),
                  "D":(np.column_stack([bt,rt,H[tr]]),np.column_stack([be,re,H[te]]),9),
                  "h_only":(H[tr].astype(np.float64),H[te].astype(np.float64),0)}
            for name,(xt,xe,nsmall) in sets.items():
                transforms=[]
                if nsmall:
                    transforms.append(("small",make_pipeline(QuantileTransformer(n_quantiles=200,
                        output_distribution="uniform",random_state=0),StandardScaler()),list(range(nsmall))))
                if xt.shape[1]>nsmall:
                    transforms.append(("hidden",StandardScaler(),list(range(nsmall,xt.shape[1]))))
                pipe=make_pipeline(ColumnTransformer(transforms),Ridge(solver="cholesky"))
                cv=GridSearchCV(pipe,{"ridge__alpha":CFG["alphas"]},
                    cv=KFold(3,shuffle=True,random_state=42),scoring="neg_mean_squared_error",n_jobs=1)
                cv.fit(xt,y[tr]); preds[name][te]=cv.predict(xe)
                logs.append(dict(fold=fold,predictor=name,alpha=float(cv.best_params_["ridge__alpha"]),
                                 inner_mse=float(-cv.best_score_)))
            # Reproduce prior fixed-alpha comparisons, with prior transform settings.
            for name,xt,xe in [("A_fixed1",bt,be),("B_fixed1",sets["B"][0],sets["B"][1])]:
                est=make_pipeline(QuantileTransformer(n_quantiles=min(1000,len(tr)),
                    output_distribution="uniform",random_state=0),StandardScaler(),Ridge(alpha=1.))
                est.fit(xt,y[tr]);preds[name][te]=est.predict(xe)
            print(f"raw-state probe fold {fold+1}/5 complete",flush=True)
    def rho(p,yy): return float(spearmanr(p,yy).statistic)
    metrics=[dict(predictor=k,mse=float(np.mean((y-p)**2)),rho=rho(p,y),
                  r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))) for k,p in preds.items()]
    comparisons=[]
    for base,aug in [("A","B"),("B","C"),("C","D"),("B","h_only")]:
        pb,pa=preds[base],preds[aug];diff=(y-pb)**2-(y-pa)**2
        rng=np.random.default_rng(12345); bg=[];rg=[]
        for _ in range(2000):
            ix=rng.integers(0,len(y),len(y)); bg.append(diff[ix].mean())
            rg.append(rho(pa[ix],y[ix])-rho(pb[ix],y[ix]))
        lo,hh=np.quantile(bg,[.025,.975]); rlo,rhi=np.quantile(rg,[.025,.975])
        comparisons.append(dict(baseline=base,augmented=aug,mse_gain=float(diff.mean()),
            mse_gain_lo=float(lo),mse_gain_hi=float(hh),relative_mse_reduction=float(diff.mean()/np.mean((y-pb)**2)),
            rho_gain=rho(pa,y)-rho(pb,y),rho_gain_lo=float(rlo),rho_gain_hi=float(rhi),
            fold_gains=[float(diff[folds==f].mean()) for f in range(5)]))
    return dict(protocol=CFG,metrics=metrics,comparisons=comparisons,alpha_selection=logs,
                oof=[dict(item_id=int(ids[i]),fold=int(folds[i]),target=float(y[i]),
                          **{k:float(p[i]) for k,p in preds.items()}) for i in range(len(y))])


@app.local_entrypoint()
def main():
    dest=Path("arr_results/rs_value/h_probe");dest.mkdir(parents=True,exist_ok=True)
    (dest/"protocol_before_run.json").write_text(json.dumps(CFG,indent=2)+"\n")
    result=evaluate.remote()
    prior=json.loads(Path("arr_results/rs_value/qwen2_5_7b__pavlick_nli__first.json").read_text())
    import numpy as np
    for row,old in zip(result["oof"],prior["oof"]):
        assert row["item_id"]==old["item_id"] and row["fold"]==old["fold"]
        assert np.isclose(row["A_fixed1"],old["baseline"],rtol=1e-9,atol=1e-9)
        assert np.isclose(row["B_fixed1"],old["baseline_all_rs"],rtol=1e-9,atol=1e-9)
    path=dest/"qwen2_5_7b__pavlick_nli.json"
    if path.exists() and not (dest/"qwen2_5_7b__pavlick_nli.pre_float64.json").exists():
        (dest/"qwen2_5_7b__pavlick_nli.pre_float64.json").write_text(path.read_text())
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ["metrics","comparisons"]},indent=2))
