"""Check numerical consistency and fold isolation in completed RS experiments."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr


def main():
    root=Path(__file__).resolve().parent/"arr_results/rs_value"
    completed={}
    for path in sorted(root.glob("*__*__*.json")):
        r=json.loads(path.read_text())
        rows=r["oof"]; y=np.array([x["target"] for x in rows])
        ids=np.array([x["item_id"] for x in rows]); f=np.array([x["fold"] for x in rows])
        assert len(np.unique(ids))==len(ids)
        assert set(f)==set(range(5))
        assert np.isfinite(y).all()
        metrics={x["predictor"]:x for x in r["metrics"]}
        for name,m in metrics.items():
            p=np.array([x[name] for x in rows])
            assert np.isfinite(p).all()
            assert np.isclose(m["mse"],np.mean((y-p)**2),rtol=1e-12)
            assert np.isclose(m["r2"],1-np.sum((y-p)**2)/np.sum((y-y.mean())**2),rtol=1e-12,atol=1e-12)
            assert np.isclose(m["rho"],spearmanr(p,y).statistic,rtol=1e-12)
        for g in r["comparisons"]:
            base=np.array([x[g["baseline"]] for x in rows])
            aug=np.array([x[g["augmented"]] for x in rows])
            diff=(y-base)**2-(y-aug)**2
            assert np.isclose(g["mse_gain"],diff.mean(),rtol=1e-12,atol=1e-12)
            assert np.allclose(g["fold_gains"],[diff[f==k].mean() for k in range(5)])
            assert g["mse_gain_lo"]<=g["mse_gain_hi"]
        for a in r["frame_audit"]:
            assert a["train_n"]==int((f!=a["fold"]).sum())
            assert a["test_n"]==int((f==a["fold"]).sum())
            assert a["K"]==200
        assert max(x["equal_size_W_range"] for x in r["mechanisms"])<1e-12
        completed[(r["model"],r["dataset"],r["scheme"])]=r
    for (model,dataset,scheme),r in completed.items():
        if scheme!="first" or (model,dataset,"span_mean") not in completed:
            continue
        other=completed[(model,dataset,"span_mean")]
        for a,b in zip(r["oof"],other["oof"]):
            assert a["item_id"]==b["item_id"] and a["fold"]==b["fold"]
            assert a["target"]==b["target"]
            # Pooling changes representation features, never the output baseline.
            assert np.isclose(a["output"],b["output"],rtol=1e-12)
            assert a["mean"]==b["mean"]
    # Verify the analytical temperature derivative against finite differences.
    d=np.array([1.,3.,7.]); c=np.array([.2,.7,.9]); t=2.
    m=softmax(-d/t)
    analytic=(np.sum(m*c*d)-np.sum(m*c)*np.sum(m*d))/t**2
    eps=1e-5
    numeric=(softmax(-d/(t+eps))@c-softmax(-d/(t-eps))@c)/(2*eps)
    assert np.isclose(analytic,numeric,rtol=1e-8)
    print(f"Validated metrics, paired gains, fold audits, and construction identities for {len(completed)}/18 cells")


if __name__=="__main__":
    main()
