"""Standalone scientific summary of completed RS comparisons."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"arr_results/rs_value/.mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    dest=ROOT/"arr_results/rs_value"
    p=pd.read_csv(dest/"pooling_density_audit.csv")
    g=pd.read_csv(dest/"heldout_gains.csv")
    m=pd.read_csv(dest/"heldout_metrics.csv")
    fig,axes=plt.subplots(1,3,figsize=(17,7),gridspec_kw={"width_ratios":[.8,1.3,1.3]})
    colors={"span_mean":"#566A89","first":"#B34E30"}
    ax=axes[0]
    for j,model in enumerate(["llama3_8b","mistral_7b","qwen2_5_7b"]):
        q=p[(p.dataset=="chaosnli")&(p.model==model)].set_index("scheme")
        vals=[q.loc[s,"rho_knn"] for s in ["span_mean","first_token"]]
        ax.plot(vals,[j,j],color="#c4c4c4",lw=2)
        for val,scheme in zip(vals,["span_mean","first"]):
            ax.scatter(val,j,c=colors[scheme],s=65,label={"span_mean":"Span mean","first":"Prompt boundary"}[scheme] if j==0 else None,zorder=3)
    ax.set_yticks(range(3),["Llama","Mistral","Qwen"])
    ax.invert_yaxis(); ax.set_xlim(-1,-.5); ax.set_xlabel("Spearman ρ(width, distance)")
    ax.set_title("A. Density association persists\nChaosNLI, original full frames")
    ax.legend(frameon=False,loc="upper right")
    models=["llama3_8b","mistral_7b","qwen2_5_7b"]
    datasets=["chaosnli","chaosnli_alpha","pavlick_nli"]
    for ax,base,aug,title in [
        (axes[1],"baseline","baseline_all_rs","B. RS added to output + distance"),
        (axes[2],"knn","knn_width","C. Width added to distance")]:
        sub=g[(g.baseline==base)&(g.augmented==aug)]
        labels=[]
        for j,(model,dataset) in enumerate((a,b) for b in datasets for a in models):
            labels.append({"llama3_8b":"Llama","mistral_7b":"Mistral","qwen2_5_7b":"Qwen"}[model]+" / "+dataset.replace("chaosnli_alpha","α").replace("chaosnli","Chaos").replace("pavlick_nli","Pavlick"))
            for scheme,offset in [("span_mean",-.12),("first",.12)]:
                q=sub[(sub.model==model)&(sub.dataset==dataset)&(sub.scheme==scheme)]
                if q.empty: continue
                row=q.iloc[0]
                denominator=m[(m.model==model)&(m.dataset==dataset)&(m.scheme==scheme)&(m.predictor==base)].iloc[0].mse
                val,lo,hi=100*np.array([row.mse_gain,row.mse_gain_lo,row.mse_gain_hi])/denominator
                ax.errorbar(val,j+offset,xerr=[[val-lo],[hi-val]],fmt="o",ms=5,
                    color=colors[scheme],capsize=2,label=scheme if j==0 else None)
        ax.axvline(0,color="#777",ls="--",lw=1)
        ax.set_yticks(range(9),labels); ax.invert_yaxis()
        ax.set_title(title+"\nCross-fitted frames and predictors")
        ax.set_xlabel("MSE reduction (%) — positive favours RS")
        ax.grid(axis="x",alpha=.15)
    fig.suptitle("Where random-set features add predictive value",fontsize=16,y=.99)
    fig.text(.5,.01,"Exploratory 95% paired item-bootstrap intervals conditional on fixed OOF predictions; teacher-forced answer states.",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.955])
    fig.savefig(dest/"rs_value_overview.png",dpi=200)
    fig.savefig(dest/"rs_value_overview.svg")
    plt.close(fig)
    print("Saved rs_value_overview.png and .svg")


if __name__=="__main__":
    main()
