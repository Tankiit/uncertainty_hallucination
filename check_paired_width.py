#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np


def load_pairs(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    hc = d["h_correct"].float().numpy().astype(np.float64)
    hw = d["h_wrong"].float().numpy().astype(np.float64)
    assert hc.shape == hw.shape, "pos/neg must be paired"
    return hc, hw, str(d.get("model_key", "?")), str(d.get("dataset", "?"))


def width(H, centroids, sizes, tau, min_size=1):
    from scipy.spatial.distance import cdist
    d = cdist(H, centroids)                               
    z = -(d - d.min(-1, keepdims=True)) / tau
    e = np.exp(z)
    m = e / e.sum(-1, keepdims=True)                             
    coef = 1.0 - 1.0 / np.maximum(sizes, 1.0)                              

    keep = sizes >= min_size
    info = {"min_size": min_size, "n_clusters_kept": int(keep.sum()),
            "n_clusters_dropped": int((~keep).sum())}
    if min_size > 1:
        kept_mass = m[:, keep].sum(-1)
        info["mean_mass_dropped"] = float(1.0 - kept_mass.mean())
        info["n_items_all_mass_dropped"] = int((kept_mass < 1e-6).sum())
        m = m[:, keep] / np.maximum(kept_mass, 1e-300)[:, None]
        coef = coef[keep]
    return m @ coef, m, info


def paired_report(W_c, W_w, label, rng):
    from scipy.stats import binomtest, wilcoxon
    n = len(W_c)
    gt = int((W_w > W_c).sum())
    ties = int((W_w == W_c).sum())
                                                                       
                                                                          
                                                                          
                                     
    n_eff = n - ties
    p_hat = gt / n_eff
    bt = binomtest(gt, n_eff, 0.5)
    ci = bt.proportion_ci(0.95)
    dW = W_w - W_c
    try:
        w_p = wilcoxon(dW).pvalue
    except ValueError:
        w_p = float("nan")
    direction = ("W HIGHER on correct" if p_hat < 0.5 else
                 "W higher on wrong" if p_hat > 0.5 else "exactly chance")
    print(f"  {label:26s} n={n_eff:5d}  P(W_wrong>W_correct)={p_hat:.4f} "
          f"CI[{ci.low:.4f},{ci.high:.4f}]  binom p={bt.pvalue:.3g}  "
          f"wilcoxon p={w_p:.3g}  ties={ties}")
    print(f"  {'':26s} mean dW={dW.mean():+.3e}  median dW={np.median(dW):+.3e}"
          f"  -> {direction}")
    return {"n": n, "p_hat": p_hat, "ci": [ci.low, ci.high],
            "binom_p": bt.pvalue, "mean_dW": float(dW.mean())}


def fit_frame(H, K, seed):
    from sklearn.cluster import KMeans
    km = KMeans(K, random_state=seed, n_init=5).fit(H)
    return km.cluster_centers_, np.bincount(km.labels_, minlength=K).astype(float)


def run_one(hc, hw, K, tau, seed, verbose=True, frame=None, min_size=1):
    H = np.vstack([hc, hw])
    centroids, sizes = frame if frame is not None else fit_frame(H, K, seed)
    W, m, info = width(H, centroids, sizes, tau, min_size)
    N = len(hc)
    W_c, W_w = W[:N], W[N:]
    if verbose:
        print(f"\n--- K={K} tau={tau} seed={seed} min_size={min_size} ---")
        print(f"  frame: sizes min/med/max = {sizes.min():.0f}/"
              f"{np.median(sizes):.0f}/{sizes.max():.0f}   "
              f"singletons={int((sizes==1).sum())}   "
              f"clusters kept={info['n_clusters_kept']}/"
              f"{info['n_clusters_kept']+info['n_clusters_dropped']}")
        if min_size > 1:
            print(f"  trimming: mean mass dropped={info['mean_mass_dropped']:.4f}"
                  f"  items losing all mass={info['n_items_all_mass_dropped']}")
        print(f"  W: mean={W.mean():.6f} std={W.std():.3e} "
              f"range=[{W.min():.6f},{W.max():.6f}]")
        print(f"  mass concentration: mean max_k m_k = {m.max(-1).mean():.4f}")
    rng = np.random.default_rng(seed)
    res = paired_report(W_c, W_w, "OBSERVED", rng)
    res.update({"K": K, "tau": tau, "seed": seed, "W_std": float(W.std()),
                "W_mean": float(W.mean()), **info})

    if verbose:
                                                                      
                                                                              
                                                           
        flip = rng.random(N) < 0.5
        c2 = np.where(flip, W_w, W_c)
        w2 = np.where(flip, W_c, W_w)
        res["control_swap"] = paired_report(c2, w2, "CONTROL swap-labels", rng)

                                                                              
                                                                   
        res["control_break"] = paired_report(
            W_c, W_w[rng.permutation(N)], "CONTROL break-pairing", rng)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="outputs/llama3_8b/halueval_qa/hidden_states.pt")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_size", type=int, default=1)
    ap.add_argument("--sweep_tau", action="store_true")
    ap.add_argument("--sweep_k", action="store_true")
    ap.add_argument("--sweep_grid", action="store_true",
                    help="full K x tau sweep; each (K, seed) frame is fit once")
    ap.add_argument("--sweep_trim", action="store_true")
    ap.add_argument("--seeds", default=None,
                    help="comma list; K sweep repeats over these seeds")
    ap.add_argument("--out", default=None, help="dump results as JSON")
    args = ap.parse_args()

    hc, hw, model, dataset = load_pairs(args.cache)
    print(f"cell = {model} / {dataset}   {len(hc)} pairs, d={hc.shape[1]}")
    print("null hypothesis: P(W_wrong > W_correct) = 0.5")
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])
    out = []

    if args.sweep_grid:
        print("\n=== full K x tau sweep (each frame reused across tau) ===")
        for K in (50, 100, 200, 400):
            for sd in seeds:
                fr = fit_frame(np.vstack([hc, hw]), K, sd)
                for tau in (0.1, 0.5, 1.0, 2.0, 10.0):
                    out.append(run_one(hc, hw, K, tau, sd, frame=fr,
                                       min_size=args.min_size))
    elif args.sweep_tau:
        print("\n=== tau sweep (one shared frame; KMeans does not depend on tau) ===")
        fr = fit_frame(np.vstack([hc, hw]), args.k, args.seed)
        for tau in (0.1, 0.5, 1.0, 2.0, 10.0):
            out.append(run_one(hc, hw, args.k, tau, args.seed, frame=fr,
                               min_size=args.min_size))
    elif args.sweep_k:
        print("\n=== K sweep (tau fixed) ===")
        for K in (50, 100, 200, 400):
            for sd in seeds:
                fr = fit_frame(np.vstack([hc, hw]), K, sd)
                                                                             
                                                                
                for ms in (1, args.min_size) if args.min_size > 1 else (1,):
                    out.append(run_one(hc, hw, K, args.tau, sd, frame=fr,
                                       min_size=ms))
    elif args.sweep_trim:
        print("\n=== trim sweep (one shared frame; trimming does not refit KMeans) ===")
        fr = fit_frame(np.vstack([hc, hw]), args.k, args.seed)
        for ms in (1, 2, 5, 10, 30, 100):
            out.append(run_one(hc, hw, args.k, args.tau, args.seed,
                               frame=fr, min_size=ms))
    else:
        out.append(run_one(hc, hw, args.k, args.tau, args.seed,
                           min_size=args.min_size))

    if args.out:
        import json
        json.dump(out, open(args.out, "w"), indent=2, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
