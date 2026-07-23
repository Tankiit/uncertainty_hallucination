#!/usr/bin/env python3
                            
                                                                       
                                                                    
                                                                  
 
                                                        
                                                                          
                                                                        
                                                                            
                                                                             
                                             
                                                                             
                                                                           
                                                                         
                                                                             
                                                                              
                                                    

import argparse
import numpy as np
from scipy import stats


                                                                                  
def stratified_diagnostic(W, landing_size, target, conf, size_only,
                          small_thresh=10):
    small = landing_size < small_thresh
    strata = {"small_|F|": small, "large_|F|": ~small}

    results = {}
    for name, mask in strata.items():
        n = int(mask.sum())
        if n < 30:
            results[name] = {"n": n, "note": "too few items, skip"}
            continue
        r_w_size = stats.pearsonr(W[mask], size_only[mask])[0]
        rho_w = stats.spearmanr(W[mask], target[mask])[0]
        rho_size = stats.spearmanr(size_only[mask], target[mask])[0]
                                                            
        rho_w_partial = partial_corr(W[mask], target[mask], conf[mask])
        results[name] = dict(n=n, r_W_size=r_w_size, rho_W=rho_w,
                             rho_size=rho_size, rho_W_partial_conf=rho_w_partial)
    return results


def partial_corr(x, y, z):
                                                   
    bx = np.polyfit(z, x, 1); rx = x - np.polyval(bx, z)
    by = np.polyfit(z, y, 1); ry = y - np.polyval(by, z)
    return stats.pearsonr(rx, ry)[0]


                                                                                 
def fit_frame(H, K, seed):
    from sklearn.cluster import KMeans
    km = KMeans(K, random_state=seed, n_init=5).fit(H)
    sizes = np.bincount(km.labels_, minlength=K).astype(float)
    sizes[sizes == 0] = 1.0
    return km.cluster_centers_, sizes


def width_and_landing(H, centroids, sizes, tau):
    from scipy.spatial.distance import cdist
    d = cdist(H, centroids)
    e = np.exp(-(d - d.min(-1, keepdims=True)) / tau)
    m = e / e.sum(-1, keepdims=True)
    coef = 1.0 - 1.0 / sizes
    W = m @ coef
    land = m.argmax(-1)
    return W, sizes[land], coef[land], m.max(-1)                                           


                                                                        
def load_halueval(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    hc = d["h_correct"].float().numpy().astype(np.float64)
    hw = d["h_wrong"].float().numpy().astype(np.float64)
    lpc = d["lp_correct"].float().numpy().astype(np.float64)
    lpw = d["lp_wrong"].float().numpy().astype(np.float64)
    return hc, hw, lpc, lpw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="outputs/llama3_8b/halueval_qa/hidden_states.pt")
    ap.add_argument("--k", type=int, nargs="+", default=[200, 400])
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.1, 0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--small_thresh", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hc, hw, lpc, lpw = load_halueval(args.cache)
    N = len(hc)
    H = np.vstack([hc, hw])                                                         
    target = np.concatenate([np.zeros(N), np.ones(N)])                              
    conf = np.concatenate([lpc, lpw])                                          
    print(f"cache: {args.cache}")
    print(f"{N} pairs -> {2*N} answer-rows | d={hc.shape[1]}")
    print(f"target = per-answer err (0 correct, 1 wrong); conf = generation logprob")
    print(f"small_|F| threshold: landing cluster size < {args.small_thresh}\n")

    import json
    dump = []
    for K in args.k:
                                                                             
        frames = {sd: fit_frame(H, K, sd) for sd in args.seeds}
                                                                     
        for tau in args.taus:
            agg = {"small_|F|": [], "large_|F|": []}
            ns = {"small_|F|": [], "large_|F|": []}
            for sd in args.seeds:
                centroids, sizes = frames[sd]
                W, Fland, size_only, maxmass = width_and_landing(
                    H, centroids, sizes, tau)
                res = stratified_diagnostic(W, Fland, target, conf, size_only,
                                            args.small_thresh)
                for name in ("small_|F|", "large_|F|"):
                    r = res[name]
                    ns[name].append(r["n"])
                    if "r_W_size" in r:
                        agg[name].append(r)
                    dump.append(dict(K=K, tau=tau, seed=sd, stratum=name, **r))

            print(f"===== K={K}  tau={tau} =====", flush=True)
            for name in ("small_|F|", "large_|F|"):
                rows = agg[name]
                nmean = np.mean(ns[name])
                if not rows:
                    print(f"  {name:9s}: n~{nmean:.0f}  (too few, skipped)")
                    continue
                def ms(key):
                    v = np.array([r[key] for r in rows])
                    return v.mean(), v.std()
                r_ws_m, r_ws_s = ms("r_W_size")
                rw_m, rw_s = ms("rho_W")
                rs_m, rs_s = ms("rho_size")
                rp_m, rp_s = ms("rho_W_partial_conf")
                print(f"  {name:9s}: n~{nmean:6.0f}  "
                      f"r(W,size)={r_ws_m:+.4f}+/-{r_ws_s:.4f}  "
                      f"rho_W={rw_m:+.4f}+/-{rw_s:.4f}  "
                      f"rho_size={rs_m:+.4f}+/-{rs_s:.4f}  "
                      f"rho_W|conf={rp_m:+.4f}+/-{rp_s:.4f}")
            print()

    if args.out:
        json.dump(dump, open(args.out, "w"), indent=2, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
