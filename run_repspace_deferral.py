#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, glob
import numpy as np

                                                                            
                                                                       
                                                                          
                                                   
from screen import (aurc, stc, compose_aurc_fixed, aurc_gap_bootstrap,
                    verdict_from_ci)

                                                                             
                                                                   
                                                                          
                                                                        
                             
try:
    from rsuq.extract import collect_states, logits_from_states, chunk_texts
    _HAS_EXTRACT = True
except Exception:
    _HAS_EXTRACT = False


                                                                             
def load_cache(path: str) -> dict:
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)

    def np_(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    model = str(d.get("model_key", d.get("model", "model")))
    dataset = str(d.get("dataset", os.path.basename(os.path.dirname(path)) or "cell"))
    cell = {"h_pos": np_(d["h_correct"]).astype(np.float32),
            "h_neg": np_(d["h_wrong"]).astype(np.float32)}
    if "lp_correct" in d and "lp_wrong" in d:
        cell["lp_pos"] = np_(d["lp_correct"]).astype(np.float32)
        cell["lp_neg"] = np_(d["lp_wrong"]).astype(np.float32)
    return {(model, dataset): cell}


                                                                                 
def load_live_model(model_id: str, device: str = "cuda", dtype: str | None = None):
    if not _HAS_EXTRACT:
        raise ImportError("rsuq.extract not available; cannot use --model mode")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch as _torch
    dt_map = {"fp16": _torch.float16, "bf16": _torch.bfloat16,
              "fp32": _torch.float32, None: None}
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dt_map.get(dtype)).to(device)
    return tok, model


def collect_qa_cells(tok, model, texts_correct, texts_wrong,
                     device: str = "cuda", block_size: int = 256,
                     batch_size: int = 32, pool: str = "last",
                     max_blocks: int | None = None,
                     model_key: str = "live", dataset_key: str = "qa"):
                                              
    blocks_c = chunk_texts(tok, texts_correct, block_size, max_blocks)
    blocks_w = chunk_texts(tok, texts_wrong, block_size, max_blocks)
    sc = collect_states(model, blocks_c, batch_size, device, pool)
    sw = collect_states(model, blocks_w, batch_size, device, pool)
                                                         
    def _lp(sc):
        z = logits_from_states(model, sc.h.to(device))
        logp = _torch_log_softmax(z, dim=-1)
        return logp.gather(-1, sc.gold.to(logp.device).unsqueeze(-1)).squeeze(-1).float().cpu().numpy()

    h_pos = sc.h.numpy()
    h_neg = sw.h.numpy()
    lp_pos = _lp(sc)
    lp_neg = _lp(sw)
    return {(model_key, dataset_key): {
        "h_pos": h_pos.astype(np.float32),
        "h_neg": h_neg.astype(np.float32),
        "lp_pos": lp_pos.astype(np.float32),
        "lp_neg": lp_neg.astype(np.float32),
    }}


def _torch_log_softmax(z, dim=-1):
    import torch
    return torch.log_softmax(z, dim=dim)


                                                                                   
def build_repspace_frame(H_all: np.ndarray, K: int, seed: int):
    from sklearn.cluster import KMeans
    km = KMeans(K, random_state=seed, n_init=5).fit(H_all)
    sizes = np.bincount(km.labels_, minlength=K).astype(float)
    return {"centroids": km.cluster_centers_, "sizes": sizes,
            "width_coef": 1 - 1 / np.maximum(sizes, 1.0)}


def mass_geometric(H, frame, tau=1.0):
    from scipy.spatial.distance import cdist
    d = cdist(H, frame["centroids"])
    e = np.exp(-(d - d.min(-1, keepdims=True)) / tau)
    return e / e.sum(-1, keepdims=True)


def width_from_mass(M, frame):
    return M @ frame["width_coef"]


def train_mass_head(H_tr, frame, mode, y_tr=None, seed=0):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    geo = mass_geometric(H_tr, frame)
    if mode == "unsupervised":
        reg = MLPRegressor(hidden_layer_sizes=(64,), max_iter=300,
                           random_state=seed).fit(H_tr, geo)
        def f(H):
            r = np.clip(reg.predict(H), 1e-6, None)
            return r / r.sum(-1, keepdims=True)
        return f
    tgt = geo.argmax(-1)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                        random_state=seed).fit(H_tr, tgt)
    K = frame["sizes"].shape[0]
    def f(H):
        p = clf.predict_proba(H)
        full = np.zeros((len(H), K))
        full[:, clf.classes_] = p
        full = np.clip(full, 1e-6, None)
        return full / full.sum(-1, keepdims=True)
    return f


                                                                            
def assert_cell_ok(h_pos, h_neg, y, lp_pos=None, lp_neg=None):
    assert h_pos.shape == h_neg.shape, "pos/neg must be paired, equal shape"
    assert set(np.unique(y).tolist()) == {0, 1}, "y must be the contrastive split"
                                                                                 
    N = len(h_pos)
    assert (y[:N] == 0).all() and (y[N:] == 1).all(), "label/order mismatch"
    if lp_pos is not None:
        assert len(lp_pos) == N and len(lp_neg) == N, "logprob foil misaligned"


                                                                                
def run_cell(h_pos, h_neg, K, seed, head_mode, lp_pos=None, lp_neg=None):
    N, d = h_pos.shape
    H_all = np.vstack([h_pos, h_neg])
    y = np.r_[np.zeros(N), np.ones(N)].astype(int)                           
    assert_cell_ok(h_pos, h_neg, y, lp_pos, lp_neg)

                                                                       
                                                 
    rng = np.random.default_rng(seed)
    idx = rng.permutation(2 * N)
    n_tr = int(1.2 * N)
    tr, te = idx[:n_tr], idx[n_tr:]

                                                                           
    from sklearn.linear_model import LogisticRegression
    if lp_pos is not None and lp_neg is not None:
        conf = -np.r_[lp_pos, lp_neg]
    else:
        fpred = LogisticRegression(max_iter=500).fit(H_all[tr], y[tr])
        conf = -fpred.decision_function(H_all)

                                   
    frame = build_repspace_frame(H_all, K, seed)
    W_geo = width_from_mass(mass_geometric(H_all, frame), frame)

    res = {}

                             
    a_conf, unc_conf = compose_aurc_fixed([conf[tr]], y[tr], [conf[te]], y[te])
    a_cw,  unc_confW = compose_aurc_fixed([conf[tr], W_geo[tr]], y[tr],
                                          [conf[te], W_geo[te]], y[te])
    arm = {"STC": stc(W_geo[te], y[te]),
           "aurc_conf": a_conf,
           "aurc_conf_plus_W": a_cw,
           "aurc_W_alone": aurc(W_geo[te], y[te])}
    if unc_conf is not None and unc_confW is not None:
        g_mean, g_lo, g_hi = aurc_gap_bootstrap(
            unc_conf, unc_confW, y[te], B=2000, seed=seed)
        arm.update({"gap_mean": g_mean, "gap_lo": g_lo, "gap_hi": g_hi,
                    "verdict": verdict_from_ci(g_mean, g_lo, g_hi)})
    else:
        arm.update({"gap_mean": None, "gap_lo": None, "gap_hi": None,
                    "verdict": "NULL (train single-class)"})
    res["geometric"] = arm

                                                        
    modes = (["supervised", "unsupervised"] if head_mode == "both"
             else [head_mode])
    for mode in modes:
        head = train_mass_head(H_all[tr], frame, mode, y_tr=y[tr], seed=seed)
        W_trn = width_from_mass(head(H_all), frame)
        a_conf_m, unc_conf_m = compose_aurc_fixed([conf[tr]], y[tr],
                                                   [conf[te]], y[te])
        a_cw_m,  unc_confW_m = compose_aurc_fixed(
            [conf[tr], W_trn[tr]], y[tr],
            [conf[te], W_trn[te]], y[te])
        arm = {"STC": stc(W_trn[te], y[te]),
               "aurc_conf": a_conf_m,
               "aurc_conf_plus_W": a_cw_m,
               "aurc_W_alone": aurc(W_trn[te], y[te])}
        if unc_conf_m is not None and unc_confW_m is not None:
            g_mean, g_lo, g_hi = aurc_gap_bootstrap(
                unc_conf_m, unc_confW_m, y[te], B=2000, seed=seed)
            arm.update({"gap_mean": g_mean, "gap_lo": g_lo, "gap_hi": g_hi,
                        "verdict": verdict_from_ci(g_mean, g_lo, g_hi)})
        else:
            arm.update({"gap_mean": None, "gap_lo": None, "gap_hi": None,
                        "verdict": "NULL (train single-class)"})
        res[f"trained_{mode}"] = arm

    return res


                                                                               
def aggregate_arms(seed_rows):
    arms = seed_rows[0].keys()
    agg = {}
    for arm in arms:
        rows = [r[arm] for r in seed_rows if arm in r]
        gaps = np.array([r["gap_mean"] for r in rows
                         if r["gap_mean"] is not None])
        stcs = np.array([r["STC"] for r in rows])
        a_conf = np.array([r["aurc_conf"] for r in rows
                           if r["aurc_conf"] is not None])
        a_cw   = np.array([r["aurc_conf_plus_W"] for r in rows
                           if r["aurc_conf_plus_W"] is not None])
        helps_strict = sum(
            1 for r in rows
            if r["verdict"].startswith("HELPS"))
        n_seeds = len(rows)
        agg[arm] = {
            "STC_mean": float(np.mean(stcs)) if len(stcs) else None,
            "STC_std":  float(np.std(stcs))  if len(stcs) else None,
            "aurc_conf_mean":          (float(np.mean(a_conf))
                                        if len(a_conf) else None),
            "aurc_conf_plus_W_mean":   (float(np.mean(a_cw))
                                        if len(a_cw) else None),
            "aurc_W_alone_mean":       float(np.mean(
                [r["aurc_W_alone"] for r in rows])),
            "gap_mean_across_seeds":   (float(np.mean(gaps))
                                        if len(gaps) else None),
            "gap_std_across_seeds":    (float(np.std(gaps))
                                        if len(gaps) else None),
            "n_seeds_gap_positive":    int((gaps > 0).sum()) if len(gaps) else 0,
            "n_seeds_gap_helps":       helps_strict,
            "n_seeds":                 n_seeds,
        }
    return agg


def verdict_aggregate(agg_row, noise_floor=0.02):
    mean_gap = agg_row["gap_mean_across_seeds"]
    n_helps  = agg_row["n_seeds_gap_helps"]
    n_seeds  = agg_row["n_seeds"]
    if mean_gap is None or n_seeds == 0:
        return "NULL (no valid gaps)"
    if mean_gap > noise_floor and n_helps >= max(1, n_seeds // 2):
        return ("DEFER: W adds info beyond logprob conf "
                f"(mean gap {mean_gap:+.4f} > {noise_floor}, "
                f"{n_helps}/{n_seeds} seeds CI-HELPS).")
    if mean_gap > 0:
        return (f"MARGINAL: mean gap {mean_gap:+.4f} > 0 but below noise "
                f"floor or inconsistent ({n_helps}/{n_seeds} seeds CI-HELPS).")
    return ("ABSTAIN: W does not consistently beat logprob alone "
            f"(mean gap {mean_gap:+.4f}).")


                                                                        
def _parse_k_sweep(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            ks = list(range(start, stop + 1))
        elif len(parts) == 3:
            start, stop, step = (int(p) for p in parts)
            ks = list(range(start, stop + 1, step))
        else:
            raise ValueError(f"bad --k_sweep range spec: {raw!r}")
    else:
        ks = [int(x) for x in raw.split(",") if x.strip()]
    if not ks:
        raise ValueError(f"--k_sweep parsed to empty list: {raw!r}")
    return sorted(set(ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None,
                    help="path or glob to cache file(s); required unless --model")
    ap.add_argument("--model", default=None,
                    help="HF model id for live state collection (e.g. 'gpt2', "
                         "'meta-llama/Llama-3-8B'). Requires rsuq.extract.")
    ap.add_argument("--dataset", default="live",
                    help="dataset label for the live-model cell")
    ap.add_argument("--device", default="cuda", help="torch device for live mode")
    ap.add_argument("--dtype", default=None,
                    choices=[None, "fp16", "bf16", "fp32"],
                    help="model dtype for live mode")
    ap.add_argument("--pool", default="last", choices=["last", "last4_mean"],
                    help="hidden-state pooling strategy")
    ap.add_argument("--block_size", type=int, default=256,
                    help="token block size for chunking")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="batch size for live forward passes")
    ap.add_argument("--k", type=int, default=50,
                    help="single K (ignored if --k_sweep is given)")
    ap.add_argument("--k_sweep", default=None,
                    help="comma list '20,50,100' or range '20:201:40'; "
                         "supersedes --k")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--head_mode", default="both",
                    choices=["supervised", "unsupervised", "both"])
    ap.add_argument("--out", default="results_repspace_deferral.json")
    args = ap.parse_args()

    if not args.cache and not args.model:
        ap.error("one of --cache or --model is required")

    ks = _parse_k_sweep(args.k_sweep) or [args.k]

                                                                      
    cells = {}
    if args.model:
        if not _HAS_EXTRACT:
            raise ImportError(
                "rsuq.extract not available — cannot use --model mode. "
                "Run from the rsuq/ directory or pip install -e .")
        print(f"loading live model: {args.model} ({args.dtype or 'default'})")
        tok, model = load_live_model(args.model, args.device, args.dtype)
                                                                    
        raise NotImplementedError(
            "Live-model collection requires a QA data loader to provide "
            "texts_correct / texts_wrong. Wire collect_qa_cells to your "
            "dataset (e.g. rsuq.qa_data.load_qa_instances) before using "
            "--model. See collect_qa_cells() signature for the contract.")
    else:
        paths = sorted(glob.glob(args.cache))
        if not paths:
            raise FileNotFoundError(f"no cache files match {args.cache!r}")
        for p in paths:
            cells.update(load_cache(p))
        print(f"loaded {len(cells)} (model,dataset) cells from {len(paths)} file(s); "
              f"K sweep = {ks}")

    results = {}
    verdicts = []
    for meta, data in cells.items():
        key = str(meta)
        results[key] = {"by_K": {}}
        h_pos, h_neg = data["h_pos"], data["h_neg"]
        lp_pos = data.get("lp_pos"); lp_neg = data.get("lp_neg")
        N = h_pos.shape[0]
        print(f"\n=== cell {key}  N={N} d={h_pos.shape[1]} ===")

        for K in ks:
            print(f"\n --- K={K} ---")
            seed_rows = []
            for seed in range(args.seeds):
                cell_res = run_cell(h_pos, h_neg, K, seed, args.head_mode,
                                    lp_pos=lp_pos, lp_neg=lp_neg)
                seed_rows.append(cell_res)
                for arm in ("geometric", "trained_supervised",
                            "trained_unsupervised"):
                    if arm not in cell_res:
                        continue
                    a = cell_res[arm]
                    gap_str = (f"{a['gap_mean']:+.4f}"
                               if a["gap_mean"] is not None else "  n/a")
                    print(f"  K={K} seed={seed} arm={arm:<22} "
                          f"STC={a['STC']:+.4f}  gap={gap_str}  "
                          f"{a['verdict']}")

            agg = aggregate_arms(seed_rows)
            results[key]["by_K"][K] = {"per_seed": seed_rows, "agg": agg}
            for arm, a in agg.items():
                v = verdict_aggregate(a)
                print(f"  [K={K}|{arm}] {v}")
                verdicts.append({
                    "cell": key, "K": K, "arm": arm,
                    "verdict": v,
                    "gap_mean_across_seeds": a["gap_mean_across_seeds"],
                    "n_seeds_gap_helps": a["n_seeds_gap_helps"],
                    "n_seeds": a["n_seeds"],
                    "STC_mean": a["STC_mean"],
                })

                                     
    print("\n" + "=" * 78)
    print("K-SWEEP SUMMARY  (gap = AURC_conf - AURC_conf+W;  positive = W helps)")
    print("=" * 78)
    hdr = f"{'cell':<32} {'K':>5} {'arm':<22} {'gap_mean':>9} {'helps':>6} {'STC':>7}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for v in verdicts:
        gap = v["gap_mean_across_seeds"]
        gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
        stc_str = (f"{v['STC_mean']:+.4f}"
                   if v["STC_mean"] is not None else "n/a")
                           
        vtag = ("DEFER" if v["verdict"].startswith("DEFER")
                else "MARG" if v["verdict"].startswith("MARGINAL")
                else "ABST")
        print(f"{v['cell']:<32} {v['K']:>5} {v['arm']:<22} "
              f"{gap_str:>9} {v['n_seeds_gap_helps']:>2}/{v['n_seeds']:<2} "
              f"{stc_str:>7}  {vtag}")

    summary = {
        "meta": {"source": args.cache or args.model,
                 "K_sweep": ks, "seeds": args.seeds,
                 "head_mode": args.head_mode, "n_cells": len(cells),
                 "bootstrap_B": 2000, "noise_floor": 0.02},
        "results": results,
        "verdicts": verdicts,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()