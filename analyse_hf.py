import modal, json, os, glob

app = modal.App("rsuq-analyse-hf")
vol = modal.Volume.from_name("rsuq-latent", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers", "safetensors", "numpy",
                      "pandas", "pyarrow", "scikit-learn", "scipy"))

CELLS = "/vol/hf/cells"


                                                                         
def _root(model, dataset):
    return f"{CELLS}/{model}__{dataset}"


def _meta(root):
    return json.load(open(f"{root}/meta.json"))


def _index(root):
    import pandas as pd
    return pd.read_parquet(f"{root}/index.parquet")


def _pool(span, scheme):
    return {"first": span[0], "last": span[-1],
            "span_mean": span.mean(0), "span_max": span.max(0)}[scheme]


def _arms(meta_or_idx_cols):
    return ("correct", "wrong")


def load_states(root, meta, layer, arm, scheme):
    import numpy as np
    from safetensors import safe_open
    li = meta["captured_layers"].index(layer)                              
    ids, vecs = [], []
    for p in sorted(glob.glob(f"{root}/states/shard-*.safetensors")):
        with safe_open(p, framework="np") as f:
            for k in f.keys():
                i, a, field = k.split(".")
                if a != arm or field != "h":
                    continue
                span = f.get_slice(k)[li]                       
                span = np.asarray(span, dtype=np.float32)
                if span.shape[0] == 0:                                      
                    continue
                ids.append(int(i)); vecs.append(_pool(span, scheme))
    if not vecs:
        raise RuntimeError(
            f"no states for arm={arm!r} layer={layer} in {root}. "
            f"Available arms are set at extraction time; a disagreement cell "
            f"uses 'main'.")
    o = np.argsort(ids)
    return np.asarray(ids)[o], np.stack(vecs)[o]


def load_logits(root, arm, field):
    import numpy as np
    from safetensors.numpy import load_file
    out = {}
    for p in sorted(glob.glob(f"{root}/logits/shard-*.safetensors")):
        for k, v in load_file(p).items():
            i, a, f_ = k.split(".")
            if a == arm and f_ == field:
                out[int(i)] = v
    return out


def has_dispersion(meta):
    return meta.get("kind") == "disagreement" or meta.get("has_dispersion_target", False)


def annotator_entropy_from_index(idx):
    import numpy as np
    col = ("soft_labels" if "soft_labels" in idx.columns
           else "annotator_labels" if "annotator_labels" in idx.columns else None)
    if col is None:
        raise RuntimeError("index.parquet carries no soft_labels/annotator_labels")
    ent, ids = [], []
    for iid, raw in zip(idx.item_id, idx[col]):
        v = json.loads(raw) if isinstance(raw, str) else raw
        p = np.asarray(v, float).ravel()
        if col == "annotator_labels" and p.size and float(p.max()) == p.max().astype(int):
            lab = p.astype(int)
            if lab.min() >= 0 and lab.size > 1 and p.sum() != 1.0:
                p = np.bincount(lab, minlength=int(lab.max()) + 1).astype(float)
        p = p / max(p.sum(), 1e-12)
        ent.append(float(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
        ids.append(int(iid))
    o = np.argsort(ids)
    return np.asarray(ids)[o], np.asarray(ent)[o]


def credal_width(m, sizes):
    import numpy as np
    return (m * (1.0 - 1.0 / np.asarray(sizes, float))).sum(-1)


def soft_mass(H, centroids, tau=1.0):
    import numpy as np
    from scipy.spatial.distance import cdist
    d = cdist(H, centroids)
    e = np.exp(-(d - d.min(-1, keepdims=True)) / tau)
    return e / e.sum(-1, keepdims=True)


def auc(x_correct, x_wrong):
    import numpy as np
    from scipy.stats import rankdata
    n1, n2 = len(x_wrong), len(x_correct)
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(np.concatenate([x_wrong, x_correct]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def boot_rho(s, ent, B=500, seed=0):
    import numpy as np
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)                                        
    n = len(s)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        out[b] = spearmanr(s[i], ent[i])[0]
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def pooled_gain_ci(per_seed, ent, B=2000, seed=0):
    import numpy as np
    from scipy.stats import spearmanr

    def gain(w, so, idx=None):
        if idx is None:
            return abs(spearmanr(w, ent)[0]) - abs(spearmanr(so, ent)[0])
        return (abs(spearmanr(w[idx], ent[idx])[0])
                - abs(spearmanr(so[idx], ent[idx])[0]))

    per_seed_obs = [gain(w, so) for w, so in per_seed]
    obs = float(np.mean(per_seed_obs))
    rng = np.random.default_rng(seed)
    n = len(ent)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)                                         
        out[b] = float(np.mean([gain(w, so, i) for w, so in per_seed]))
                                                                             
                                   
    n_pos = sum(1 for g in per_seed_obs if g > 0)
    return (obs, float(np.nanpercentile(out, 2.5)),
            float(np.nanpercentile(out, 97.5)),
            float(np.std(per_seed_obs)), n_pos)


def boot_rho_gain(w, so, ent, B=2000, seed=0):
    import numpy as np
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    n = len(ent)
    g = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)                                              
        g[b] = abs(spearmanr(w[i], ent[i])[0]) - abs(spearmanr(so[i], ent[i])[0])
    obs = abs(spearmanr(w, ent)[0]) - abs(spearmanr(so, ent)[0])
    return (float(obs), float(np.nanpercentile(g, 2.5)),
            float(np.nanpercentile(g, 97.5)), float(np.mean(g > 0)))


def _signals(m, sizes):
    import numpy as np
    return {"credal_width": credal_width(m, sizes),
            "mass_entropy": -(m * np.log(np.clip(m, 1e-12, None))).sum(-1),
            "one_minus_max": 1 - m.max(-1),
            "size_only": 1 - 1 / sizes[m.argmax(-1)]}                        


                                                                     
@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 2)
def dispersion(model: str, dataset: str, scheme: str = "span_mean",
               layer: int = -1, n_clusters: int = 200, taus: str = "0.5,1,2",
               seeds: str = "0,1,2", arm: str = "main"):
    import numpy as np, pandas as pd
    from sklearn.cluster import KMeans
    from scipy.stats import spearmanr

    root = _root(model, dataset); meta = _meta(root)
    assert has_dispersion(meta), f"{dataset} is not a disagreement cell"
    if layer < 0:
        layer = meta["n_total_layers"] + 1 + layer
    assert layer in meta["captured_layers"],\
        f"layer {layer} not captured; have {meta['captured_layers']}"

    ids_h, H = load_states(root, meta, layer, arm, scheme)
    ids_e, ent = annotator_entropy_from_index(_index(root))
    keep = np.intersect1d(ids_h, ids_e)
    H = H[np.isin(ids_h, keep)]
    ent = ent[np.isin(ids_e, keep)]
    print(f"{len(keep)} items | H={H.shape} | layer={layer} arm={arm}")
    if ent.std() < 1e-6:
        raise RuntimeError("annotator entropy is constant — no dispersion target")
                                                                          
                                                               
    q = np.percentile(ent, [0, 5, 25, 50, 75, 95, 100])
    print(f"  annot-entropy mean={ent.mean():.4f} std={ent.std():.4f}  "
          f"pcts 0/5/25/50/75/95/100 = " + "/".join(f"{x:.3f}" for x in q))
    print(f"  frac within +/-0.1 of median: "
          f"{float(np.mean(np.abs(ent - np.median(ent)) < 0.1)):.3f}  "
          f"(high => concentrated target, rho is weakly identified)")

    seed_list = [int(s) for s in seeds.split(",")]
    tau_list = [float(t) for t in taus.split(",")]
    rows = []
                                                                           
                                                                             
    cache = {tau: [] for tau in tau_list}
    for sd in seed_list:
        km = KMeans(n_clusters, random_state=sd, n_init=4).fit(H)
        sizes = np.bincount(km.labels_, minlength=n_clusters).astype(float)
        sizes[sizes == 0] = 1
        for tau in tau_list:
            m = soft_mass(H, km.cluster_centers_, tau)
            sig = _signals(m, sizes)
                                                                          
                                                                              
                                                                             
            r_ws = float(np.corrcoef(sig["credal_width"], sig["size_only"])[0, 1])
            cache[tau].append((sig["credal_width"], sig["size_only"]))
            for name, s in sig.items():
                rho, p = spearmanr(s, ent)
                rows.append(dict(seed=sd, tau=tau, signal=name,
                                 spearman_rho=float(rho), p=float(p),
                                 corr_W_vs_size_only=r_ws,
                                 median_max_mass=float(np.median(m.max(-1)))))

                                                           
    pooled = {}
    for tau in tau_list:
        pooled[tau] = pooled_gain_ci(cache[tau], ent, seed=0)

    df = pd.DataFrame(rows)
    dst = f"/vol/derived/{model}__{dataset}"; os.makedirs(dst, exist_ok=True)
    df.to_parquet(f"{dst}/A1_dispersion_L{layer}_{scheme}.parquet"); vol.commit()

                                                                            
                                                                             
                                                                             
                                                                             
                                    
     
                                                                           
                                                                          
                                                                        
                                                                          
                                                                          
                                                                 
    print("\n=== per-tau verdict (never averaged over tau) ===")
    print(f"{'tau':>6} {'med max_m':>10} {'r(W,size)':>10} {'rho(W)':>9} "
          f"{'rho(size)':>10} {'gain':>8} {'pooled 95% CI':>18} {'seedSD':>7}  status")
    verdicts = []
    for tau in sorted(df.tau.unique()):
        sub = df[df.tau == tau]
        cw = sub[sub.signal == "credal_width"].spearman_rho.mean()
        so = sub[sub.signal == "size_only"].spearman_rho.mean()
        mm = sub.median_max_mass.mean()
        rws = sub.corr_W_vs_size_only.mean()
        gain, glo, ghi, seed_sd, n_pos = pooled[tau]
        n_seeds = len(seed_list)
        if mm > 0.90 or rws > 0.98:
            status = "UNINFORMATIVE (W == size_only algebraically)"
        elif glo > 0 and n_pos == n_seeds:
            status = "W BEATS density null (pooled CI>0, all seeds agree)"
        elif glo > 0:
            status = f"W beats, but only {n_pos}/{n_seeds} seeds positive"
        elif ghi < 0 and n_pos == 0:
            status = "W WORSE than density null (pooled CI<0, all seeds agree)"
        elif ghi < 0:
            status = f"W worse, but {n_pos}/{n_seeds} seeds positive"
        else:
            status = "tie -> W is a size readout (CI spans 0)"
        verdicts.append(dict(tau=float(tau), median_max_mass=float(mm),
                             corr_W_vs_size_only=float(rws),
                             rho_W=float(cw), rho_size=float(so),
                             gain=float(gain), gain_lo=float(glo),
                             gain_hi=float(ghi), between_seed_sd=float(seed_sd),
                             n_seeds_positive=int(n_pos), status=status))
        print(f"{tau:6.2f} {mm:10.4f} {rws:10.4f} {cw:+9.4f} {so:+10.4f} "
              f"{gain:+8.4f} [{glo:+.4f},{ghi:+.4f}] {seed_sd:7.4f}  {status}")

                                                                 
    informative = [v for v in verdicts
                   if v["median_max_mass"] <= 0.90
                   and v["corr_W_vs_size_only"] <= 0.98]
    print(f"\ninformative taus (max-mass <= 0.90 AND r(W,size_only) <= 0.98): "
          f"{[v['tau'] for v in informative] or 'NONE'}")
    if not informative:
        print("  -> at every tau tested the mass is peaked, so this cell cannot")
        print("     separate W from the density null. Push tau higher — but note")
        print("     that as mass goes uniform W collapses to a constant, so a")
        print("     window that is both informative and non-degenerate may not")
        print("     exist. THAT would itself be the finding.")
    return {"rows": df.to_dict("records"), "verdicts": verdicts}


                                                                      
@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 3)
def uq_panel(model: str, dataset: str, scheme: str = "span_mean",
             layer: int = -1, n_clusters: int = 200, tau: float = 10.0,
             seeds: str = "0,1,2", arm: str = "correct", knn_k: int = 10):
    import numpy as np, pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import spearmanr

    root = _root(model, dataset); meta = _meta(root)
    assert has_dispersion(meta), f"{dataset} is not a disagreement cell"
    if layer < 0:
        layer = meta["n_total_layers"] + 1 + layer

    ids_h, H = load_states(root, meta, layer, arm, scheme)
    ids_e, ent = annotator_entropy_from_index(_index(root))
    lp = load_logits(root, arm, "topk_logprobs")
    rl = load_logits(root, arm, "realised_logprob")
    keep = np.intersect1d(np.intersect1d(ids_h, ids_e),
                          np.array(sorted(set(lp) & set(rl))))
    sel_h = np.isin(ids_h, keep); sel_e = np.isin(ids_e, keep)
    H = H[sel_h]; ent = ent[sel_e]
    order = [int(i) for i in keep]
    print(f"{len(keep)} items | layer={layer} arm={arm} tau={tau}")

                                                                            
                                                                              
                                                                            
                                                                            
                                                           
    pe, pm, pg, nl = [], [], [], []
    for i in order:
        L = np.asarray(lp[i], dtype=np.float64)                  
        if L.shape[0] == 0:
            pe.append(np.nan); pm.append(np.nan); pg.append(np.nan)
            nl.append(np.nan); continue
        p = np.exp(L)
        tail = np.clip(1.0 - p.sum(-1), 1e-12, None)
        H_tok = -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)\
                - tail * np.log(tail)
        srt = np.sort(p, -1)[:, ::-1]
        pe.append(float(np.mean(H_tok)))
        pm.append(float(np.mean(1.0 - srt[:, 0])))
        pg.append(float(np.mean(1.0 - (srt[:, 0] - srt[:, 1]))))
        r = np.asarray(rl[i], dtype=np.float64)
        nl.append(float(-np.mean(r)) if r.size else np.nan)

                                                                            
    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, len(H)),
                          algorithm="brute").fit(H)
    dist, _ = nn.kneighbors(H)
    knn = dist[:, 1:].mean(1)

                                                                            
    acc = {}
    for sd in [int(s) for s in seeds.split(",")]:
        km = KMeans(n_clusters, random_state=sd, n_init=4).fit(H)
        sizes = np.bincount(km.labels_, minlength=n_clusters).astype(float)
        sizes[sizes == 0] = 1
        sig = _signals(soft_mass(H, km.cluster_centers_, tau), sizes)
        for k, v in sig.items():
            acc.setdefault(k, []).append(v)
    metrics = {k: np.mean(v, axis=0) for k, v in acc.items()}
    metrics.update({"knn_dist": knn,
                    "pred_entropy": np.asarray(pe),
                    "pred_maxprob": np.asarray(pm),
                    "pred_margin": np.asarray(pg),
                    "neg_mean_logprob": np.asarray(nl)})

    GROUP = {"credal_width": "mass", "mass_entropy": "mass",
             "one_minus_max": "mass", "size_only": "density",
             "knn_dist": "density", "pred_entropy": "token",
             "pred_maxprob": "token", "pred_margin": "token",
             "neg_mean_logprob": "token"}

    rows = []
    for name, v in metrics.items():
        ok = np.isfinite(v) & np.isfinite(ent)
        if ok.sum() < 30:
            continue
        rho, p = spearmanr(v[ok], ent[ok])
        lo, hi = boot_rho(v[ok], ent[ok], B=1000, seed=0)
        rows.append(dict(metric=name, group=GROUP.get(name, "?"),
                         rho=float(rho), p=float(p), ci_lo=lo, ci_hi=hi,
                         abs_rho=abs(float(rho)), n=int(ok.sum())))
    df = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)

    print(f"\n{'metric':18s} {'group':8s} {'rho':>9} {'95% CI':>20} {'|rho|':>8}")
    for _, r in df.iterrows():
        print(f"{r.metric:18s} {r.group:8s} {r.rho:+9.4f} "
              f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}] {r.abs_rho:8.4f}")

                                                                             
    print(f"\n=== credal_width vs each baseline (paired bootstrap of the "
          f"difference in |rho|) ===")
    w = metrics["credal_width"]
    okw = np.isfinite(w) & np.isfinite(ent)
    comp = []
    for name, v in metrics.items():
        if name == "credal_width":
            continue
        ok = okw & np.isfinite(v)
        g, lo, hi, _, _ = pooled_gain_ci([(w[ok], v[ok])], ent[ok], B=1000)
        verdict = ("W better" if lo > 0 else
                   "W worse" if hi < 0 else "tie")
        comp.append(dict(baseline=name, gain=g, lo=lo, hi=hi, verdict=verdict))
        print(f"  vs {name:18s} gain={g:+.4f} [{lo:+.4f},{hi:+.4f}]  {verdict}")

    dst = f"/vol/derived/{model}__{dataset}"; os.makedirs(dst, exist_ok=True)
    df.to_parquet(f"{dst}/A1b_uq_panel_L{layer}_tau{tau}.parquet"); vol.commit()
    return {"panel": df.to_dict("records"), "vs_width": comp}


                                                                      
@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 4)
def uq_panel_correctness(model: str, dataset: str, scheme: str = "span_mean",
                         layer: int = -1, n_clusters: int = 200,
                         tau: float = 10.0, seeds: str = "0,1,2",
                         knn_k: int = 10, match_len: bool = True):
    import numpy as np, pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    root = _root(model, dataset); meta = _meta(root)
    if layer < 0:
        layer = meta["n_total_layers"] + 1 + layer
    idx = _index(root)
    ids_c, Hc = load_states(root, meta, layer, "correct", scheme)
    ids_w, Hw = load_states(root, meta, layer, "wrong", scheme)
    keep = np.intersect1d(ids_c, ids_w)
    Hc = Hc[np.isin(ids_c, keep)]; Hw = Hw[np.isin(ids_w, keep)]
    n = len(keep)
    H = np.vstack([Hc, Hw])                                             
    print(f"{n} paired items | H={H.shape} | layer={layer} tau={tau}")

    def token_metrics(arm):
        lp = load_logits(root, arm, "topk_logprobs")
        rl = load_logits(root, arm, "realised_logprob")
        pe, pm, pg, nl = [], [], [], []
        for i in [int(x) for x in keep]:
            L = np.asarray(lp.get(i, np.zeros((0, 1))), dtype=np.float64)
            if L.shape[0] == 0:
                pe += [np.nan]; pm += [np.nan]; pg += [np.nan]; nl += [np.nan]
                continue
            p = np.exp(L)
            tail = np.clip(1.0 - p.sum(-1), 1e-12, None)
            Ht = -(p * np.log(np.clip(p, 1e-12, None))).sum(-1) - tail * np.log(tail)
            srt = np.sort(p, -1)[:, ::-1]
            pe.append(float(np.mean(Ht)))
            pm.append(float(np.mean(1.0 - srt[:, 0])))
            pg.append(float(np.mean(1.0 - (srt[:, 0] - srt[:, 1]))))
            r = np.asarray(rl.get(i, []), dtype=np.float64)
            nl.append(float(-np.mean(r)) if r.size else np.nan)
        return dict(pred_entropy=np.array(pe), pred_maxprob=np.array(pm),
                    pred_margin=np.array(pg), neg_mean_logprob=np.array(nl))

    tok_c, tok_w = token_metrics("correct"), token_metrics("wrong")

    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, len(H)),
                          algorithm="brute").fit(H)
    d, _ = nn.kneighbors(H)
    knn = d[:, 1:].mean(1)

    acc = {}
    for sd in [int(s) for s in seeds.split(",")]:
        km = KMeans(n_clusters, random_state=sd, n_init=4).fit(H)
        sizes = np.bincount(km.labels_, minlength=n_clusters).astype(float)
        sizes[sizes == 0] = 1
        for k, v in _signals(soft_mass(H, km.cluster_centers_, tau), sizes).items():
            acc.setdefault(k, []).append(v)
    metrics = {k: np.mean(v, axis=0) for k, v in acc.items()}
    metrics["knn_dist"] = knn
    pairs = {k: (v[:n], v[n:]) for k, v in metrics.items()}
    for k in tok_c:
        pairs[k] = (tok_c[k], tok_w[k])

                           
    sub = None
    if match_len and {"n_ans_tok_correct", "n_ans_tok_wrong"} <= set(idx.columns):
        j = idx.set_index("item_id").loc[[int(x) for x in keep]]
        sub = (j.n_ans_tok_correct.to_numpy() == j.n_ans_tok_wrong.to_numpy())
        print(f"  equal-length arms: {sub.mean()*100:.1f}% of items "
              f"({int(sub.sum())}/{n})")

    def theta_ci(c, w, mask=None, B=1000, seed=0):
        ok = np.isfinite(c) & np.isfinite(w)
        if mask is not None:
            ok &= mask
        if ok.sum() < 30:
            return (float("nan"),) * 3
        c2, w2 = c[ok], w[ok]
        rng = np.random.default_rng(seed)
        bs = np.array([auc(c2[i], w2[i]) for i in
                       (rng.integers(0, len(c2), len(c2)) for _ in range(B))])
        return (auc(c2, w2), float(np.nanpercentile(bs, 2.5)),
                float(np.nanpercentile(bs, 97.5)))

    GROUP = {"credal_width": "mass", "mass_entropy": "mass",
             "one_minus_max": "mass", "size_only": "density",
             "knn_dist": "density", "pred_entropy": "token",
             "pred_maxprob": "token", "pred_margin": "token",
             "neg_mean_logprob": "token"}
    rows = []
    for k, (c, w) in pairs.items():
        th, lo, hi = theta_ci(c, w)
        tm = theta_ci(c, w, sub)[0] if sub is not None else float("nan")
        rows.append(dict(metric=k, group=GROUP.get(k, "?"), theta=th,
                         ci_lo=lo, ci_hi=hi, theta_lenmatched=tm,
                         disc=abs(th - 0.5)))
    df = pd.DataFrame(rows).sort_values("disc", ascending=False)
    print(f"\n{'metric':18s} {'group':8s} {'theta':>8} {'95% CI':>20} "
          f"{'|d-.5|':>7} {'len-matched':>12}")
    for _, r in df.iterrows():
        print(f"{r.metric:18s} {r.group:8s} {r.theta:8.4f} "
              f"[{r.ci_lo:.4f},{r.ci_hi:.4f}] {r.disc:7.4f} {r.theta_lenmatched:12.4f}")

    print("\n=== credal_width vs each baseline (paired bootstrap of "
          "|theta-.5| difference) ===")
    wc, ww = pairs["credal_width"]
    for k, (c, w) in pairs.items():
        if k == "credal_width":
            continue
        ok = np.isfinite(c) & np.isfinite(w) & np.isfinite(wc) & np.isfinite(ww)
        rng = np.random.default_rng(0)
        g = np.array([abs(auc(wc[ok][i], ww[ok][i]) - .5) - abs(auc(c[ok][i], w[ok][i]) - .5)
                      for i in (rng.integers(0, int(ok.sum()), int(ok.sum()))
                                for _ in range(1000))])
        obs = abs(auc(wc[ok], ww[ok]) - .5) - abs(auc(c[ok], w[ok]) - .5)
        lo, hi = np.nanpercentile(g, [2.5, 97.5])
        print(f"  vs {k:18s} gain={obs:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"{'W better' if lo > 0 else 'W worse' if hi < 0 else 'tie'}")

    dst = f"/vol/derived/{model}__{dataset}"; os.makedirs(dst, exist_ok=True)
    df.to_parquet(f"{dst}/A1d_uq_correctness_L{layer}_tau{tau}.parquet"); vol.commit()
    return df.to_dict("records")


                                                                      
@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 2)
def ensemble_disagreement(dataset: str, models: str = "llama3_8b,mistral_7b,qwen2_5_7b",
                          scheme: str = "span_mean", layer: int = -1,
                          n_clusters: int = 200, tau: float = 10.0, seed: int = 0):
    import numpy as np, pandas as pd
    from scipy.stats import spearmanr

    ms = models.split(",")
    prefs, common = {}, None
    for m in ms:
        root = _root(m, dataset)
        rc = load_logits(root, "correct", "realised_logprob")
        rw = load_logits(root, "wrong", "realised_logprob")
        ids = sorted(set(rc) & set(rw))
        d = {i: float(np.mean(rc[i]) - np.mean(rw[i]))
             for i in ids if np.size(rc[i]) and np.size(rw[i])}
        prefs[m] = d
        common = set(d) if common is None else common & set(d)
    ids_e, ent = annotator_entropy_from_index(_index(_root(ms[0], dataset)))
    common = sorted(common & set(int(i) for i in ids_e))
    ent = ent[np.isin(ids_e, np.array(common))]
    P = np.stack([[prefs[m][i] for i in common] for m in ms])              
    print(f"{dataset}: {len(common)} items shared across {len(ms)} models")

    Z = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-12)
    pref_std = Z.std(0)
    votes = (P > 0).mean(0)
    vote_entropy = -(np.where(votes > 0, votes * np.log2(np.clip(votes, 1e-12, None)), 0)
                     + np.where(votes < 1, (1 - votes) * np.log2(np.clip(1 - votes, 1e-12, None)), 0))

    for m in ms:
        print(f"  {m:12s} prefers correct on {np.mean(np.array([prefs[m][i] for i in common]) > 0)*100:5.1f}% of items")
    print(f"  all three agree on {np.mean((votes == 0) | (votes == 1))*100:.1f}% of items")

    out = []
    for name, v in [("ensemble_pref_std", pref_std),
                    ("ensemble_vote_entropy", vote_entropy)]:
        rho, p = spearmanr(v, ent)
        lo, hi = boot_rho(v, ent, B=1000, seed=0)
        out.append(dict(metric=name, rho=float(rho), p=float(p),
                        ci_lo=lo, ci_hi=hi, n=len(ent)))
        print(f"  {name:22s} rho={rho:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.2e}")

                                                                  
    from sklearn.cluster import KMeans
    root = _root(ms[0], dataset); meta = _meta(root)
    lay = meta["n_total_layers"] + 1 + layer if layer < 0 else layer
    ids_h, H = load_states(root, meta, lay, "correct", scheme)
    m_ok = np.isin(ids_h, np.array(common))
    H = H[m_ok]
    km = KMeans(n_clusters, random_state=seed, n_init=4).fit(H)
    sizes = np.bincount(km.labels_, minlength=n_clusters).astype(float)
    sizes[sizes == 0] = 1
    W = _signals(soft_mass(H, km.cluster_centers_, tau), sizes)["credal_width"]
    for name, v in [("ensemble_pref_std", pref_std),
                    ("ensemble_vote_entropy", vote_entropy)]:
        g, lo, hi, _, _ = pooled_gain_ci([(W, v)], ent, B=1000)
        print(f"  credal_width vs {name:22s} gain={g:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{'W better' if lo > 0 else 'W worse' if hi < 0 else 'tie'}")
    return out


                                                                     
@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 4)
def layer_sweep(model: str, dataset: str, scheme: str = "span_mean",
                n_clusters: int = 200, tau: float = 1.0, seed: int = 0):
    import numpy as np, pandas as pd
    from sklearn.cluster import KMeans
    from scipy.stats import spearmanr

    root = _root(model, dataset); meta = _meta(root)
    idx = _index(root)
    paired = "degenerate_first_token" in idx.columns
    ent = None
    if has_dispersion(meta):
        _, ent = annotator_entropy_from_index(idx)

    rows = []
    for L in meta["captured_layers"]:
        if paired:
            _, Hc = load_states(root, meta, L, "correct", scheme)
            _, Hw = load_states(root, meta, L, "wrong", scheme)
            H = np.vstack([Hc, Hw]); n = len(Hc)
        else:
            _, H = load_states(root, meta, L, "main", scheme); n = None
        km = KMeans(n_clusters, random_state=seed, n_init=4).fit(H)
        sizes = np.bincount(km.labels_, minlength=n_clusters).astype(float)
        sizes[sizes == 0] = 1
        m = soft_mass(H, km.cluster_centers_, tau)
        Wv = credal_width(m, sizes)
        r = dict(layer=L, median_max_mass=float(np.median(m.max(-1))),
                 size_min=float(sizes.min()), size_med=float(np.median(sizes)),
                 size_max=float(sizes.max()),
                                                               
                 corr_W_sizeonly=float(np.corrcoef(
                     Wv, 1 - 1 / sizes[m.argmax(-1)])[0, 1]))
        if paired:
            r["theta_marginal"] = auc(Wv[:n], Wv[n:])                        
        if ent is not None and len(ent) == len(Wv):
            r["rho_W_annotator_entropy"] = float(spearmanr(Wv, ent)[0])
        rows.append(r)
        print(f"  L{L:3d} maxm={r['median_max_mass']:.3f} "
              f"corr_size={r['corr_W_sizeonly']:+.4f}"
              + (f" theta={r['theta_marginal']:.4f}" if paired else "")
              + (f" rho_ann={r.get('rho_W_annotator_entropy', float('nan')):+.4f}"
                 if "rho_W_annotator_entropy" in r else ""), flush=True)

    df = pd.DataFrame(rows)
    dst = f"/vol/derived/{model}__{dataset}"; os.makedirs(dst, exist_ok=True)
    df.to_parquet(f"{dst}/A2_layer_sweep_{scheme}.parquet"); vol.commit()
    return df.to_dict("records")


                                                                     
@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 2,
              secrets=[modal.Secret.from_name("huggingface")])
def decoder_frame(model: str, dataset: str, topk: int = 20):
    import numpy as np, pandas as pd
    from transformers import AutoTokenizer

    root = _root(model, dataset); meta = _meta(root)
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    idx = _index(root)
    paired = "degenerate_first_token" in idx.columns
    arms = ("correct", "wrong") if paired else ("main",)

                                                              
    cache = {}
    def norm(t):
        t = int(t)
        if t not in cache:
            cache[t] = tok.convert_ids_to_tokens(t).lstrip("Ġ▁ ").lower()
        return cache[t]

    def frame_width(tk_ids, tk_lp):
        ws, ncls = [], []
        for pos in range(tk_ids.shape[0]):
            ids = tk_ids[pos][:topk]
            lp = tk_lp[pos][:topk].astype(np.float64)
            p = np.exp(lp - lp.max()); p /= p.sum()
            cls = {}
            for j, t in enumerate(ids):
                cls.setdefault(norm(t), []).append(j)
            mass = np.array([p[ix].sum() for ix in cls.values()])
            size = np.array([len(ix) for ix in cls.values()], float)
            ws.append(float((mass * (1 - 1 / size)).sum()))
            ncls.append(len(cls))
        return float(np.mean(ws)), float(np.mean(ncls))

    per_arm = {}
    for arm in arms:
        ids_lp = load_logits(root, arm, "topk_logprobs")
        ids_id = load_logits(root, arm, "topk_ids")
        rec = {}
        for i in sorted(ids_id):
            if ids_id[i].shape[0] == 0:
                continue
            rec[i] = frame_width(ids_id[i], ids_lp[i])
        per_arm[arm] = rec

    common = sorted(set.intersection(*[set(v) for v in per_arm.values()]))
    df = pd.DataFrame([
        dict(item_id=i, **{f"{stat}_{arm}": per_arm[arm][i][j]
                           for arm in arms for j, stat in enumerate(("W", "nclass"))})
        for i in common])

    nc = df.filter(like="nclass_").to_numpy().mean()
    print(f"decoder frame: mean {nc:.1f} surface classes per position "
          f"of top-{topk}  -> focal sets are SMALL (the saturation precondition)")
    print(f"  mean |F_k| implied = {topk / max(nc, 1e-9):.2f}; "
          f"coefficient 1-1/|F| has real range only below ~10")
    if paired:
        th = auc(df.W_correct.to_numpy(), df.W_wrong.to_numpy())
        print(f"  theta (marginal MWU) P(W_wrong > W_correct) = {th:.4f}")
    dst = f"/vol/derived/{model}__{dataset}"; os.makedirs(dst, exist_ok=True)
    df.to_parquet(f"{dst}/A3_decoder_frame_k{topk}.parquet"); vol.commit()
    return df.head(20).to_dict("records")


                                                                         
@app.function(image=image, volumes={"/vol": vol})
def report():
    import pandas as pd
    for p in sorted(glob.glob("/vol/derived/*/*.parquet")):
        print(f"\n=== {p} ===")
        print(pd.read_parquet(p).head(12).to_string(index=False))


@app.function(image=image, volumes={"/vol": vol})
def cells():
    import pandas as pd
    rows = []
    for mp in sorted(glob.glob(f"{CELLS}/*/meta.json")):
        m = json.load(open(mp)); root = os.path.dirname(mp)
        ns = len(glob.glob(f"{root}/states/shard-*.safetensors"))
        gb = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fs in os.walk(root) for f in fs) / 1e9
        rows.append(dict(cell=m["cell"], kind=m.get("kind"), tier=m.get("tier"),
                         layers=len(m["captured_layers"]), n=m["n_items"],
                         shards=ns, GB=round(gb, 2),
                         dispersion=has_dispersion(m)))
    if not rows:
        print(f"no cells under {CELLS} — run extract_hf.py first"); return
    print(pd.DataFrame(rows).to_string(index=False))
