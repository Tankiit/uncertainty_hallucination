"""ARR ablations -- the analyses the paper promises but does not yet have.

Companion to arr_modal.py; same app, same volume, same conventions (one
file, heavy imports inside function bodies so `modal run` works from a login
node). Import the shared helpers from arr_modal rather than re-deriving
them: a second copy of _rho or _partial_rho will drift, and then two numbers
in the paper will disagree for a reason nobody can reconstruct.

    modal run arr_ablations.py::length_panel   --model=… --dataset=…
    modal run arr_ablations.py::pool_ablation  --model=… --dataset=…
    modal run arr_ablations.py::shuffle_control --model=… --dataset=…
    modal run arr_ablations.py::grid_ablation  --model=… --dataset=…
    modal run arr_ablations.py::appendix_dump

    modal run arr_ablations.py --stage=length   # fan out over the 9 cells

PRIORITY. Run in this order; each answers a question the paper currently
asserts rather than shows.

  1. length_panel   -- does the token-count dependence generalise beyond
                       ChaosNLI, and does the inversion against annotator
                       entropy survive controlling for it? The mechanism
                       section covers only the cells where the divergence is
                       computable, which excludes BOTH Pavlick cells -- and
                       Pavlick carries the largest inversion in the paper
                       (-0.396). This is the hole.
  2. pool_ablation  -- first-token vs span-mean. Turns the pooling claim
                       from an argument into a demonstration.
  3. shuffle_control-- the negative control the paper does not have.
  4. grid_ablation  -- K, layer, neighbourhood size. Expected, not decisive.
  5. appendix_dump  -- assemble what is already computed into the appendix
                       the paper promises twice and does not have.
"""
import modal, os, glob, json

from arr_modal import (                    # noqa: F401  -- shared, not copied
    app, vol, image, CELLS, DERIVED, MODELS, DATASETS,
    _root, _dst, _load_peritem, _rho, _partial_rho, _item_bootstrap, _bh,
)

LADDER = ["knn_dist", "size_only", "credal_width"]
shared_image = image.add_local_python_source("arr_modal")
ablation_image = (image.pip_install("safetensors")
                  .add_local_python_source("arr_modal", "analyse_hf"))

# --------------------------------------------------------------------------
# PRE-REGISTRATION. Freeze before running. Each function asserts on its own.
# --------------------------------------------------------------------------
LENGTH_NULL = 0.05      # FROZEN 2026-08-02, before running these ablations.
                        #   against annotator entropy is accounted for by
                        #   length alone in that cell. Suggest 0.05, matching
                        #   NULL_THRESHOLD in arr_modal.
POOL_DROP = 0.20        # FROZEN 2026-08-02, before running these ablations.
                        #   first-token pooling to call the dependence a
                        #   pooling artefact. Suggest: below 0.20, from 0.85.
SHUFFLE_MAX = 0.05      # FROZEN 2026-08-02, before running these ablations.
                        #   before the pipeline is considered suspect.
                        #   Suggest 0.05 at these n.
# --------------------------------------------------------------------------


def _ans_tok(model, dataset, ids):
    """Answer token count per item, aligned to the per-item dump order.

    TODO: n_ans_tok_* lives in index.parquet (compile_all.py:472). There may
          be one column per arm (correct/wrong). DECIDE WHICH before running
          and record it:
            - correct arm only   -> matches what span_mean pooled over, so it
              is the right control for the SIGNAL side;
            - the difference     -> matches what shifts p_mdl, so it is the
              right control for the DIVERGENCE side.
          These are different controls and answer different questions. The
          paper's mechanism section needs the second; this ablation, whose
          target is annotator entropy, needs the first.
    TODO: uq_panel reorders by `keep`. Align on item_id, never on row order.
    """
    import numpy as np, pandas as pd
    idx = pd.read_parquet(f"{_root(model, dataset)}/index.parquet")
    col = "n_ans_tok_correct"
    if col not in idx.columns:
        raise KeyError(f"{model}/{dataset}: {col} absent from index.parquet")
    vals = {int(i): float(v) for i, v in zip(idx.item_id, idx[col])}
    missing = [int(i) for i in ids if int(i) not in vals]
    if missing:
        raise KeyError(f"{len(missing)} score ids absent from index, e.g. {missing[:5]}")
    return np.asarray([vals[int(i)] for i in ids], dtype=float)


# ==========================================================================
# 1. LENGTH PANEL -- the hole in the mechanism section
# ==========================================================================
@app.function(image=shared_image, volumes={"/vol": vol}, timeout=60 * 60 * 2)
def length_panel(model: str, dataset: str, layer: int = -1,
                 scheme: str = "span_mean", n_clusters: int = 200,
                 tau: float = 10.0, seeds: str = "0,1,2",
                 n_boot: int = 2000):
    """Does the token-count dependence generalise, and does the inversion
    against ANNOTATOR ENTROPY survive controlling for it?

    Runs on all nine cells, unlike the mechanism analysis, because the target
    here is annotator entropy rather than the divergence. The two Pavlick
    cells carry the largest inversions in the paper (-0.396, -0.246) and the
    mechanism section reaches neither.

    Reports per signal:
        rho(signal, n_ans_tok)      is the dependence there at all?
        rho(signal, H)              the panel value, for reference
        rho(signal, H | n_ans_tok)  does the association survive?

    Reading, pre-registered:
      - dependence present on ChaosNLI only  -> the length story is
        dataset-specific and the abstract must say so;
      - present everywhere and the partial collapses -> the length story
        generalises and is the main mechanism, not a co-driver;
      - present everywhere but the partial SURVIVES on Pavlick -> the sign
        result stands independent of length in the cells where it is
        strongest, which strengthens the paper. Say which.
    """
    assert LENGTH_NULL is not None, "freeze LENGTH_NULL first"
    import numpy as np, pandas as pd
    scores, ent, ids = _load_peritem(model, dataset, layer, scheme,
                                     n_clusters, tau, seeds)
    ell = _ans_tok(model, dataset, ids)

    rows = []
    for sig, v in scores.items():
        r_len = _rho(v, ell)
        r_h = _rho(v, ent)
        r_par = _partial_rho(v, ent, ell)
        lo, hi, p = _item_bootstrap(
            lambda i: _partial_rho(v[i], ent[i], ell[i]), len(ent), B=n_boot)
        rows.append(dict(model=model, dataset=dataset, tau=tau, signal=sig,
                         rho_len=r_len, rho_H=r_h, partial=r_par,
                         ci_lo=lo, ci_hi=hi, p=p,
                         explained_by_length=bool(abs(r_par) < LENGTH_NULL
                                                  and lo <= 0 <= hi)))
    df = pd.DataFrame(rows)
    print(df[df.signal.isin(LADDER)].to_string(index=False))
    df.to_parquet(f"{_dst(model, dataset)}/arr_length_tau{tau}.parquet")
    vol.commit()
    return rows


# ==========================================================================
# 2. POOLING ABLATION -- turns the claim into a demonstration
# ==========================================================================
@app.function(image=ablation_image, volumes={"/vol": vol}, timeout=60 * 60 * 4,
              cpu=8, memory=32768)
def pool_ablation(model: str, dataset: str, layer: int = -1,
                  n_clusters: int = 200, tau: float = 10.0,
                  seeds: str = "0,1,2",
                  schemes: str = "span_mean,first_token"):
    """Recompute the whole signal panel under each pooling scheme.

    The paper argues that mean pooling over an answer span introduces a
    token-count term because the number of positions averaged varies with the
    label word. First-token pooling reads ONE position and cannot. If the
    argument is right, rho(W, n_ans_tok) should fall sharply and the
    inversion should weaken with it.

    This is the single ablation that converts the mechanism from an
    explanation into a controlled result. It is also the one that could
    embarrass the paper: if the dependence survives first-token pooling, the
    pooling explanation is wrong and Sec. Mechanism needs rewriting.

    TODO: extract_hf.py stores SPANS, not poolings, in the safetensors
          shards, so re-pooling needs no re-extraction. Reuse analyse_hf's
          _pool with the alternative scheme rather than writing a second
          pooling function.
    TODO: the frame must be REFIT per scheme. A frame fitted on span_mean
          states and applied to first_token states compares nothing. This is
          the expensive part -- budget for a full K-means refit per cell per
          scheme.
    TODO: report, per scheme: rho(W, n_ans_tok), rho(W, H), rho(W, knn),
          and the sign. The last is what the paper's spine rests on.
    """
    assert POOL_DROP is not None, "freeze POOL_DROP first"
    import numpy as np, pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    import analyse_hf as A

    root = _root(model, dataset); meta = A._meta(root)
    lyr = meta["n_total_layers"] + 1 + layer if layer < 0 else layer
    _, ent, canonical_ids = _load_peritem(model, dataset, lyr, "span_mean",
                                           n_clusters, tau, seeds)
    ell = _ans_tok(model, dataset, canonical_ids)
    rows = []
    for pooling in [s.strip() for s in schemes.split(",") if s.strip()]:
        scheme_name = "first" if pooling == "first_token" else pooling
        ids_h, H0 = A.load_states(root, meta, lyr, "correct", scheme_name)
        pos = {int(i): j for j, i in enumerate(ids_h)}
        missing = [int(i) for i in canonical_ids if int(i) not in pos]
        if missing:
            raise KeyError(f"{pooling}: {len(missing)} ids absent from states")
        H = H0[[pos[int(i)] for i in canonical_ids]]
        nn = NearestNeighbors(n_neighbors=min(11, len(H)), algorithm="brute").fit(H)
        dist, _ = nn.kneighbors(H)
        metrics = {"knn_dist": dist[:, 1:].mean(1)}
        acc = {}
        for sd in [int(s) for s in seeds.split(",")]:
            km = KMeans(n_clusters, random_state=sd, n_init=4).fit(H)
            sizes = np.bincount(km.labels_, minlength=n_clusters).astype(float)
            sizes[sizes == 0] = 1
            sig = A._signals(A.soft_mass(H, km.cluster_centers_, tau), sizes)
            for name, value in sig.items():
                acc.setdefault(name, []).append(value)
        metrics.update({name: np.mean(value, axis=0) for name, value in acc.items()})
        for name, value in metrics.items():
            rows.append(dict(model=model, dataset=dataset, layer=lyr,
                             scheme=pooling, signal=name,
                             rho_len=_rho(value, ell), rho_H=_rho(value, ent),
                             rho_knn=_rho(value, metrics["knn_dist"])))
    df = pd.DataFrame(rows)
    w = df[df.signal == "credal_width"].set_index("scheme")
    base = abs(float(w.loc["span_mean", "rho_len"]))
    first = abs(float(w.loc["first_token", "rho_len"]))
    verdict = bool(first < POOL_DROP and first < base)
    df["pooling_artifact"] = verdict
    print(df[df.signal.isin(LADDER)].round(4).to_string(index=False))
    print(f"W |rho(length)| {base:.3f} -> {first:.3f}; "
          f"POOL_DROP={POOL_DROP:.2f}: {'PASS' if verdict else 'FAIL'}")
    df.to_parquet(f"{_dst(model, dataset)}/arr_pool_ablation.parquet")
    vol.commit()
    return df.to_dict("records")


# ==========================================================================
# 3. SHUFFLE CONTROL -- the negative control the paper lacks
# ==========================================================================
@app.function(image=shared_image, volumes={"/vol": vol}, timeout=60 * 60)
def shuffle_control(model: str, dataset: str, layer: int = -1,
                    scheme: str = "span_mean", n_clusters: int = 200,
                    tau: float = 10.0, seeds: str = "0,1,2",
                    n_perm: int = 200, seed: int = 0):
    """Permute the target across items and rerun the panel.

    Every association should collapse. Cheapest possible sanity check on a
    paper whose central finding is the sign of a correlation, and its absence
    is conspicuous.

    Also reports rho(n_ans_tok, H) directly: if a bare token count predicts
    annotator entropy at any appreciable level, several results need
    reframing, because then length is not only a property of the signal but
    of the target as well.
    """
    assert SHUFFLE_MAX is not None, "freeze SHUFFLE_MAX first"
    import numpy as np, pandas as pd
    scores, ent, ids = _load_peritem(model, dataset, layer, scheme,
                                     n_clusters, tau, seeds)
    rng = np.random.default_rng(seed)
    rows = []
    for sig, v in scores.items():
        null = np.array([_rho(v, rng.permutation(ent)) for _ in range(n_perm)])
        obs = _rho(v, ent)
        rows.append(dict(model=model, dataset=dataset, signal=sig,
                         rho_obs=obs,
                         null_absmax=float(np.nanmax(np.abs(null))),
                         null_p95=float(np.nanpercentile(np.abs(null), 95)),
                         perm_p=float(np.nanmean(np.abs(null) >= abs(obs))),
                         passes=bool(np.nanmax(np.abs(null)) < SHUFFLE_MAX)))
    # TODO: add rho(n_ans_tok, ent) here once _ans_tok is implemented.
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_parquet(f"{_dst(model, dataset)}/arr_shuffle.parquet"); vol.commit()
    return rows


# ==========================================================================
# 4. GRID ABLATION -- K, layer, neighbourhood size
# ==========================================================================
@app.function(image=ablation_image, volumes={"/vol": vol}, timeout=60 * 60 * 6,
              cpu=8, memory=32768)
def grid_ablation(model: str, dataset: str, tau: float = 10.0,
                  seeds: str = "0,1,2", scheme: str = "span_mean",
                  K_grid: str = "50,200,500", layers: str = "-1,-5,-9",
                  k_grid: str = "10,25,50"):
    """Does the sign survive the design choices the paper fixed?

    K matters most: the construction is cardinality-based and cluster size
    scales with K, so 'does the sign hold at K=50 and K=500' is the first
    thing a reviewer will ask about a frame-based signal.

    TODO: K requires a frame refit per value; layer requires re-pooling from
          the cached spans; k is free (neighbourhood.csv already has the
          band, it is simply not reported).
    TODO: do NOT average across tau anywhere in the summary. size_only is
          exactly tau-invariant. RESULTS_ALL defect 6.
    TODO: report the SIGN grid, not just magnitudes. A table of |rho| across
          K would hide exactly what this paper is about.
    """
    import numpy as np, pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    import analyse_hf as A

    root = _root(model, dataset); meta = A._meta(root)
    _, ent0, canonical_ids = _load_peritem(model, dataset, -1, scheme, 200,
                                            tau, seeds)
    wanted_layers = [int(x) for x in layers.split(",")]
    targets = [meta["n_total_layers"] + 1 + x if x < 0 else x
               for x in wanted_layers]
    captured = sorted(meta["captured_layers"])
    # The extraction cached eight probe layers rather than every layer. Snap
    # each requested target to the nearest captured layer, deterministically
    # preferring the lower layer on a tie, and record both values below.
    resolved = [min(captured, key=lambda c: (abs(c - target), c))
                for target in targets]

    rows = []
    for requested, lyr in zip(wanted_layers, resolved):
        ids_h, H0 = A.load_states(root, meta, lyr, "correct", scheme)
        pos = {int(i): j for j, i in enumerate(ids_h)}
        H = H0[[pos[int(i)] for i in canonical_ids]]
        for k in [int(x) for x in k_grid.split(",")]:
            nn = NearestNeighbors(n_neighbors=min(k + 1, len(H)),
                                  algorithm="brute").fit(H)
            d, _ = nn.kneighbors(H)
            v = d[:, 1:].mean(1)
            rows.append(dict(model=model, dataset=dataset, layer=lyr,
                             requested_layer=requested, K=np.nan, k=k,
                             signal="knn_dist", rho=_rho(v, ent0),
                             sign=int(np.sign(_rho(v, ent0)))))
        for K in [int(x) for x in K_grid.split(",")]:
            K_eff = min(K, len(H) - 1)
            acc = {}
            for sd in [int(s) for s in seeds.split(",")]:
                km = KMeans(K_eff, random_state=sd, n_init=4).fit(H)
                sizes = np.bincount(km.labels_, minlength=K_eff).astype(float)
                sizes[sizes == 0] = 1
                sig = A._signals(A.soft_mass(H, km.cluster_centers_, tau), sizes)
                for name, value in sig.items():
                    acc.setdefault(name, []).append(value)
            metrics = {name: np.mean(value, axis=0) for name, value in acc.items()}
            for name in ("size_only", "credal_width"):
                r = _rho(metrics[name], ent0)
                rows.append(dict(model=model, dataset=dataset, layer=lyr,
                                 requested_layer=requested, K=K,
                                 K_effective=K_eff, k=np.nan,
                                 signal=name, rho=r,
                                 sign=int(np.sign(r)) if np.isfinite(r) else 0))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_parquet(f"{_dst(model, dataset)}/arr_grid_ablation.parquet")
    vol.commit()
    return df.to_dict("records")


# ==========================================================================
# 5. APPENDIX DUMP -- assemble what already exists
# ==========================================================================
@app.function(image=shared_image, volumes={"/vol": vol}, timeout=60 * 30)
def appendix_dump(fmt: str = "latex"):
    """Emit the tables the paper promises and does not have.

    Nothing here needs new compute. All of it is on the volume already.

      A1  all nine signals x nine cells, with intervals
          (the paper says twice that 'the remainder appear in the appendix')
      A2  full dispersion panel, 27 rows, with the uninformative-row rule
      A3  full sign-stability grid, 9 cells x 3 signals x 3 tau
      A4  neighbourhood table at every k in {10,25,50}
      A5  target variants: the panel recomputed against h_norm and h_mm
      A6  seed SD per cell, AS A FRACTION of the effect size
      A7  the components analysis in full

    TODO: A6 is the one most likely to be read against the paper. Report the
          ratio, not the raw SD -- a reviewer computing it themselves and
          finding it large is worse than stating it.
    TODO: emit booktabs LaTeX directly, so the appendix is regenerated from
          the volume rather than transcribed. Transcription is how the
          0.9996-vs-0.85 discrepancy survived as long as it did.
    """
    import pandas as pd, glob
    parts = {
        "A1_full_panel":   f"{DERIVED}/*/peritem_*.parquet",
        "A2_dispersion":   f"{DERIVED}/*/A1b_uq_panel_*.parquet",
        "A3_sign":         f"{DERIVED}/*/arr_sign_stability.parquet",
        "A4_neighbourhood": f"{DERIVED}/*/arr_nbhd_*.parquet",
        "A5_variants":     f"{DERIVED}/*/arr_audit.parquet",
        "A7_components":   f"{DERIVED}/*/arr_components_*.parquet",
    }
    summary = []
    for name, pat in parts.items():
        fs = sorted(glob.glob(pat))
        print(f"{name:18s} {len(fs):3d} files")
        if not fs:
            print("    (missing -- run the corresponding stage first)")
        summary.append(dict(table=name, n_files=len(fs), files=fs))
    out = pd.DataFrame([{k: v for k, v in x.items() if k != "files"}
                        for x in summary])
    os.makedirs(DERIVED, exist_ok=True)
    out.to_parquet(f"{DERIVED}/arr_appendix_inventory.parquet")
    vol.commit()
    return summary


@app.function(image=shared_image, volumes={"/vol": vol}, timeout=60 * 30)
def ablation_summary():
    """Compact persisted-result audit used to build results/ablations.md."""
    import pandas as pd
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            root = f"{DERIVED}/{model}__{dataset}"
            rec = dict(model=model, dataset=dataset)
            paths = {
                "length": f"{root}/arr_length_tau10.0.parquet",
                "shuffle": f"{root}/arr_shuffle.parquet",
                "pool": f"{root}/arr_pool_ablation.parquet",
                "grid": f"{root}/arr_grid_ablation.parquet",
            }
            rec["present"] = {k: os.path.exists(v) for k, v in paths.items()}
            if rec["present"]["length"]:
                d = pd.read_parquet(paths["length"])
                w = d[d.signal == "credal_width"].iloc[0]
                rec["length"] = {k: float(w[k]) for k in
                                 ("rho_len", "rho_H", "partial", "ci_lo", "ci_hi")}
                rec["length"]["explained"] = bool(w.explained_by_length)
            if rec["present"]["shuffle"]:
                d = pd.read_parquet(paths["shuffle"])
                w = d[d.signal == "credal_width"].iloc[0]
                rec["shuffle"] = {k: float(w[k]) for k in
                                  ("rho_obs", "null_absmax", "null_p95", "perm_p")}
                rec["shuffle"]["passes"] = bool(w.passes)
            if rec["present"]["pool"]:
                d = pd.read_parquet(paths["pool"])
                w = d[d.signal == "credal_width"].set_index("scheme")
                rec["pool"] = {
                    s: {k: float(w.loc[s, k]) for k in
                        ("rho_len", "rho_H", "rho_knn")}
                    for s in ("span_mean", "first_token")}
                rec["pool"]["artifact"] = bool(d.pooling_artifact.iloc[0])
            if rec["present"]["grid"]:
                d = pd.read_parquet(paths["grid"])
                w = d[d.signal == "credal_width"]
                rec["grid"] = dict(n=int(len(d)),
                                   w_min=float(w.rho.min()),
                                   w_max=float(w.rho.max()),
                                   w_signs=sorted(int(x) for x in w.sign.unique()))
            rows.append(rec)
    print(json.dumps(rows, indent=2, allow_nan=True))
    return rows


@app.function(image=shared_image, volumes={"/vol": vol}, timeout=60 * 30)
def export_ablations():
    """Combine all ablation parquets into one tidy export for local plots."""
    import pandas as pd
    parts = []
    specs = {
        "length": "arr_length_tau10.0.parquet",
        "shuffle": "arr_shuffle.parquet",
        "pool": "arr_pool_ablation.parquet",
        "grid": "arr_grid_ablation.parquet",
    }
    for model in MODELS:
        for dataset in DATASETS:
            root = f"{DERIVED}/{model}__{dataset}"
            for ablation, filename in specs.items():
                path = f"{root}/{filename}"
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                d = pd.read_parquet(path).copy()
                d["ablation"] = ablation
                parts.append(d)
    out = pd.concat(parts, ignore_index=True, sort=False)
    path = f"{DERIVED}/arr_ablations_all.parquet"
    out.to_parquet(path); vol.commit()
    print(f"wrote {len(out)} rows to {path}")
    return dict(rows=len(out), path=path)


@app.function(image=ablation_image, volumes={"/vol": vol}, timeout=60 * 60 * 3,
              cpu=8, memory=32768)
def seed_variability(model: str, dataset: str, layer: int = -1,
                     scheme: str = "span_mean", n_clusters: int = 200,
                     tau: float = 10.0, seeds: str = "0,1,2"):
    """Seedwise correlations and SD/|mean effect| at the primary setting."""
    import numpy as np, pandas as pd
    from sklearn.cluster import KMeans
    import analyse_hf as A

    root = _root(model, dataset); meta = A._meta(root)
    lyr = meta["n_total_layers"] + 1 + layer if layer < 0 else layer
    _, ent, ids = _load_peritem(model, dataset, lyr, scheme,
                                n_clusters, tau, seeds)
    ids_h, H0 = A.load_states(root, meta, lyr, "correct", scheme)
    pos = {int(i): j for j, i in enumerate(ids_h)}
    H = H0[[pos[int(i)] for i in ids]]
    per_signal = {name: [] for name in ("size_only", "credal_width")}
    seed_list = [int(s) for s in seeds.split(",")]
    for sd in seed_list:
        km = KMeans(n_clusters, random_state=sd, n_init=4).fit(H)
        sizes = np.bincount(km.labels_, minlength=n_clusters).astype(float)
        sizes[sizes == 0] = 1
        sig = A._signals(A.soft_mass(H, km.cluster_centers_, tau), sizes)
        for name in per_signal:
            per_signal[name].append(_rho(sig[name], ent))
    rows = []
    for name, vals in per_signal.items():
        mean = float(np.mean(vals)); sd = float(np.std(vals))
        rows.append(dict(model=model, dataset=dataset, layer=lyr,
                         scheme=scheme, K=n_clusters, tau=tau, signal=name,
                         seed_0=vals[0], seed_1=vals[1], seed_2=vals[2],
                         mean_rho=mean, seed_sd=sd,
                         sd_fraction_effect=(sd / abs(mean)
                                             if abs(mean) > 1e-12 else np.nan)))
    out = pd.DataFrame(rows)
    print(out.round(5).to_string(index=False))
    out.to_parquet(f"{_dst(model, dataset)}/arr_seed_variability.parquet")
    vol.commit()
    return rows


@app.local_entrypoint()
def ablations_main(stage: str = "length"):
    """Fan out. Order matters: length first, it is the open hole.

        modal run arr_ablations.py --stage=length
        modal run arr_ablations.py --stage=shuffle
        modal run arr_ablations.py --stage=pool
    """
    cells = [(m, d) for m in MODELS for d in DATASETS]
    if stage == "grid_remaining":
        cells = [(m, d) for m in MODELS for d in
                 ("chaosnli", "chaosnli_alpha")]
        fn = grid_ablation
        for r in fn.starmap(cells):
            print(r if not isinstance(r, list) else f"{len(r)} rows")
        return
    if stage == "seed":
        for r in seed_variability.starmap(cells):
            print(r if not isinstance(r, list) else f"{len(r)} rows")
        return
    fn = {"length": length_panel, "shuffle": shuffle_control,
          "pool": pool_ablation, "grid": grid_ablation}.get(stage)
    if fn is None:
        raise SystemExit(f"unknown stage: {stage}")
    for r in fn.starmap(cells):
        print(r if not isinstance(r, list) else f"{len(r)} rows")
