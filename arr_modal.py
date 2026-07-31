"""ARR disagreement experiments, on Modal.

Companion to analyse_hf.py. Same volume, same cell layout, same conventions:
one file, all heavy imports INSIDE function bodies so `modal run` works from
a login node with only the modal client installed.

    modal run arr_modal.py::audit_target    --model=llama3_8b --dataset=chaosnli
    modal run arr_modal.py::fold_check      --model=llama3_8b --dataset=chaosnli_alpha
    modal run arr_modal.py::ladder          --model=llama3_8b --dataset=chaosnli
    modal run arr_modal.py::sign_stability  --model=llama3_8b --dataset=chaosnli
    modal run arr_modal.py::neighbourhood   --model=llama3_8b --dataset=chaosnli
    modal run arr_modal.py::family          --model=llama3_8b --dataset=chaosnli
    modal run arr_modal.py::collect                       # BH across the panel

    modal run arr_modal.py                                # the whole grid

DEPENDENCY. Everything except `audit_target` reads the per-item score dump
written by Patch 1 in PATCHES_analyse_hf.md. That patch edits
analyse_hf.uq_panel rather than recomputing the signals here, deliberately:
a second implementation of the same signals would eventually drift from the
first, and then two numbers in the paper would disagree for reasons nobody
could reconstruct. Apply the patch, re-run uq_panel once per cell, then come
back here.

FENCE. Nothing here derives the saturation condition (ICLR), proves the
non-redundancy result (theory paper), or makes deferral claims (AAAI).
`nbhd_oracle` observes the target and is never reported as a method.
"""
import modal, json, os, glob

app = modal.App("rsuq-arr")
vol = modal.Volume.from_name("rsuq-latent", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "pandas", "pyarrow", "scikit-learn", "scipy"))

# neighbourhood needs the raw pooled states, so it adds safetensors and reuses
# analyse_hf's loaders (load_states/_meta/_index) rather than a second copy of
# the pooling + keep-intersection logic that would eventually drift from it.
nbhd_image = image.pip_install("safetensors").add_local_python_source("analyse_hf")

CELLS = "/vol/hf/cells"
DERIVED = "/vol/derived"

MODELS = ["llama3_8b", "mistral_7b", "qwen2_5_7b"]
DATASETS = ["chaosnli", "chaosnli_alpha", "pavlick_nli"]
BINARY_DATASETS = {"chaosnli_alpha"}          # folded target, see fold_check

# Collapsed from analyse_hf.py GROUP, which was fixed before any result on
# these datasets was inspected. Cite that dict as the pre-registration.
REPR = {"credal_width", "mass_entropy", "one_minus_max",
        "size_only", "knn_dist"}
OUTPUT = {"pred_entropy", "pred_maxprob", "pred_margin", "neg_mean_logprob"}
LADDER = ["knn_dist", "size_only", "credal_width"]

# --------------------------------------------------------------------------
# PRE-REGISTRATION. Fill in and freeze BEFORE running anything. Every
# function that needs one of these asserts on it and will refuse to start.
# Choosing a threshold once the numbers are visible is choosing the result.
# --------------------------------------------------------------------------
NULL_THRESHOLD = 0.05      # FROZEN 2026-07-31. ladder: |partial rho| below this
                           #   AND CI containing 0 -> the rung adds nothing.
MIN_CELLS_LADDER = 5       # FROZEN 2026-07-31. ladder: majority of 9 cells
                           #   before it is a general result, not a per-cell one.
MATERIAL_RISE = 0.05       # FROZEN 2026-07-31. fold: rep-space |rho| must rise
                           #   to within 0.05 of the token family's rho on the
                           #   same cell to call the split a target artefact.
FAMILY_STAT = "mean"       # FROZEN 2026-07-31. family: mean primary (no
                           #   selection problem); max is the robustness row.
MIN_CELLS_FAMILY = 5       # FROZEN 2026-07-31. family: 5 of 9 must clear BH.
PRIMARY_TARGET = "h_plugin"  # FROZEN 2026-07-31. Primary target = plug-in
                           #   Shannon, what every existing number uses (no
                           #   drift). h_norm and h_mm are robustness rows;
                           #   audit_target reports all three.
# --------------------------------------------------------------------------


def _root(model, dataset):
    return f"{CELLS}/{model}__{dataset}"


def _dst(model, dataset):
    d = f"{DERIVED}/{model}__{dataset}"
    os.makedirs(d, exist_ok=True)
    return d


def _peritem_path(model, dataset, layer, scheme, K, tau, seeds):
    """Must match the filename written by Patch 1 exactly.

    Carries model, dataset, layer, scheme, K, tau, seeds. RESULTS_ALL.md
    defect 7 is a filename that omitted these and let a smoke run silently
    overwrite a paper-setting parquet.
    """
    return (f"{_dst(model, dataset)}/peritem_{model}__{dataset}"
            f"_L{layer}_{scheme}_K{K}_tau{tau}_s{seeds.replace(',', '-')}"
            f".parquet")


def _load_peritem(model, dataset, layer, scheme, K, tau, seeds):
    """Long -> (scores dict of per-item arrays, target array, item ids)."""
    import pandas as pd, numpy as np
    p = _peritem_path(model, dataset, layer, scheme, K, tau, seeds)
    if not os.path.exists(p) and layer < 0:
        # layer<0 is the sentinel 'the default final layer', which uq_panel
        # resolves to a concrete integer (e.g. 32 for llama/mistral, 28 for
        # qwen2.5-7b) and writes into the dump filename. Recover it by globbing
        # every OTHER paper-setting field and demanding exactly one match --
        # so this never silently loosens K/tau/scheme/seeds (defect 7).
        pat = (f"{_dst(model, dataset)}/peritem_{model}__{dataset}"
               f"_L*_{scheme}_K{K}_tau{tau}_s{seeds.replace(',', '-')}.parquet")
        hits = sorted(glob.glob(pat))
        if len(hits) == 1:
            p = hits[0]
        elif len(hits) > 1:
            raise RuntimeError(
                f"ambiguous peritem dumps for layer<0, refusing to guess:\n  "
                + "\n  ".join(hits))
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p}\nApply Patch 1 and re-run uq_panel for this cell first. "
            f"Do not substitute a recomputation here.")
    df = pd.read_parquet(p)
    wide = df.pivot_table(index="item_id", columns="signal", values="value")
    ent = wide.pop("annot_entropy").to_numpy()
    ids = wide.index.to_numpy()
    return {c: wide[c].to_numpy() for c in wide.columns}, ent, ids


# ---------------------------------------------------------------- statistics

def _rho(x, y, signed=True):
    import numpy as np
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return float("nan")
    r = spearmanr(x[ok], y[ok])[0]
    return float(r if signed else abs(r))


def _partial_rho(x, y, z):
    """Partial Spearman rho(x, y | z), computed on ranks.

    Rank-transform, then Pearson partial on the ranks. Tie-tolerant, which
    matters: size_only is discrete and heavily tied, and annotator entropy
    has a large mass at exactly zero on the binary dataset.
    """
    import numpy as np
    from scipy.stats import rankdata
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 30:
        return float("nan")
    rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))

    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta

    ex, ey = resid(rx, rz), resid(ry, rz)
    den = np.linalg.norm(ex) * np.linalg.norm(ey)
    return float(ex @ ey / den) if den > 0 else float("nan")


def _item_bootstrap(fn, n, B=2000, seed=0):
    """Bootstrap fn(idx) over ITEM indices. Returns (lo, hi, p_two_sided).

    Items, never signals or seeds. Paired comparisons recompute both sides
    on the same resample -- that is what makes them paired.

    The p is a bootstrap-percentile inversion, not a permutation p. It is an
    approximation, and the paper should say so. ARR_run_summary.md section 8
    correctly refused to read a p off a CI crossing zero; this is not that,
    but it is not a permutation test either.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        out[b] = fn(rng.integers(0, n, n))
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    fpos = float(np.nanmean(out > 0))
    return float(lo), float(hi), float(min(2.0 * min(fpos, 1 - fpos), 1.0))


def _bh(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up. Returns (reject_mask, qvalues)."""
    import numpy as np
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    q = np.full_like(p, np.nan)
    if m == 0:
        return np.zeros_like(p, bool), q
    idx = np.where(ok)[0][np.argsort(p[ok])]
    ranked = p[idx] * m / np.arange(1, m + 1)
    q[idx] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return (q <= alpha) & ok, q


# ============================================================== experiments

@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 30)
def audit_target(model: str, dataset: str):
    """Audit the annotation target. Run this FIRST, before anything else.

    analyse_hf.annotator_entropy_from_index computes plug-in Shannon entropy
    with no bias correction, and the three datasets have three different
    label-set sizes. Three consequences that must be measured, not assumed:

      1. Cross-dataset comparisons of rho are not on a common footing.
      2. The binary dataset's target is folded (see fold_check).
      3. The counts-vs-labels branch can silently misparse.

    On (3): the heuristic tests `max(p) == int(max(p))` and `sum(p) != 1`,
    which cannot distinguish 100 per-annotator labels [0,0,1,2,...] from a
    count vector [78,15,7]. Fed the second it bincounts them as labels and
    returns garbage without raising. Discriminator: a label vector has
    size >> max(value); a count vector has size << max(value).
    """
    import numpy as np, pandas as pd
    idx = pd.read_parquet(f"{_root(model, dataset)}/index.parquet")
    col = ("soft_labels" if "soft_labels" in idx.columns
           else "annotator_labels" if "annotator_labels" in idx.columns
           else None)
    if col is None:
        raise RuntimeError("index.parquet carries no soft_labels/annotator_labels")

    raw = [json.loads(v) if isinstance(v, str) else v for v in idx[col]]
    sizes = np.array([len(np.ravel(v)) for v in raw])
    maxes = np.array([float(np.max(np.ravel(v))) if len(np.ravel(v)) else 0.0
                      for v in raw])
    looks_like = ("per-annotator labels" if np.median(sizes) > np.median(maxes)
                  else "counts or probabilities")

    ent, n_annot, k_obs = [], [], []
    for v in raw:
        c = np.asarray(v, float).ravel()
        # TODO: apply the SAME parse analyse_hf uses, then assert the two
        #       agree. Divergence here is the silent-failure case and it is
        #       the whole reason this function exists.
        N = c.sum()
        p = c / max(N, 1e-12)
        ent.append(float(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
        n_annot.append(float(N))
        k_obs.append(int((c > 0).sum()))
    ent = np.asarray(ent); n_annot = np.asarray(n_annot)
    k_obs = np.asarray(k_obs)

    # All three target variants, per item (annotation_audit.entropy_variants).
    # PRIMARY_TARGET selects which one downstream reports as primary; the
    # other two are robustness rows. h_mm needs a per-item N, which we have
    # here, so all three are computable without blocking on anything.
    support = int(max(k_obs)) if k_obs.size else 1
    h_plugin = ent
    h_norm = ent / np.log(support) if support > 1 else np.full_like(ent, np.nan)
    h_mm = np.where(n_annot > 0,
                    ent + (k_obs - 1) / (2.0 * np.maximum(n_annot, 1e-12)),
                    np.nan)
    variants = {"h_plugin": h_plugin, "h_norm": h_norm, "h_mm": h_mm}
    assert PRIMARY_TARGET in variants, f"bad PRIMARY_TARGET {PRIMARY_TARGET!r}"

    out = dict(
        model=model, dataset=dataset, n_items=int(len(idx)),
        column=col, looks_like=looks_like,
        support_observed=support,
        size_mode=int(np.bincount(sizes).argmax()),
        size_varies=bool(sizes.min() != sizes.max()),
        n_annot_min=float(n_annot.min()), n_annot_max=float(n_annot.max()),
        n_annot_median=float(np.median(n_annot)),
        n_annot_varies=bool(n_annot.min() != n_annot.max()),
        primary_target=PRIMARY_TARGET,
        mean_H=float(ent.mean()), max_H=float(ent.max()),
        frac_zero=float((ent <= 1e-12).mean()),
        mean_h_plugin=float(np.nanmean(h_plugin)),
        mean_h_norm=float(np.nanmean(h_norm)),
        mean_h_mm=float(np.nanmean(h_mm)),
        max_h_plugin=float(np.nanmax(h_plugin)),
        max_h_norm=float(np.nanmax(h_norm)),
        max_h_mm=float(np.nanmax(h_mm)),
    )
    print(json.dumps(out, indent=2))
    print("\n### Table 1 row")
    print(f"{dataset} & {out['n_items']} & {out['support_observed']} & "
          f"{out['n_annot_median']:.0f} & {out['mean_H']:.3f} & "
          f"{out['max_H']:.3f} & {out['frac_zero']:.3f} \\\\")
    if out["n_annot_varies"]:
        print("\n!! annotator count VARIES within this dataset. The plug-in "
              "bias is then an ITEM-LEVEL confound in the target, not just a "
              "cross-dataset nuisance. Sec. Setup needs a paragraph, not a row.")
    pd.DataFrame([out]).to_parquet(
        f"{_dst(model, dataset)}/arr_audit.parquet"); vol.commit()
    return out


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60)
def fold_check(model: str, dataset: str, layer: int = -1,
               scheme: str = "span_mean", n_clusters: int = 200,
               tau: float = 10.0, seeds: str = "0,1,2"):
    """Does the binary target explain the alpha-NLI result?

    On a two-element label set, H(p) is symmetric about p = 0.5 and discards
    which option annotators favoured. alpha-NLI is the only binary cell AND
    the only cell where the output family leads, so task type and target
    geometry are perfectly confounded.

    Cheapest experiment in the package; it can invalidate a headline.

    Reading, pre-registered:
      (a) rep-space |rho| rises materially against `lean` -> the family split
          is an artefact of the target summary. CUT it from the abstract and
          report the fold as the finding instead.
      (b) rep-space stays flat -> fold ruled out; proceed to `family`.

    `split` = |p1 - 0.5| is a NULL CONTROL, not a variant: it is a monotone
    reparameterisation of H, so its Spearman must match H's up to sign. If it
    does not, something upstream is wrong.
    """
    assert MATERIAL_RISE is not None, "freeze MATERIAL_RISE first"
    if dataset not in BINARY_DATASETS:
        raise ValueError(f"{dataset} is not binary; the fold does not apply")
    import numpy as np, pandas as pd
    scores, ent, ids = _load_peritem(model, dataset, layer, scheme,
                                     n_clusters, tau, seeds)
    idx = pd.read_parquet(f"{_root(model, dataset)}/index.parquet")
    col = ("soft_labels" if "soft_labels" in idx.columns
           else "annotator_labels" if "annotator_labels" in idx.columns
           else None)
    if col is None:
        raise RuntimeError("index.parquet carries no soft_labels/annotator_labels")

    # p1 = P(option 2), normalised, aligned to `ids`. uq_panel reorders items
    # by `keep`, so JOIN on item_id -- never assume index row order matches
    # the dump order. Same parse as annotator_entropy_from_index.
    p1_by_id = {}
    for iid, raw in zip(idx.item_id, idx[col]):
        v = json.loads(raw) if isinstance(raw, str) else raw
        p = np.asarray(v, float).ravel()
        p = p / max(p.sum(), 1e-12)
        if p.size != 2:
            raise ValueError(
                f"{dataset} item {iid} has a {p.size}-element label dist, not "
                f"2; fold_check applies only to a binary target")
        p1_by_id[int(iid)] = float(p[1])
    missing = [int(i) for i in ids if int(i) not in p1_by_id]
    if missing:
        raise KeyError(f"{len(missing)} dump ids absent from index.parquet, "
                       f"e.g. {missing[:5]}")
    p1 = np.array([p1_by_id[int(i)] for i in ids])
    split = np.abs(p1 - 0.5)          # NULL CONTROL: monotone reparam of H

    rows = [dict(model=model, dataset=dataset, signal=s,
                 rho_entropy=_rho(v, ent),
                 rho_lean=_rho(v, p1),
                 rho_split_control=_rho(v, split))
            for s, v in scores.items()]
    df = pd.DataFrame(rows)

    # Null control: |rho(split)| must reproduce |rho(entropy)| up to sign,
    # because split is a monotone reparameterisation of H. A large gap means
    # something upstream is wrong -- report it rather than trust the rest.
    ctrl = float(np.nanmax(
        np.abs(df.rho_split_control.abs() - df.rho_entropy.abs())))

    def fam_max(family, colname):
        sub = df[df.signal.isin(family)][colname].abs()
        return float(sub.max()) if len(sub) else float("nan")

    rep_ent, rep_lean = fam_max(REPR, "rho_entropy"), fam_max(REPR, "rho_lean")
    out_ent, out_lean = fam_max(OUTPUT, "rho_entropy"), fam_max(OUTPUT, "rho_lean")
    # Pre-registered: the split is a target artefact if rep-space, read
    # against the UNFOLDED target, rises to within MATERIAL_RISE of the token
    # family's lead on entropy. Otherwise the fold is ruled out.
    artefact = bool(rep_lean >= out_ent - MATERIAL_RISE)

    print(f"\n##### fold_check {model} / {dataset}")
    print(df.to_string(index=False))
    print(f"\nREPR   max|rho|: entropy={rep_ent:.4f}  lean={rep_lean:.4f}")
    print(f"OUTPUT max|rho|: entropy={out_ent:.4f}  lean={out_lean:.4f}")
    print(f"null-control max||rho_split|-|rho_entropy|| = {ctrl:.4f}  "
          f"({'OK' if ctrl < 0.02 else 'FAIL: split must reproduce H'})")
    print(f"MATERIAL_RISE={MATERIAL_RISE} -> " + (
        "ARTEFACT: rep-space is competitive on the unfolded target. CUT the "
        "split from the abstract; report the fold as the finding." if artefact
        else "fold ruled out: rep-space stays flat. Proceed to family."))

    dst = _dst(model, dataset)
    df.to_parquet(f"{dst}/arr_fold_{model}__{dataset}_L{layer}_{scheme}"
                  f"_K{n_clusters}_tau{tau}_s{seeds.replace(',', '-')}.parquet")
    vol.commit()
    return dict(model=model, dataset=dataset, rep_ent=rep_ent, rep_lean=rep_lean,
                out_ent=out_ent, out_lean=out_lean, null_control=ctrl,
                artefact=artefact)


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60)
def fold_gold(model: str, dataset: str, layer: int = -1,
              scheme: str = "span_mean", n_clusters: int = 200,
              tau: float = 10.0, seeds: str = "0,1,2"):
    """The fold test with a GENUINELY un-folded target.

    fold_check's `lean` = p(option 2) is degenerate on this data: our pipeline
    canonicalised chaosnli_alpha by CROWD MAJORITY, so p(option 2) <= 0.5 for
    every item and lean is a monotone reparam of H -- rho_lean == rho_entropy
    identically, testing nothing (verified: p1 in [0, 0.5], frac>0.5 = 0.000).

    This function instead orients on the alpha-NLI GOLD axis, recovered from
    the ORIGINAL ChaosNLI release (chaosNLI_alphanli.jsonl) joined positionally
    and verified n_correct == raw count@majority for all 1532 items. The
    resulting lean_gold = p(crowd -> gold) spans [0, 1]: on 10.6% of items the
    crowd majority DISAGREES with gold, so H(lean_gold) is non-monotone and the
    fold is finally testable. Map built by arr_build_leangold.py ->
    /vol/source/chaosnli_alpha_leangold_byitem.json.

    Pre-registered reading is unchanged (MATERIAL_RISE): rep-space |rho| rising
    to within MATERIAL_RISE of the token family's lead against lean_gold means
    the split is a target-summary artefact; staying flat rules the fold out.
    """
    assert MATERIAL_RISE is not None, "freeze MATERIAL_RISE first"
    if dataset not in BINARY_DATASETS:
        raise ValueError(f"{dataset} is not binary; the fold does not apply")
    import numpy as np, pandas as pd
    scores, ent, ids = _load_peritem(model, dataset, layer, scheme,
                                     n_clusters, tau, seeds)
    fp = "/vol/source/chaosnli_alpha_leangold_byitem.json"
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"{fp}\nRun `modal run arr_build_leangold.py` first to recover the "
            f"un-canonicalised gold-oriented lean from the ChaosNLI release.")
    lg = json.load(open(fp))
    missing = [int(i) for i in ids if str(int(i)) not in lg]
    if missing:
        raise KeyError(f"{len(missing)} dump ids absent from lean_gold map, "
                       f"e.g. {missing[:5]}")
    lean = np.array([lg[str(int(i))] for i in ids])
    split = np.abs(lean - 0.5)          # NULL CONTROL: monotone reparam of H

    rows = [dict(model=model, dataset=dataset, signal=s,
                 rho_entropy=_rho(v, ent),
                 rho_lean_gold=_rho(v, lean),
                 rho_split_control=_rho(v, split))
            for s, v in scores.items()]
    df = pd.DataFrame(rows)
    ctrl = float(np.nanmax(
        np.abs(df.rho_split_control.abs() - df.rho_entropy.abs())))

    def fam_max(family, colname):
        sub = df[df.signal.isin(family)][colname].abs()
        return float(sub.max()) if len(sub) else float("nan")

    rep_ent, rep_lean = fam_max(REPR, "rho_entropy"), fam_max(REPR, "rho_lean_gold")
    out_ent, out_lean = fam_max(OUTPUT, "rho_entropy"), fam_max(OUTPUT, "rho_lean_gold")
    artefact = bool(rep_lean >= out_ent - MATERIAL_RISE)
    # how much any single signal gains from the unfolded target
    df["gain_lean"] = df.rho_lean_gold.abs() - df.rho_entropy.abs()

    print(f"\n##### fold_gold {model} / {dataset}  (gold-oriented, un-folded)")
    print(df.round(4).to_string(index=False))
    print(f"\nfrac(lean_gold<0.5) crowd-vs-gold disagreement drives the test")
    print(f"REPR   max|rho|: entropy={rep_ent:.4f}  lean_gold={rep_lean:.4f}")
    print(f"OUTPUT max|rho|: entropy={out_ent:.4f}  lean_gold={out_lean:.4f}")
    print(f"max per-signal gain from unfolding = {df.gain_lean.max():+.4f}")
    print(f"null-control max||rho_split|-|rho_entropy|| = {ctrl:.4f}  "
          f"({'OK' if ctrl < 0.02 else 'FAIL: split must reproduce H'})")
    print(f"MATERIAL_RISE={MATERIAL_RISE} -> " + (
        "ARTEFACT: rep-space is competitive on the unfolded target. CUT the "
        "split from the abstract; report the fold as the finding." if artefact
        else "fold ruled out: rep-space stays flat even with real direction. "
        "The split survives this objection."))

    dst = _dst(model, dataset)
    df.to_parquet(f"{dst}/arr_foldgold_{model}__{dataset}_L{layer}_{scheme}"
                  f"_K{n_clusters}_tau{tau}_s{seeds.replace(',', '-')}.parquet")
    vol.commit()
    return dict(model=model, dataset=dataset, rep_ent=rep_ent, rep_lean=rep_lean,
                out_ent=out_ent, out_lean=out_lean, null_control=ctrl,
                max_gain=float(df.gain_lean.max()), artefact=artefact)


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 2)
def ladder(model: str, dataset: str, layer: int = -1,
           scheme: str = "span_mean", n_clusters: int = 200,
           tau: float = 10.0, seeds: str = "0,1,2", n_boot: int = 2000):
    """Does elaboration add information over the rung below?

    Three representation-space signals read the same set of vectors with
    increasing machinery:
        rung 0  knn_dist      distance to k nearest neighbours   (no frame)
        rung 1  size_only     1 - 1/|F_argmax|                   (frame)
        rung 2  credal_width  sum_k m_k (1 - 1/|F_k|)     (frame + mass)

    Claim: both nested partial correlations are indistinguishable from zero.
    That converts "elaboration buys nothing" from assertion to measurement.
    """
    assert NULL_THRESHOLD is not None, "freeze NULL_THRESHOLD first"
    import numpy as np, pandas as pd
    scores, ent, _ = _load_peritem(model, dataset, layer, scheme,
                                   n_clusters, tau, seeds)
    rows = []
    for lo_, up_ in zip(LADDER[:-1], LADDER[1:]):
        x, z = scores[up_], scores[lo_]
        obs = _partial_rho(x, ent, z)
        cl, ch, p = _item_bootstrap(
            lambda i: _partial_rho(x[i], ent[i], z[i]), len(ent), B=n_boot)
        rows.append(dict(model=model, dataset=dataset, tau=tau,
                         lower=lo_, upper=up_, partial_rho=obs,
                         ci_lo=cl, ci_hi=ch, p=p,
                         adds_nothing=bool(abs(obs) < NULL_THRESHOLD
                                           and cl <= 0 <= ch),
                         rho_upper=_rho(x, ent), rho_lower=_rho(z, ent)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_parquet(f"{_dst(model, dataset)}/arr_ladder_tau{tau}.parquet")
    vol.commit()
    return rows


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 2)
def sign_stability(model: str, dataset: str, layer: int = -1,
                   scheme: str = "span_mean", n_clusters: int = 200,
                   taus: str = "5,10,25", seeds: str = "0,1,2"):
    """Is the credal_width sign a property of the construction, or of tau=10?

    The whole reframe rests on this. The published panel is a single
    temperature; a sign that held only there would be a property of that
    setting.

    DO NOT AVERAGE OVER TAU. size_only is exactly tau-invariant (argmax is
    tau-free), so a tau-averaged width compared against it compares
    quantities of different kinds. RESULTS_ALL.md defect 6.

    Reading, pre-registered:
      sign constant outside the binary-target cells -> mechanism claim holds
      any flip -> weaken to a per-dataset claim and report the flip. Do not
                  smooth it.
    """
    import numpy as np, pandas as pd
    rows = []
    for tau in [float(t) for t in taus.split(",")]:
        try:
            scores, ent, _ = _load_peritem(model, dataset, layer, scheme,
                                           n_clusters, tau, seeds)
        except FileNotFoundError as e:
            print(f"[skip tau={tau}] {e}")
            continue
        for sig in LADDER:
            r = _rho(scores[sig], ent)
            rows.append(dict(model=model, dataset=dataset, tau=tau,
                             signal=sig, rho=r,
                             sign=int(np.sign(r)) if np.isfinite(r) else 0))
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["signal"], columns="tau",
                         values="rho").to_string())
    flips = (df[df.signal == "credal_width"].groupby("dataset")["sign"]
             .nunique().max() if len(df) else 0)
    print(f"\ncredal_width sign classes within this cell: {flips} "
          f"({'STABLE' if flips <= 1 else 'FLIPS -- weaken the claim'})")
    df.to_parquet(f"{_dst(model, dataset)}/arr_sign_stability.parquet")
    vol.commit()
    return rows


@app.function(image=nbhd_image, volumes={"/vol": vol}, timeout=60 * 60 * 3)
def neighbourhood(model: str, dataset: str, layer: int = -1,
                  scheme: str = "span_mean", arm: str = "correct",
                  k_grid: str = "10,25,50", n_folds: int = 5, seed: int = 0,
                  n_clusters: int = 200, tau: float = 10.0, seeds: str = "0,1,2"):
    """Alternative readings of the same neighbourhood, plus the ceiling.

    Cardinality is one summary among several, so the ladder result invites
    the objection that a different summary would work. This answers it.

      nbhd_spread          mean pairwise distance WITHIN the neighbourhood
      nbhd_heterogeneity   gold-label entropy over the neighbourhood
      nbhd_oracle_CEILING  cross-fitted kNN regression of the target

    The oracle is NOT a signal. It observes the target and is not
    deployable. Every table, caption and sentence mentioning it must say
    'oracle'. Its job is to bound how much of the annotator entropy is
    locally smooth in this space, which separates "the information is
    absent" from "our constructions fail to read it".

    Neighbour construction matches analyse_hf.uq_panel exactly (brute-force
    NearestNeighbors, self excluded) so knn_dist and these read the same
    neighbourhoods. Do not switch metric or backend. k is NOT tuned against
    the target -- the grid is a robustness band.

    Gold labels for nbhd_heterogeneity are the majority argmax of the
    annotator distribution. Where that is degenerate (the binary cell is
    canonicalised correct-first, so argmax is constant) or the column is
    absent, heterogeneity is reported N/A rather than silently filled.
    Target and item order come from the per-item dump; H is loaded via
    analyse_hf.load_states and aligned to the dump by item_id.
    """
    import numpy as np, pandas as pd
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import spearmanr
    import analyse_hf as A

    root = _root(model, dataset)
    meta = A._meta(root)
    lyr = meta["n_total_layers"] + 1 + layer if layer < 0 else layer

    # target (annotator entropy) and canonical item order from the dump
    scores, target, ids = _load_peritem(model, dataset, lyr, scheme,
                                        n_clusters, tau, seeds)
    ids = np.asarray([int(i) for i in ids])

    # pooled states, aligned to the dump ids (never assume row order matches)
    ids_h, H = A.load_states(root, meta, lyr, arm, scheme)
    pos = {int(i): r for r, i in enumerate(ids_h)}
    miss = [int(i) for i in ids if int(i) not in pos]
    if miss:
        raise KeyError(f"{len(miss)} dump ids absent from states, e.g. {miss[:5]}")
    H = H[[pos[int(i)] for i in ids]]
    n = len(ids)

    # gold = majority argmax of the annotator distribution, if well defined
    idx = A._index(root)
    col = ("soft_labels" if "soft_labels" in idx.columns
           else "annotator_labels" if "annotator_labels" in idx.columns else None)
    gold = None
    if col is not None:
        gmap = {}
        for iid, raw in zip(idx.item_id, idx[col]):
            v = json.loads(raw) if isinstance(raw, str) else raw
            p = np.asarray(v, float).ravel()
            gmap[int(iid)] = int(np.argmax(p)) if p.size else -1
        g = np.array([gmap.get(int(i), -1) for i in ids])
        if np.all(g >= 0) and len(np.unique(g)) >= 2:
            gold = g
    if gold is None:
        why = "column absent" if col is None else "argmax degenerate (canonicalised)"
        print(f"[{dataset}] nbhd_heterogeneity = N/A  ({why})")

    def nbrs(k):
        nn = NearestNeighbors(n_neighbors=min(k + 1, n),
                              algorithm="brute").fit(H)
        d, i = nn.kneighbors(H)
        return i[:, 1:], d[:, 1:]

    def sp(v):
        ok = np.isfinite(v) & np.isfinite(target)
        return float(spearmanr(v[ok], target[ok])[0]) if ok.sum() > 30 else np.nan

    rows, summary = [], []
    for k in [int(x) for x in k_grid.split(",")]:
        nb, _ = nbrs(k)

        spread = np.empty(n)
        for i, ix in enumerate(nb):
            P = H[ix]
            d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
            spread[i] = d[np.triu_indices(len(P), 1)].mean()

        if gold is not None:
            nl = int(np.max(gold)) + 1
            het = np.empty(n)
            for i, ix in enumerate(nb):
                c = np.bincount(gold[ix], minlength=nl).astype(float)
                p = c / max(c.sum(), 1e-12)
                het[i] = -(p * np.log(np.clip(p, 1e-12, None))).sum()
        else:
            het = np.full(n, np.nan)

        # Cross-fitted: neighbours searched only within training folds, so no
        # item contributes to its own prediction. Plain leave-one-out kNN on
        # the full set leaks through mutual neighbourhoods and inflates the
        # ceiling.
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, n_folds, n)
        oracle = np.full(n, np.nan)
        for f in range(n_folds):
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            if len(tr) <= k or len(te) == 0:
                continue
            nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(H[tr])
            _, j = nn.kneighbors(H[te])
            oracle[te] = target[tr][j].mean(1)

        for name, v in (("nbhd_spread", spread),
                        ("nbhd_heterogeneity", het),
                        ("nbhd_oracle_CEILING", oracle)):
            rows.append(pd.DataFrame(dict(item_id=ids, signal=name, k=k, value=v)))

        # Oracle reported as BOTH rank correlation AND R^2 -- a ceiling given
        # only as a rank correlation is hard to read as "how much is recoverable".
        ok = np.isfinite(oracle) & np.isfinite(target)
        ss_res = float(np.sum((target[ok] - oracle[ok]) ** 2))
        ss_tot = float(np.sum((target[ok] - target[ok].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        summary.append(dict(k=k, rho_spread=sp(spread),
                            rho_het=(sp(het) if gold is not None else np.nan),
                            rho_oracle_CEILING=sp(oracle), oracle_R2_CEILING=r2))

    long = pd.concat(rows, ignore_index=True)
    long["model"] = model; long["dataset"] = dataset
    long["layer"] = lyr; long["scheme"] = scheme
    long.to_parquet(f"{_dst(model, dataset)}/arr_nbhd_L{lyr}_{scheme}"
                    f"_k{k_grid.replace(',', '-')}_s{seed}.parquet")
    vol.commit()

    sdf = pd.DataFrame(summary)
    print(f"\n##### neighbourhood {model} / {dataset}  (layer {lyr}, n={n})")
    print(sdf.round(4).to_string(index=False))
    print("nbhd_oracle_CEILING is an ORACLE: it sees the target and is never "
          "reported as a method. R^2 bounds how much annotator entropy is "
          "locally recoverable in this representation.")
    return sdf.to_dict("records")


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 2)
def family(model: str, dataset: str, layer: int = -1,
           scheme: str = "span_mean", n_clusters: int = 200,
           tau: float = 10.0, seeds: str = "0,1,2", n_boot: int = 2000):
    """Does the representation-vs-output family split survive inference?

    Currently the split is selection-on-max with no machinery: each cell's
    "best metric" is an argmax over nine signals, and comparing
    best-of-family against best-of-family inflates both sides by an unknown
    amount.

    Family membership is NOT chosen here -- it is a collapse of the GROUP
    dict in analyse_hf.py, fixed before results were inspected.

    |rho| is used deliberately: here the question is which family is more
    predictive, not which direction it points. This is the one place in this
    file where abs() is correct.

    Run this only AFTER fold_check clears. Even a significant split is still
    confounded with target support size until it does. Both gates.
    """
    assert FAMILY_STAT is not None, "freeze FAMILY_STAT first"
    import numpy as np, pandas as pd
    scores, ent, _ = _load_peritem(model, dataset, layer, scheme,
                                   n_clusters, tau, seeds)

    def stat(sub, e, fam):
        v = [abs(_rho(sub[s], e)) for s in fam if s in sub]
        v = [x for x in v if np.isfinite(x)]
        if not v:
            return float("nan")
        return float(np.mean(v) if FAMILY_STAT == "mean" else np.max(v))

    def delta(i=None):
        sub = {k: (v[i] if i is not None else v) for k, v in scores.items()}
        e = ent[i] if i is not None else ent
        return stat(sub, e, REPR) - stat(sub, e, OUTPUT)

    obs = delta()
    lo, hi, p = _item_bootstrap(delta, len(ent), B=n_boot)
    row = dict(model=model, dataset=dataset, tau=tau, stat=FAMILY_STAT,
               delta=obs, ci_lo=lo, ci_hi=hi, p=p)
    print(json.dumps(row, indent=2))
    pd.DataFrame([row]).to_parquet(
        f"{_dst(model, dataset)}/arr_family_tau{tau}.parquet"); vol.commit()
    return row


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 30)
def collect(alpha: float = 0.05):
    """Gather every arr_* parquet on the volume and apply BH per panel.

    Corrections are applied WITHIN a panel, never pooled across panels that
    test different things:
      ladder  -> 9 cells x 2 nested gains = 18 rows
      family  -> 9 rows

    Record which rows were excluded and why. The ARR dispersion panel
    corrects over 22 informative rows of 27 with a documented exclusion
    rule; match that discipline.
    """
    import pandas as pd, numpy as np
    for kind, pat in (("ladder", "arr_ladder_*.parquet"),
                      ("family", "arr_family_*.parquet")):
        fs = sorted(glob.glob(f"{DERIVED}/*/{pat}"))
        if not fs:
            print(f"[{kind}] nothing on the volume yet"); continue
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        rej, q = _bh(df["p"].to_numpy(), alpha=alpha)
        df["reject"], df["q"] = rej, q
        print(f"\n===== {kind}: {len(df)} rows, BH at alpha={alpha} =====")
        print(df.to_string(index=False))
        df.to_parquet(f"{DERIVED}/arr_{kind}_corrected.parquet")
    vol.commit()


@app.function(image=image, volumes={"/vol": vol})
def status():
    """What exists on the volume. Cheap, read-only. Run before long jobs."""
    for m in MODELS:
        for d in DATASETS:
            have = sorted(os.path.basename(p) for p in
                          glob.glob(f"{DERIVED}/{m}__{d}/peritem_*.parquet"))
            arr = sorted(os.path.basename(p) for p in
                         glob.glob(f"{DERIVED}/{m}__{d}/arr_*.parquet"))
            print(f"{m:12s} {d:16s} peritem={len(have)} arr={len(arr)}")
            if not have:
                print("    ^ Patch 1 not applied or uq_panel not re-run here")


@app.local_entrypoint()
def main(stage: str = "audit"):
    """Fan out across the nine cells.

        modal run arr_modal.py --stage=audit
        modal run arr_modal.py --stage=ladder
        modal run arr_modal.py --stage=sign
        modal run arr_modal.py --stage=family

    Order matters. audit -> fold (binary cell only) -> ladder -> nbhd ->
    family. Nothing downstream is trustworthy until audit passes, and
    family should not run until fold clears.
    """
    cells = [(m, d) for m in MODELS for d in DATASETS]
    if stage == "audit":
        for r in audit_target.starmap(cells):
            print(r["dataset"], r["model"], "ok")
    elif stage == "fold":
        for r in fold_check.starmap(
                [(m, d) for m, d in cells if d in BINARY_DATASETS]):
            print(r)
    elif stage == "ladder":
        for r in ladder.starmap(cells):
            print(r)
    elif stage == "sign":
        for r in sign_stability.starmap(cells):
            print(len(r), "rows")
    elif stage == "nbhd":
        for r in neighbourhood.starmap(cells):
            print(r, "rows")
    elif stage == "family":
        for r in family.starmap(cells):
            print(r)
    else:
        raise SystemExit(f"unknown stage: {stage}")
