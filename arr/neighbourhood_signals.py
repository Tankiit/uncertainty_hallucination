"""Neighbourhood signals -- the delegated module (see BRIEF).

Four readings of the same set of vectors around an item. All computed from
cached pooled hidden states plus index.parquet. No re-extraction, no GPU.
Owner: collaborator. Does NOT touch the frame, K, tau, the mass assignment,
or credal_width. Those are fixed for this paper.

  S1 nbhd_spread          how diffuse the neighbourhood is
  S2 nbhd_heterogeneity   gold-label entropy over the neighbourhood
  S3 nbhd_oracle          CEILING, not a method. It sees the target.
  S4 nbhd_eff_rank        optional, low priority, robustness row only

Neighbour construction matches analyse_hf.uq_panel exactly (brute-force
NearestNeighbors, self excluded) so that knn_dist and these signals are
reading the same neighbourhoods. Do not switch metric or backend.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

K_HEADLINE = 10          # matches the existing knn_dist baseline
K_GRID = (10, 25, 50)    # robustness band. NOT tuned against the target.


def neighbours(H, k):
    """Indices and distances of the k nearest neighbours, self excluded."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(H)),
                          algorithm="brute").fit(H)
    dist, idx = nn.kneighbors(H)
    return idx[:, 1:], dist[:, 1:]


def s1_spread(H, k=K_HEADLINE):
    """Mean pairwise distance WITHIN each neighbourhood.

    Distinguishes 'far from everything' (knn_dist already measures that)
    from 'sitting in a diffuse region'. This is the summary credal width
    was intended to proxy.

    TODO: O(n k^2 d) as written below. Fine at n=3000, check at n=13k.
          If too slow, the trace of the local covariance is a cheap
          equivalent up to a constant -- but pick ONE and report which.
    """
    idx, _ = neighbours(H, k)
    out = np.empty(len(H))
    for i, nb in enumerate(idx):
        P = H[nb]
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        out[i] = d[np.triu_indices(len(P), 1)].mean()
    return out


def s2_heterogeneity(H, gold, k=K_HEADLINE):
    """Entropy of the GOLD label distribution over each neighbourhood.

    Semantic rather than geometric. Uses gold labels only -- never the
    annotator distribution. If this were allowed to see annotator labels it
    would be a second oracle, not a signal.
    """
    idx, _ = neighbours(H, k)
    n_lab = int(np.max(gold)) + 1
    out = np.empty(len(H))
    for i, nb in enumerate(idx):
        c = np.bincount(gold[nb], minlength=n_lab).astype(float)
        p = c / max(c.sum(), 1e-12)
        out[i] = -(p * np.log(np.clip(p, 1e-12, None))).sum()
    return out


def s3_oracle(H, target, k=K_HEADLINE, n_folds=5, seed=0):
    """CEILING. Cross-fitted kNN regression of the target on the neighbourhood.

    Predicts item i's annotator entropy from the annotator entropies of its
    neighbours, with i held out. Bounds how much of the target is locally
    smooth in this representation space.

    THIS IS NOT A SIGNAL. It observes the target. Every table, caption and
    sentence mentioning it must say 'oracle'. If it ever appears in a row
    adjacent to the nine signals without that label, the paper is making a
    claim it cannot support.

    Cross-fitting: neighbours are searched only within the training folds,
    so no item ever contributes to its own prediction. A plain
    leave-one-out kNN on the full set would leak through mutual
    neighbourhoods and inflate the ceiling.

    TODO: report BOTH the Spearman of prediction against truth AND an R^2.
          The paper's other numbers are rank correlations, but a ceiling
          expressed only as a rank correlation is hard to interpret as
          'how much is recoverable'.
    """
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(H))
    pred = np.full(len(H), np.nan)
    for f in range(n_folds):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        if len(tr) <= k or len(te) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(H[tr])
        _, nb = nn.kneighbors(H[te])
        pred[te] = target[tr][nb].mean(1)
    return pred


def s4_eff_rank(H, k=K_HEADLINE):
    """OPTIONAL. Participation ratio of the local covariance spectrum.

    Build only if S1-S3 are done and there is time. Sits close to published
    work (EigenScore/INSIDE, local intrinsic dimensionality), so this is a
    robustness row, never a contribution.
    """
    raise NotImplementedError("low priority -- see BRIEF")


def compute_all(H, gold, target, k_grid=K_GRID, with_oracle=True):
    """Return a tidy long frame: item_id, signal, k, value.

    Nothing else. No correlations, no verdicts -- the existing panel code
    consumes this and does the statistics.

    TODO: filename convention for the caller. MUST carry model, dataset,
          layer, pooling, k-grid, seed. RESULTS_ALL.md defect 7 is a
          filename that omitted these and let a smoke run silently
          overwrite a paper-setting parquet.
    """
    import pandas as pd
    rows = []
    for k in k_grid:
        rows.append(pd.DataFrame(dict(signal="nbhd_spread", k=k,
                                      value=s1_spread(H, k))))
        rows.append(pd.DataFrame(dict(signal="nbhd_heterogeneity", k=k,
                                      value=s2_heterogeneity(H, gold, k))))
        if with_oracle:
            rows.append(pd.DataFrame(dict(signal="nbhd_oracle_CEILING", k=k,
                                          value=s3_oracle(H, target, k))))
    out = pd.concat(rows, ignore_index=True)
    out["item_id"] = np.tile(np.arange(len(H)), len(out) // len(H))
    return out
