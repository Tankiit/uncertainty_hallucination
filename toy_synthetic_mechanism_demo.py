"""
toy_synthetic_mechanism_demo.py

FROZEN for handoff -- see BRIEF_toy_synthetic_demo.md for the task.
Don't change the generation/construction logic; only vary K and tau via
sweep()'s arguments.

Shows, on a small synthetic universe, whether r(W, density) differs
between GEOMETRIC focal sets (KMeans, mirrors the real pipeline) and
CONFUSION focal sets (membership independent of spatial position, by
construction) -- testing whether the AAAI paper's density-collapse
finding is a property of geometric frame construction specifically.

THE ALGEBRAIC POINT:
  coefficient(F_k) = 1 - 1/|F_k| saturates toward 1 as |F_k| grows.
  Geometric (KMeans) focal sets are mechanically LARGER in dense
  regions and SMALLER in sparse ones, so |F_k| -- and hence W --
  inherits the density signal. A confusion structure whose membership
  rule ignores spatial position has no such mechanical entanglement.

LOCKED DESIGN DECISIONS (do not vary these -- see brief for why):
  - Synthetic universe: one dense Gaussian blob + one sparse Gaussian
    blob (N=500 total by default), 2D.
  - Density proxy: mean distance to k=10 nearest neighbors.
  - Confusion focal sets: uniform random label per point, independent
    of spatial position -- the "decoupled by construction" reading.
  - Mass model: soft (softmax over -distance/tau to cluster centroids
    for geometric; Dirichlet noise peaked on the true label for
    confusion). |F_k| is always the HARD-partition cardinality.

WHAT TO VARY: K (K_geometric == K_confusion) and tau, via sweep().
"""

import csv

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def credal_width(mass, sizes):
    """W(m) = sum_k m_k * (1 - 1/|F_k|). mass: (N,K), sizes: (K,)."""
    coeff = 1.0 - 1.0 / sizes
    return mass @ coeff


# ---------------------------------------------------------------------
# 1. SYNTHETIC UNIVERSE -- LOCKED, do not modify
# ---------------------------------------------------------------------
def make_synthetic_points(n_dense=400, n_sparse=100, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.normal(loc=0.0, scale=0.3, size=(n_dense, dim))
    sparse = rng.normal(loc=6.0, scale=3.0, size=(n_sparse, dim))
    X = np.vstack([dense, sparse])
    region = np.array([0] * n_dense + [1] * n_sparse)  # 0=dense, 1=sparse
    return X, region


# ---------------------------------------------------------------------
# 2. DENSITY PROXY -- LOCKED, do not modify
# ---------------------------------------------------------------------
def knn_density(X, k=10):
    """Mean distance to k nearest neighbors. Larger = sparser."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    return dists[:, 1:].mean(axis=1)  # drop self-distance


# ---------------------------------------------------------------------
# 3. GEOMETRIC-NEIGHBORHOOD FOCAL SETS -- LOCKED, do not modify
# ---------------------------------------------------------------------
def geometric_focal_sets(X, k_clusters=8, tau=1.0, seed=0):
    km = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10).fit(X)
    hard_assign = km.labels_
    sizes = np.bincount(hard_assign, minlength=k_clusters).astype(float)

    d = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=-1)
    logits = -d / tau
    mass = np.exp(logits - logits.max(axis=1, keepdims=True))
    mass /= mass.sum(axis=1, keepdims=True)

    return mass, sizes


# ---------------------------------------------------------------------
# 4. "TRUE CONFUSION" FOCAL SETS -- LOCKED, do not modify
# ---------------------------------------------------------------------
def confusion_focal_sets(X, k_confusion=8, tau=1.0, seed=0):
    rng = np.random.default_rng(seed + 1000)  # separate stream from geometry
    n = X.shape[0]
    hard_assign = rng.integers(0, k_confusion, size=n)
    sizes = np.bincount(hard_assign, minlength=k_confusion).astype(float)

    alpha = np.full(k_confusion, 0.5)
    mass = rng.dirichlet(alpha, size=n)
    mass[np.arange(n), hard_assign] *= 3.0
    mass /= mass.sum(axis=1, keepdims=True)

    return mass, sizes


# ---------------------------------------------------------------------
# 5. THE SWEEP -- this is what you run
# ---------------------------------------------------------------------
def sweep(K_values=(4, 8, 16, 32), tau_values=(0.5, 1.0, 2.0, 5.0),
          n_dense=400, n_sparse=100, knn_k=10, seed=0,
          out_csv="sweep_results.csv"):
    X, region = make_synthetic_points(n_dense, n_sparse, seed=seed)
    density = knn_density(X, k=knn_k)

    rows = []
    for K in K_values:
        for tau in tau_values:
            mass_geo, sizes_geo = geometric_focal_sets(X, k_clusters=K, tau=tau, seed=seed)
            W_geo = credal_width(mass_geo, sizes_geo)

            mass_conf, sizes_conf = confusion_focal_sets(X, k_confusion=K, tau=tau, seed=seed)
            W_conf = credal_width(mass_conf, sizes_conf)

            r_geo, p_geo = stats.pearsonr(W_geo, density)
            r_conf, p_conf = stats.pearsonr(W_conf, density)

            row = dict(K=K, tau=tau, r_geo=r_geo, p_geo=p_geo,
                       r_conf=r_conf, p_conf=p_conf,
                       median_size_geo=float(np.median(sizes_geo)),
                       median_size_conf=float(np.median(sizes_conf)))
            rows.append(row)
            print(f"K={K:>3} tau={tau:>4}: r_geo={r_geo:+.4f}  r_conf={r_conf:+.4f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten {out_csv}")

    return rows, X, density


# ---------------------------------------------------------------------
# 6. SANITY PLOT -- eyeball check, not a paper figure
# ---------------------------------------------------------------------
def sanity_plot(X, density, K=8, tau=1.0, seed=0, out_png="sanity_plot.png"):
    import matplotlib.pyplot as plt

    mass_geo, sizes_geo = geometric_focal_sets(X, k_clusters=K, tau=tau, seed=seed)
    W_geo = credal_width(mass_geo, sizes_geo)

    mass_conf, sizes_conf = confusion_focal_sets(X, k_confusion=K, tau=tau, seed=seed)
    W_conf = credal_width(mass_conf, sizes_conf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(density, W_geo, s=8, alpha=0.5)
    axes[0].set_title(f"Geometric (K={K}, tau={tau})")
    axes[0].set_xlabel("density (knn dist)")
    axes[0].set_ylabel("W")

    axes[1].scatter(density, W_conf, s=8, alpha=0.5, color="orange")
    axes[1].set_title(f"Confusion (K={K}, tau={tau})")
    axes[1].set_xlabel("density (knn dist)")
    axes[1].set_ylabel("W")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Written {out_png}")


if __name__ == "__main__":
    rows, X, density = sweep()
    sanity_plot(X, density)
