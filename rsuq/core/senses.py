
from __future__ import annotations
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@torch.no_grad()
def pool_last4(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack(hidden_states[-4:], 0).mean(0)[0]


def detect_polysemous(occ_states: dict[int, torch.Tensor],
                      disp_pct: float = 90.0, min_occ: int = 30
                      ) -> torch.Tensor:
    disp = {}
    for w, X in occ_states.items():
        if X.shape[0] < min_occ:
            continue
        c = X.mean(0, keepdim=True)
        disp[w] = float((X - c).norm(dim=-1).mean())
    if not disp:
        return torch.empty(0, dtype=torch.long)
    thr = torch.quantile(torch.tensor(list(disp.values())), disp_pct / 100.0)
    return torch.tensor([w for w, d in disp.items() if d > thr],
                        dtype=torch.long)


def build_senses(occ_states: dict[int, torch.Tensor],
                 poly: torch.Tensor, s_range=(2, 3, 4), seed: int = 0
                 ) -> dict[int, torch.Tensor]:
    out = {}
    for w in poly.tolist():
        X = occ_states[w].numpy()
        best, best_s = None, -1.0
        for S in s_range:
            if X.shape[0] <= S:
                continue
            km = KMeans(n_clusters=S, random_state=seed, n_init=5).fit(X)
            sc = silhouette_score(X, km.labels_)
            if sc > best_s:
                best, best_s = km, sc
        out[w] = torch.from_numpy(best.cluster_centers_).float()
    return out


@torch.no_grad()
def cluster_prototypes(fixed_frame, occ_states: dict[int, torch.Tensor],
                       d_model: int) -> torch.Tensor:
    K = fixed_frame.K
    proto = torch.zeros(K, d_model)
    count = torch.zeros(K)
    for w, X in occ_states.items():
        k = int(fixed_frame.kappa[w])
        proto[k] += X.mean(0)
        count[k] += 1
    missing = count == 0
    proto[~missing] /= count[~missing].unsqueeze(-1)
    if missing.any():
        proto[missing] = proto[~missing].mean(0)                       
    return proto


def map_senses_to_clusters(sense_centroids: dict[int, torch.Tensor],
                           prototypes: torch.Tensor
                           ) -> dict[int, torch.Tensor]:
    return {w: torch.cdist(C, prototypes).argmin(-1)
            for w, C in sense_centroids.items()}
