
from __future__ import annotations
import torch
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@torch.inference_mode()
def collect_occurrences(model, tokenizer, blocks, candidate_ids: set[int],
                        device="cuda", pool="last4_mean",
                        max_per_token: int = 200) -> dict[int, torch.Tensor]:
    model.eval()
    base = model.base_model
    store = defaultdict(list)
    for i in range(blocks.shape[0]):
        ids = blocks[i:i+1].to(device)
        out = base(ids, output_hidden_states=(pool == "last4_mean"))
        if pool == "last4_mean":
            h = torch.stack(out.hidden_states[-4:], 0).mean(0)[0]           
        else:
            h = out.last_hidden_state[0]
        row = ids[0]
        for t in range(row.shape[0]):
            tid = int(row[t])
            if tid in candidate_ids and len(store[tid]) < max_per_token:
                store[tid].append(h[t].cpu())
    return {k: torch.stack(v) for k, v in store.items() if len(v) >= 30}


def separation_score(X: np.ndarray, max_senses=4, seed=0):
    best = (-1.0, 1, None)
    for S in range(2, max_senses + 1):
        if X.shape[0] <= S:
            break
        lab = KMeans(S, random_state=seed, n_init=5).fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(X, lab)
        if s > best[0]:
            best = (s, S, lab)
    return best


@torch.inference_mode()
def cluster_prototypes(fixed_frame, occ: dict[int, torch.Tensor], d_model):
    K = fixed_frame.K
    proto = torch.zeros(K, d_model)
    cnt = torch.zeros(K)
    for tid, X in occ.items():
        k = int(fixed_frame.kappa[tid])
        proto[k] += X.mean(0)
        cnt[k] += 1
    miss = cnt == 0
    proto[~miss] /= cnt[~miss].unsqueeze(-1)
    if miss.any():
        proto[miss] = proto[~miss].mean(0)
    return proto


def build_inventory(fixed_frame, occ: dict[int, torch.Tensor], d_model,
                    tau: float = 0.5, max_senses=4, seed=0):
    proto = cluster_prototypes(fixed_frame, occ, d_model)
    poly, cents, s2c, report = [], {}, {}, {}
    for tid, X in occ.items():
        Xn = X.numpy()
        score, S, lab = separation_score(Xn, max_senses, seed)
        report[tid] = {"separation": float(score), "n_senses": int(S),
                       "n_occ": int(X.shape[0]),
                       "in_stratum_P": bool(score > tau)}
        if score <= tau:
            continue                                                         
        poly.append(tid)
        centroids = torch.stack([X[lab == s].mean(0) for s in range(S)])
        cents[tid] = centroids
        s2c[tid] = torch.cdist(centroids, proto).argmin(-1)                  
    return {"poly_tokens": torch.tensor(poly, dtype=torch.long),
            "sense_centroids": cents, "sense_cluster": s2c,
            "prototypes": proto, "separation_report": report,
            "n_stratum_P": len(poly)}
