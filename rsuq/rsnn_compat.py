
from __future__ import annotations
import numpy as np
import torch


                                                                              
def betp_matrix(focal_sets: list[set], classes: list) -> np.ndarray:
    M = np.zeros((len(focal_sets), len(classes)))
    for j, A in enumerate(focal_sets):
        if not A:
            continue
        inv = 1.0 / len(A)
        for i, c in enumerate(classes):
            if c in A:
                M[j, i] = inv
    return M


def final_betp(mass: np.ndarray, betp_mat: np.ndarray) -> np.ndarray:
    return mass @ betp_mat


                                                                               
def groundtruth_belief_encode(labels, focal_sets: list[set]) -> np.ndarray:
    Y = np.zeros((len(labels), len(focal_sets)), dtype=np.int64)
    for i, lab in enumerate(labels):
        for j, A in enumerate(focal_sets):
            if lab in A:
                Y[i, j] = 1
    return Y


                                                                              
def mobius_inverse(belief_preds: np.ndarray, focal_sets: list[set],
                   add_universal: bool = True) -> np.ndarray:
    n = len(focal_sets)
    coeff = np.zeros((n, n))
    for i, A in enumerate(focal_sets):
        for j, B in enumerate(focal_sets):
            if B.issubset(A):
                coeff[j, i] = (-1) ** (len(A) - len(B))
    mass = belief_preds @ coeff
    mass[mass < 0] = 0
    if add_universal:
        resid = np.clip(1 - mass.sum(-1), 0, None)
        mass = np.concatenate([mass, resid[:, None]], -1)
    return mass / mass.sum(-1, keepdims=True)


                                                                               
def assert_matches_scatter_pignistic(frame, m: torch.Tensor, classes=None):
    from rsuq.core.beliefs import pignistic
    V = frame.V
    classes = classes or list(range(V))
    fs = [set(frame.members[k].tolist()) for k in range(frame.K)]
    Bm = betp_matrix(fs, classes)
    bp_matrix = final_betp(m.detach().cpu().numpy(), Bm)
    bp_scatter = pignistic(m, frame.kappa, frame.sizes).detach().cpu().numpy()
    assert np.allclose(bp_matrix, bp_scatter, atol=1e-5),\
        "RS-NN matrix pignistic disagrees with RSUQ scatter on a disjoint frame"
    return True
