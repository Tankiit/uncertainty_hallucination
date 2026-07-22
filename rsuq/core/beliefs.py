
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12


def _expand_kappa(kappa: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return kappa.expand_as(like) if kappa.dim() < like.dim() else kappa


class MassHead(nn.Module):

    def __init__(self, d_model: int, K: int, hidden: int | None = None):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                  nn.Linear(hidden, K))
                    if hidden else nn.Linear(d_model, K))

    def forward(self, h):  return self.net(h)
    def masses(self, h):   return F.softmax(self.net(h), dim=-1)


def cluster_prob_mass(p: torch.Tensor, kappa: torch.Tensor, K: int):
    kap = _expand_kappa(kappa, p)
    s = torch.zeros(*p.shape[:-1], K, device=p.device, dtype=p.dtype)
    s.scatter_add_(-1, kap, p)
    return s


def within_cluster_q(p: torch.Tensor, kappa: torch.Tensor, K: int):
    kap = _expand_kappa(kappa, p)
    s = cluster_prob_mass(p, kappa, K)
    return p / (s.gather(-1, kap) + EPS)


def pignistic(m: torch.Tensor, kappa: torch.Tensor, sizes: torch.Tensor):
    kap = _expand_kappa(kappa, m.new_empty(*m.shape[:-1], kappa.shape[-1]))
    sz = sizes if sizes.dim() == m.dim() else sizes.expand(*m.shape[:-1], -1)
    return m.gather(-1, kap) / sz.gather(-1, kap)


def ranked_pignistic(m, q, kappa, sizes, lam: float = 1.0):
    kap = _expand_kappa(kappa, q)
    sz = sizes if sizes.dim() == m.dim() else sizes.expand(*m.shape[:-1], -1)
    inv = 1.0 / sz.gather(-1, kap)
    return m.gather(-1, kap) * ((1 - lam) * inv + lam * q)
