
from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


                                                                            
@runtime_checkable
class FrameProtocol(Protocol):
    K: int
    V: int

    def assignments(self, h: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


                                                                            
def _width_coef(sizes: torch.Tensor) -> torch.Tensor:
    return 1.0 - 1.0 / sizes.clamp(min=1.0)


                                                                            
@dataclass
class FixedFrame:
    kappa: torch.Tensor                    
    K: int

    def __post_init__(self):
        counts = torch.bincount(self.kappa, minlength=self.K)
        if (counts == 0).any():
            raise ValueError("empty clusters in fixed frame")
        self.V = int(self.kappa.numel())
        self.sizes = counts.float()
        self.width_coef = _width_coef(self.sizes)
        self._members = None

    def assignments(self, h=None):
        return self.kappa, self.sizes

    @property
    def members(self) -> list[torch.Tensor]:
        if self._members is None:
            order = torch.argsort(self.kappa, stable=True)
            self._members = list(torch.split(order, self.sizes.long().tolist()))
        return self._members

                                                                      
    @staticmethod
    def from_model(model, K: int = 200, pca_dim: int = 64,
                   seed: int = 0) -> "FixedFrame":
        emb = model.get_input_embeddings().weight.detach().cpu().numpy()
        return FixedFrame.from_embeddings(emb, K, pca_dim, seed)

    @staticmethod
    def from_embeddings(emb: np.ndarray, K: int = 200, pca_dim: int = 64,
                        seed: int = 0) -> "FixedFrame":
        from sklearn.decomposition import PCA
        from sklearn.cluster import MiniBatchKMeans
        Z = PCA(n_components=min(pca_dim, emb.shape[1]),
                random_state=seed).fit_transform(emb)
        km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3,
                             batch_size=4096).fit(Z)
        return FixedFrame(kappa=torch.from_numpy(km.labels_).long(), K=K)

    def save(self, path):  torch.save({"kappa": self.kappa, "K": self.K}, path)
    @staticmethod
    def load(path):        d = torch.load(path); return FixedFrame(d["kappa"], d["K"])


                                                                            
@dataclass
class ContextFrame:
    base: FixedFrame
    poly_tokens: torch.Tensor
    sense_centroids: dict = field(default_factory=dict)
    sense_cluster: dict = field(default_factory=dict)

    def __post_init__(self):
        self.K, self.V = self.base.K, self.base.V
        self._base_out = self.base.kappa[self.poly_tokens]                 

    def _select(self, h: torch.Tensor) -> torch.Tensor:
        cols = []
        for w in self.poly_tokens.tolist():
            C = self.sense_centroids[w].to(h.device)                    
            s = torch.cdist(h, C).argmin(-1)                        
            cols.append(self.sense_cluster[w].to(h.device)[s])
        return torch.stack(cols, dim=-1)                              

    def assignments(self, h: torch.Tensor):
        if h is None:
            raise ValueError("ContextFrame requires the hidden state h")
        if h.dim() == 1:
            h = h.unsqueeze(0)
        N = h.shape[0]
        new = self._select(h)                                         
                                                                          
                                                                       
        kappa_t = self.base.kappa.to(h.device).unsqueeze(0).repeat(N, 1)
        kappa_t[:, self.poly_tokens.to(h.device)] = new
                                                                            
        sizes_t = self.base.sizes.to(h.device).unsqueeze(0).repeat(N, 1)
        ones = torch.ones(N, len(self.poly_tokens), device=h.device)
        sizes_t.scatter_add_(-1, self._base_out.to(h.device).expand(N, -1), -ones)
        sizes_t.scatter_add_(-1, new, ones)
        if (sizes_t <= 0).any():
            raise RuntimeError("context reassignment emptied a cluster; "
                               "guard polysemous selection or merge clusters")
        return kappa_t.squeeze(0) if N == 1 else kappa_t,\
               sizes_t.squeeze(0) if N == 1 else sizes_t
