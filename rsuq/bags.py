
from __future__ import annotations
import torch
import torch.nn as nn


@torch.no_grad()
def mc_dropout_bag(model, input_ids, n: int) -> torch.Tensor:
    was_training = model.training
    model.eval()
    _enable_dropout(model)                                                   
    try:
        ids = input_ids.expand(n, -1) if input_ids.dim() == 2\
              else input_ids.unsqueeze(0).expand(n, -1)              
        return model.base_model(ids).last_hidden_state                   
    finally:
        model.train(was_training)


def _enable_dropout(model):
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()


@torch.no_grad()
def token_perturb_bag(model, input_ids, n: int, p: float = 0.1,
                      vocab_size: int | None = None,
                      protect_last: int = 1) -> torch.Tensor:
    model.eval()
    V = vocab_size or model.config.vocab_size
    x = (input_ids if input_ids.dim() == 2 else
         input_ids.unsqueeze(0)).expand(n, -1).clone()               
    mask = torch.rand(x.shape, device=x.device) < p
    if protect_last:
        mask[:, -protect_last:] = False
    x[mask] = torch.randint(0, V, (int(mask.sum()),), device=x.device)
    return model.base_model(x).last_hidden_state                        


@torch.no_grad()
def embed_noise_bag(model, input_ids, n: int, sigma: float = 0.05
                    ) -> torch.Tensor:
    model.eval()
    wte = model.get_input_embeddings()
    base = wte(input_ids)                                         
    scale = sigma * base.std()
    noisy = base.expand(n, -1, -1) + scale * torch.randn(
        n, *base.shape[1:], device=base.device, dtype=base.dtype)
    return model.base_model(inputs_embeds=noisy).last_hidden_state


GENERATORS = {"mc_dropout": mc_dropout_bag,
              "token_perturb": token_perturb_bag,
              "embed_noise": embed_noise_bag}


def make_bag(model, input_ids, n: int, kind: str = "mc_dropout", **kw):
    return GENERATORS[kind](model, input_ids, n, **kw)


def bag_mean(bag: torch.Tensor) -> torch.Tensor:
    return bag.mean(0)
