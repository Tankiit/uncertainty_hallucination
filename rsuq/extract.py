
from __future__ import annotations
import torch
from dataclasses import dataclass


                                                                           
def chunk_texts(tokenizer, texts, block_size: int = 256,
                max_blocks: int | None = None) -> torch.Tensor:
    enc = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    flat = [t for seq in enc for t in seq]
    M = len(flat) // block_size
    if max_blocks:
        M = min(M, max_blocks)
    return torch.tensor(flat[:M * block_size]).view(M, block_size)


                                                                          
@dataclass
class StateCache:
    h: torch.Tensor                                                          
    gold: torch.Tensor                          

    def __len__(self):
        return self.h.shape[0]


@torch.inference_mode()
def collect_states(model, blocks: torch.Tensor, batch_size: int = 32,
                   device: str = "cuda", pool: str = "last",
                   dtype=None) -> StateCache:
    model.eval()
    if dtype is not None:
        model.to(dtype)
    hs, ys = [], []
    base = model.base_model                                          
    need_all = pool == "last4_mean"
    for i in range(0, blocks.shape[0], batch_size):
        ids = blocks[i:i + batch_size].to(device)
        if need_all:
            out = base(ids, output_hidden_states=True)
            h = torch.stack(out.hidden_states[-4:], 0).mean(0)
        else:
            h = base(ids).last_hidden_state                        
        hs.append(h[:, :-1].reshape(-1, h.shape[-1]).float().cpu())
        ys.append(ids[:, 1:].reshape(-1).cpu())
    return StateCache(torch.cat(hs), torch.cat(ys))


                                                                          
@torch.inference_mode()
def logits_from_states(model, h: torch.Tensor) -> torch.Tensor:
    head = model.get_output_embeddings()
    return head(h.to(next(head.parameters()).device,
                     next(head.parameters()).dtype)).float()


def embedding_matrix(model) -> torch.Tensor:
    return model.get_input_embeddings().weight.detach()
