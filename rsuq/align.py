
from __future__ import annotations


def _require_fast(tokenizer):
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "rsuq.align needs a fast tokenizer (offset mapping). Load with "
            "AutoTokenizer.from_pretrained(..., use_fast=True).")


def locate_word_positions(tokenizer, context: str, word: str,
                          which: str = "last") -> dict:
    _require_fast(tokenizer)
    start = context.find(word)
    if start < 0:
        raise ValueError(f"{word!r} not a substring of context")
    end = start + len(word)
    enc = tokenizer(context, return_offsets_mapping=True,
                    return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()                        
    toks = [i for i, (c0, c1) in enumerate(offsets)
            if c0 < end and c1 > start and c1 > c0]                            
    if not toks:
        raise ValueError(f"no token overlaps span [{start},{end}) for {word!r}")
    idx = toks[-1] if which == "last" else toks[0]
    return {"input_ids": enc["input_ids"], "char_span": (start, end),
            "token_span": (toks[0], toks[-1]), "score_index": idx,
            "n_subtokens": len(toks)}


def map_span_to_tokens(tokenizer, text: str, span_start: int, span_end: int,
                       encoding=None) -> list[int]:
    _require_fast(tokenizer)
    if encoding is None:
        encoding = tokenizer(text, return_offsets_mapping=True,
                             return_tensors="pt")
    offsets = encoding["offset_mapping"][0].tolist()
    return [i for i, (c0, c1) in enumerate(offsets)
            if c0 < span_end and c1 > span_start and c1 > c0]


def find_answer_span(context: str, answer: str) -> tuple[int, int] | None:
    i = context.find(answer)
    return (i, i + len(answer)) if i >= 0 else None
