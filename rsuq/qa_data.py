
from __future__ import annotations
import random


def _appended_span(question: str, response: str) -> tuple[int, int]:
    start = len(question) + 1                                                 
    return (start, start + len(response))


def load_halueval(n: int, task: str = "qa", seed: int = 0):
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", task, split="data")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= n:
            break
        item = ds[i]
        q = item["question"].strip()
        for key, lab in (("right_answer", 0), ("hallucinated_answer", 1)):
            resp = item[key].strip()
            out.append({"question": q, "response": resp, "label": lab,
                        "answer_char_span": _appended_span(q, resp)})
    return out


def load_triviaqa(n: int, seed: int = 0):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    pool = idx[:max(n, 200)]
    out = []
    for k, i in enumerate(pool):
        if len(out) >= n:
            break
        item = ds[i]
        q = item["question"].strip()
        gold = item["answer"]["value"].strip()
                                                         
        j = pool[(k + 1) % len(pool)]
        distractor = ds[j]["answer"]["value"].strip()
        aliases = set(a.lower() for a in item["answer"].get("aliases", [gold]))
        if distractor.lower() in aliases:                                   
            continue
        for resp, lab in ((gold, 0), (distractor, 1)):
            out.append({"question": q, "response": resp, "label": lab,
                        "answer_char_span": _appended_span(q, resp)})
    return out


LOADERS = {"halueval": load_halueval, "triviaqa": load_triviaqa}


def load_qa_instances(dataset: str, n: int, seed: int = 0):
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; have {list(LOADERS)}")
    return LOADERS[dataset](n, seed=seed)
