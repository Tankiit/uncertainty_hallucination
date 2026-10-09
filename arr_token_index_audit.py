"""Read-only boundary audit of the persisted ARR hidden-state spans."""
import json
from pathlib import Path
import modal

app=modal.App("rsuq-arr-token-boundary-audit")
vol=modal.Volume.from_name("rsuq-latent",create_if_missing=False)
image=(modal.Image.debian_slim(python_version="3.11")
       .pip_install("numpy","pandas","pyarrow","safetensors","scipy")
       .add_local_python_source("analyse_hf"))


@app.function(image=image,volumes={"/vol":vol},timeout=1200,memory=8192)
def audit(model: str,dataset: str):
    import numpy as np
    import glob
    from safetensors import safe_open
    import analyse_hf as A
    root=A._root(model,dataset); meta=A._meta(root); idx=A._index(root)
    layer=meta["n_total_layers"]
    ci,C=A.load_states(root,meta,layer,"correct","first")
    wi,W=A.load_states(root,meta,layer,"wrong","first")
    pos={int(i):j for j,i in enumerate(wi)}
    assert all(int(i) in pos for i in ci)
    W=W[[pos[int(i)] for i in ci]]
    diff=np.max(np.abs(C-W),axis=1)
    exact=np.all(C==W,axis=1)
    scale=np.maximum(np.maximum(abs(C).max(axis=1),abs(W).max(axis=1)),1e-12)
    cosine=np.sum(C*W,axis=1)/(np.linalg.norm(C,axis=1)*np.linalg.norm(W,axis=1))
    imap=idx.set_index("item_id")
    equal_length=np.array([imap.loc[int(i),"n_ans_tok_correct"]==imap.loc[int(i),"n_ans_tok_wrong"] for i in ci])
    eci,EC=A.load_states(root,meta,0,"correct","first")
    ewi,EW=A.load_states(root,meta,0,"wrong","first")
    epos={int(i):j for j,i in enumerate(ewi)}
    EW=EW[[epos[int(i)] for i in eci]]
    embedding_equal=np.all(EC==EW,axis=1)
    answer_ids={}
    for path in sorted(glob.glob(f"{root}/logits/shard-*.safetensors")):
        with safe_open(path,framework="np") as f:
            for key in f.keys():
                iid,arm,field=key.split(".")
                if field=="ans_ids":
                    answer_ids[(int(iid),arm)]=np.asarray(f.get_tensor(key)).ravel()
    unequal_first_answer_ids=0
    same_boundary_embedding_despite_different_answer=0
    for j,iid in enumerate(eci):
        a=answer_ids[(int(iid),"correct")]; b=answer_ids[(int(iid),"wrong")]
        if len(a) and len(b) and a[0]!=b[0]:
            unequal_first_answer_ids+=1
            same_boundary_embedding_despite_different_answer+=int(embedding_equal[j])
    # At span[1], grouping embedding vectors by first answer id must be
    # one-to-one if that is the first answer-token embedding.
    token_vectors={}; vector_tokens={}; second_rows=0
    for path in sorted(glob.glob(f"{root}/states/shard-*.safetensors")):
        with safe_open(path,framework="np") as f:
            for key in f.keys():
                iid,arm,field=key.split(".")
                if field!="h": continue
                span=np.asarray(f.get_slice(key)[meta["captured_layers"].index(0)])
                if len(span)<2: continue
                aid=int(answer_ids[(int(iid),arm)][0]); vec=span[1].tobytes()
                token_vectors.setdefault(aid,set()).add(vec)
                vector_tokens.setdefault(vec,set()).add(aid)
                second_rows+=1
    columns=list(idx.columns)
    boundary_cols=[c for c in columns if any(t in c.lower() for t in ["ctx","tok","span","start","end","offset","degenerate"])]
    row_examples=[]
    for _,r in idx.head(3).iterrows():
        row_examples.append({c:(r[c].item() if hasattr(r[c],"item") else r[c])
                             for c in ["item_id"]+boundary_cols})
    return dict(model=model,dataset=dataset,layer=layer,n=len(ci),
        exact_equal_rows=int(exact.sum()),exact_equal_fraction=float(exact.mean()),
        max_abs_difference=float(diff.max()),median_abs_difference=float(np.median(diff)),
        max_relative_difference=float((diff/scale).max()),min_cosine=float(cosine.min()),
        same_length_n=int(equal_length.sum()),
        same_length_exact_fraction=float(exact[equal_length].mean()) if equal_length.any() else None,
        different_length_exact_fraction=float(exact[~equal_length].mean()) if (~equal_length).any() else None,
        embedding_first_equal_fraction=float(embedding_equal.mean()),
        first_embedding_unique_correct=int(len(np.unique(EC,axis=0))),
        different_first_answer_id_n=unequal_first_answer_ids,
        equal_embedding_with_different_answer_id_n=same_boundary_embedding_despite_different_answer,
        second_position_rows=second_rows,
        second_position_token_embedding_bijection=all(len(v)==1 for v in token_vectors.values()) and all(len(v)==1 for v in vector_tokens.values()),
        second_position_unique_first_answer_ids=len(token_vectors),
        meta=meta,index_columns=columns,boundary_examples=row_examples)


@app.local_entrypoint()
def main():
    dest=Path("arr_results/rs_value/token_index")
    dest.mkdir(parents=True,exist_ok=True)
    cells=[(m,d) for m in ["llama3_8b","mistral_7b","qwen2_5_7b"]
           for d in ["chaosnli","chaosnli_alpha","pavlick_nli"]]
    for result in audit.starmap(cells,order_outputs=False):
        path=dest/f"{result['model']}__{result['dataset']}.json"
        path.write_text(json.dumps(result,indent=2,default=str)+"\n")
        print(json.dumps({k:result[k] for k in ["model","dataset","n","exact_equal_fraction","max_abs_difference"]}),flush=True)
