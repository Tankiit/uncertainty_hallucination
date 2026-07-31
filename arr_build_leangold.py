"""Recover the un-canonicalised, gold-oriented crowd preference for
chaosnli_alpha and save it for arr_modal.fold_gold.

Our pipeline canonicalised chaosnli_alpha by CROWD MAJORITY, so p(option 2)
<= 0.5 for every item and the naive fold target (fold_check) is a monotone
reparam of H -- it tests nothing. This downloads the ORIGINAL ChaosNLI release
(chaosNLI_alphanli.jsonl), joins it POSITIONALLY to our cell, and verifies the
alignment directionally: our canonical soft_labels[i][0] (= n at the crowd
MAJORITY option) must equal the raw label_count at the majority index. That
held for all 1532 items. lean_gold = p(crowd -> alpha-NLI gold) is then well
oriented and spans [0, 1] (the crowd majority-disagrees with gold on ~10.6% of
items), so H(lean_gold) is non-monotone and the fold is finally testable.

Writes /vol/source/chaosnli_alpha_leangold_byitem.json (keyed by item_id) and
..._leangold.json (keyed by example_id). Idempotent; safe to re-run.

    modal run arr_build_leangold.py
"""
import modal, json, io, zipfile
app = modal.App("arr-build-leangold")
vol = modal.Volume.from_name("rsuq-latent")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "pandas", "pyarrow", "requests"))
URL = "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1"

@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 20)
def build():
    import numpy as np, requests
    zf = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=600).content))
    fn = [n for n in zf.namelist() if "alpha" in n.lower() and n.endswith(".jsonl")][0]
    rawlines = zf.read(fn).decode().splitlines()

    def parse(e):
        e = json.loads(e)
        lc = e["label_count"] if isinstance(e["label_count"], list) else eval(str(e["label_count"]))
        ld = e["label_dist"] if isinstance(e["label_dist"], list) else eval(str(e["label_dist"]))
        gold = str(e.get("old_label") or e.get("majority_label"))
        maj = str(e["majority_label"])
        return np.asarray(lc, float), np.asarray(ld, float), gold, maj
    raw = [parse(l) for l in rawlines]

    src = json.loads(open("/vol/source/chaosnli_alpha.json").read())
    SOFT, EID = src["soft_labels"], src["example_id"]
    n = len(SOFT)
    assert len(raw) == n, (len(raw), n)

    leang = np.empty(n); gold_ne_maj = 0; dir_ok = 0
    for i in range(n):
        lc, ld, gold, maj = raw[i]
        gi = 0 if gold == "1" else 1
        mi = 0 if maj == "1" else 1
        leang[i] = float(ld[gi])                 # p(crowd -> alpha gold)
        gold_ne_maj += int(gold != maj)
        # our correct_ans is the crowd MAJORITY, so n_correct (our soft[0])
        # must equal the raw count at the MAJORITY index. Proves alignment.
        ncorrect = float(np.asarray(SOFT[i], float)[0])
        if abs(ncorrect - lc[mi]) <= 1e-6:
            dir_ok += 1

    print(f"n={n}")
    print(f"alignment (n_correct == raw count@majority): {dir_ok}/{n}")
    print(f"gold != crowd-majority in {gold_ne_maj}/{n} items "
          f"({100*gold_ne_maj/n:.1f}%) -- these are what make the target unfoldable")
    print(f"lean_gold span {leang.min():.3f}..{leang.max():.3f} | "
          f"mean={leang.mean():.3f} | frac<0.5={float((leang<0.5).mean()):.3f} | "
          f"frac>0.5={float((leang>0.5).mean()):.3f} | frac==0.5={float((leang==0.5).mean()):.3f}")

    if dir_ok == n:
        with open("/vol/source/chaosnli_alpha_leangold_byitem.json", "w") as f:
            json.dump({str(i): float(leang[i]) for i in range(n)}, f)
        with open("/vol/source/chaosnli_alpha_leangold.json", "w") as f:
            json.dump({str(EID[i]): float(leang[i]) for i in range(n)}, f)
        vol.commit()
        print("SAVED lean_gold (by item_id and by example_id) to /vol/source/")
    else:
        print("directional check FAILED -- not saving; orientation not proven")

@app.local_entrypoint()
def main():
    build.remote()
