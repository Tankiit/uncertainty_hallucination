# RSUQ — Random-Set Uncertainty Quantification

One library, four papers. Frames are the interface; papers are thin drivers.

    rsuq/core/frame.py    FrameProtocol; FixedFrame (NeurIPS, Stage II,
                          Barycenter); ContextFrame (AAAI)
    rsuq/core/senses.py   sense inventory for ContextFrame (OD-1..OD-3 marked)
    rsuq/core/beliefs.py  mass heads, q, pignistics (per-position aware)
    rsuq/core/signals.py  W, H_tc, H(BetP), size-controlled W
    rsuq/diagnostics/     orthogonality gate, redundancy screen, smoking-gun
                          pairs, AUROC/sep-ratio/ECE
    papers/aaai_driver.py the controlled frame comparison (claims + ablations
                          pre-registered in-file)
    tests/test_core.py    identities I1-I8, pinned

Design rule: if a paper driver exceeds ~200 lines, the abstraction leaked —
move the logic into the library.

## Collaborator handoffs

### ARR disagreement-panel verification

The active collaborator brief is
[`naim_arr.md`](naim_arr.md). It covers regeneration of the 9-cell
human-disagreement panel, multiple-comparisons correction, and reconciliation
of the 3-win versus 2-win summaries.

This task requires access to the Modal workspace containing the
`rsuq-latent` volume. Repository access alone does not grant Modal access.
Before running an expensive analysis, verify authentication and inspect the
available extraction cells:

```bash
modal profile current
modal run analyse_hf.py::cells
```

If either command fails for authorization or the expected volume/cells are not
visible, stop and report that to Tanmoy. Do not substitute local or downloaded
datasets for the volume-backed data.

Once access is confirmed, follow `naim_arr.md` exactly. The analysis functions
write machine-readable parquet files under `/vol/derived/` and emit
compile-compatible text blocks beginning with `##### model / dataset`.
Concatenate those text blocks into `a1_pooled.log`, `uq_panel.log`, and
`correctness_panel.log`; do not derive or alter numerical results while
formatting them.
