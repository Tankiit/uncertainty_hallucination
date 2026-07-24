# ARR disagreement panel — run summary (Naim)

Regenerated per `naim_arr.md` §4–§7. Status: §5–§7 done, **§8 blocked** (no p-value for the paired gain — reported, awaiting decision).

## Configuration

| | |
|---|---|
| Models | llama3_8b, mistral_7b, qwen2_5_7b |
| Datasets (disagreement) | chaosnli, chaosnli_alpha, pavlick_nli |
| Grid | K=200, tau ∈ {5,10,25}, seeds {0,1,2}, scheme=span_mean, final layer, arm=correct |
| Target | Spearman |rho| vs human annotator entropy; baseline `size_only = 1 − 1/|F_argmax|` |
| Hardware | Modal, CPU (no GPU) |

## Headline: 3 win / 1 lose / 5 tie  (reproduces a1()'s 3/1/5)

Cell-level verdict (win = beats density null at ≥1 informative tau):

| model \ dataset | chaosnli | chaosnli_alpha | pavlick_nli |
|---|---|---|---|
| llama3_8b | tie | tie | tie |
| mistral_7b | LOSE | tie | WIN |
| qwen2_5_7b | WIN | tie | WIN |

## Full dispersion panel (27 rows)

| model | dataset | tau | gain | 95% CI (bootstrap) | verdict |
|---|---|---|---|---|---|
| llama3_8b | chaosnli | 5.00 | -0.0022 | [-0.0162,+0.0121] | tie |
| llama3_8b | chaosnli | 10.00 | -0.0038 | [-0.0194,+0.0120] | tie |
| llama3_8b | chaosnli | 25.00 | -0.0135 | [-0.0336,+0.0060] | tie |
| llama3_8b | chaosnli_alpha | 5.00 | +0.0009 | [-0.0187,+0.0198] | tie |
| llama3_8b | chaosnli_alpha | 10.00 | +0.0029 | [-0.0249,+0.0383] | tie |
| llama3_8b | chaosnli_alpha | 25.00 | +0.0074 | [-0.0279,+0.0462] | tie |
| llama3_8b | pavlick_nli | 5.00 | -0.0127 | [-0.0441,+0.0391] | tie |
| llama3_8b | pavlick_nli | 10.00 | -0.0186 | [-0.0522,+0.0444] | tie |
| llama3_8b | pavlick_nli | 25.00 | -0.0167 | [-0.0503,+0.0501] | tie |
| mistral_7b | chaosnli | 5.00 | +0.0074 | [-0.0034,+0.0181] | uninformative |
| mistral_7b | chaosnli | 10.00 | -0.0156 | [-0.0312,-0.0001] | **LOSE** |
| mistral_7b | chaosnli | 25.00 | -0.0449 | [-0.0640,-0.0259] | **LOSE** |
| mistral_7b | chaosnli_alpha | 5.00 | +0.0078 | [-0.0067,+0.0140] | uninformative |
| mistral_7b | chaosnli_alpha | 10.00 | +0.0184 | [-0.0088,+0.0317] | tie |
| mistral_7b | chaosnli_alpha | 25.00 | +0.0087 | [-0.0191,+0.0307] | tie |
| mistral_7b | pavlick_nli | 5.00 | +0.0375 | [+0.0109,+0.0660] | uninformative |
| mistral_7b | pavlick_nli | 10.00 | +0.0750 | [+0.0268,+0.1260] | **WIN** |
| mistral_7b | pavlick_nli | 25.00 | +0.0458 | [-0.0279,+0.1182] | tie |
| qwen2_5_7b | chaosnli | 5.00 | +0.0323 | [+0.0252,+0.0396] | uninformative |
| qwen2_5_7b | chaosnli | 10.00 | +0.0402 | [+0.0294,+0.0509] | **WIN** |
| qwen2_5_7b | chaosnli | 25.00 | +0.0348 | [+0.0209,+0.0478] | **WIN** |
| qwen2_5_7b | chaosnli_alpha | 5.00 | +0.0052 | [-0.0111,+0.0150] | uninformative |
| qwen2_5_7b | chaosnli_alpha | 10.00 | +0.0013 | [-0.0192,+0.0202] | tie |
| qwen2_5_7b | chaosnli_alpha | 25.00 | +0.0228 | [-0.0194,+0.0456] | tie |
| qwen2_5_7b | pavlick_nli | 5.00 | +0.0738 | [+0.0442,+0.1021] | **WIN** |
| qwen2_5_7b | pavlick_nli | 10.00 | +0.0849 | [+0.0374,+0.1290] | **WIN** |
| qwen2_5_7b | pavlick_nli | 25.00 | +0.0071 | [-0.0632,+0.0794] | tie |

## Baseline UQ panel (9 cells, tau=10)

| cell | best metric | \|rho\| | W | knn | size | pred_ent | W vs knn |
|---|---|---|---|---|---|---|---|
| llama3_8b / chaosnli | **size_only** | 0.201 | -0.197 | +0.197 | -0.201 | +0.042 | tie |
| llama3_8b / chaosnli_alpha | **neg_mean_logprob** | 0.148 | -0.026 | +0.051 | +0.014 | +0.089 | tie |
| llama3_8b / pavlick_nli | **pred_maxprob** | 0.191 | +0.005 | +0.058 | -0.019 | +0.058 | tie |
| mistral_7b / chaosnli | **size_only** | 0.267 | -0.236 | +0.249 | -0.267 | +0.080 | W worse |
| mistral_7b / chaosnli_alpha | **neg_mean_logprob** | 0.185 | +0.024 | +0.050 | +0.009 | +0.132 | tie |
| mistral_7b / pavlick_nli | **credal_width** | 0.246 | -0.246 | +0.153 | -0.187 | +0.189 | W better |
| qwen2_5_7b / chaosnli | **credal_width** | 0.272 | -0.272 | +0.249 | -0.248 | -0.095 | W better |
| qwen2_5_7b / chaosnli_alpha | **neg_mean_logprob** | 0.149 | +0.007 | +0.021 | +0.004 | +0.111 | tie |
| qwen2_5_7b / pavlick_nli | **credal_width** | 0.396 | -0.396 | +0.375 | -0.327 | +0.201 | tie |

## Blocker (§8)

BH/Bonferroni must correct the **paired gain** `|rho(W)| − |rho(size_only)|`, but `pooled_gain_ci()` returns only a bootstrap CI (2.5/97.5) + seedSD + n_pos — **no p-value**. Per brief §8, reported as a missing statistic; not inferring p from the CI crossing zero. Awaiting: report as-is, or implement a validated bootstrap p-value.
