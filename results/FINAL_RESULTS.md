# Frozen final SAVRec V7 results

These values are copied from the final cleaned evaluation run after the clean-test gate
reporting correction. The correction affected only the reported clean evidence-gate value;
recommendation metrics were unchanged.

## Main test comparison

| Model | HR@20 mean | NDCG@20 mean | MRR@20 mean |
|---|---:|---:|---:|
| SMORE | 0.069520 | 0.034509 | 0.024836 |
| SAVRec | 0.055556 | 0.021186 | 0.011970 |
| Backbone-only | 0.050698 | 0.020301 | 0.012054 |
| Text-Only | 0.042198 | 0.016577 | 0.009763 |
| Most-Popular | 0.034608 | 0.013703 | 0.007871 |
| MMGCN | 0.028537 | 0.010433 | 0.005478 |
| LightGCN | 0.019126 | 0.008367 | 0.005475 |
| VBPR | 0.023376 | 0.008189 | 0.004111 |
| BM3 | 0.020947 | 0.007445 | 0.003886 |

## Evidence ablation (NDCG@20)

| Variant | NDCG@20 |
|---|---:|
| SAVRec | 0.021186 |
| SAVRec − Coherence | 0.021316 |
| SAVRec − Availability | 0.021064 |
| SAVRec − History | 0.020535 |
| SAVRec + Shuffled Evidence | 0.020458 |
| SAVRec − Popularity | 0.015097 |
| Backbone-only | 0.020301 |

## Multi-seed visual stress (NDCG@20)

| Image removal | Backbone-only | SAVRec | SAVRec clean/test evidence gate |
|---:|---:|---:|---:|
| 0% | 0.020301 | 0.021186 | 0.211483 |
| 25% | 0.019929 | 0.020829 | 0.209167 |
| 50% | 0.020007 | 0.020965 | 0.206439 |
| 75% | 0.018844 | 0.020356 | 0.201079 |

## Exploratory validation findings

- Lowest image-availability quartile: SAVRec NDCG@20 0.027271 vs backbone 0.022752.
- Highest image-availability quartile: SAVRec 0.024036 vs backbone 0.029010.
- Parameter overhead vs matched backbone: 58,689 parameters, approximately 0.1683%.

The subgroup analysis is exploratory/post-hoc and should be reported as such.
