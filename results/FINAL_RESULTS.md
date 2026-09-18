# Final SAVRec Results

This file summarizes the final experimental results reported for SAVRec.

All trainable models are evaluated using three model seeds:

```text
42, 1, 7
```

Evaluation uses full-catalog ranking over all 3,322 hotels. Unless otherwise stated,
results are reported as mean ± sample standard deviation across the three model seeds.

---

## Main test comparison

| Model | HR@10 | HR@20 | NDCG@10 | NDCG@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| Most-Popular | 0.0246 | 0.0346 | 0.0112 | 0.0137 | 0.0079 |
| LightGCN | 0.0118 ± 0.0042 | 0.0191 ± 0.0027 | 0.0066 ± 0.0015 | 0.0084 ± 0.0008 | 0.0055 ± 0.0007 |
| Text-Only | 0.0228 ± 0.0042 | 0.0422 ± 0.0080 | 0.0118 ± 0.0026 | 0.0166 ± 0.0035 | 0.0098 ± 0.0025 |
| VBPR | 0.0121 ± 0.0011 | 0.0234 ± 0.0014 | 0.0053 ± 0.0007 | 0.0082 ± 0.0010 | 0.0041 ± 0.0010 |
| MMGCN | 0.0176 ± 0.0043 | 0.0285 ± 0.0074 | 0.0076 ± 0.0026 | 0.0104 ± 0.0033 | 0.0055 ± 0.0022 |
| BM3 | 0.0091 ± 0.0046 | 0.0209 ± 0.0042 | 0.0045 ± 0.0032 | 0.0074 ± 0.0027 | 0.0039 ± 0.0026 |
| SMORE | 0.0468 ± 0.0041 | 0.0695 ± 0.0029 | 0.0288 ± 0.0027 | 0.0345 ± 0.0012 | 0.0248 ± 0.0019 |
| Backbone-only | 0.0301 ± 0.0055 | 0.0507 ± 0.0076 | 0.0151 ± 0.0022 | 0.0203 ± 0.0027 | 0.0121 ± 0.0015 |
| SAVRec | 0.0288 ± 0.0032 | 0.0556 ± 0.0096 | 0.0144 ± 0.0014 | 0.0212 ± 0.0031 | 0.0120 ± 0.0015 |

The exact three-seed means for the primary matched comparison are:

```text
Backbone-only NDCG@20 : 0.020301
SAVRec NDCG@20        : 0.021186
Relative difference   : +4.36%

Backbone-only HR@20   : 0.050698
SAVRec HR@20          : 0.055556
Relative difference   : +9.58%
```

These are descriptive mean differences. Paired user-level tests do not establish
statistically significant superiority of SAVRec over the matched backbone.

SMORE remains the strongest clean-ranking model among the evaluated baselines.

---

## Clean-ranking statistical analysis

For the planned inferential analysis, each user's metric contribution is first
averaged across the three model seeds before paired testing.

For SAVRec versus the matched backbone:

```text
NDCG@20 mean difference  : +0.000886
95% bootstrap CI         : [-0.003673, 0.005223]
Holm-corrected Wilcoxon p: 1.000

HR@20 mean difference    : +0.004857
95% bootstrap CI         : [-0.005161, 0.014572]
Holm-corrected Wilcoxon p: 1.000
```

The confidence intervals include zero. The clean-ranking differences are therefore
not statistically established.

---

## Evidence ablation

Each evidence ablation is independently trained for all three model seeds using the
same SAVRec training protocol.

For a removed evidence variable, the corresponding channel is replaced by its
canonical training/catalog mean during both training and inference while preserving
the evidence-vector dimensionality and model parameter count.

For the shuffled-evidence control, user-side and item-side evidence are independently
permuted using deterministic seed-specific permutations before training.

| Variant | NDCG@20 mean ± SD | Change vs full SAVRec |
|---|---:|---:|
| Backbone-only | 0.020301 ± 0.002736 | -4.18% |
| Shuffled evidence | 0.020458 ± 0.002767 | -3.44% |
| SAVRec - History | 0.020535 ± 0.001714 | -3.07% |
| SAVRec - Popularity | 0.015097 ± 0.002957 | -28.74% |
| SAVRec - Availability | 0.021064 ± 0.004233 | -0.58% |
| SAVRec - Coherence | 0.021316 ± 0.001243 | +0.61% |
| Full SAVRec | 0.021186 ± 0.003051 | — |

### Per-seed NDCG@20

| Variant | Seed 42 | Seed 1 | Seed 7 |
|---|---:|---:|---:|
| Backbone-only | 0.022459 | 0.021220 | 0.017223 |
| Full SAVRec | 0.018718 | 0.020243 | 0.024598 |
| Shuffled evidence | 0.022687 | 0.021324 | 0.017361 |
| SAVRec - History | 0.022020 | 0.018660 | 0.020925 |
| SAVRec - Popularity | 0.015283 | 0.017957 | 0.012052 |
| SAVRec - Availability | 0.025726 | 0.020006 | 0.017460 |
| SAVRec - Coherence | 0.022695 | 0.020971 | 0.020282 |

Aligned evidence has a 3.44% higher three-seed mean NDCG@20 than shuffled evidence.

However, the per-seed ordering is not uniform: shuffled evidence is higher for seeds
42 and 1, whereas full SAVRec is higher for seed 7. The result should therefore be
interpreted as evidence that pair-specific association may carry useful information,
rather than as evidence that aligned evidence consistently outperforms shuffled
evidence.

Item popularity is the strongest individual evidence signal on this benchmark.

---

## Learned gate analysis

Mean Spearman correlations between the learned SAVRec gate and evidence variables are:

```text
Item popularity    : -0.640
User history       : +0.049
Image availability : +0.102
Visual coherence   : +0.074
```

Popularity is therefore the dominant learned gate regulator among the four evidence
variables on this benchmark.

### Gate by popularity quartile

```text
Q1 : 0.404680
Q2 : 0.215481
Q3 : 0.151835
Q4 : 0.067985
```

The learned gate decreases substantially as item popularity increases.

This is a learned association and should not be interpreted as a causal relationship.

---

## Controlled visual-availability degradation

The robustness experiment removes 25%, 50%, and 75% of each item's available images.

Five deterministic corruption seeds are used:

```text
2026, 2027, 2028, 2029, 2030
```

The SAVRec and matched-backbone checkpoints remain frozen. No model is retrained or
retuned after visual corruption.

| Image removal | Backbone NDCG@20 | SAVRec NDCG@20 | SAVRec evidence gate |
|---:|---:|---:|---:|
| 0% | 0.020301 | 0.021186 | 0.209290 |
| 25% | 0.019929 ± 0.002478 | 0.020829 ± 0.003140 | 0.209167 |
| 50% | 0.020007 ± 0.002847 | 0.020965 ± 0.003117 | 0.206439 |
| 75% | 0.018844 ± 0.001700 | 0.020356 ± 0.003653 | 0.201079 |

The corrected clean evidence-gate value is:

```text
0.209290
```

The earlier value `0.211483` is not used in the final reporting.

### Relative mean NDCG@20 difference

```text
Clean : +4.36%
25%   : +4.52%
50%   : +4.79%
75%   : +8.02%
```

At 75% image removal:

```text
Backbone degradation : -7.17%
SAVRec degradation    : -3.92%
```

The mean SAVRec gate also decreases as visual evidence is progressively reduced:

```text
0%  : 0.209290
25% : 0.209167
50% : 0.206439
75% : 0.201079
```

---

## Robustness uncertainty

A post-hoc hierarchical bootstrap with 20,000 resamples is used.

The procedure resamples model-training seeds first and then corruption realizations
within each sampled model seed.

### Paired SAVRec - Backbone NDCG@20

| Image removal | 95% bootstrap CI |
|---:|---:|
| 25% | [-0.003822, 0.007018] |
| 50% | [-0.003626, 0.007516] |
| 75% | [-0.002542, 0.007448] |

At 75% removal:

```text
Mean paired NDCG@20 difference : +0.001512
95% CI                         : [-0.002542, 0.007448]

Difference in absolute degradation favoring SAVRec : +0.000626
95% CI                                          : [-0.000160, 0.001484]
```

Because these intervals include zero, the experiment does not establish a
statistically significant robustness advantage.

The higher SAVRec means and smaller mean degradation are therefore interpreted as
descriptive mechanism evidence.

---

## Exploratory evidence-stratified findings

These subgroup analyses are exploratory and should not be interpreted as confirmatory
statistical evidence.

### Image availability

Lowest image-availability quartile:

```text
Backbone NDCG@20 : 0.022752
SAVRec NDCG@20   : 0.027271
Relative change  : +19.86%
```

Highest image-availability quartile:

```text
Backbone NDCG@20 : 0.029010
SAVRec NDCG@20   : 0.024036
```

The effect is therefore not monotonic across image-availability strata.

### Visual coherence

```text
Low-coherence quartile SAVRec - Backbone : +0.002842
High-coherence quartile                  : -0.000100
```

These subgroup results are exploratory/post-hoc.

---

## Model-size overhead

```text
Matched backbone parameters : 34,870,145
SAVRec parameters           : 34,928,834
Additional parameters       : 58,689
Relative overhead           : approximately 0.168%
```

The additional SAVRec components contain:

```text
Evidence encoder : 9,024 parameters
Evidence gate    : 49,665 parameters
```

---

## Interpretation

The primary contribution of SAVRec is not a claim of state-of-the-art clean-ranking
accuracy.

The results instead support SAVRec as a lightweight and inspectable mechanism for
explicitly regulating visual contribution using directly measurable user-, item-,
and image-side evidence.

Key observations are:

- SAVRec has higher three-seed mean NDCG@20 and HR@20 than its matched
  evidence-free backbone, but the paired clean-ranking differences are not
  statistically significant.
- SMORE remains the strongest clean-ranking baseline.
- Item popularity is the dominant learned evidence signal.
- Aligned evidence has a 3.44% higher three-seed mean NDCG@20 than shuffled evidence,
  although the per-seed ordering is not uniform.
- The learned visual gate decreases as available visual evidence is reduced.
- SAVRec shows smaller descriptive mean degradation than the matched backbone under
  severe image removal, but the hierarchical-bootstrap confidence intervals include
  zero.
- The evidence-guided controller adds only approximately 0.168% additional
  parameters.
