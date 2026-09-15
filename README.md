# SAVRec: Observable-Evidence-Guided Visual Recommendation under Extreme Sparsity

This repository contains the cleaned implementation used for the final SAVRec V7
journal experiments.

## Repository structure

```text
notebooks/
  01_Data_Preparation.ipynb
  02_SAVRec_Final_Experiment.ipynb
scripts/
  01_data_preparation.py
  02_savrec_final_experiment.py
data/
  README.md
results/
  FINAL_RESULTS.md
requirements.txt
.gitignore
```

## Dataset sources

The benchmark is a derived alignment of:

- **HotelRec** user–hotel review/interaction data (Kaggle parts
  `hariwh0/hotelrec-dataset-1` through `hariwh0/hotelrec-dataset-4`).
- **Hotels-50K** hotel/image metadata from `GWUvision/Hotels-50K`.

Raw third-party datasets and hotel images are not redistributed here. See
`data/README.md` for provenance and artifact requirements.

## Recommended reproducibility route

1. Obtain the frozen model-independent artifacts listed in `data/README.md`.
2. Set `SAVREC_DATA_ROOT` to the directory containing those artifacts.
3. Run `notebooks/02_SAVRec_Final_Experiment.ipynb` from top to bottom on a CUDA GPU.
4. The notebook performs validation-only checkpoint selection, reference-baseline tuning,
   ablations, visual-degradation analysis, final evaluation, significance analysis, and
   paper-ready exports.

`01_Data_Preparation.ipynb` documents the dataset-construction path. The final V6
split/catalog uses `visual_coverage_v3.csv`, and the final visual input is the
per-image CLIP ViT-B/32 artifact `clip_per_image_v4.pt`.

Because hotel image URLs are external, fresh downloads can change over time. Exact
reproduction of the paper numbers should therefore use the frozen V6/CLIP artifacts.

## Important evaluation note

The historical V6 test split was inspected during earlier development. The final V7
architecture, tuning, early stopping, and checkpoint selection use validation only.
The repository and paper therefore do **not** describe the test split as an untouched
or sealed holdout.

## Final model comparison

See `results/FINAL_RESULTS.md` for the frozen final metrics.

## Data and licensing

Follow the original HotelRec and Hotels-50K licenses/terms. Derived artifacts should be
shared separately only when redistribution is permitted.
