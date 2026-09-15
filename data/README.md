# Data used by SAVRec

SAVRec aligns two public data sources:

1. **HotelRec** — user–hotel review/interaction data. The preparation code downloads the
   four Kaggle parts used in the project:
   - `hariwh0/hotelrec-dataset-1`
   - `hariwh0/hotelrec-dataset-2`
   - `hariwh0/hotelrec-dataset-3`
   - `hariwh0/hotelrec-dataset-4`

2. **Hotels-50K** — hotel identity and image metadata used for the visual modality.
   Source repository: `GWUvision/Hotels-50K`
   (`https://github.com/GWUvision/Hotels-50K`).

The raw third-party datasets and hotel images are **not redistributed** in this repository.
Users should obtain them from their original sources and comply with the corresponding
licenses and terms.

## Frozen artifacts required by the final V7 experiment

Set the environment variable `SAVREC_DATA_ROOT` to the directory containing:

```text
train_v6.parquet
val_v6.parquet
test_v6.parquet
evaluation_catalog_items_v6.parquet
user_map_v6.json
item_map_v6.json
item_text_embeddings_sbert_v6.pt
clip_per_image_v4.pt
```

If `SAVREC_DATA_ROOT` is not set, the notebooks use the original Colab/Google Drive
project directory as a backward-compatible default.

The final experiment notebook reconstructs the cleaned V7 view, evidence features,
graphs, and all trainable models from these frozen model-independent artifacts.

## Provenance notes

- The final V6 split/catalog was constructed using `visual_coverage_v3.csv` from the
  ResNet-50 visual-coverage stage.
- The paper run used the frozen `clip_per_image_v4.pt` artifact.
- `notebooks/01_Data_Preparation.ipynb` contains a clean CLIP ViT-B/32 reconstruction
  stage that targets the final V6 catalog directly.
- Hotel image URLs are external and can disappear or change. Therefore a fresh image
  download may not reproduce the exact image slots used by the frozen paper run.
  For exact numerical reproduction, use the frozen `clip_per_image_v4.pt` when
  redistribution of that derived artifact is permitted.
