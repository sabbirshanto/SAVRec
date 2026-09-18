# SAVRec: Evidence-Guided Visual Regulation for Multimodal Recommendation under Extreme Interaction Sparsity

This repository contains the implementation, preprocessing pipeline, training code,
evaluation procedures, and statistical analyses used for the SAVRec study.

SAVRec is a multimodal recommendation framework that explicitly regulates visual
contribution using observable evidence derived from user interaction history, item
popularity, image availability, and cross-image visual coherence.

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
LICENSE
README.md
