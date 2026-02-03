
# CatBoost Bug-Prediction Pipeline (2026) — End-to-End Training, Evaluation, and Batch Inference

Author: Dr. Sebastian Raubitzek

## Overview

This repository implements a complete, reproducible pipeline for training and applying a **CatBoostClassifier** to software-project tabular datasets for **binary bug prediction**.

The workflow is designed to be practical for real project data:

- Multiple per-project CSVs can be merged into one dataset.
- Cleaning includes deterministic handling of negative values (optionally replacing with NaN).
- A train/test split is created and stored as CSV.
- A **single “best model”** is trained and selected by comparing:
  1) **Bayesian-optimized CatBoost** (BayesSearchCV)
  2) **Out-of-the-box CatBoost** (defaults, only minimal reproducibility/categorical handling)
- The trained model is evaluated with both:
  - “raw” metrics (natural imbalance)
  - “undersampled/balanced” repeated evaluation (to understand behavior under equal class frequencies)
- The trained model can be applied to **one or many new CSVs** in `to_be_tested/`, producing output CSVs in `tested/` with an appended prediction column.

This repository is intentionally script-first (no hidden frameworks). Each file is a single stage of the pipeline.

---

## Conceptual Structure

1. **Merge project CSVs**
   - Combines many project files into one dataset table.

2. **Clean / preprocess**
   - Selects features
   - Handles invalid values (e.g., negative values) deterministically

3. **Train/test split**
   - Creates `train.csv` and `test.csv` with a fixed random seed

4. **Train + model selection**
   - Internal train/validation split of `train.csv`
   - Bayesian optimization vs out-of-the-box comparison
   - Select **one winner** and save it as `best_model.cbm`

5. **Post-hoc evaluation**
   - Confusion matrices (raw + balanced)
   - Classification reports
   - Feature importance plots
   - Boxplots by predicted class
   - Outputs both PNG and EPS graphics

6. **Inference on new project files**
   - Loads `best_model.cbm` + `features.json`
   - Batch-applies model to new CSV(s)
   - Saves the same table with `{TARGET}_predicted` appended (optionally also probability)

---

## Repository Structure

Minimal core pipeline (2026-style):

- `00_merge_projectcs_to_csv.py`  
  Merge many per-project CSV files into one combined dataset.

- `01_data_analysis_clean.py`  
  Clean and prepare data (selected features + negative-value handling).

- `02_train_test_split.py`  
  Create CSV train/test split (`train_*.csv`, `test_*.csv`) with fixed random seed.

- `03_train_CatBoost_undersampling.py`  
  Train and select best CatBoost model:
  - Bayesian optimization (BayesSearchCV) vs out-of-the-box
  - Internal train/val split
  - Heavy undersampling strategy for robustness
  - Saves **only** the winner as `best_model.cbm`
  - Saves `training_report.json`, `test_report.json`, and `features.json`

- `04_post_hoc_evaluation.py`  
  Post-hoc evaluation plots and reports:
  - Confusion matrices (raw + balanced)
  - Classification reports (raw + undersampled averages)
  - Feature importance (PNG + EPS)
  - Boxplots for top features by predicted class (PNG + EPS)
  - Uses a fixed hex palette (see below)

- `05_run_model_csvs.py`  
  Batch inference:
  - Loads `best_model.cbm` + `features.json`
  - Processes **all CSV files** in `to_be_tested/`
  - Writes updated CSVs to `tested/` (same file names) with `{TARGET}_predicted`

## Dependencies:

python==3.8.19 \
numpy==1.24.3 \
pandas==2.0.3 \
scipy==1.10.1 \
scikit-learn==1.3.2 \
imbalanced-learn==0.11.0 \
catboost==1.2.7 \
scikit-optimize==0.9.0 \
matplotlib==3.7.5 \
seaborn==0.13.2 \
joblib==1.3.2 \
tqdm==4.66.1 

---

## Data and Folder Conventions

### Training-time folders (pipeline outputs)

The training pipeline writes model artifacts into:

- `./data_{TARGET}/models_{RANDOM_STATE}/`
  - `best_model.cbm`
  - `features.json`
  - `training_report.json`
  - `test_report.json`

Train/test split CSVs live in:

- `./data_{TARGET}/splits_{RANDOM_STATE}/`
  - `train.csv`
  - `test.csv`

### Inference-time folders

- `./to_be_tested/`  
  Place new project CSVs here.

- `./tested/`  
  The pipeline writes predicted versions here (same filenames), with a new appended column:
  - `{TARGET}_predicted`
  - (optionally) `{TARGET}_predicted_proba`




