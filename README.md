# CatBoost Bug-Prediction Pipeline (2026) — End-to-End Training, Evaluation, and Batch Inference

Author: Dr. Sebastian Raubitzek

## Overview

This repository implements a complete, reproducible pipeline for training and applying CatBoost models to software-project tabular datasets for **binary bug prediction**.

The workflow supports two parallel model lines:

1) **Classifier line (CatBoostClassifier)**  
   - Produces a hard class prediction `{TARGET}_predicted`  
   - Optionally produces a probability `{TARGET}_predicted_proba`

2) **Risk score line (CatBoostRegressor)**  
   - Produces a continuous **risk score** (clipped to `[0,1]`)  
   - Produces a translated hard class prediction via a fixed threshold (default `0.5`)

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
- The trained model can be applied to **one or many new CSVs**, producing output CSVs with appended prediction columns.

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
   - Saves the same table with appended predictions

---

## Repository Structure

Minimal core pipeline (2026-style):

- `00_data_analysis_merge_projects.py`  
  Merge many per-project CSV files into one combined dataset.

- `01_clean_data.py`  
  Clean and prepare data (selected features + negative-value handling).

- `02_train_test_split.py`  
  Create CSV train/test split (`train.csv`, `test.csv`) with fixed random seed.

---

## Training + Evaluation (two parallel lines)

### A) Classifier line (CatBoostClassifier)

- `03_train_CatBoost_Classifier.py`  
  Train and select best CatBoostClassifier model:
  - Bayesian optimization (BayesSearchCV) vs out-of-the-box
  - Internal train/val split
  - Heavy undersampling strategy for robustness
  - Saves **only** the winner as `best_model.cbm`
  - Saves `training_report.json`, `test_report.json`, and `features.json`

- `04_post_hoc_analysis_Classifier.py`  
  Post-hoc evaluation plots and reports (classifier outputs):
  - Confusion matrices (raw + balanced)
  - Classification reports (raw + undersampled averages)
  - Feature importance (PNG + EPS)
  - Boxplots for top features by predicted class (PNG + EPS)
  - Uses a fixed hex palette (see below)

### B) Risk score line (CatBoostRegressor)

- `03b_RiskScore_train_CatBoost_Regressor.py`  
  Train and select best CatBoostRegressor model (risk score workflow):
  - Same train/selection structure as the classifier line
  - Model output is a continuous score (interpreted as risk)
  - Risk score is translated to a class label via a threshold (default `0.5`)
  - Saves **only** the winner as `best_model.cbm`
  - Saves `training_report.json`, `test_report.json`, and `features.json`

- `04b_RiskScore_post_hoc_analysis_Regressor.py`  
  Post-hoc evaluation plots and reports (risk score + translated class):
  - Uses regressor predictions as risk score
  - Translates risk score to class via threshold for classification-style reporting
  - Confusion matrices (raw + balanced)
  - Classification reports (raw + undersampled averages)
  - Feature importance (PNG + EPS)
  - Boxplots for top features by predicted class (PNG + EPS)
  - Uses the same fixed hex palette

---

## Inference / Batch Application (two parallel lines)

### A) Classifier line inference

- `05_model_csvs_Classifier.py`  
  Batch inference (classifier):
  - Loads `best_model.cbm` + `features.json`
  - Processes CSV files in the configured inference folder(s)
  - Writes updated CSVs with:
    - `{TARGET}_predicted`
    - optionally `{TARGET}_predicted_proba`

- `06_apply_best_model_to_regarded_csvs_2026.py`  
  Batch inference + reporting (classifier, regarded/to_be_tested style):
  - Loads `best_model.cbm` + `features.json`
  - Processes **all CSV files** in the configured folder
  - Writes `*_predicted.csv` and per-file `*_report.txt`
  - Keeps class-relevant columns side-by-side at the end of the table:
    - `{TARGET}` (if present), `{TARGET}_predicted`, and (if written) probability / relabeled risk score

### B) Risk score line inference

- `05b_RiskScore_model_csvs.py`  
  Batch inference (risk score regressor):
  - Loads regressor `best_model.cbm` + `features.json`
  - Produces a continuous risk score per row (clipped to `[0,1]`)
  - Adds translated class prediction via threshold

- `06_apply_best_model_to_regarded_csvs_2026_riskscore.py`  
  Batch inference + reporting (risk score regressor, regarded style):
  - Loads regressor `best_model.cbm` + `features.json` from the `_riskscore` model folder
  - Writes `*_predicted.csv` and per-file `*_report.txt`
  - Appends two columns:
    - `{TARGET}_predicted` (translated class)
    - `{TARGET}_predicted_risk_score` (clipped to `[0,1]`)
  - Orders class-relevant columns side-by-side at the end of the table:
    - `{TARGET}` (if present), `{TARGET}_predicted`, `{TARGET}_predicted_risk_score`

---

## Dependencies

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

Classifier artifacts are written into:

- `./data_{TARGET}/models_{RANDOM_STATE}/`
  - `best_model.cbm`
  - `features.json`
  - `training_report.json`
  - `test_report.json`

Risk score (regressor) artifacts are written into:

- `./data_{TARGET}/models_{RANDOM_STATE}_riskscore/`
  - `best_model.cbm`
  - `features.json`
  - `training_report.json`
  - `test_report.json`

Train/test split CSVs live in:

- `./data_{TARGET}/splits_{RANDOM_STATE}/`
  - `train.csv`
  - `test.csv`

### Inference-time folders

Depending on which inference script you run, inference folders are:

- `./to_be_tested/`  
  Place new project CSVs here (commonly used by the batch inference scripts).

- `./tested/`  
  Output folder used by the simple batch inference scripts.

- `./regarded/`  
  Output/input folder used by the “regarded batch + report” scripts (`06_*`), if configured that way.

All inference scripts append model outputs to the CSVs and write predicted versions (plus optional per-file reports, depending on the script).
