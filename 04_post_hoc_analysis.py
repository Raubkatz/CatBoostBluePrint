#!/usr/bin/env python3
"""
05_posthoc_evaluation_plots_2026.py

Post-hoc evaluation script (plots + reports) adapted to the NEW 2026 pipeline.

This is a rebuilt version of your legacy evaluation script, but adapted to:
- Load the previously trained CatBoost model from:
    data_{TARGET}/models_{RANDOM_STATE}/best_model.cbm
- Load feature list from:
    data_{TARGET}/models_{RANDOM_STATE}/features.json
- Load test data from:
    data_{TARGET}/splits_{RANDOM_STATE}/test.csv

It intentionally keeps the SAME evaluation “outputs and artifacts” as your original script:
- Folder creation
- Full (raw) confusion matrix
- Averaged/undersampled confusion matrix + normalized confusion matrix
- Full classification report (raw)
- Averaged classification report (undersampled)
- Feature importance bar plot (PNG + EPS)
- Boxplots of top features by predicted class (PNG + EPS)
- Uses the same palette (hex codes)

Plain meaning (“this means…”):
- We do NOT retrain anything here.
  This means we are evaluating exactly what training produced.
- We do NOT use a StandardScaler anymore.
  This means we evaluate CatBoost directly on the raw feature values (CatBoost can handle NaNs).
- The “raw” report uses the test set as-is.
  This means it reflects your real class imbalance.
- The “undersampled” report creates balanced subsets of the test set.
  This means it tries to show behavior when both classes are equally frequent.
- Boxplots are grouped by *predicted* class.
  This means you visualize which features differ according to the model’s own decisions.

Run:
  python 05_posthoc_evaluation_plots_2026.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE: These are kept because your original script used them and you requested
# to keep the evaluation style. We do *not* use StandardScaler anymore because
# the new pipeline does not train a scaler artifact.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler  # Retained because you said "exactly as it is."
from catboost import CatBoostClassifier

###############################################################################
# Custom Under-Sampler (kept from legacy script)
###############################################################################
def custom_equal_under_sampler(X, y, fraction=0.8, random_state=None):
    """
    Perform under-sampling by taking 'fraction' of the minority class
    and the same number of samples from the majority class.

    Plain meaning (“this means…”):
    - Find which class is smaller (minority) in y.
    - Keep only `fraction * minority_count` samples from minority.
    - Keep the same number from the majority.
    - Combine them and shuffle.

    This creates a balanced dataset (50/50 between the classes), but smaller than
    the original. That means:
    - It reduces class imbalance
    - It also reduces sample count (more variance, hence repeated runs are often used)
    """
    rng = np.random.default_rng(seed=random_state)

    X_array = np.asarray(X)
    y_array = np.asarray(y)

    unique_classes = np.unique(y_array)
    if len(unique_classes) != 2:
        raise ValueError("This function supports only binary classification.")

    # Identify minority/majority
    class_counts = {cls: np.sum(y_array == cls) for cls in unique_classes}
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)

    minority_indices = np.where(y_array == minority_class)[0]
    majority_indices = np.where(y_array == majority_class)[0]

    num_minority = class_counts[minority_class]
    num_to_pick_minority = int(round(num_minority * fraction))

    minority_sampled = rng.choice(minority_indices, size=num_to_pick_minority, replace=False)
    majority_sampled = rng.choice(majority_indices, size=num_to_pick_minority, replace=False)

    combined_indices = np.concatenate([minority_sampled, majority_sampled])
    rng.shuffle(combined_indices)

    return X_array[combined_indices], y_array[combined_indices]


###############################################################################
# Set Parameters (adapted to 2026 pipeline paths)
###############################################################################

# Palette kept EXACTLY (hex codes)
# custom_palette = ["#188FA7", "#769FB6", "#9DBBAE", "#D5D6AA", "#E2DBBE"] #orotgonal order
custom_palette = ["#188FA7", "#E2DBBE", "#769FB6", "#9DBBAE", "#D5D6AA"]

# NEW: global font scaler for ALL plotting text (axis labels, ticks, titles, legends, annotations).
# This means:
# - Set FONT_SCALE = 1.0 for the original intended size.
# - Set FONT_SCALE > 1.0 to scale everything up, < 1.0 to scale down.
FONT_SCALE = 1.0

# Pipeline identifiers
TARGET = "isBugPresent"
RANDOM_STATE = 42

# Model artifact from the new training script (saved winner only)
MODEL_PATH = f"./data_{TARGET}/models_{str(RANDOM_STATE)}/best_model.cbm"

# Feature list saved by the training script
FEATURES_JSON_PATH = f"./data_{TARGET}/models_{str(RANDOM_STATE)}/features.json"

# Data paths from the pipeline split step
TEST_DATA_PATH = f'./data_{TARGET}/splits_{str(RANDOM_STATE)}/test.csv'

# Output base folder:
# We keep a legacy-like folder layout underneath the model folder.
# This means you still get confusion_matrices/, feature_importance/, result_txts/, etc.
RESULTS_BASE = f"./data_{TARGET}/models_{str(RANDOM_STATE)}/Results_CatBoost_Classifier"

CONF_MAT_DIR = os.path.join(RESULTS_BASE, "confusion_matrices")
RESULT_TXT_DIR = os.path.join(RESULTS_BASE, "result_txts")
FEAT_IMP_DIR = os.path.join(RESULTS_BASE, "feature_importance")

os.makedirs(CONF_MAT_DIR, exist_ok=True)
os.makedirs(RESULT_TXT_DIR, exist_ok=True)
os.makedirs(FEAT_IMP_DIR, exist_ok=True)

# A readable name for saving files (no pickle name anymore; use model folder name)
MODEL_NAME = f"best_model_{TARGET}_rs{RANDOM_STATE}"

# Check existence of model/features/test
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(FEATURES_JSON_PATH):
    raise FileNotFoundError(f"features.json not found: {FEATURES_JSON_PATH}")
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test CSV not found: {TEST_DATA_PATH}")


###############################################################################
# Global plotting font scaling (NEW)
###############################################################################

# Plain meaning:
# - matplotlib uses rcParams as global defaults.
# - By scaling these once, every plot inherits the scaled font sizes automatically.
plt.rcParams.update({
    "font.size": 12 * FONT_SCALE,
    "axes.titlesize": 20 * FONT_SCALE,
    "axes.labelsize": 18 * FONT_SCALE,
    "xtick.labelsize": 18 * FONT_SCALE,
    "ytick.labelsize": 18 * FONT_SCALE,
    "legend.fontsize": 14 * FONT_SCALE,
    "figure.titlesize": 20 * FONT_SCALE,
})


###############################################################################
# Load model + features
###############################################################################

print(f"Loading CatBoost model: {MODEL_PATH}")

# IMPORTANT:
# The new pipeline saves .cbm via CatBoost's native serializer.
# This means we should load using model.load_model, not pickle.
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# Load the feature list that was used for training
print(f"Loading features list: {FEATURES_JSON_PATH}")
with open(FEATURES_JSON_PATH, "r", encoding="utf-8") as f:
    feature_payload = pd.read_json(f)

# features.json was written as {"target": ..., "features": [...]}
# pandas read_json will interpret it as a DataFrame, so we re-load it safely:
import json
with open(FEATURES_JSON_PATH, "r", encoding="utf-8") as f:
    feature_payload = json.load(f)

selected_features = feature_payload["features"]
loaded_target = feature_payload["target"]

# Sanity check (plain meaning: avoid evaluating the wrong column)
if loaded_target != TARGET:
    raise ValueError(
        f"features.json target mismatch: expected {TARGET}, got {loaded_target}"
    )

print(f"Selected features (n={len(selected_features)}): {selected_features}")


###############################################################################
# Load and Preprocess Test Data
###############################################################################

data_test = pd.read_csv(TEST_DATA_PATH)

# Extract X and y
# Plain meaning (“this means…”):
# - X_test contains only the feature columns used in training.
# - y_test contains the target labels we want to predict.
missing_features = [c for c in selected_features if c not in data_test.columns]
if missing_features:
    raise ValueError(f"Missing feature columns in test.csv: {missing_features}")

if TARGET not in data_test.columns:
    raise ValueError(f"Missing target column '{TARGET}' in test.csv")

X_test = data_test[selected_features].copy()
y_test = data_test[TARGET].astype(int).copy()

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Predict
# Plain meaning:
# - model.predict gives hard labels (0/1) directly.
# - We do not scale inputs; CatBoost handles numeric scales and NaNs.
y_pred = model.predict(X_test)

# Compute Classification Report & Confusion Matrix
full_report = classification_report(y_test, y_pred)
full_conf_matrix = confusion_matrix(y_test, y_pred)

# Label names for plots
class_labels = ["No Bug", "Bug"]


###############################################################################
# Save Full Confusion Matrix (raw / unmodified test set)
###############################################################################

# Plain meaning:
# - This heatmap reflects the class imbalance in your real test data.
# - It is the simplest and most direct evaluation artifact.
plt.figure(figsize=(8, 6))
sns.heatmap(
    full_conf_matrix,
    annot=True,
    fmt="d",
    cmap=custom_palette,
    xticklabels=class_labels,
    yticklabels=class_labels,
    annot_kws={"fontsize": 12 * FONT_SCALE}
)
plt.title("Confusion Matrix (Raw Test Set)", fontsize=20 * FONT_SCALE)
plt.xlabel("Predicted Labels", fontsize=18 * FONT_SCALE)
plt.ylabel("True Labels", fontsize=18 * FONT_SCALE)
plt.tight_layout()
plt.savefig(os.path.join(CONF_MAT_DIR, f"{MODEL_NAME}_conf_matrix.png"))
plt.savefig(os.path.join(CONF_MAT_DIR, f"{MODEL_NAME}_conf_matrix.eps"))
plt.clf()
plt.close()


###############################################################################
# Averaged classification reports with undersampling (kept style)
###############################################################################

# These parameters control how much you undersample and how many times.
# Plain meaning:
# - n_iterations > 1 would average out randomness across different sampled subsets.
# - sampling_ratio controls how many minority samples you keep (fraction of minority).
n_iterations = 1
sampling_ratio = 0.8

acc_scores = []
avg_conf_matrix = np.zeros_like(full_conf_matrix)

aggregated_metrics = {}

for _ in range(n_iterations):
    # IMPORTANT:
    # legacy code undersampled *X_test_scaled*. Now we undersample X_test directly.
    # This means the evaluation is consistent with the new pipeline (no scaler).
    X_under, y_under = custom_equal_under_sampler(X_test, y_test, fraction=sampling_ratio, random_state=None)

    # Predict on undersampled subset
    y_pred_under = model.predict(X_under)

    # Aggregate
    acc_scores.append(accuracy_score(y_under, y_pred_under))
    avg_conf_matrix += confusion_matrix(y_under, y_pred_under)

    metrics_dict = classification_report(y_under, y_pred_under, output_dict=True)

    # Accumulate classification_report dict fields
    for label, scores in metrics_dict.items():
        if isinstance(scores, dict):
            if label not in aggregated_metrics:
                aggregated_metrics[label] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1-score": 0.0,
                    "support": 0
                }
            aggregated_metrics[label]["precision"] += scores["precision"]
            aggregated_metrics[label]["recall"]    += scores["recall"]
            aggregated_metrics[label]["f1-score"]  += scores["f1-score"]
            if "support" in scores:
                aggregated_metrics[label]["support"] += scores["support"]
        else:
            # "accuracy" key
            if label not in aggregated_metrics:
                aggregated_metrics[label] = 0.0
            aggregated_metrics[label] += scores

# Average across iterations
for label, scores in aggregated_metrics.items():
    if isinstance(scores, dict):
        scores["precision"] /= n_iterations
        scores["recall"]    /= n_iterations
        scores["f1-score"]  /= n_iterations
        scores["support"]   = scores["support"] / n_iterations
    else:
        aggregated_metrics[label] /= n_iterations


def dict_to_cr_string(agg_dict):
    """
    Convert aggregated metrics into a classification-report-like string.

    Plain meaning:
    - sklearn gives a nested dict; we print it in a fixed report layout
      similar to sklearn's classification_report string.
    """
    lines = []
    lines.append("               precision    recall  f1-score   support")

    # The keys might include '0', '1', 'accuracy', 'macro avg', 'weighted avg'
    possible_labels = ["0", "1", "2", "3", "accuracy", "macro avg", "weighted avg"]
    for lbl in possible_labels:
        if lbl in agg_dict:
            if lbl == "accuracy":
                val_str = f"{agg_dict[lbl]:.4f}"
                lines.append(f"   accuracy                           {val_str}")
            else:
                precision = agg_dict[lbl]["precision"]
                recall = agg_dict[lbl]["recall"]
                f1 = agg_dict[lbl]["f1-score"]
                support = agg_dict[lbl]["support"]
                lines.append(
                    f"       {lbl:>2}         {precision:>9.4f}   {recall:>7.4f}   {f1:>8.4f}   {int(support):>6}"
                )
    return "\n".join(lines)


avg_class_report_str = dict_to_cr_string(aggregated_metrics)
relative_conf_matrix = avg_conf_matrix / avg_conf_matrix.sum(axis=1, keepdims=True)


###############################################################################
# Save averaged confusion matrices (unnormalized + normalized)
###############################################################################

# Averaged confusion matrix (sum over iterations; in your legacy script you name it averaged)
plt.figure(figsize=(8, 6))
sns.heatmap(
    avg_conf_matrix,
    annot=True,
    fmt=".2f",
    cmap=custom_palette,
    xticklabels=class_labels,
    yticklabels=class_labels,
    annot_kws={"fontsize": 18 * FONT_SCALE}
)
plt.title("Averaged Confusion Matrix (Undersampled)", fontsize=20 * FONT_SCALE)
plt.xlabel("Predicted Labels", fontsize=18 * FONT_SCALE)
plt.ylabel("True Labels", fontsize=18 * FONT_SCALE)
plt.xticks(fontsize=18 * FONT_SCALE)
plt.yticks(fontsize=18 * FONT_SCALE)
plt.tight_layout()
plt.savefig(os.path.join(CONF_MAT_DIR, f"{MODEL_NAME}_avg_conf_matrix.png"))
plt.savefig(os.path.join(CONF_MAT_DIR, f"{MODEL_NAME}_avg_conf_matrix.eps"))
plt.clf()
plt.close()

# Normalized confusion matrix (row-normalized)
plt.figure(figsize=(8, 6))
sns.heatmap(
    relative_conf_matrix,
    annot=True,
    fmt=".2f",
    cmap=custom_palette,
    xticklabels=class_labels,
    yticklabels=class_labels,
    annot_kws={"fontsize": 18 * FONT_SCALE}
)
plt.title("Relative Confusion Matrix (Undersampled, Normalized)", fontsize=20 * FONT_SCALE)
plt.xlabel("Predicted Labels", fontsize=18 * FONT_SCALE)
plt.ylabel("True Labels", fontsize=18 * FONT_SCALE)
plt.xticks(fontsize=18 * FONT_SCALE)
plt.yticks(fontsize=18 * FONT_SCALE)
plt.tight_layout()
plt.savefig(os.path.join(CONF_MAT_DIR, f"{MODEL_NAME}_relative_conf_matrix.png"))
plt.savefig(os.path.join(CONF_MAT_DIR, f"{MODEL_NAME}_relative_conf_matrix.eps"))
plt.clf()
plt.close()


###############################################################################
# Save Results TXT (raw + undersampled summaries)
###############################################################################

results_path = os.path.join(RESULT_TXT_DIR, f"{MODEL_NAME}_results.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("Full Classification Report (without undersampling):\n")
    f.write(full_report)
    f.write("\n============================\n")
    f.write("Averaged Classification Report (with undersampling):\n")
    f.write(avg_class_report_str)
    f.write("\n============================\n")
    f.write("Averaged Accuracy: {:.4f}\n".format(np.mean(acc_scores)))
    f.write("\nAveraged Confusion Matrix:\n")
    f.write(np.array2string(avg_conf_matrix, separator=", "))
    f.write("\nRelative Confusion Matrix:\n")
    f.write(np.array2string(relative_conf_matrix, separator=", "))

print(f"Results saved to {results_path}")


###############################################################################
# Feature Importance Analysis (CatBoost built-in)
###############################################################################

# Plain meaning:
# - CatBoost can compute feature_importance for each feature used.
# - We normalize it to “relative importance” so it sums to 1.
feature_importances = model.get_feature_importance()
relative_importances = feature_importances / feature_importances.sum()
feature_names = selected_features

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": relative_importances
}).sort_values(by="Importance", ascending=False)

output_dir = os.path.join(FEAT_IMP_DIR, f"{MODEL_NAME}_importance")
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df,
    palette=custom_palette
)
plt.title("Feature Importance Analysis (Relative)", fontsize=20 * FONT_SCALE)
plt.xlabel("Relative Importance", fontsize=18 * FONT_SCALE)
plt.ylabel("Feature", fontsize=18 * FONT_SCALE)
plt.xticks(fontsize=16 * FONT_SCALE)
plt.yticks(fontsize=16 * FONT_SCALE)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()

png_path = os.path.join(output_dir, "feature_importance.png")
eps_path = os.path.join(output_dir, "feature_importance.eps")
plt.savefig(png_path, dpi=300)
plt.savefig(eps_path, format="eps")
plt.close()
print(f"Feature importance plots saved to: {png_path} and {eps_path}")


###############################################################################
# Boxplots for Top Features by Predicted Class (kept style)
###############################################################################

# 1) Identify top features
# NOTE: your legacy code said "Top 9" but used head(4).
# We keep your exact behavior: head(4).
top_9_features = importance_df.head(4)["Feature"].values

# 2) Predict all test samples (hard labels)
y_pred_test = model.predict(X_test)

# 3) Define undersampling method based on predicted classes (kept)
def custom_predicted_under_sampler(X, y_pred, fraction=0.5, random_state=None):
    """
    Perform under-sampling by taking 'fraction' of each predicted class.

    Plain meaning:
    - We look at the model's predicted labels.
    - For each predicted label, keep only a fraction of those samples.
    - Combine both predicted classes back into a smaller dataset.

    This means:
    - The distribution is balanced with respect to *predictions*, not true labels.
    - It helps plot clean boxplots without one predicted group dominating.
    """
    rng = np.random.default_rng(seed=random_state)

    X_array = np.asarray(X)
    y_pred_array = np.asarray(y_pred)

    unique_classes = np.unique(y_pred_array)
    class_counts = {cls: np.sum(y_pred_array == cls) for cls in unique_classes}

    sampled_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_pred_array == cls)[0]
        num_to_pick = int(round(class_counts[cls] * fraction))
        sampled_indices.extend(rng.choice(cls_indices, size=num_to_pick, replace=False))

    rng.shuffle(sampled_indices)

    return X_array[sampled_indices], y_pred_array[sampled_indices]

# 4) Aggregate multiple runs (kept)
n_iterations = 100
sampling_ratio = 0.5
plot_df_aggregated = []

for i in range(n_iterations):
    print(f"Iteration {i+1}")

    # Undersample using predicted class distribution
    X_under_array, y_under_pred = custom_predicted_under_sampler(
        X_test, y_pred_test, fraction=sampling_ratio, random_state=i
    )

    # Convert back to DataFrame with original column names
    X_under_df = pd.DataFrame(X_under_array, columns=X_test.columns)

    # Build DataFrame with unscaled features + predicted class label
    iteration_df = X_under_df.copy()
    iteration_df["PredictedClass"] = y_under_pred
    iteration_df["PredictedClass"] = iteration_df["PredictedClass"].map({0: "No Bug", 1: "Bug"})
    iteration_df["Iteration"] = i

    plot_df_aggregated.append(iteration_df)

# 5) Combine all runs
plot_df_aggregated = pd.concat(plot_df_aggregated, ignore_index=True)

# 6) Adjust font sizes
plt.rcParams.update({"font.size": 16 * FONT_SCALE, "xtick.labelsize": 14 * FONT_SCALE, "ytick.labelsize": 14 * FONT_SCALE})

# 7) Generate and save vertical boxplots
for feat in top_9_features:
    plt.figure(figsize=(6, 10))
    sns.boxplot(
        data=plot_df_aggregated,
        x="PredictedClass",
        y=feat,
        palette=custom_palette,
        showfliers=False
    )
    plt.xlabel("Predicted Class", fontsize=24 * FONT_SCALE)
    plt.ylabel(feat, fontsize=24 * FONT_SCALE)
    plt.xticks(fontsize=22 * FONT_SCALE)
    plt.yticks(fontsize=22 * FONT_SCALE)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{feat}_distribution_by_predicted_class_aggregated_no.png")
    plot_path_eps = os.path.join(output_dir, f"{feat}_distribution_by_predicted_class_aggregated_no.eps")
    plt.savefig(plot_path, dpi=300)
    plt.savefig(plot_path_eps, dpi=300)
    plt.close()
    print(f"Boxplot for {feat} (aggregated) saved to: {plot_path}")
