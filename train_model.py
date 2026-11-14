import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "breast-cancer-wisconsin.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "breast_cancer_model.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# START
# ============================================================

print("\n==============================================")
print("WISCONSIN FNA BREAST CANCER MODEL TRAINING")
print("==============================================")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")
print("Dataset:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Original rows:", len(df))
print("Original columns:", len(df.columns))


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nCSV Columns:")

for column in df.columns:
    print("-", column)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "ID",
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
    "Class"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print("-", column)

    print("\nColumns found:")
    print(list(df.columns))

    raise SystemExit


# ============================================================
# MODEL FEATURES
# ============================================================

feature_columns = [
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses"
]


# ============================================================
# CLEAN DATA
# ============================================================

print("\n==============================================")
print("CLEANING DATA")
print("==============================================")


# Convert '?' to missing values

df = df.replace("?", pd.NA)


# Convert feature columns to numeric

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Convert class to numeric

df["Class"] = pd.to_numeric(
    df["Class"],
    errors="coerce"
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing values before preprocessing:")

print(
    df[feature_columns].isnull().sum()
)


missing_total = df[feature_columns].isnull().sum().sum()

print(
    "\nTotal missing feature values:",
    missing_total
)


# ============================================================
# REMOVE EXACT DUPLICATE ROWS
# ============================================================

print("\n==============================================")
print("DUPLICATE CHECK")
print("==============================================")

exact_duplicates = df.duplicated().sum()

print(
    "Exact duplicate rows:",
    exact_duplicates
)


if exact_duplicates > 0:

    df = df.drop_duplicates().reset_index(
        drop=True
    )


print(
    "Rows after removing exact duplicates:",
    len(df)
)


# ============================================================
# DO NOT REMOVE DUPLICATE IDs
# ============================================================

duplicate_ids = df["ID"].duplicated().sum()

print(
    "\nDuplicate IDs:",
    duplicate_ids
)

print(
    "Duplicate IDs are retained because ID is only an identifier."
)


# ============================================================
# CLASS MAPPING
# ============================================================

print("\n==============================================")
print("CLASS MAPPING")
print("==============================================")

print("Original classes:")
print(df["Class"].value_counts())


# 2 = Benign
# 4 = Malignant

df["Class"] = df["Class"].map({
    2: 0,
    4: 1
})


# Remove unexpected class values

df = df.dropna(
    subset=["Class"]
)

df["Class"] = df["Class"].astype(int)


print("\nFinal class mapping:")
print("0 = Benign")
print("1 = Malignant")


print("\nFinal class distribution:")
print(df["Class"].value_counts())


# ============================================================
# FEATURES AND TARGET
# ============================================================

print("\n==============================================")
print("PREPARING FEATURES")
print("==============================================")


# ID is deliberately excluded.

X = df[feature_columns]

y = df["Class"]


print("\nNumber of features:", len(feature_columns))

print("\nFeatures:")

for feature in feature_columns:
    print("-", feature)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n==============================================")
print("DATASET SPLIT")
print("==============================================")


# First split:
# 80% training
# 20% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Second split:
# 10% validation
# 10% testing

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nTraining samples   :", len(X_train))
print("Validation samples :", len(X_val))
print("Test samples       :", len(X_test))


print("\nTraining distribution:")
print(y_train.value_counts())


print("\nValidation distribution:")
print(y_val.value_counts())


print("\nTest distribution:")
print(y_test.value_counts())


# ============================================================
# RANDOM FOREST PIPELINE
# ============================================================

print("\n==============================================")
print("CREATING RANDOM FOREST")
print("==============================================")


model = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )
    )
])


print("Random Forest created.")
print("Number of trees: 300")


# ============================================================
# TRAIN
# ============================================================

print("\n==============================================")
print("TRAINING MODEL")
print("==============================================")


model.fit(
    X_train,
    y_train
)


print("\nTraining completed successfully.")


# ============================================================
# VALIDATION
# ============================================================

print("\n==============================================")
print("VALIDATION RESULTS")
print("==============================================")


val_predictions = model.predict(
    X_val
)


val_probabilities = model.predict_proba(
    X_val
)[:, 1]


val_accuracy = accuracy_score(
    y_val,
    val_predictions
)


val_precision = precision_score(
    y_val,
    val_predictions,
    zero_division=0
)


val_recall = recall_score(
    y_val,
    val_predictions,
    zero_division=0
)


val_f1 = f1_score(
    y_val,
    val_predictions,
    zero_division=0
)


val_auc = roc_auc_score(
    y_val,
    val_probabilities
)


print(
    f"Validation Accuracy : {val_accuracy * 100:.2f}%"
)

print(
    f"Validation Precision: {val_precision * 100:.2f}%"
)

print(
    f"Validation Recall   : {val_recall * 100:.2f}%"
)

print(
    f"Validation F1       : {val_f1 * 100:.2f}%"
)

print(
    f"Validation ROC-AUC  : {val_auc:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n==============================================")
print("FINAL TEST RESULTS")
print("==============================================")


test_predictions = model.predict(
    X_test
)


test_probabilities = model.predict_proba(
    X_test
)[:, 1]


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


# ============================================================
# SPECIFICITY
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)


tn, fp, fn, tp = cm.ravel()


specificity = tn / (tn + fp)


# ============================================================
# PRINT METRICS
# ============================================================

print(
    f"\nTest Accuracy    : {test_accuracy * 100:.2f}%"
)

print(
    f"Test Precision   : {test_precision * 100:.2f}%"
)

print(
    f"Test Recall      : {test_recall * 100:.2f}%"
)

print(
    f"Test Specificity : {specificity * 100:.2f}%"
)

print(
    f"Test F1 Score    : {test_f1 * 100:.2f}%"
)

print(
    f"Test ROC-AUC     : {test_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================================")
print("CLASSIFICATION REPORT")
print("==============================================")


print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Benign",
            "Malignant"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================")


print(cm)


print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n==============================================")
print("SAVING MODEL")
print("==============================================")


joblib.dump(
    {
        "model": model,
        "features": feature_columns,
        "class_mapping": {
            0: "Benign",
            1: "Malignant"
        }
    },
    MODEL_PATH
)


print("\nModel saved successfully.")

print("Model path:")
print(MODEL_PATH)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

INFO_PATH = os.path.join(
    MODEL_DIR,
    "model_info.txt"
)


with open(INFO_PATH, "w") as file:

    file.write(
        "WISCONSIN FNA BREAST CANCER MODEL\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        "Model: Random Forest\n"
    )

    file.write(
        "Number of Trees: 300\n\n"
    )

    file.write(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Test Precision: "
        f"{test_precision * 100:.2f}%\n"
    )

    file.write(
        f"Test Recall: "
        f"{test_recall * 100:.2f}%\n"
    )

    file.write(
        f"Test Specificity: "
        f"{specificity * 100:.2f}%\n"
    )

    file.write(
        f"Test F1 Score: "
        f"{test_f1 * 100:.2f}%\n"
    )

    file.write(
        f"Test ROC-AUC: "
        f"{test_auc:.4f}\n\n"
    )

    file.write(
        "Class Mapping:\n"
    )

    file.write(
        "0 = Benign\n"
    )

    file.write(
        "1 = Malignant\n\n"
    )

    file.write(
        "Features:\n"
    )

    for feature in feature_columns:

        file.write(
            f"- {feature}\n"
        )


print("\nModel information saved to:")
print(INFO_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("TRAINING COMPLETED")
print("==============================================")


print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print(
    f"Test Recall: "
    f"{test_recall * 100:.2f}%"
)


print(
    f"Test Specificity: "
    f"{specificity * 100:.2f}%"
)


print(
    f"Test F1 Score: "
    f"{test_f1 * 100:.2f}%"
)


print(
    f"Test ROC-AUC: "
    f"{test_auc:.4f}"
)


print("\nClass mapping:")
print("0 = Benign")
print("1 = Malignant")


print("\n==============================================")