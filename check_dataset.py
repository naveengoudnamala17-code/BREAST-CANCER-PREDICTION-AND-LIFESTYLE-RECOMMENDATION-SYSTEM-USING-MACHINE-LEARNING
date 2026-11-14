import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "breast-cancer-wisconsin.csv"
)


# ============================================================
# START
# ============================================================

print("\n==============================================")
print("BREAST CANCER MODEL COMPARISON")
print("==============================================")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded.")
print("Original rows:", len(df))


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# FEATURES
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
print("DATA CLEANING")
print("==============================================")


# Convert ? to missing values

df = df.replace("?", np.nan)


# Convert feature columns to numeric

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Convert class

df["Class"] = pd.to_numeric(
    df["Class"],
    errors="coerce"
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print(
    "Exact duplicate rows:",
    duplicates
)

df = df.drop_duplicates().reset_index(
    drop=True
)

print(
    "Rows after duplicate removal:",
    len(df)
)


# ============================================================
# CLASS MAPPING
# ============================================================

# 2 = Benign
# 4 = Malignant

df["Class"] = df["Class"].map({
    2: 0,
    4: 1
})


df = df.dropna(
    subset=["Class"]
)

df["Class"] = df["Class"].astype(int)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[feature_columns]

y = df["Class"]


print("\nClass distribution:")

print(
    y.value_counts()
)

print("\n0 = Benign")
print("1 = Malignant")


# ============================================================
# SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n==============================================")
print("DATASET SPLIT")
print("==============================================")


# 80% training
# 20% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 10% validation
# 10% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Test samples:",
    len(X_test)
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),


    "SVM": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                probability=True,
                random_state=42
            )
        )
    ]),


    "Random Forest": Pipeline([

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
    ]),


    "Gradient Boosting": Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ])
}


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

results = []


# ============================================================
# TRAIN AND COMPARE
# ============================================================

for name, model in models.items():

    print("\n==============================================")
    print("TRAINING:", name)
    print("==============================================")


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_predictions = model.predict(
        X_val
    )


    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )


    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )


    # Probability for malignant class

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )


    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )


    auc = roc_auc_score(
        y_test,
        test_probabilities
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_predictions
    )


    tn, fp, fn, tp = cm.ravel()


    specificity = tn / (
        tn + fp
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Test Accuracy    : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision        : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall           : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Specificity      : "
        f"{specificity * 100:.2f}%"
    )

    print(
        f"F1 Score         : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC          : "
        f"{auc:.4f}"
    )


    print("\nConfusion Matrix:")

    print(cm)


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "Specificity": specificity,

        "F1 Score": f1,

        "ROC-AUC": auc
    })


# ============================================================
# COMPARISON TABLE
# ============================================================

print("\n\n==============================================")
print("FINAL MODEL COMPARISON")
print("==============================================")


results_df = pd.DataFrame(
    results
)


# Convert values to percentages

display_df = results_df.copy()


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1 Score"
]:

    display_df[column] = (
        display_df[column] * 100
    ).round(2)


display_df["ROC-AUC"] = (
    display_df["ROC-AUC"]
    .round(4)
)


print(
    "\n"
)

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# FIND BEST MODELS
# ============================================================

best_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax()
]


best_auc = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]


best_f1 = results_df.loc[
    results_df["F1 Score"].idxmax()
]


print("\n==============================================")
print("BEST MODELS")
print("==============================================")


print(
    "\nBest Accuracy:"
)

print(
    best_accuracy["Model"],
    f"({best_accuracy['Accuracy'] * 100:.2f}%)"
)


print(
    "\nBest ROC-AUC:"
)

print(
    best_auc["Model"],
    f"({best_auc['ROC-AUC']:.4f})"
)


print(
    "\nBest F1 Score:"
)

print(
    best_f1["Model"],
    f"({best_f1['F1 Score'] * 100:.2f}%)"
)


# ============================================================
# SAVE COMPARISON
# ============================================================

RESULT_PATH = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)


display_df.to_csv(
    RESULT_PATH,
    index=False
)


print(
    "\nComparison saved to:"
)

print(
    RESULT_PATH
)


print("\n==============================================")
print("MODEL COMPARISON COMPLETED")
print("==============================================")