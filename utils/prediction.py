import os
import joblib
import pandas as pd


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "breast_cancer_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )


saved_model = joblib.load(MODEL_PATH)


# The training script saved a dictionary containing
# the trained model and feature names.

model = saved_model["model"]

FEATURES = saved_model["features"]


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAPPING = {
    0: "Benign",
    1: "Malignant"
}


# ============================================================
# FNA PREDICTION
# ============================================================

def predict_fna(data):

    # --------------------------------------------------------
    # Convert dictionary to DataFrame
    # --------------------------------------------------------

    if isinstance(data, dict):

        data = pd.DataFrame([data])


    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing FNA features: "
            + ", ".join(missing_features)
        )


    # --------------------------------------------------------
    # Select features in training order
    # --------------------------------------------------------

    X = data[FEATURES].copy()


    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for feature in FEATURES:

        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(X)[0]


    # --------------------------------------------------------
    # Prediction probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)[0]

    benign_probability = float(
        probabilities[0]
    )

    malignant_probability = float(
        probabilities[1]
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = CLASS_MAPPING.get(
        int(prediction),
        "Unknown"
    )


    return {

        "prediction": result,

        "prediction_code": int(
            prediction
        ),

        "benign_probability": round(
            benign_probability * 100,
            2
        ),

        "malignant_probability": round(
            malignant_probability * 100,
            2
        )
    }