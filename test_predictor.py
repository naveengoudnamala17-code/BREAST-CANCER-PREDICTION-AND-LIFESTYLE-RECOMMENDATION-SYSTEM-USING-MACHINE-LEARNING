from utils.prediction import predict_fna

# Sample FNA input
sample = {
    "Clump Thickness": 5,
    "Uniformity of Cell Size": 1,
    "Uniformity of Cell Shape": 1,
    "Marginal Adhesion": 1,
    "Single Epithelial Cell Size": 2,
    "Bare Nuclei": 1,
    "Bland Chromatin": 3,
    "Normal Nucleoli": 1,
    "Mitoses": 1
}


# Run prediction
result = predict_fna(sample)


print("\n================================")
print("FNA PREDICTION TEST")
print("================================")

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Benign Probability:",
    result["benign_probability"],
    "%"
)

print(
    "Malignant Probability:",
    result["malignant_probability"],
    "%"
)

print("================================")