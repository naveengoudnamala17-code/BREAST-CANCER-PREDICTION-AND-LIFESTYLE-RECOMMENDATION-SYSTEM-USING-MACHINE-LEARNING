# BREAST CANCER PREDICTION AND LIFESTYLE RECOMMENDATION SYSTEM USING MACHINE LEARNING

An MCA final-year project that uses Machine Learning to predict breast cancer and provide general health and lifestyle recommendations.

## About the Project

This project aims to classify breast tumours as Benign or Malignant using the Wisconsin Breast Cancer Dataset (WBCD).

A Random Forest Classifier is trained using nine diagnostic features. The trained model is integrated into a Flask-based web application that allows users to enter patient details and obtain prediction results.

The application also provides prediction confidence, an estimated risk level, general lifestyle recommendations, prediction history, and PDF reports.

## Key Features

- Breast cancer prediction (Benign or Malignant)
- Random Forest machine learning model
- Prediction confidence
- Estimated risk level
- General health and lifestyle recommendations
- Patient prediction history
- Dashboard with prediction statistics
- PDF report generation

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- SQLite
- HTML
- CSS
- JavaScript
- Joblib
- ReportLab

## Dataset

The project uses the Wisconsin Breast Cancer Dataset (WBCD).

The model uses the following nine diagnostic features:

1. Clump Thickness
2. Uniformity of Cell Size
3. Uniformity of Cell Shape
4. Marginal Adhesion
5. Single Epithelial Cell Size
6. Bare Nuclei
7. Bland Chromatin
8. Normal Nucleoli
9. Mitoses

## Machine Learning Model

The Random Forest Classifier is used for binary classification.

The model predicts two classes:

- Benign
- Malignant

## Model Performance

The Random Forest model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| Validation Accuracy | 95.65% |
| Test Accuracy | 97.14% |
| Precision | 95.83% |
| Recall | 95.83% |
| Specificity | 97.83% |
| F1 Score | 95.83% |
| ROC-AUC | 0.9955 |

## Project Workflow

1. Dataset loading
2. Data preprocessing
3. Feature selection
4. Training the Random Forest model
5. Model evaluation
6. Model integration with Flask
7. Prediction through the web application
8. Displaying results and recommendations

## Installation and Execution

1. Clone the repository:

   `git clone https://github.com/naveengoudnamala17-code/BREAST-CANCER-PREDICTION-AND-LIFESTYLE-RECOMMENDATION-SYSTEM-USING-MACHINE-LEARNING.git`

2. Navigate to the project folder.

3. Install the required dependencies:

   `pip install -r requirements.txt`

4. Run the application:

   `python app.py`

5. Open the local application address displayed in the terminal in your web browser.

## Project Purpose and Disclaimer

This application is developed for academic and educational purposes. It is intended as a preliminary machine learning decision-support project.

The predictions, estimated risk levels, and recommendations are not professional medical diagnoses and should not replace consultation with qualified healthcare professionals.
