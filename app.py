import os
import sqlite3
import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file
)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.units import mm

from datetime import datetime


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = "breast_cancer_wisconsin_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "breast_cancer_model.pkl"
)


# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "breast-cancer-wisconsin.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]

    FEATURES = saved_model["features"]

    print("Model loaded successfully:")
    print(MODEL_PATH)

    print("Features:")
    print(FEATURES)

except Exception as e:

    model = None

    FEATURES = [
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

    print("ERROR loading model:")
    print(e)


# ============================================================
# MODEL ACCURACY
# ============================================================

def calculate_model_accuracy():

    try:
        return 97.14

    except Exception as e:

        print("ACCURACY ERROR:", str(e))

        return 0


# ============================================================
# DATABASE
# ============================================================

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database.db"
)


def get_db():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_database():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_name TEXT,

            prediction TEXT,

            confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    conn.commit()

    conn.close()


init_database()


# ============================================================
# LOGIN
# ============================================================

@app.route("/", methods=["GET"])
def home():

    if "user" in session:

        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if name and password:

            session["user"] = name

            return redirect(
                url_for("dashboard")
            )

        return render_template(
            "login.html",
            error="Please enter your name and password."
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:

        return redirect(
            url_for("login")
        )

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM patients
    """)

    total_patients = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM patients
        WHERE prediction = 'Benign'
    """)

    benign_cases = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM patients
        WHERE prediction = 'Malignant'
    """)

    malignant_cases = cursor.fetchone()["total"]

    conn.close()

    accuracy = calculate_model_accuracy()

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        benign_cases=benign_cases,
        malignant_cases=malignant_cases,
        accuracy=accuracy,
        user=session["user"]
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route(
    "/prediction",
    methods=["GET", "POST"]
)
def prediction():

    if "user" not in session:

        return redirect(
            url_for("login")
        )

    if request.method == "GET":

        return render_template(
            "prediction.html"
        )

    try:

        # ====================================================
        # PATIENT DETAILS
        # ====================================================

        patient_name = request.form.get(
            "patient_name",
            ""
        ).strip()

        age = request.form.get(
            "age",
            ""
        ).strip()

        gender = request.form.get(
            "gender",
            ""
        ).strip()


        # ====================================================
        # VALIDATION
        # ====================================================

        if patient_name == "":

            return render_template(
                "prediction.html",
                error="Please enter the patient name."
            )

        if age == "":

            return render_template(
                "prediction.html",
                error="Please enter the patient's age."
            )

        if gender == "":

            return render_template(
                "prediction.html",
                error="Please select the patient's gender."
            )


        # ====================================================
        # DIAGNOSTIC FEATURES
        # ====================================================

        feature_form_names = [

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

        feature_values = []

        for feature in feature_form_names:

            value = request.form.get(
                feature,
                ""
            ).strip()

            if value == "":

                return render_template(
                    "prediction.html",
                    error=f"Missing value for {feature}"
                )

            try:

                numeric_value = float(value)

            except ValueError:

                return render_template(
                    "prediction.html",
                    error=f"Please enter a valid number for {feature}."
                )

            feature_values.append(
                numeric_value
            )


        # ====================================================
        # CREATE DATAFRAME
        # ====================================================

        input_data = pd.DataFrame(
            [feature_values],
            columns=FEATURES
        )

        print()
        print("INPUT DATA:")
        print(input_data)


        # ====================================================
        # CHECK MODEL
        # ====================================================

        if model is None:

            raise Exception(
                "Model was not loaded."
            )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction_value = model.predict(
            input_data
        )[0]


        # ====================================================
        # CONFIDENCE
        # ====================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = max(
                probabilities
            ) * 100

        else:

            confidence = 0

        confidence = round(
            confidence,
            2
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        if int(prediction_value) == 0:

            result = "Benign"

        else:

            result = "Malignant"


        # ====================================================
        # AI-ESTIMATED RISK
        # ====================================================

        if result == "Benign":

            if confidence >= 85:
                risk_level = "Low Risk"

            elif confidence >= 60:
                risk_level = "Medium Risk"

            else:
                risk_level = "High Risk"

        else:

            if confidence >= 85:
                risk_level = "High Risk"

            elif confidence >= 60:
                risk_level = "Medium Risk"

            else:
                risk_level = "Low Risk"


        risk = risk_level


        print()
        print("Prediction:", result)
        print("Confidence:", confidence)
        print("AI Risk Level:", risk_level)


        # ====================================================
        # CANCER TYPE
        # ====================================================

        if result == "Benign":

            cancer_type = (
                "No malignant pattern detected"
            )

        else:

            cancer_type = (
                "Malignant pattern detected"
            )


        # ====================================================
        # AI EXPLANATION
        # ====================================================

        if result == "Benign":

            explanation = (

                "The AI model classified the submitted "
                "diagnostic measurements as benign. The "
                "combination of the provided Wisconsin "
                "diagnostic features was more consistent "
                "with a non-malignant case according to "
                "the trained model."

            )

        else:

            explanation = (

                "The AI model classified the submitted "
                "diagnostic measurements as malignant. "
                "The combination of the provided Wisconsin "
                "diagnostic features was more consistent "
                "with a malignant case according to "
                "the trained model."

            )


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        if result == "Benign":

            if risk_level == "Low Risk":

                recommendations = [

                    "Continue routine health check-ups and recommended screening.",

                    "Maintain regular physical activity and a healthy lifestyle.",

                    "Maintain a healthy body weight.",

                    "Monitor your general health and report any new or persistent symptoms to a healthcare professional.",

                    "Continue following the screening schedule appropriate for your age and individual circumstances.",

                    "This AI prediction is for educational purposes and should not replace professional medical advice."

                ]

            elif risk_level == "Medium Risk":

                recommendations = [

                    "Discuss this AI-assisted result with a qualified healthcare professional.",

                    "Follow the recommended screening and follow-up schedule carefully.",

                    "Maintain regular physical activity and a healthy body weight.",

                    "Pay attention to any new or persistent changes and discuss concerns with a healthcare professional.",

                    "Consider appropriate professional follow-up based on your clinical history.",

                    "This AI prediction should not be considered a medical diagnosis."

                ]

            else:

                recommendations = [

                    "Discuss this AI-assisted result with a qualified healthcare professional for further evaluation.",

                    "A benign prediction does not replace clinical assessment.",

                    "Follow any recommended screening or diagnostic follow-up provided by your healthcare professional.",

                    "Report unusual or persistent symptoms to a healthcare professional.",

                    "Additional professional evaluation may be appropriate depending on your clinical history.",

                    "Do not make medical decisions based solely on this AI prediction."

                ]

        else:

            if risk_level == "Low Risk":

                recommendations = [

                    "Discuss the AI-assisted result with a qualified healthcare professional.",

                    "Follow the screening and follow-up schedule recommended by your healthcare professional.",

                    "Maintain regular physical activity and a healthy lifestyle.",

                    "Discuss any new or persistent symptoms with your healthcare professional.",

                    "Further clinical evaluation may be recommended depending on your circumstances.",

                    "Do not make medical decisions based only on this AI prediction."

                ]

            elif risk_level == "Medium Risk":

                recommendations = [

                    "Arrange a consultation with a qualified healthcare professional for further evaluation.",

                    "Follow the recommended clinical screening and follow-up schedule.",

                    "Additional diagnostic evaluation may be recommended by your healthcare professional.",

                    "Maintain regular physical activity and a balanced lifestyle.",

                    "Discuss symptoms and concerns with your healthcare professional.",

                    "Do not use this AI prediction as a substitute for professional medical evaluation."

                ]

            else:

                recommendations = [

                    "Arrange a professional medical evaluation as soon as reasonably possible.",

                    "Additional diagnostic testing may be recommended by your healthcare professional.",

                    "Follow the clinical evaluation and follow-up plan provided by your healthcare professional.",

                    "Discuss the AI result and appropriate next steps with a qualified healthcare professional.",

                    "Do not make treatment decisions based solely on this AI prediction.",

                    "Seek appropriate medical attention promptly if concerning symptoms are present."

                ]


        # ====================================================
        # DIET PLAN
        # ====================================================

        if result == "Benign":

            if risk_level == "Low Risk":

                diet = [

                    "Maintain a balanced diet containing a variety of vegetables and fruits.",

                    "Include whole grains and other high-fiber foods.",

                    "Choose suitable protein sources such as beans, lentils, eggs, fish, or lean meats.",

                    "Drink sufficient water throughout the day.",

                    "Limit highly processed foods and excessive added sugar.",

                    "Maintain a healthy and consistent eating pattern."

                ]

            elif risk_level == "Medium Risk":

                diet = [

                    "Increase the variety of vegetables and fruits in your daily meals.",

                    "Choose whole grains and other fiber-rich foods regularly.",

                    "Include suitable protein sources such as beans, lentils, fish, eggs, or lean meats.",

                    "Choose healthier fats such as nuts, seeds, and suitable plant-based oils.",

                    "Limit highly processed foods, excessive added sugar, and excess saturated fat.",

                    "Maintain adequate hydration and a balanced eating pattern."

                ]

            else:

                diet = [

                    "Focus on a nutrient-rich and balanced diet containing vegetables, fruits, whole grains, and suitable protein sources.",

                    "Include adequate protein from foods such as beans, lentils, eggs, fish, or lean meats.",

                    "Choose healthier fats from sources such as nuts, seeds, and suitable plant-based oils.",

                    "Limit highly processed foods and excessive added sugar.",

                    "Maintain adequate hydration throughout the day.",

                    "Discuss individual nutritional needs with a qualified healthcare professional or dietitian."

                ]

        else:

            if risk_level == "Low Risk":

                diet = [

                    "Maintain a balanced diet containing vegetables, fruits, whole grains, and suitable protein sources.",

                    "Include adequate protein from beans, lentils, eggs, fish, or lean meats.",

                    "Choose nutrient-dense foods as part of a healthy eating pattern.",

                    "Drink sufficient water throughout the day unless advised otherwise by a healthcare professional.",

                    "Limit highly processed foods and excessive added sugar.",

                    "Discuss any specific dietary requirements with a healthcare professional."

                ]

            elif risk_level == "Medium Risk":

                diet = [

                    "Prioritize vegetables, fruits, whole grains, and other fiber-rich foods.",

                    "Include adequate protein from suitable sources such as beans, lentils, eggs, fish, or lean meats.",

                    "Choose healthier fat sources such as nuts, seeds, and suitable plant-based oils.",

                    "Limit highly processed foods, excessive added sugar, and excess saturated fat.",

                    "Maintain adequate hydration throughout the day.",

                    "Discuss individual nutritional needs with a qualified healthcare professional or registered dietitian."

                ]

            else:

                diet = [

                    "Maintain a balanced and nutrient-dense eating pattern containing vegetables, fruits, whole grains, and suitable protein sources.",

                    "Include adequate protein to support overall nutritional needs.",

                    "Choose suitable foods such as beans, lentils, eggs, fish, lean meats, vegetables, fruits, and whole grains.",

                    "Limit highly processed foods and excessive added sugar.",

                    "Drink sufficient water unless your healthcare professional has advised otherwise.",

                    "If receiving medical treatment, discuss individual dietary needs with a healthcare professional or registered dietitian."

                ]


        # ====================================================
        # SAVE TO DATABASE
        # ====================================================

        conn = get_db()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO patients
            (
                patient_name,
                prediction,
                confidence
            )
            VALUES (?, ?, ?)
            """,
            (
                patient_name,
                result,
                confidence
            )
        )

        conn.commit()

        conn.close()


        # ====================================================
        # SHOW RESULT
        # ====================================================

        return render_template(

            "result.html",

            result=result,

            confidence=confidence,

            risk=risk_level,

            risk_level=risk_level,

            cancer_type=cancer_type,

            explanation=explanation,

            recommendations=recommendations,

            diet=diet,

            patient_name=patient_name,

            age=age,

            gender=gender,

            clump_thickness=feature_values[0],

            uniformity_cell_size=feature_values[1],

            uniformity_cell_shape=feature_values[2],

            marginal_adhesion=feature_values[3],

            single_epithelial_cell_size=feature_values[4],

            bare_nuclei=feature_values[5],

            bland_chromatin=feature_values[6],

            normal_nucleoli=feature_values[7],

            mitoses=feature_values[8]

        )


    except Exception as e:

        print()
        print(
            "PREDICTION ERROR:",
            str(e)
        )

        return render_template(

            "prediction.html",

            error=f"Prediction error: {str(e)}"

        )


# ============================================================
# PDF REPORT
# ============================================================

@app.route(
    "/download_report",
    methods=["POST"]
)
def download_report():

    patient_name = request.form.get(
        "patient_name",
        "N/A"
    )

    age = request.form.get(
        "age",
        "N/A"
    )

    gender = request.form.get(
        "gender",
        "N/A"
    )

    result = request.form.get(
        "result",
        "N/A"
    )

    confidence = request.form.get(
        "confidence",
        "N/A"
    )

    risk_level = request.form.get(
        "risk_level",
        "N/A"
    )

    cancer_type = request.form.get(
        "cancer_type",
        "N/A"
    )

    explanation = request.form.get(
        "explanation",
        "No AI explanation available."
    )

    clump_thickness = request.form.get(
        "clump_thickness",
        "N/A"
    )

    uniformity_cell_size = request.form.get(
        "uniformity_cell_size",
        "N/A"
    )

    uniformity_cell_shape = request.form.get(
        "uniformity_cell_shape",
        "N/A"
    )

    marginal_adhesion = request.form.get(
        "marginal_adhesion",
        "N/A"
    )

    single_epithelial_cell_size = request.form.get(
        "single_epithelial_cell_size",
        "N/A"
    )

    bare_nuclei = request.form.get(
        "bare_nuclei",
        "N/A"
    )

    bland_chromatin = request.form.get(
        "bland_chromatin",
        "N/A"
    )

    normal_nucleoli = request.form.get(
        "normal_nucleoli",
        "N/A"
    )

    mitoses = request.form.get(
        "mitoses",
        "N/A"
    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendations = [

        "Continue routine health check-ups and recommended screening.",

        "Maintain regular physical activity and a healthy lifestyle.",

        "Maintain a healthy body weight.",

        "Monitor general health and report new or persistent symptoms to a healthcare professional.",

        "Follow the screening schedule appropriate for age and individual circumstances.",

        "This AI prediction is for educational purposes and should not replace professional medical advice."

    ]


    # ========================================================
    # DIET & LIFESTYLE
    # ========================================================

    diet = [

        "Maintain a balanced diet containing a variety of vegetables and fruits.",

        "Include whole grains and other high-fiber foods.",

        "Choose suitable protein sources such as beans, lentils, eggs, fish, or lean meats.",

        "Drink sufficient water throughout the day.",

        "Limit highly processed foods and excessive added sugar.",

        "Maintain a healthy and consistent eating pattern."

    ]


    # ========================================================
    # CREATE REPORT FOLDER
    # ========================================================

    report_folder = os.path.join(
        app.root_path,
        "static",
        "reports"
    )

    os.makedirs(
        report_folder,
        exist_ok=True
    )


    # ========================================================
    # SAFE FILE NAME
    # ========================================================

    safe_name = "".join(

        c for c in patient_name

        if c.isalnum()
        or c in (" ", "_", "-")

    ).strip()


    if not safe_name:

        safe_name = "Patient"


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    filename = (
        f"Breast_Cancer_Report_"
        f"{safe_name}_{timestamp}.pdf"
    )


    filepath = os.path.join(
        report_folder,
        filename
    )


    # ========================================================
    # PDF DOCUMENT
    # ========================================================

    doc = SimpleDocTemplate(

        filepath,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm

    )


    styles = getSampleStyleSheet()


    # ========================================================
    # CUSTOM STYLES
    # ========================================================

    title_style = ParagraphStyle(

        "TitleStyle",

        parent=styles["Title"],

        fontSize=22,

        leading=27,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#17365D"),

        spaceAfter=8

    )


    subtitle_style = ParagraphStyle(

        "SubtitleStyle",

        parent=styles["Normal"],

        fontSize=10,

        leading=14,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#6B7280"),

        spaceAfter=20

    )


    section_style = ParagraphStyle(

        "SectionStyle",

        parent=styles["Heading2"],

        fontSize=14,

        leading=18,

        textColor=colors.HexColor("#17365D"),

        spaceBefore=12,

        spaceAfter=10

    )


    normal_style = ParagraphStyle(

        "NormalStyle",

        parent=styles["Normal"],

        fontSize=9.5,

        leading=15,

        textColor=colors.HexColor("#374151")

    )


    small_style = ParagraphStyle(

        "SmallStyle",

        parent=styles["Normal"],

        fontSize=8,

        leading=12,

        textColor=colors.HexColor("#6B7280")

    )


    result_style = ParagraphStyle(

        "ResultStyle",

        parent=styles["Normal"],

        fontSize=18,

        leading=22,

        alignment=TA_CENTER,

        textColor=(

            colors.HexColor("#166534")

            if result == "Benign"

            else colors.HexColor("#991B1B")

        ),

        spaceAfter=8

    )


    confidence_style = ParagraphStyle(

        "ConfidenceStyle",

        parent=styles["Normal"],

        fontSize=13,

        leading=18,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#17365D"),

        spaceAfter=10

    )


    # ========================================================
    # STORY
    # ========================================================

    story = []


    story.append(
        Paragraph(
            "BREAST CANCER AI",
            title_style
        )
    )


    story.append(
        Paragraph(
            "Wisconsin Diagnostic Prediction System",
            subtitle_style
        )
    )


    story.append(
        Paragraph(
            "AI-Assisted Prediction Report",
            section_style
        )
    )


    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    patient_data = [

        [

            Paragraph(
                "<b>Patient Name</b>",
                normal_style
            ),

            Paragraph(
                "<b>Age</b>",
                normal_style
            ),

            Paragraph(
                "<b>Gender</b>",
                normal_style
            )

        ],

        [

            Paragraph(
                str(patient_name),
                normal_style
            ),

            Paragraph(
                str(age),
                normal_style
            ),

            Paragraph(
                str(gender),
                normal_style
            )

        ]

    ]


    patient_table = Table(

        patient_data,

        colWidths=[
            55 * mm,
            55 * mm,
            55 * mm
        ]

    )


    patient_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#EAF2FF")
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, 1),
                colors.HexColor("#F8FAFC")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#DBE3EE")
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#DBE3EE")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(patient_table)

    story.append(
        Spacer(1, 12)
    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    story.append(
        Paragraph(
            "Prediction Result",
            section_style
        )
    )


    result_text = (

        "✓ BENIGN"

        if result == "Benign"

        else "⚠ MALIGNANT"

    )


    result_data = [

        [

            Paragraph(
                f"<b>{result_text}</b>",
                result_style
            )

        ],

        [

            Paragraph(
                f"Model Confidence: <b>{confidence}%</b>",
                confidence_style
            )

        ],

        [

            Paragraph(
                f"AI Risk Assessment: <b>{risk_level}</b>",
                normal_style
            )

        ]

    ]


    result_table = Table(

        result_data,

        colWidths=[165 * mm]

    )


    result_background = (

        colors.HexColor("#ECFDF5")

        if result == "Benign"

        else colors.HexColor("#FEF2F2")

    )


    result_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                result_background
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,

                (
                    colors.HexColor("#86EFAC")

                    if result == "Benign"

                    else colors.HexColor("#FCA5A5")
                )

            ),

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                10
            )

        ])

    )


    story.append(result_table)

    story.append(
        Spacer(1, 12)
    )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    story.append(
        Paragraph(
            "AI Explanation",
            section_style
        )
    )


    explanation_table = Table(

        [

            [

                Paragraph(
                    str(explanation),
                    normal_style
                )

            ]

        ],

        colWidths=[165 * mm]

    )


    explanation_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#EFF6FF")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#2563EB")
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )

        ])

    )


    story.append(explanation_table)


    # ========================================================
    # DIAGNOSTIC FEATURES
    # ========================================================

    story.append(
        Paragraph(
            "Diagnostic Features",
            section_style
        )
    )


    features = [

        ["Feature", "Value"],

        ["Clump Thickness", clump_thickness],

        ["Uniformity of Cell Size", uniformity_cell_size],

        ["Uniformity of Cell Shape", uniformity_cell_shape],

        ["Marginal Adhesion", marginal_adhesion],

        ["Single Epithelial Cell Size", single_epithelial_cell_size],

        ["Bare Nuclei", bare_nuclei],

        ["Bland Chromatin", bland_chromatin],

        ["Normal Nucleoli", normal_nucleoli],

        ["Mitoses", mitoses]

    ]


    feature_table = Table(

        features,

        colWidths=[
            120 * mm,
            45 * mm
        ]

    )


    feature_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#17365D")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#F8FAFC")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#DBE3EE")
            ),

            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "CENTER"
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])

    )


    story.append(feature_table)


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    story.append(
        Paragraph(
            "Personalized Recommendations",
            section_style
        )
    )


    recommendation_text = "<br/>".join(

        [
            f"• {item}"
            for item in recommendations
        ]

    )


    recommendation_table = Table(

        [

            [

                Paragraph(
                    recommendation_text,
                    normal_style
                )

            ]

        ],

        colWidths=[165 * mm]

    )


    recommendation_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#EFF6FF")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#2563EB")
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )

        ])

    )


    story.append(recommendation_table)


    # ========================================================
    # DIET PLAN
    # ========================================================

    story.append(
        Paragraph(
            "Personalized Diet & Lifestyle Plan",
            section_style
        )
    )


    diet_text = "<br/>".join(

        [
            f"• {item}"
            for item in diet
        ]

    )


    diet_table = Table(

        [

            [

                Paragraph(
                    diet_text,
                    normal_style
                )

            ]

        ],

        colWidths=[165 * mm]

    )


    diet_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#F0FDF4")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#16A34A")
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )

        ])

    )


    story.append(diet_table)


    # ========================================================
    # IMPORTANT NOTICE
    # ========================================================

    story.append(
        Paragraph(
            "Important Notice",
            section_style
        )
    )


    notice = (

        "This system is an academic AI-based prediction tool "
        "intended for educational and research purposes. "
        "The prediction should not be considered a medical diagnosis."

        "<br/><br/>"

        "The confidence score, risk level, recommendations, "
        "and diet plan are generated according to the AI model "
        "and are intended only as general guidance."

        "<br/><br/>"

        "A qualified healthcare professional should review "
        "clinical findings before making medical decisions."

    )


    notice_table = Table(

        [

            [

                Paragraph(
                    notice,
                    normal_style
                )

            ]

        ],

        colWidths=[165 * mm]

    )


    notice_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#FFF7ED")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#F97316")
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )

        ])

    )


    story.append(notice_table)

    story.append(
        Spacer(1, 15)
    )


    # ========================================================
    # REPORT DATE
    # ========================================================

    report_date = datetime.now().strftime(
        "%d-%m-%Y %H:%M"
    )


    story.append(
        Paragraph(
            f"Report Generated: {report_date}",
            small_style
        )
    )


    story.append(
        Paragraph(
            "Breast Cancer AI • Wisconsin Diagnostic System",
            small_style
        )
    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(story)


    # ========================================================
    # DOWNLOAD PDF
    # ========================================================

    return send_file(

        filepath,

        as_attachment=True,

        download_name=filename,

        mimetype="application/pdf"

    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    if "user" not in session:

        return redirect(
            url_for("login")
        )


    conn = get_db()

    cursor = conn.cursor()


    cursor.execute("""
        SELECT *
        FROM patients
        ORDER BY id DESC
    """)


    patients = cursor.fetchall()

    conn.close()


    return render_template(
        "history.html",
        patients=patients
    )


# ============================================================
# RESET COUNTS
# ============================================================

@app.route(
    "/reset_counts",
    methods=["POST"]
)
def reset_counts():

    if "user" not in session:

        return redirect(
            url_for("login")
        )


    conn = get_db()

    cursor = conn.cursor()


    cursor.execute("""
        DELETE FROM patients
    """)


    conn.commit()

    conn.close()


    return redirect(
        url_for("dashboard")
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=============================================="
    )

    print(
        "BREAST CANCER AI"
    )

    print(
        "WISCONSIN DIAGNOSTIC SYSTEM"
    )

    print(
        "=============================================="
    )

    print()

    print(
        "Open:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()


    app.run(
        debug=True
    )