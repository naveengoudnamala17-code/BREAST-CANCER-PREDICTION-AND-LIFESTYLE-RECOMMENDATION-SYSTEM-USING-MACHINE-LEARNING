from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
import os


# ============================================================
# COLORS
# ============================================================

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2563A6")
LIGHT_BLUE = colors.HexColor("#EEF5FF")
VERY_LIGHT_BLUE = colors.HexColor("#F8FBFF")

GREEN = colors.HexColor("#166534")
LIGHT_GREEN = colors.HexColor("#DCFCE7")

RED = colors.HexColor("#991B1B")
LIGHT_RED = colors.HexColor("#FEE2E2")

ORANGE = colors.HexColor("#9A3412")
LIGHT_ORANGE = colors.HexColor("#FFEDD5")

GREY = colors.HexColor("#6B7280")
LIGHT_GREY = colors.HexColor("#F3F4F6")
BORDER = colors.HexColor("#D9E2EC")
DARK = colors.HexColor("#1F2937")

WHITE = colors.white


# ============================================================
# FONT
# ============================================================

try:

    font_path = "C:/Windows/Fonts/arial.ttf"

    if os.path.exists(font_path):

        pdfmetrics.registerFont(
            TTFont("Arial", font_path)
        )

        FONT = "Arial"

    else:

        FONT = "Helvetica"

except Exception:

    FONT = "Helvetica"


# ============================================================
# PAGE HEADER / FOOTER
# ============================================================

def draw_page(canvas, doc):

    canvas.saveState()

    width, height = A4

    # --------------------------------------------------------
    # TOP HEADER
    # --------------------------------------------------------

    canvas.setFillColor(NAVY)

    canvas.rect(
        0,
        height - 24 * mm,
        width,
        24 * mm,
        fill=1,
        stroke=0
    )

    canvas.setFillColor(WHITE)

    canvas.setFont(
        FONT,
        15
    )

    canvas.drawString(
        20 * mm,
        height - 14 * mm,
        "Breast Cancer AI"
    )

    canvas.setFont(
        FONT,
        8
    )

    canvas.drawString(
        20 * mm,
        height - 19 * mm,
        "Wisconsin Diagnostic Prediction System"
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    canvas.setStrokeColor(BORDER)

    canvas.line(
        20 * mm,
        14 * mm,
        width - 20 * mm,
        14 * mm
    )

    canvas.setFillColor(GREY)

    canvas.setFont(
        FONT,
        7
    )

    canvas.drawString(
        20 * mm,
        9 * mm,
        "Breast Cancer AI • Academic / Research Use"
    )

    canvas.drawRightString(
        width - 20 * mm,
        9 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ============================================================
# PDF GENERATOR
# ============================================================

def generate_pdf(
    output_path,
    patient_name,
    age,
    gender,
    result,
    confidence,
    risk_level,
    cancer_type,
    explanation,
    recommendations,
    diet,
    clump_thickness,
    uniformity_cell_size,
    uniformity_cell_shape,
    marginal_adhesion,
    single_epithelial_cell_size,
    bare_nuclei,
    bland_chromatin,
    normal_nucleoli,
    mitoses
):

    # ========================================================
    # DOCUMENT
    # ========================================================

    doc = SimpleDocTemplate(

        output_path,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=32 * mm,

        bottomMargin=20 * mm
    )


    styles = getSampleStyleSheet()


    # ========================================================
    # STYLES
    # ========================================================

    title_style = ParagraphStyle(

        "Title",

        parent=styles["Title"],

        fontName=FONT,

        fontSize=24,

        leading=28,

        textColor=NAVY,

        alignment=TA_CENTER,

        spaceAfter=6
    )


    subtitle_style = ParagraphStyle(

        "Subtitle",

        parent=styles["Normal"],

        fontName=FONT,

        fontSize=10,

        leading=14,

        textColor=GREY,

        alignment=TA_CENTER,

        spaceAfter=18
    )


    section_style = ParagraphStyle(

        "Section",

        parent=styles["Heading2"],

        fontName=FONT,

        fontSize=14,

        leading=18,

        textColor=NAVY,

        spaceBefore=8,

        spaceAfter=10
    )


    normal_style = ParagraphStyle(

        "NormalCustom",

        parent=styles["Normal"],

        fontName=FONT,

        fontSize=9,

        leading=14,

        textColor=DARK
    )


    small_style = ParagraphStyle(

        "Small",

        parent=styles["Normal"],

        fontName=FONT,

        fontSize=8,

        leading=12,

        textColor=GREY
    )


    center_style = ParagraphStyle(

        "Center",

        parent=normal_style,

        alignment=TA_CENTER
    )


    result_style = ParagraphStyle(

        "Result",

        parent=styles["Heading1"],

        fontName=FONT,

        fontSize=20,

        leading=24,

        alignment=TA_CENTER
    )


    # ========================================================
    # STORY
    # ========================================================

    story = []


    # ========================================================
    # REPORT TITLE
    # ========================================================

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )

    story.append(
        Paragraph(
            "Prediction Result",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-assisted diagnostic assessment",
            subtitle_style
        )
    )


    # ========================================================
    # REPORT DATE
    # ========================================================

    current_date = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    date_table = Table(

        [[
            Paragraph(
                "<b>Report Generated</b>",
                small_style
            ),

            Paragraph(
                current_date,
                small_style
            )
        ]],

        colWidths=[
            45 * mm,
            125 * mm
        ]
    )

    date_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                LIGHT_GREY
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
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

    story.append(date_table)

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    story.append(
        Paragraph(
            "AI Prediction",
            section_style
        )
    )


    if result == "Benign":

        result_color = GREEN
        result_background = LIGHT_GREEN
        result_text = "✓ BENIGN"

    else:

        result_color = RED
        result_background = LIGHT_RED
        result_text = "⚠ MALIGNANT"


    prediction_table = Table(

        [[
            Paragraph(
                result_text,
                ParagraphStyle(
                    "ResultStatus",
                    parent=result_style,
                    textColor=result_color
                )
            ),

            Paragraph(
                f"<b>Model Confidence</b><br/><font size='18'>{confidence}%</font>",
                center_style
            ),

            Paragraph(
                f"<b>AI Risk Level</b><br/><font size='12'>{risk_level}</font>",
                center_style
            )
        ]],

        colWidths=[
            55 * mm,
            55 * mm,
            55 * mm
        ]
    )


    prediction_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, 0),
                result_background
            ),

            (
                "BACKGROUND",
                (1, 0),
                (-1, 0),
                VERY_LIGHT_BLUE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
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
                15
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                15
            )

        ])
    )

    story.append(prediction_table)

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )


    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "Patient Information",
            section_style
        )
    )


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
            56 * mm,
            56 * mm,
            56 * mm
        ]
    )


    patient_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                LIGHT_BLUE
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, 1),
                VERY_LIGHT_BLUE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
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

        [[
            Paragraph(
                explanation,
                normal_style
            )
        ]],

        colWidths=[
            168 * mm
        ]
    )


    explanation_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                LIGHT_BLUE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                BLUE
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

        ("Clump Thickness", clump_thickness),

        ("Uniformity of Cell Size", uniformity_cell_size),

        ("Uniformity of Cell Shape", uniformity_cell_shape),

        ("Marginal Adhesion", marginal_adhesion),

        ("Single Epithelial Cell Size", single_epithelial_cell_size),

        ("Bare Nuclei", bare_nuclei),

        ("Bland Chromatin", bland_chromatin),

        ("Normal Nucleoli", normal_nucleoli),

        ("Mitoses", mitoses)

    ]


    feature_data = []


    for i in range(0, len(features), 3):

        row = []

        for name, value in features[i:i + 3]:

            row.append(
                Paragraph(
                    f"<b>{name}</b><br/><font size='14'>{value}</font>",
                    normal_style
                )
            )

        while len(row) < 3:

            row.append("")


        feature_data.append(row)


    feature_table = Table(

        feature_data,

        colWidths=[
            56 * mm,
            56 * mm,
            56 * mm
        ],

        rowHeights=25 * mm
    )


    feature_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                VERY_LIGHT_BLUE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
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


    recommendation_text = ""


    for item in recommendations:

        recommendation_text += (
            "• "
            + str(item)
            + "<br/>"
        )


    recommendation_table = Table(

        [[
            Paragraph(
                recommendation_text,
                normal_style
            )
        ]],

        colWidths=[
            168 * mm
        ]
    )


    recommendation_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                LIGHT_BLUE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                BLUE
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
    # DIET & LIFESTYLE
    # ========================================================

    story.append(
        Paragraph(
            "Personalized Diet & Lifestyle Plan",
            section_style
        )
    )


    diet_text = ""


    for item in diet:

        diet_text += (
            "• "
            + str(item)
            + "<br/>"
        )


    diet_table = Table(

        [[
            Paragraph(
                diet_text,
                normal_style
            )
        ]],

        colWidths=[
            168 * mm
        ]
    )


    diet_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                LIGHT_GREEN
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                GREEN
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


    notice_text = (

        "<b>Academic / Research Use Only</b><br/><br/>"

        "This system is an academic AI-based prediction "
        "tool intended for educational and research purposes. "
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

        [[
            Paragraph(
                notice_text,
                normal_style
            )
        ]],

        colWidths=[
            168 * mm
        ]
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
                1,
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


    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(

        story,

        onFirstPage=draw_page,

        onLaterPages=draw_page

    )


# ============================================================
# END OF PDF GENERATOR
# ============================================================