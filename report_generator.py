"""
PDF report generator using ReportLab.
Creates downloadable disease analysis reports with image, prediction,
severity, and disease information.
"""

import io
import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from disease_info import get_disease_info


def _save_temp_image(image: np.ndarray) -> str:
    """Save a numpy image to a temporary file and return the path."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"cotton_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
    cv2.imwrite(temp_path, image)
    return temp_path


def generate_pdf_report(
    image: np.ndarray,
    disease: str,
    confidence: float,
    severity_level: str,
    severity_percentage: float,
    lang: str = "en",
) -> bytes:
    """
    Generate a PDF report for a cotton disease prediction.

    Args:
        image: BGR numpy array of the analyzed leaf image.
        disease: Predicted disease name.
        confidence: Prediction confidence (0-100).
        severity_level: Severity classification string.
        severity_percentage: Infected area percentage.
        lang: Language code for disease information.

    Returns:
        PDF file contents as bytes.
    """
    disease_info = get_disease_info(disease, lang)
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#2E7D32"),
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1B5E20"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=16,
    )
    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )

    story = []

    # Title
    story.append(Paragraph("Cotton Plant Disease Detection Report", title_style))
    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
            meta_style,
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # Uploaded image
    temp_image_path = _save_temp_image(image)
    try:
        img_element = RLImage(temp_image_path, width=3.5 * inch, height=3.5 * inch)
        img_element.hAlign = "CENTER"
        story.append(img_element)
    except Exception:
        story.append(Paragraph("Image could not be embedded in report.", body_style))
    story.append(Spacer(1, 0.3 * inch))

    # Prediction summary table
    status = "Healthy Plant" if disease == "Healthy" else "Disease Detected"
    summary_data = [
        ["Field", "Value"],
        ["Disease", disease_info["name"]],
        ["Confidence", f"{confidence:.1f}%"],
        ["Status", status],
        ["Severity Level", severity_level],
        ["Severity Percentage", f"{severity_percentage:.1f}%"],
    ]

    summary_table = Table(summary_data, colWidths=[2.5 * inch, 3.5 * inch])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E8F5E9")]),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))

    # Disease information sections
    sections = [
        ("Symptoms", disease_info["symptoms"]),
        ("Causes", disease_info["causes"]),
        ("Prevention Methods", disease_info["prevention"]),
        ("Treatment Recommendations", disease_info["treatment"]),
        ("Impact on Yield", disease_info["impact_on_yield"]),
    ]

    for section_title, section_content in sections:
        story.append(Paragraph(section_title, heading_style))
        story.append(Paragraph(section_content, body_style))

    # Footer
    story.append(Spacer(1, 0.5 * inch))
    story.append(
        Paragraph(
            "CottonGuard AI - Smart Crop Advisory System using CNN",
            meta_style,
        )
    )
    story.append(
        Paragraph(
            "This report is generated by an AI system and should be used as advisory only. "
            "Consult agricultural experts for critical decisions.",
            meta_style,
        )
    )

    doc.build(story)

    # Clean up temp image
    try:
        os.remove(temp_image_path)
    except OSError:
        pass

    buffer.seek(0)
    return buffer.getvalue()
