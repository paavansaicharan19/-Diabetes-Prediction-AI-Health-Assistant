"""
report_generator.py

This file contains the ReportAgent responsible for generating a professional,
well-styled PDF health report for the patient. It compiles:
- The patient's clinical measurements
- The Machine Learning prediction and risk probability
- The Gemini Reasoning Agent's explanation
- The Recommendation Agent's lifestyle guidance

It uses the `reportlab` library to build the PDF document programmatically.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

class ReportAgent:
    """
    Agent responsible for generating a styled PDF health report from patient
    diagnostics and AI explanations.
    """
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _clean_text(self, text: str) -> str:
        """
        Alternately replaces markdown double-asterisks with proper HTML bold tags,
        escapes XML characters, and normalizes bullet symbols for ReportLab.
        """
        # Escape ampersands first to prevent XML parser errors in ReportLab
        text_escaped = text.replace("&", "&amp;")
        
        # Alternately replace ** with <b> and </b>
        parts = text_escaped.split("**")
        html_text = ""
        for i, part in enumerate(parts):
            if i % 2 == 1:
                html_text += f"<b>{part}</b>"
            else:
                html_text += part
                
        # Normalize list bullets
        stripped = html_text.strip()
        if stripped.startswith("- "):
            html_text = html_text.replace("- ", "• ", 1)
        elif stripped.startswith("* "):
            html_text = html_text.replace("* ", "• ", 1)
            
        return html_text

    def generate_pdf_report(
        self, 
        patient_data: dict, 
        prediction: dict, 
        ai_explanation: str, 
        ai_recommendations: str
    ) -> str:
        """
        Compiles health metrics and AI insights into a styled PDF report.
        Returns the path to the generated PDF file.
        """
        # Create filename based on timestamp to keep them unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Diabetes_Assessment_Report_{timestamp}.pdf"
        file_path = os.path.join(self.output_dir, filename)
        
        # Setup document template
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=54,  # 0.75 in margin
            leftMargin=54,
            topMargin=54,
            bottomMargin=54
        )
        
        # Build stylesheet
        styles = getSampleStyleSheet()
        
        # Define custom professional styles
        title_style = ParagraphStyle(
            name='ReportTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1A365D"),  # Navy blue primary
            alignment=TA_LEFT,
            spaceAfter=15
        )
        
        subtitle_style = ParagraphStyle(
            name='ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4A5568"),  # Cool grey
            spaceAfter=20
        )
        
        h1_style = ParagraphStyle(
            name='SectionHeading',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2C5282"),  # Lighter blue secondary
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            name='BodyTextCustom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#2D3748")  # Charcoal body text
        )
        
        disclaimer_style = ParagraphStyle(
            name='DisclaimerText',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#718096"),
            spaceBefore=15,
            alignment=TA_CENTER
        )

        story = []

        # ----------------------------------------------------
        # 1. HEADER SECTION
        # ----------------------------------------------------
        story.append(Paragraph("Patient Health Assessment Report", title_style))
        current_time = datetime.now().strftime("%B %d, %Y - %H:%M")
        story.append(Paragraph(f"Generated on: {current_time} | Agentic AI Predictor & Assistant", subtitle_style))
        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # 2. DIAGNOSTIC SUMMARY CARD (ML Results)
        # ----------------------------------------------------
        story.append(Paragraph("1. Diagnostic Results (Machine Learning)", h1_style))
        
        prob_percent = prediction["probability"] * 100
        risk_level = prediction["risk_level"]
        diag_text = "Diabetic (High Likelihood)" if prediction["prediction"] == 1 else "Non-Diabetic (Low Likelihood)"
        
        # Color coordinate according to risk
        if risk_level == "Low Risk":
            risk_color = colors.HexColor("#27AE60") # Green
        elif risk_level == "Medium Risk":
            risk_color = colors.HexColor("#F2994A") # Orange
        else:
            risk_color = colors.HexColor("#EB5757") # Red
            
        summary_data = [
            [
                Paragraph("<b>Predicted Classification:</b>", body_style),
                Paragraph(diag_text, body_style)
            ],
            [
                Paragraph("<b>Prediction Probability:</b>", body_style),
                Paragraph(f"{prob_percent:.2f}%", body_style)
            ],
            [
                Paragraph("<b>Assessed Risk Level:</b>", body_style),
                Paragraph(f"<font color='{risk_color.hexval()}'><b>{risk_level}</b></font>", body_style)
            ]
        ]
        
        summary_table = Table(summary_data, colWidths=[180, 324])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
            ('PADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LINELEFT', (0,0), (0,-1), 5, risk_color), # Colored indicator bar on the left
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 15))

        # ----------------------------------------------------
        # 3. CLINICAL METRICS TABLE
        # ----------------------------------------------------
        story.append(Paragraph("2. Input Clinical Measurements", h1_style))
        
        metrics_data = [
            [Paragraph("<b>Clinical Metric</b>", body_style), Paragraph("<b>Patient Value</b>", body_style), Paragraph("<b>Normal Range Guide</b>", body_style)]
        ]
        
        # Table of inputs with normal guidelines
        metric_guide = {
            "Pregnancies": (patient_data["Pregnancies"], "< 3"),
            "Glucose": (f"{patient_data['Glucose']} mg/dL", "< 100 mg/dL (Fasting)"),
            "BloodPressure": (f"{patient_data['BloodPressure']} mmHg", "< 80 mmHg (Diastolic)"),
            "SkinThickness": (f"{patient_data['SkinThickness']} mm", "10 - 30 mm"),
            "Insulin": (f"{patient_data['Insulin']} mu U/ml", "< 140 mu U/ml (2-Hour)"),
            "BMI": (f"{patient_data['BMI']}", "18.5 - 24.9"),
            "DiabetesPedigreeFunction": (f"{patient_data['DiabetesPedigreeFunction']}", "< 0.50"),
            "Age": (f"{patient_data['Age']} Years", "Adult range (21+)")
        }
        
        for k, (val, guide) in metric_guide.items():
            metrics_data.append([
                Paragraph(k, body_style),
                Paragraph(str(val), body_style),
                Paragraph(guide, body_style)
            ])
            
        metrics_table = Table(metrics_data, colWidths=[180, 140, 184])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('PADDING', (0,0), (-1,-1), 6),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 15))

        # ----------------------------------------------------
        # 4. AI CLINICAL EXPLANATION
        # ----------------------------------------------------
        story.append(Paragraph("3. Detailed AI Clinical Analysis", h1_style))
        # Split explanations by newline and build paragraphs
        for p_text in ai_explanation.strip().split("\n"):
            if p_text.strip():
                clean_text = self._clean_text(p_text)
                story.append(Paragraph(clean_text, body_style))
                story.append(Spacer(1, 6))
                
        story.append(Spacer(1, 10))

        # ----------------------------------------------------
        # 5. PERSONALIZED HEALTH PLAN
        # ----------------------------------------------------
        story.append(Paragraph("4. Personalized Action Plan & Recommendations", h1_style))
        for p_text in ai_recommendations.strip().split("\n"):
            if p_text.strip():
                clean_text = self._clean_text(p_text)
                story.append(Paragraph(clean_text, body_style))
                story.append(Spacer(1, 6))

        # ----------------------------------------------------
        # 6. DISCLAIMER & FOOTER
        # ----------------------------------------------------
        story.append(Spacer(1, 15))
        story.append(Paragraph(
            "⚠️ DISCLAIMER: This health assessment report is powered by machine learning and AI algorithms. "
            "It is for educational and illustrative purposes only and does not constitute medical advice, "
            "diagnosis, or treatment. Always consult a qualified medical professional for clinical guidance.",
            disclaimer_style
        ))

        # Build PDF
        doc.build(story)
        
        return file_path
