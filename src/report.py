"""
report.py — AgriVision Decision Support System
==============================================
Handles PDF Analytics Generation.
"""

from fpdf import FPDF
from datetime import datetime
import os

class AgriVisionReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(42, 157, 143) # Teal
        self.cell(0, 10, 'AgriVision DSS - Biomass & Yield Analysis', 0, 1, 'L')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'L')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_pdf_report(
    counts: list[int],
    mean_density: float,
    cv_pct: float,
    yield_est: float,
    revenue: float,
    health_msg: str,
    currency: str,
    conf_thresh: float,
    iou_thresh: float
) -> bytes:
    """
    Generate a formatted PDF report summarizing the detection run.
    Returns the raw bytes of the PDF.
    """
    
    pdf = AgriVisionReport()
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'Executive Summary', 0, 1)
    
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 8, (
        f"A total of {len(counts)} image(s) underwent spatial analysis using the YOLOv8s Multi-Scale architecture. "
        f"The model detected an aggregate of {sum(counts)} wheat heads. "
        f"{health_msg}"
    ))
    pdf.ln(5)

    # 2. Key Metrics Table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Agronomic KPIs', 0, 1)
    
    pdf.set_font('Arial', '', 11)
    line_h = 8
    
    metrics = [
        ("Images Processed:", f"{len(counts)}"),
        ("Total Bounding Boxes:", f"{sum(counts)}"),
        ("Mean Detections per Image:", f"{mean_density:.1f}"),
        ("Coefficient of Variation (CV):", f"{cv_pct:.1f}%"),
        ("Estimated Yield (t/ha):", f"{yield_est:.2f} t/ha"),
        ("Estimated Revenue:", f"{revenue:,.2f} {currency}")
    ]
    
    for k, v in metrics:
        pdf.cell(80, line_h, k, border=0)
        pdf.cell(40, line_h, v, border=0, ln=1)
        
    pdf.ln(5)
    
    # 3. Engine Parameters
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Engine Configuration', 0, 1)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(80, line_h, "Confidence Threshold:", border=0)
    pdf.cell(40, line_h, f"{conf_thresh:.2f}", border=0, ln=1)
    
    pdf.cell(80, line_h, "IoU Threshold (NMS):", border=0)
    pdf.cell(40, line_h, f"{iou_thresh:.2f}", border=0, ln=1)
    
    # Dump to bytes (Explicitly cast to bytes to prevent Streamlit bytearray exception)
    return pdf.output(dest='S').encode('latin-1') 
