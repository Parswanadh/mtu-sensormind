import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf(engine_id: int, plan: dict, analysis: str, rul: int) -> str:
    """
    Generates a professional PDF Work Order.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(base_dir, 'dashboard', 'work_orders')
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wo_id = f"WO-{timestamp}-{engine_id}"
    filename = f"work_order_{engine_id}_{timestamp}.pdf"
    filepath = os.path.join(out_dir, filename)
    
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    subtitle_style = styles['Heading2']
    subtitle_style.textColor = colors.HexColor('#002b5c') # Dark blue
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    urgent_style = ParagraphStyle(
        'Urgent',
        parent=styles['Normal'],
        textColor=colors.red,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Header
    story.append(Paragraph("MTU SensorMind", title_style))
    story.append(Paragraph("Automated Maintenance Work Order", title_style))
    story.append(Spacer(1, 20))
    
    # Meta Data Table
    meta_data = [
        ['Work Order ID:', wo_id, 'Date:', datetime.now().strftime("%Y-%m-%d %H:%M")],
        ['Engine ID:', str(engine_id), 'Estimated RUL:', f"{rul} cycles"]
    ]
    
    t = Table(meta_data, colWidths=[100, 150, 50, 150])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('PADDING', (0,0), (-1,-1), 6)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Fault Summary
    story.append(Paragraph("1. Fault Summary & Analysis", subtitle_style))
    story.append(Paragraph(analysis, normal_style))
    story.append(Spacer(1, 15))
    
    # Action Plan
    story.append(Paragraph("2. Recommended Maintenance Action", subtitle_style))
    
    urgency = plan.get('urgency', 'SCHEDULED')
    p_style = urgent_style if 'IMMEDIATE' in urgency else normal_style
    
    story.append(Paragraph(f"<b>Urgency:</b> {urgency}", p_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"<b>Action:</b> {plan.get('action', 'N/A')}", normal_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"<b>Estimated Downtime:</b> {plan.get('estimated_downtime_hrs', 0)} hours", normal_style))
    story.append(Spacer(1, 15))
    
    # Parts List
    story.append(Paragraph("3. Required Parts", subtitle_style))
    parts = plan.get('parts_required', [])
    if parts:
        for part in parts:
            story.append(Paragraph(f"• {part}", normal_style))
    else:
        story.append(Paragraph("No specific parts listed.", normal_style))
    story.append(Spacer(1, 15))
    
    # Technician Notes
    story.append(Paragraph("4. Technician Notes", subtitle_style))
    story.append(Paragraph(plan.get('technician_notes', 'N/A'), normal_style))
    
    # Build PDF
    doc.build(story)
    return filepath
