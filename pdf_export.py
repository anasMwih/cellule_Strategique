import os
import re
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus.tables import Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def generate_pdf(report_md: str, pdf_path: str):
    """
    Génère un PDF professionnel à partir du contenu Markdown.
    """
    # ENREGISTREMENT POLICE UTF-8
    # Assurez-vous que le fichier DejaVuSans.ttf est accessible dans l'environnement
    try:
        pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
        font_name = 'DejaVu'
    except Exception as e:
        print("Erreur de chargement de police DejaVu:", e)
        # Fallback to default
        font_name = 'Helvetica'

    # STYLES
    styles = getSampleStyleSheet()
    styles["BodyText"].fontName = font_name
    styles["BodyText"].leading = 14
    styles["Heading1"].fontName = font_name
    styles["Heading2"].fontName = font_name
    styles["Heading3"].fontName = font_name

    # Style spécifique pour forcer le retour à la ligne dans les tableaux
    styles.add(ParagraphStyle(name='TableText', parent=styles['BodyText'], fontSize=9, leading=12))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=30
    )

    story = []

    # NETTOYAGE TEXTE
    report_md = report_md.replace("–", "-").replace("—", "-").replace("<br>", "\n")

    # TRAITEMENT LIGNE PAR LIGNE
    lines = report_md.split("\n")

    in_table = False
    table_data = []

    def flush_table():
        nonlocal in_table, table_data
        if in_table and table_data:
            max_cols = max(len(row) for row in table_data)
            
            # S'assurer que toutes les lignes ont le même nombre de colonnes
            for row in table_data:
                while len(row) < max_cols:
                    row.append(Paragraph("", styles["TableText"]))
                    
            # Calculer la largeur des colonnes (A4 = 595 pts, Marges = 80 pts -> 515 pts utilisables)
            col_width = 515.0 / max_cols
            
            t = Table(table_data, colWidths=[col_width] * max_cols)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ecf0f1')), # En-tête gris clair
                ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('FONTNAME', (0,0),(-1,-1), font_name),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 15))
        in_table = False
        table_data = []

    for line in lines:
        line = line.strip()
        line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)

        if "|" in line:
            if "---" in line:
                continue
                
            cells = line.split("|")
            if cells and cells[0].strip() == "":
                cells = cells[1:]
            if cells and cells[-1].strip() == "":
                cells = cells[:-1]
                
            cells = [c.strip() for c in cells]
            
            row = [Paragraph(c, styles["TableText"]) for c in cells]
            table_data.append(row)
            in_table = True
            continue
        else:
            flush_table()

        if not line:
            story.append(Spacer(1, 10))
            continue
        
        # TITRES
        if line.startswith("# "):
            txt = line.replace("# ", "")
            p = Paragraph(f"<font size=20 color='#2c3e50'><b>{txt}</b></font>", styles["Heading1"])
            story.append(p)
            story.append(Spacer(1, 18))
        elif line.startswith("## "):
            txt = line.replace("## ", "")
            p = Paragraph(f"<font size=16 color='#2980b9'><b>{txt}</b></font>", styles["Heading2"])
            story.append(p)
            story.append(Spacer(1, 14))
        elif line.startswith("### "):
            txt = line.replace("### ", "")
            p = Paragraph(f"<font size=13 color='#34495e'><b>{txt}</b></font>", styles["Heading3"])
            story.append(p)
            story.append(Spacer(1, 10))
        # LISTES A PUCES
        elif line.startswith("- ") or line.startswith("* "):
            txt = line[2:]
            p = Paragraph(f"• {txt}", styles["BodyText"])
            story.append(p)
            story.append(Spacer(1, 5))
        # TEXTE NORMAL
        else:
            p = Paragraph(line, styles["BodyText"])
            story.append(p)
            story.append(Spacer(1, 8))

    flush_table()

    # GENERATION PDF
    doc.build(story)
    return pdf_path
