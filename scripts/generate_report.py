#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de generación automatizada del reporte final en PDF.
Convierte docs/informe_final.md a reports/final/informe_final.pdf
utilizando ReportLab con estilos tipográficos premium y maquetación académica.
"""

import re
import os
from loguru import logger

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Configuración de loguru
logger.add("config/logging/pdf_generation.log", rotation="1 MB", level="INFO")

def format_math(text):
    """
    Reemplaza expresiones LaTeX en markdown por su representación limpia en HTML
    que ReportLab Paragraph es capaz de interpretar de forma nativa.
    """
    replacements = {
        # Ecuaciones en bloque (deben ser centradas)
        r"$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$": "<b>y(t) = g(t) + s(t) + h(t) + &epsilon;<sub>t</sub></b>",
        r"$$P(t) = \frac{dE(t)}{dt}$$": "<b>P(t) = dE(t) / dt</b>",
        r"$$E = \int_{t_1}^{t_2} P(t) dt$$": "<b>E = &int;<sub>t<sub>1</sub></sub><sup>t<sub>2</sub></sup> P(t) dt</b>",
        r"$$g(t) = (k + a(t)^T \delta) t + (m + a(t)^T \gamma)$$": "<b>g(t) = (k + a(t)<sup>T</sup> &delta;) t + (m + a(t)<sup>T</sup> &gamma;)</b>",
        r"$$s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right) \right)$$": "<b>s(t) = &Sigma;<sub>n=1</sub><sup>N</sup> (a<sub>n</sub> cos(2&pi;nt/P) + b<sub>n</sub> sin(2&pi;nt/P))</b>",
        r"$$\hat{y}(x) = \frac{1}{M} \sum_{i=1}^{M} T_i(x)$$": "<b>ŷ(x) = (1/M) &Sigma;<sub>i=1</sub><sup>M</sup> T<sub>i</sub>(x)</b>",
        r"$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$": "<b>ŷ<sub>i</sub><sup>(t)</sup> = ŷ<sub>i</sub><sup>(t-1)</sup> + f<sub>t</sub>(x<sub>i</sub>)</b>",
        r"$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$": "<b>L<sup>(t)</sup> = &Sigma;<sub>i=1</sub><sup>n</sup> l(y<sub>i</sub>, ŷ<sub>i</sub><sup>(t-1)</sup> + f<sub>t</sub>(x<sub>i</sub>)) + &Omega;(f<sub>t</sub>)</b>",
        r"$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$": "<b>&Omega;(f<sub>t</sub>) = &gamma;T + 1/2 &lambda; &Sigma;<sub>j=1</sub><sup>T</sup> w<sub>j</sub><sup>2</sup></b>",
        r"$$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$": "<b>MAE = (1/N) &Sigma;<sub>i=1</sub><sup>N</sup> |y<sub>i</sub> - ŷ<sub>i</sub>|</b>",
        r"$$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$": "<b>RMSE = &radic;[ (1/N) &Sigma;<sub>i=1</sub><sup>N</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup> ]</b>",
        r"$$MAPE = \frac{100\%}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$": "<b>MAPE = (100% / N) &Sigma;<sub>i=1</sub><sup>N</sup> | (y<sub>i</sub> - ŷ<sub>i</sub>) / y<sub>i</sub> |</b>",
        r"$$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$": "<b>R<sup>2</sup> = 1 - [ &Sigma;<sub>i=1</sub><sup>N</sup> (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup> / &Sigma;<sub>i=1</sub><sup>N</sup> (y<sub>i</sub> - ŷ<sub>mean</sub>)<sup>2</sup> ]</b>",
        r"$$\text{rolling\_mean\_24}_t = \frac{1}{24} \sum_{i=1}^{24} y_{t-i}$$": "<b>rolling_mean_24<sub>t</sub> = (1/24) &Sigma;<sub>i=1</sub><sup>24</sup> y<sub>t-i</sub></b>",
        
        # Ecuaciones inline
        r"$\Delta f < 0$": "<i>&Delta;f &lt; 0</i>",
        r"$\Delta f = 0$": "<i>&Delta;f = 0</i>",
        r"$\bar{y}$": "ŷ<sub>mean</sub>",
        r"$\text{ kW}$": " kW",
        r"$\text{ MWh}$": " MWh",
        r"$\text{ Hz}$": " Hz",
        r"$P$": "<i>P</i>",
        r"$P_{dem}$": "<i>P<sub>dem</sub></i>",
        r"$P_{gen}$": "<i>P<sub>gen</sub></i>",
        r"$t_1$": "<i>t<sub>1</sub></i>",
        r"$t_2$": "<i>t<sub>2</sub></i>",
        r"$y(t)$": "<i>y(t)</i>",
        r"$g(t)$": "<i>g(t)</i>",
        r"$s(t)$": "<i>s(t)</i>",
        r"$h(t)$": "<i>h(t)</i>",
        r"$\epsilon_t$": "<i>&epsilon;<sub>t</sub></i>",
        r"$P=24$": "<i>P=24</i>",
        r"$P=168$": "<i>P=168</i>",
        r"$P=8766\text{ horas}$": "<i>P=8766 horas</i>",
        r"$N$": "<i>N</i>",
        r"$a_n$": "<i>a<sub>n</sub></i>",
        r"$b_n$": "<i>b<sub>n</sub></i>",
        r"$M$": "<i>M</i>",
        r"$m \ll d$": "<i>m &laquo; d</i>",
        r"$d$": "<i>d</i>",
        r"$\hat{y}$": "ŷ",
        r"$f_t$": "<i>f<sub>t</sub></i>",
        r"$f_t(x_i)$": "<i>f<sub>t</sub>(x<sub>i</sub>)</i>",
        r"$\Omega(f_t)$": "<i>&Omega;(f<sub>t</sub>)</i>",
        r"$\gamma T$": "<i>&gamma;T</i>",
        r"$T$": "<i>T</i>",
        r"$w_j$": "<i>w<sub>j</sub></i>",
        r"$y_i$": "<i>y<sub>i</sub></i>",
        r"$\hat{y}_i$": "ŷ<sub>i</sub>",
        r"$\hat{y}_t$": "ŷ<sub>t</sub>",
        r"$\hat{y}_{t-1}$": "ŷ<sub>t-1</sub>",
        r"$\hat{y}_{t+1}$": "ŷ<sub>t+1</sub>",
        r"$y_{t-1}$": "y<sub>t-1</sub>",
        r"$y_{t-24}$": "y<sub>t-24</sub>",
        r"$y_{t-168}$": "y<sub>t-168</sub>",
        r"$R^2$": "R<sup>2</sup>",
        r"$\Delta f$": "&Delta;f",
    }
    
    # Aplicar reemplazos exactos
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    # Reemplazo de variables de un caracter entre $ como variables cursivas
    text = re.sub(r'\$([a-zA-Z0-9_\-\+\%\\/]+)\$', r'<i>\1</i>', text)
    
    # Limpieza final de backslashes sobrantes de LaTeX
    text = text.replace(r"\text", "").replace("{", "").replace("}", "")
    
    return text

def format_line(line):
    """Aplica formato de negrita, cursiva y matemáticas a una línea."""
    # Negritas
    line = re.sub(r'\*\frac{(.*?)}{(.*?)}', r'\1/\2', line)  # Simplificar fracciones sencillas
    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
    # Cursivas
    line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
    # Fórmulas y caracteres especiales
    line = format_math(line)
    return line.strip()

def resolve_image_path(path):
    """Resuelve la ruta absoluta de una imagen dada su ruta relativa o URI."""
    if path.startswith("file:///"):
        path = path.replace("file:///", "")
        from urllib.parse import unquote
        path = unquote(path)
        
    # Ruta directa
    if os.path.exists(path):
        return os.path.abspath(path)
        
    # Normalización
    path_norm = os.path.normpath(path)
    if os.path.exists(path_norm):
        return os.path.abspath(path_norm)
        
    # Relativa al workspace root
    clean_path = path.lstrip('/')
    if os.path.exists(clean_path):
        return os.path.abspath(clean_path)
        
    # Búsqueda basada en el directorio del script
    workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(workspace_path, clean_path)
    if os.path.exists(full_path):
        return os.path.abspath(full_path)
        
    # Búsqueda recursiva en la subcarpeta reports/figures
    basename = os.path.basename(path)
    fallback = os.path.join(workspace_path, "reports", "figures", basename)
    if os.path.exists(fallback):
        return os.path.abspath(fallback)
        
    return None

class NumberedCanvas(canvas.Canvas):
    """
    Canvas personalizado para calcular dinámicamente el número total de páginas
    y añadir cabecera y pie de página en cada página (excepto la portada).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # La portada es la página 1; omitir decoraciones en ella
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#2a5298"))
            
            # Cabecera
            self.drawString(54, 750, "CÁTEDRA PEDRO NEL GÓMEZ: LA ENERGÍA EN EL DESARROLLO DE COLOMBIA")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawRightString(558, 750, "Informe de Investigación Final")
            
            # Línea divisoria superior
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
            # Línea divisoria inferior
            self.line(54, 55, 558, 55)
            
            # Pie de página
            self.drawString(54, 40, "Universidad Nacional de Colombia - Sede Medellín")
            page_text = f"Página {self._pageNumber} de {page_count}"
            self.drawRightString(558, 40, page_text)
            self.restoreState()


def build_pdf(md_path, pdf_path):
    """Lee el markdown y compila el PDF estructurado."""
    logger.info(f"Iniciando compilación de {md_path} a {pdf_path}")
    
    # Cargar y verificar archivo origen
    if not os.path.exists(md_path):
        logger.error(f"El archivo markdown no existe en {md_path}")
        return False
        
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Inicializar documento de ReportLab (tamaño Carta, márgenes de 0.75" y 1" en cabecera/pie)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Definición de Hoja de Estilos Corporativos/Académicos (Azul y Gris Oscuro)
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=8,
        spaceBefore=0
    )
    
    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#2d3748"),
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=4,
        spaceBefore=2
    )
    
    h1_style = ParagraphStyle(
        'ReportH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e3c72"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'ReportH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#2a5298"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'ReportH3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor("#4a5568"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    math_style = ParagraphStyle(
        'ReportMath',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#1e3c72"),
        alignment=1,  # Centrado
        spaceBefore=8,
        spaceAfter=8
    )
    
    code_block_style = ParagraphStyle(
        'ReportCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1a202c"),
        backColor=colors.HexColor("#f7fafc"),
        borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=8,
        spaceAfter=8,
        keepWithNext=False
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2d3748")
    )
    
    table_cell_style_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell_style,
        alignment=1
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )
    
    table_header_style_center = ParagraphStyle(
        'TableHeaderCenter',
        parent=table_header_style,
        alignment=1
    )
    
    # ----------------------------------------------------
    # Construcción de la Portada en la Story
    # ----------------------------------------------------
    story = []
    
    story.append(Spacer(1, 40))
    story.append(HRFlowable(width="100%", thickness=5, color=colors.HexColor("#1e3c72"), spaceAfter=40))
    
    cover_title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1e3c72"),
        spaceAfter=15,
        alignment=1
    )
    
    cover_subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=180,
        alignment=1
    )
    
    cover_meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=16,
        textColor=colors.HexColor("#2d3748"),
        alignment=1
    )
    
    story.append(Paragraph("INFORME DE INVESTIGACIÓN ACADÉMICA", cover_title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Predicción de la Demanda Eléctrica del Sistema Interconectado Nacional (SIN) mediante Técnicas de Machine Learning", cover_title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Evaluación Comparativa de Modelos Prophet, Random Forest y XGBoost", cover_subtitle_style))
    
    story.append(HRFlowable(width="60%", thickness=0.5, color=colors.HexColor("#cbd5e0"), spaceAfter=30))
    
    meta_html = """
    <b>Autor:</b> Santiago Macias Ruiz<br/>
    <b>Asignatura:</b> Cátedra Pedro Nel Gómez: La energía en el desarrollo económico, social y tecnológico de Colombia<br/>
    <b>Institución:</b> Universidad Nacional de Colombia - Sede Medellín<br/>
    <b>Fecha de Publicación:</b> Junio 2026
    """
    story.append(Paragraph(meta_html, cover_meta_style))
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # Parseador Linea por Linea del Markdown
    # ----------------------------------------------------
    
    def emit_table(rows):
        """Función auxiliar para formatear y estilizar tablas de ReportLab."""
        if not rows:
            return None
        
        # Parsear las celdas como Paragraphs
        formatted_rows = []
        for r in rows:
            formatted_row = [Paragraph(format_line(cell), table_cell_style) for cell in r]
            formatted_rows.append(formatted_row)
            
        # Obtener columnas máximas
        max_cols = max(len(r) for r in formatted_rows)
        
        # Rellenar celdas en caso de que falten columnas
        for i, r in enumerate(formatted_rows):
            while len(r) < max_cols:
                r.append(Paragraph("", table_cell_style))
                rows[i].append("")
                
        # Definición de anchos de columna basados en la estructura
        if max_cols == 5:
            # Tabla de métricas de modelos: [Modelo, MAE, RMSE, MAPE, R2]
            col_widths = [124, 95, 95, 95, 95]
        elif max_cols == 2:
            # Tabla comparativa UPME vs Proyecto
            col_widths = [252, 252]
        else:
            col_widths = [504 / max_cols] * max_cols
            
        # Estilos base de la tabla
        t_styles = [
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]
        
        # Caso 1: Tabla Comparativa UPME vs Proyecto (2 columnas)
        if max_cols == 2 and "COMPARACIÓN" in rows[0][0]:
            # Encabezado principal (título fusionado)
            t_styles.append(('SPAN', (0,0), (1,0)))
            t_styles.append(('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3c72")))
            formatted_rows[0][0] = Paragraph(f"<font color='white'><b>{rows[0][0]}</b></font>", table_header_style_center)
            
            # Sub-encabezados (UPME y Proyecto)
            t_styles.append(('BACKGROUND', (0,1), (-1,1), colors.HexColor("#2a5298")))
            formatted_rows[1][0] = Paragraph(f"<font color='white'><b>{rows[1][0]}</b></font>", table_header_style)
            formatted_rows[1][1] = Paragraph(f"<font color='white'><b>{rows[1][1]}</b></font>", table_header_style)
            
            # Filas de contenido
            for r_idx in range(2, len(rows)):
                bg_color = "#f7fafc" if r_idx % 2 == 0 else "#ffffff"
                t_styles.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor(bg_color)))
                formatted_rows[r_idx][0] = Paragraph(format_line(rows[r_idx][0]), table_cell_style)
                formatted_rows[r_idx][1] = Paragraph(format_line(rows[r_idx][1]), table_cell_style)
                
        # Caso 2: Tabla de Métricas de Validación (5 columnas)
        elif max_cols == 5:
            # Fila de cabecera
            t_styles.append(('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3c72")))
            for c_idx in range(5):
                align_style = table_header_style if c_idx == 0 else table_header_style_center
                formatted_rows[0][c_idx] = Paragraph(f"<font color='white'><b>{rows[0][c_idx]}</b></font>", align_style)
                
            # Filas de datos
            for r_idx in range(1, len(rows)):
                bg_color = "#f7fafc" if r_idx % 2 == 1 else "#ffffff"
                t_styles.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor(bg_color)))
                
                for c_idx in range(5):
                    cell_text = rows[r_idx][c_idx]
                    align_style = table_cell_style if c_idx == 0 else table_cell_style_center
                    formatted_rows[r_idx][c_idx] = Paragraph(format_line(cell_text), align_style)
                    
        t = Table(formatted_rows, colWidths=col_widths)
        t.setStyle(TableStyle(t_styles))
        return t

    # Variables de control del parser
    in_code_block = False
    in_table = False
    code_lines = []
    table_rows = []
    skipped_header = False
    
    for line in lines:
        stripped = line.strip()
        
        # Saltar las líneas de título inicial y metadatos de la portada para no repetirlos en el cuerpo
        if not skipped_header:
            if stripped.startswith("# Informe de Investigación"):
                continue
            if stripped.startswith("**Asignatura:") or stripped.startswith("**Institución:") or stripped.startswith("**Fecha:") or stripped.startswith("**Autor:"):
                continue
            if stripped == "---":
                # La primera barra horizontal indica el final del encabezado inicial
                skipped_header = True
                continue
            # Si no es ninguna de estas, pero aún no hemos saltado el encabezado, saltamos por seguridad
            if stripped.startswith("## "):
                skipped_header = True
            else:
                continue
                
        # 1. Manejo de bloques de código (```)
        if stripped.startswith("```"):
            if in_code_block:
                # Terminar acumulación de bloque de código
                # Validar si el bloque de código es una tabla ASCII (como la comparación UPME)
                is_ascii_table = any("+" in l and "-" in l for l in code_lines) and any("|" in l for l in code_lines)
                
                if is_ascii_table:
                    # Limpiar y parsear tabla ASCII
                    table_data = []
                    for cl in code_lines:
                        # Saltar filas de bordes +---------+
                        if "+" in cl and "-" in cl:
                            continue
                        if cl.startswith("|"):
                            cells = [c.strip() for c in cl.split("|")]
                            if cells[0] == "": cells.pop(0)
                            if cells and cells[-1] == "": cells.pop()
                            table_data.append(cells)
                    
                    t_flowable = emit_table(table_data)
                    if t_flowable:
                        story.append(Spacer(1, 5))
                        story.append(t_flowable)
                        story.append(Spacer(1, 8))
                else:
                    # Bloque de código o diagrama normal
                    code_text = "\n".join(code_lines)
                    # Escapar caracteres HTML básicos y reemplazar espacios para conservar layout en Courier
                    escaped_code = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    escaped_code = escaped_code.replace(" ", "&nbsp;").replace("\n", "<br/>")
                    code_para = Paragraph(escaped_code, code_block_style)
                    story.append(Spacer(1, 5))
                    story.append(KeepTogether([code_para]))
                    story.append(Spacer(1, 8))
                
                # Resetear variables de bloque
                in_code_block = False
                code_lines = []
            else:
                in_code_block = True
            continue
            
        if in_code_block:
            code_lines.append(line.rstrip('\n'))
            continue
            
        # 1.5. Manejo de imágenes de Markdown ![alt](path)
        image_match = re.match(r'^!\[(.*?)\]\((.*?)\)', stripped)
        if image_match:
            caption_text = image_match.group(1)
            img_path_raw = image_match.group(2)
            img_path = resolve_image_path(img_path_raw)
            if img_path:
                try:
                    from reportlab.lib.utils import ImageReader
                    from reportlab.platypus import Image as RLImage
                    
                    img_reader = ImageReader(img_path)
                    orig_w, orig_h = img_reader.getSize()
                    aspect = orig_h / float(orig_w)
                    
                    # Ajustar ancho al área imprimible (504 pt máx). Usamos 420 pt para darle aire.
                    img_w = 420
                    img_h = img_w * aspect
                    
                    img_flowable = RLImage(img_path, width=img_w, height=img_h)
                    
                    caption_style = ParagraphStyle(
                        'ImageCaption',
                        parent=styles['Normal'],
                        fontName='Helvetica-Oblique',
                        fontSize=8.5,
                        leading=11,
                        textColor=colors.HexColor("#718096"),
                        alignment=1,  # Centrado
                        spaceBefore=6,
                        spaceAfter=12
                    )
                    caption_para = Paragraph(f"Gráfico: {caption_text}", caption_style)
                    
                    story.append(Spacer(1, 10))
                    story.append(KeepTogether([img_flowable, caption_para]))
                    story.append(Spacer(1, 10))
                except Exception as img_err:
                    logger.error(f"Error cargando imagen {img_path}: {img_err}")
            else:
                logger.warning(f"No se pudo resolver la ruta de la imagen: {img_path_raw}")
            continue

        # 2. Manejo de tablas de Markdown (fuera de bloques de código)
        if stripped.startswith("|"):
            if not in_table:
                in_table = True
                table_rows = []
            
            cells = [c.strip() for c in stripped.split("|")]
            if cells[0] == "": cells.pop(0)
            if cells and cells[-1] == "": cells.pop()
            
            # Saltar la fila separadora de alineación | :--- | :---: |
            is_separator = all(re.match(r'^[\s\:\-]+$', c) for c in cells)
            if not is_separator:
                table_rows.append(cells)
            continue
        else:
            if in_table:
                # Emitir tabla acumulada al salir del bloque de tabla
                t_flowable = emit_table(table_rows)
                if t_flowable:
                    story.append(Spacer(1, 5))
                    story.append(t_flowable)
                    story.append(Spacer(1, 8))
                in_table = False
                table_rows = []
                
        # 3. Saltar líneas vacías
        if not stripped:
            continue
            
        # 4. Saltos de sección explícitos (---) -> Salto de página académico
        if stripped == "---":
            story.append(PageBreak())
            continue
            
        # 5. Encabezados (H1, H2, H3)
        if stripped.startswith("### "):
            title = format_line(stripped[4:])
            story.append(Paragraph(title, h2_style))
            continue
        elif stripped.startswith("## "):
            title = format_line(stripped[3:])
            story.append(Paragraph(title, h1_style))
            continue
        elif stripped.startswith("#### "):
            title = format_line(stripped[5:])
            story.append(Paragraph(title, h3_style))
            continue
            
        # 6. Ecuaciones matemáticas en bloque independientes ($$...$$)
        if stripped.startswith("$$") and stripped.endswith("$$"):
            math_content = format_math(stripped)
            story.append(Paragraph(math_content, math_style))
            continue
            
        # 7. Elementos de lista ordenada o viñeta
        ordered_match = re.match(r'^(\d+)\.\s+(.*)', stripped)
        unordered_match = re.match(r'^([\*\-\+])\s+(.*)', stripped)
        
        if ordered_match:
            num = ordered_match.group(1)
            content = format_line(ordered_match.group(2))
            story.append(Paragraph(f"<b>{num}.</b>&nbsp;&nbsp;{content}", bullet_style))
        elif unordered_match:
            content = format_line(unordered_match.group(2))
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{content}", bullet_style))
        else:
            # 8. Párrafo normal
            content = format_line(stripped)
            story.append(Paragraph(content, body_style))
            
    # Compilar archivo PDF final
    try:
        doc.build(story, canvasmaker=NumberedCanvas)
        logger.info(f"PDF generado de forma exitosa en {pdf_path}")
        return True
    except Exception as e:
        logger.exception("Error al compilar el documento PDF con ReportLab")
        return False

if __name__ == "__main__":
    import sys
    
    # Rutas por defecto o pasadas por argumentos
    if len(sys.argv) >= 3:
        MD_FILE = sys.argv[1]
        PDF_FILE = sys.argv[2]
    else:
        MD_FILE = "docs/informe_final.md"
        PDF_FILE = "reports/final/informe_final.pdf"
    
    # Crear carpeta destino de forma segura
    os.makedirs(os.path.dirname(PDF_FILE), exist_ok=True)
    
    success = build_pdf(MD_FILE, PDF_FILE)
    if success:
        print(f"\n[ÉXITO] Documento PDF compilado correctamente en: {os.path.abspath(PDF_FILE)}")
    else:
        print("\n[ERROR] Ocurrió un error en la generación del PDF. Ver log en config/logging/pdf_generation.log")