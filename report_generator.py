
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


class ProfessionalReportGenerator:
    """
    Enterprise-grade PDF report generator for Plotiva
    """
    
    def __init__(self, config):
        """
        config = {
            'title': str,
            'subtitle': str,
            'author': str,
            'company': str,
            'logo_path': str (optional),
            'page_size': 'letter' or 'A4',
            'font_family': str,
            'primary_color': tuple (R,G,B),
            'accent_color': tuple (R,G,B),
            'include_toc': bool,
            'include_cover': bool,
            'watermark': str (optional)
        }
        """
        self.config = config
        self.buffer = BytesIO()
        self.elements = []
        self.styles = self._create_styles()
        
    def _create_styles(self):
        """Professional typography styles"""
        styles = getSampleStyleSheet()
        
        # Helper to add style safely
        def add_style(name, parent, **kwargs):
            if name in styles:
                return # Already exists
            style = ParagraphStyle(name=name, parent=parent, **kwargs)
            styles.add(style)
        
        # Corporate Title Style
        add_style(
            'CorporateTitle',
            styles['Title'],
            fontName=self.config.get('font_family', 'Helvetica-Bold'),
            fontSize=28,
            textColor=colors.HexColor(self._rgb_to_hex(self.config.get('primary_color', (0,51,102)))),
            spaceAfter=12,
            alignment=TA_CENTER,
            leading=34
        )
        
        # SectionHeader Style
        # Check if 'SectionHeader' already exists (unlikely in fresh sheet but safe)
        if 'SectionHeader' not in styles:
            add_style(
                'SectionHeader',
                styles['Heading1'],
                fontName=self.config.get('font_family', 'Helvetica-Bold'),
                fontSize=18,
                textColor=colors.HexColor(self._rgb_to_hex(self.config.get('primary_color', (0,51,102)))),
                spaceBefore=20,
                spaceAfter=12,
                borderWidth=2,
                borderColor=colors.HexColor(self._rgb_to_hex(self.config.get('accent_color', (0,102,204)))),
                borderPadding=5,
                leftIndent=0
            )

        # Subsection Style
        add_style(
            'SubSection',
            styles['Heading2'],
            fontName=self.config.get('font_family', 'Helvetica-Bold'),
            fontSize=14,
            textColor=colors.HexColor(self._rgb_to_hex(self.config.get('accent_color', (0,102,204)))),
            spaceBefore=14,
            spaceAfter=8
        )
        
        # BodyText Style
        # 'BodyText' comes with getSampleStyleSheet() often, so check first
        if 'BodyText' in styles:
            s = styles['BodyText']
            s.fontName = 'Helvetica'
            s.fontSize = 11
            s.leading = 16
            s.alignment = TA_JUSTIFY
            s.spaceAfter = 10
            s.textColor = colors.HexColor('#333333')
        else:
            add_style(
                'BodyText',
                styles['Normal'],
                fontName='Helvetica',
                fontSize=11,
                leading=16,
                alignment=TA_JUSTIFY,
                spaceAfter=10,
                textColor=colors.HexColor('#333333')
            )
        
        # Executive Summary Style
        add_style(
            'ExecutiveSummary',
            styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            leading=18,
            alignment=TA_JUSTIFY,
            leftIndent=20,
            rightIndent=20,
            spaceBefore=10,
            spaceAfter=10,
            backColor=colors.HexColor('#F5F5F5'),
            borderWidth=1,
            borderColor=colors.HexColor('#CCCCCC'),
            borderPadding=15
        )
        
        # Key Insight Style
        add_style(
            'KeyInsight',
            styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#006400'),
            leftIndent=30,
            bulletIndent=15,
            spaceAfter=8
        )
        
        return styles
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex color"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def add_cover_page(self):
        """Professional cover page with logo and branding"""
        # Logo
        if self.config.get('logo_path'):
            try:
                logo = Image(self.config['logo_path'], width=2*inch, height=1*inch)
                logo.hAlign = 'CENTER'
                self.elements.append(logo)
                self.elements.append(Spacer(1, 0.5*inch))
            except Exception:
                pass # Skip logo if not found
        
        # Title
        title = Paragraph(self.config.get('title', 'Analysis Report'), 
                         self.styles['CorporateTitle'])
        self.elements.append(title)
        self.elements.append(Spacer(1, 0.3*inch))
        
        # Subtitle
        if self.config.get('subtitle'):
            subtitle = Paragraph(f"<i>{self.config['subtitle']}</i>", 
                                self.styles['Heading2'])
            subtitle.alignment = TA_CENTER
            self.elements.append(subtitle)
            self.elements.append(Spacer(1, 0.5*inch))
        
        # Horizontal line
        self._add_horizontal_line(self.config.get('accent_color', (0,102,204)))
        self.elements.append(Spacer(1, 1*inch))
        
        # Metadata table
        metadata = [
            ['Generated By:', self.config.get('author', 'Plotiva Analytics')],
            ['Company:', self.config.get('company', '')],
            ['Date:', datetime.now().strftime('%B %d, %Y')],
            ['Time:', datetime.now().strftime('%I:%M %p')]
        ]
        
        meta_table = Table(metadata, colWidths=[2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        meta_table.hAlign = 'CENTER'
        self.elements.append(meta_table)
        
        # Page break
        self.elements.append(PageBreak())
    
    def add_executive_summary(self, summary_text, key_metrics=None):
        """
        Add executive summary section
        
        summary_text: str - main summary paragraph
        key_metrics: dict - {'Metric Name': 'Value'} for highlight boxes
        """
        # Section header
        self.elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.2*inch))
        
        # Summary text
        summary = Paragraph(summary_text, self.styles['ExecutiveSummary'])
        self.elements.append(summary)
        self.elements.append(Spacer(1, 0.3*inch))
        
        # Key metrics (if provided)
        if key_metrics:
            self.elements.append(Paragraph("Key Metrics", self.styles['SubSection']))
            self.elements.append(Spacer(1, 0.1*inch))
            
            # Create metric cards
            metric_data = [[k, v] for k, v in key_metrics.items()]
            # Split into chunks of 2 for table
            # Actually simplest is just a list
            metric_table = Table(metric_data, colWidths=[3*inch, 3*inch])
            metric_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E8F4F8')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#003366')),
                ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#006699')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ]))
            self.elements.append(metric_table)
        
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_section(self, title, content, level=1):
        """
        Add a content section
        
        title: str
        content: str (can include HTML tags)
        level: 1 (Section) or 2 (Subsection)
        """
        style = self.styles['SectionHeader'] if level == 1 else self.styles['SubSection']
        self.elements.append(Paragraph(title, style))
        self.elements.append(Spacer(1, 0.15*inch))
        
        if content:
            para = Paragraph(content, self.styles['BodyText'])
            self.elements.append(para)
            self.elements.append(Spacer(1, 0.2*inch))
    
    def add_chart(self, fig, title=None, caption=None, width=6*inch, height=4*inch):
        """
        Add Plotly chart to PDF
        
        fig: plotly figure object
        title: str (optional)
        caption: str (optional)
        """
        if title:
            self.elements.append(Paragraph(title, self.styles['SubSection']))
            self.elements.append(Spacer(1, 0.1*inch))
        
        try:
            # Convert Plotly to image
            img_bytes = fig.to_image(format="png", width=800, height=550)
            img_buffer = BytesIO(img_bytes)
            
            # Add to PDF
            img = Image(img_buffer, width=width, height=height)
            img.hAlign = 'CENTER'
            self.elements.append(img)
            
            if caption:
                cap = Paragraph(f"<i>{caption}</i>", self.styles['Normal'])
                cap.alignment = TA_CENTER
                self.elements.append(Spacer(1, 0.05*inch))
                self.elements.append(cap)
            
            self.elements.append(Spacer(1, 0.3*inch))
            
        except Exception as e:
            # Fallback if image conversion fails
            self.elements.append(Paragraph(f"[Chart Image Placeholder - conversion failed: {str(e)}]", self.styles['BodyText']))

    def add_table(self, data, headers=None, title=None, col_widths=None):
        """
        Add professional data table
        
        data: list of lists
        headers: list of column names
        title: str (optional)
        col_widths: list of widths (optional)
        """
        if title:
            self.elements.append(Paragraph(title, self.styles['SubSection']))
            self.elements.append(Spacer(1, 0.1*inch))
        
        # Prepare table data
        if headers:
            table_data = [headers] + data
        else:
            table_data = data
        
        if not table_data:
            return

        # Create table
        if col_widths:
            table = Table(table_data, colWidths=col_widths)
        else:
            table = Table(table_data)
        
        # Professional styling
        style_commands = [
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9F9F9')]),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]
        
        table.setStyle(TableStyle(style_commands))
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.3*inch))
    
    def add_key_insights(self, insights_list):
        """
        Add bulleted key insights with special formatting
        
        insights_list: list of strings
        """
        self.elements.append(Paragraph("Key Insights", self.styles['SubSection']))
        self.elements.append(Spacer(1, 0.1*inch))
        
        for insight in insights_list:
            bullet = Paragraph(f"• {insight}", self.styles['KeyInsight'])
            self.elements.append(bullet)
        
        self.elements.append(Spacer(1, 0.2*inch))
    
    def _add_horizontal_line(self, color=(0,102,204), thickness=2):
        """Add decorative horizontal line"""
        from reportlab.platypus import HRFlowable
        hr = HRFlowable(
            width="80%",
            thickness=thickness,
            color=colors.HexColor(self._rgb_to_hex(color)),
            spaceAfter=10,
            spaceBefore=10,
            hAlign='CENTER'
        )
        self.elements.append(hr)
    
    def add_page_break(self):
        """Add page break"""
        self.elements.append(PageBreak())
    
    def generate(self):
        """
        Generate the PDF and return bytes
        
        Returns:
            bytes: PDF file content
        """
        # Page size
        page_size = A4 if self.config.get('page_size') == 'A4' else letter
        
        # Create PDF document
        doc = SimpleDocTemplate(
            self.buffer,
            pagesize=page_size,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title=self.config.get('title', 'Report')
        )
        
        # Build PDF with custom header/footer
        doc.build(
            self.elements,
            onFirstPage=self._create_page_template,
            onLaterPages=self._create_page_template
        )
        
        # Get PDF bytes
        self.buffer.seek(0)
        return self.buffer.getvalue()
    
    def _create_page_template(self, canvas, doc):
        """Create header and footer for each page"""
        canvas.saveState()
        
        # Footer
        footer_text = f"Generated by Plotiva Analytics | {datetime.now().strftime('%B %d, %Y')}"
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawCentredString(
            doc.pagesize[0] / 2,
            0.5 * inch,
            footer_text
        )
        
        # Page number
        page_num = f"Page {doc.page}"
        canvas.drawRightString(
            doc.pagesize[0] - 0.75 * inch,
            0.5 * inch,
            page_num
        )
        
        # Watermark (if specified)
        if self.config.get('watermark'):
            canvas.setFont('Helvetica-Bold', 60)
            canvas.setFillColor(colors.HexColor('#F0F0F0'))
            canvas.saveState()
            canvas.translate(doc.pagesize[0] / 2, doc.pagesize[1] / 2)
            canvas.rotate(45)
            canvas.drawCentredString(0, 0, self.config['watermark'])
            canvas.restoreState()
        
        canvas.restoreState()
