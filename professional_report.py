import io
import queue
import threading
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Flowable, HRFlowable, Image, KeepTogether,
                                PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


class GradientBar(Flowable):
    """Custom flowable for a section header with a gradient-like bar."""
    def __init__(self, text, primary_hex, secondary_hex):
        Flowable.__init__(self)
        self.text = text
        self.primary_hex = primary_hex
        self.secondary_hex = secondary_hex
        self.width = 7*inch
        self.height = 0.5*inch
    
    def draw(self):
        # Background
        self.canv.setFillColor(colors.HexColor(self.primary_hex))
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        
        # Accent line
        self.canv.setFillColor(colors.HexColor(self.secondary_hex))
        self.canv.rect(0, 0, self.width, 4, fill=1, stroke=0)
        
        # Text
        self.canv.setFillColor(colors.white)
        self.canv.setFont('Helvetica-Bold', 16)
        self.canv.drawString(15, 0.18*inch, self.text)

class PremiumPlotivaReport:
    """
    Enterprise-grade PDF Report Generator for Plotiva.
    """
    def __init__(self, config):
        self.config = config
        self.buffer = io.BytesIO()
        
        # Page size
        self.pagesize = A4 if config.get('page_size') == 'a4' else LETTER
        
        # Color Palette (Stored as Hex strings for flexibility)
        self.colors = {
            'primary': '#1a2332',
            'secondary': '#4ecdc4',
            'accent': '#ff6b6b',
            'text_main': '#111827',
            'text_light': '#6B7280',
            'background': '#F9FAFB'
        }
        
        # Update colors based on scheme config if needed
        scheme = config.get('color_scheme', 'corporate_blue')
        if scheme == 'modern_teal':
            self.colors['primary'] = '#0F766E'
            self.colors['secondary'] = '#14B8A6'
        elif scheme == 'professional_gray':
            self.colors['primary'] = '#374151'
            self.colors['secondary'] = '#9CA3AF'

        # Styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Content elements
        self.elements = []
        self.plot_counter = 0

    def _setup_custom_styles(self):
        """Defines custom paragraph styles for the report."""
        
        # Helper to safely add or update styles
        def add_or_update_style(style):
            if style.name in self.styles:
                # Update existing style attributes
                existing = self.styles[style.name]
                existing.parent = style.parent
                existing.fontSize = style.fontSize
                existing.leading = style.leading
                existing.spaceAfter = style.spaceAfter
                existing.textColor = style.textColor
                if hasattr(style, 'alignment'):
                    existing.alignment = style.alignment
                if hasattr(style, 'fontName'):
                    existing.fontName = style.fontName
                if hasattr(style, 'backColor'):
                    existing.backColor = style.backColor
                if hasattr(style, 'borderPadding'):
                    existing.borderPadding = style.borderPadding
            else:
                self.styles.add(style)

        # Main Title
        add_or_update_style(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=32,
            leading=40,
            spaceAfter=20,
            textColor=colors.HexColor(self.colors['primary']),
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))

        # Section Header with Background
        add_or_update_style(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            leading=24,
            spaceAfter=12,
            textColor=colors.HexColor(self.colors['primary']),
            fontName='Helvetica-Bold'
        ))
        
        # Subsection
        add_or_update_style(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            leading=18,
            spaceAfter=10,
            textColor=colors.HexColor(self.colors['primary']),
            fontName='Helvetica-Bold'
        ))
        
        add_or_update_style(ParagraphStyle(
            name='NormalJustified',
            parent=self.styles['Normal'],
            alignment=TA_JUSTIFY,
            fontSize=10,
            leading=14,
            spaceAfter=8,
            textColor=colors.HexColor(self.colors['text_main'])
        ))
        
        add_or_update_style(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=13,
            spaceAfter=8,
            textColor=colors.HexColor(self.colors['text_main'])
        ))
        
        add_or_update_style(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor(self.colors['text_light']),
            alignment=TA_CENTER
        ))
        
        add_or_update_style(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=22,
            textColor=colors.HexColor(self.colors['primary']),
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            leading=26
        ))

    def add_dataset_overview(self, overview_data):
        """Adds a professional overview of the dataset."""
        self.add_colored_section("Data Profile Overview")
        
        # Intro intro
        intro = "This report analyzes a dataset containing {rows:,} records and {cols} attributes. The analysis period covers {start_date} to {end_date}, providing a comprehensive view of the underlying trends and patterns.".format(
            rows=overview_data.get('rows', 0),
            cols=overview_data.get('cols', 0),
            start_date=overview_data.get('start_date', 'N/A'),
            end_date=overview_data.get('end_date', 'N/A')
        )
        self.elements.append(Paragraph(intro, self.styles['NormalJustified']))
        self.elements.append(Spacer(1, 0.2*inch))
        
        # Summary Grid
        data = [
            ["TOTAL RECORDS", "ATTRIBUTES", "MISSING RATE", "DUPLICATES"],
            [f"{overview_data.get('rows', 0):,}", str(overview_data.get('cols', 0)), overview_data.get('missing_rate', '0%'), str(overview_data.get('duplicates', 0))]
        ]
        
        # Style the summary table
        t = Table(data, colWidths=[1.8*inch]*4)
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('FONTSIZE', (0,1), (-1,1), 12),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor(self.colors['text_light'])),
            ('TEXTCOLOR', (0,1), (-1,1), colors.HexColor(self.colors['text_main'])),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('linebelow', (0,0), (-1,0), 1, colors.HexColor('#E5E7EB')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 8),
        ]))
        self.elements.append(t)
        self.elements.append(Spacer(1, 0.4*inch))


    def add_colored_section(self, title):
        """Adds a section header with a colored gradient bar."""
        self.elements.append(GradientBar(title, self.colors['primary'], self.colors['secondary']))
        self.elements.append(Spacer(1, 0.2*inch))


    def add_cover_page(self):
        """Adds a professional cover page."""
        # Visual Strip
        self.elements.append(Spacer(1, 1*inch))
        
        # Title Block
        self.elements.append(Paragraph(self.config.get('title', 'Report'), self.styles['Title']))
        
        if self.config.get('subtitle'):
            subtitle_style = ParagraphStyle('Subtitle', parent=self.styles['Normal'], fontSize=16, leading=22, alignment=TA_CENTER, textColor=colors.HexColor(self.colors['text_light']), spaceAfter=30)
            self.elements.append(Paragraph(self.config.get('subtitle'), subtitle_style))
        
        self.elements.append(Spacer(1, 1*inch))
        
        # Divider
        self.elements.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor(self.colors['secondary']), spaceBefore=10, spaceAfter=30))
        
        # Metadata Table (Centered and clean)
        meta_data = []
        if self.config.get('company'):
            meta_data.append([Paragraph(self.config.get('company'), ParagraphStyle('MetaComp', parent=self.styles['Normal'], alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=14))])
        
        if self.config.get('author'):
             meta_data.append([Paragraph(f"Prepared by: {self.config.get('author')}", ParagraphStyle('MetaAuth', parent=self.styles['Normal'], alignment=TA_CENTER, fontSize=12, textColor=colors.HexColor(self.colors['text_light'])))])
        
        meta_data.append([Paragraph(datetime.now().strftime("%B %d, %Y"), ParagraphStyle('MetaDate', parent=self.styles['Normal'], alignment=TA_CENTER, fontSize=11, textColor=colors.HexColor(self.colors['text_light'])))])
        
        t = Table(meta_data, colWidths=[6*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ]))
        self.elements.append(t)
        self.elements.append(PageBreak())

    def add_executive_dashboard(self, kpis):
        """Adds a grid of KPI cards."""
        self.add_colored_section("Executive Dashboard")
        
        if not kpis:
            return

        # Build 2x2 grid
        for i in range(0, len(kpis), 2):
            row_kpis = kpis[i:i+2]
            cards = []
            
            for kpi in row_kpis:
                # Determine color snippet based on change/status
                status_color = colors.HexColor(self.colors['text_main'])
                if kpi.get('status') == 'success': status_color = colors.HexColor('#10B981')
                elif kpi.get('status') == 'warning': status_color = colors.HexColor('#F59E0B')
                elif kpi.get('status') == 'danger': status_color = colors.HexColor('#EF4444')
                
                # KPI Content
                kpi_content = [
                    Paragraph(kpi['name'].upper(), self.styles['MetricLabel']),
                    Paragraph(kpi['value'], self.styles['MetricValue']),
                    Paragraph(f"<font color='{status_color.hexval()}'>{kpi['change']}</font>", self.styles['MetricLabel'])
                ]
                
                # Create a mini table for the card to control border/padding
                card = Table([[c] for c in kpi_content], colWidths=[3*inch])
                card.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
                    ('TOPPADDING', (0,0), (-1,-1), 12),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ]))
                cards.append(card)
            
            # Pad to 2 columns if only 1 card in last row
            if len(cards) < 2:
                cards.append('') # Spacer
            
            # Row table
            row_table = Table([cards], colWidths=[3.5*inch, 3.5*inch])
            row_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'), # Center row content
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ]))
            row_table.hAlign = 'CENTER' # Center the table itself
            self.elements.append(row_table)
            self.elements.append(Spacer(1, 0.4*inch))

    def add_executive_summary(self, summary_text, key_findings=None):
        """Professional executive summary with highlights"""
        
        self.add_colored_section("Executive Summary")
        
        # Summary paragraph with emphasis box
        summary_style = ParagraphStyle(
            name='Summary',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            leading=18,
            textColor=colors.HexColor('#1F2937'),
            backColor=colors.HexColor('#F9FAFB'),
            leftIndent=20, # Indent text inside
            rightIndent=20,
            spaceBefore=10,
            spaceAfter=10,
            borderWidth=1,
            borderColor=colors.HexColor('#D1D5DB'),
            borderPadding=20
        )
        
        summary_para = Paragraph(summary_text, summary_style)
        
        summary_table = Table([[summary_para]], colWidths=[6.5*inch])
        summary_table.hAlign = 'CENTER'
        # Table style for outer container if needed, but per-paragraph styling is cleaner for shading text block
        # Actually ReportLab Paragraph backColor works well.
        
        self.elements.append(summary_table)
        self.elements.append(Spacer(1, 0.3*inch))
        
        # Key findings bullets
        if key_findings:
            self.elements.append(Paragraph("<b>Key Findings:</b>", self.styles['SubSection']))
            self.elements.append(Spacer(1, 0.1*inch))
            
            for finding in key_findings:
                bullet = Paragraph(f"• {finding}", ParagraphStyle(
                    name='Finding',
                    parent=self.styles['Normal'],
                    fontName='Helvetica',
                    fontSize=10,
                    textColor=colors.HexColor('#374151'),
                    leftIndent=30,
                    spaceAfter=8
                ))
                self.elements.append(bullet)
            
            self.elements.append(Spacer(1, 0.3*inch))

    def add_data_summary_table(self, df):
        """Professional data summary with stats"""
        
        self.add_colored_section("Data Overview")
        
        # Summary stats
        summary_data = [
            ['Total Records', f"{len(df):,}"],
            ['Features', f"{len(df.columns)}"],
            ['Date Range', f"{str(df.index.min())} to {str(df.index.max())}" if hasattr(df.index, 'min') else "N/A"],
            ['Completeness', f"{(1 - df.isnull().sum().sum() / df.size) * 100:.1f}%" if df.size > 0 else "0%"],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch], hAlign='LEFT')
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F3F4F6')),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        
        self.elements.append(summary_table)
        self.elements.append(Spacer(1, 0.3*inch))

    def add_page_break(self):
        self.elements.append(PageBreak())

    def add_callout_box(self, text, type='info'):
        """Adds a colored callout box for findings."""
        bg_map = {
            'info': '#EFF6FF',
            'success': '#ECFDF5',
            'warning': '#FFFBEB',
            'danger': '#FEF2F2'
        }
        border_map = {
            'info': '#BFDBFE',
            'success': '#A7F3D0',
            'warning': '#FDE68A',
            'danger': '#FECACA'
        }
        
        bg_color = bg_map.get(type, '#F9FAFB')
        border_color = border_map.get(type, '#E5E7EB')
        
        # Paragraph with word wrapping using splitLongWords
        p_style = ParagraphStyle(
            name=f'Callout_{type}',
            parent=self.styles['Normal'],
            textColor=colors.HexColor('#1F2937'),
            splitLongWords=True
        )
        p = Paragraph(text, p_style)
        
        # Reduced width to safe zone ~6.2 inch to avoid bleeding
        t = Table([[p]], colWidths=[6.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor(border_color)),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
            ('RIGHTPADDING', (0,0), (-1,-1), 15),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        
        self.elements.append(KeepTogether([t, Spacer(1, 0.15*inch)]))

    def analyze_chart_insight(self, fig, df):
        """Generate executive-level insights from chart"""
        
        if not fig.data:
            return "Data visualization showing key performance metrics and trends."
            
        chart_type = fig.data[0].type
        
        try:
            if chart_type == 'scatter':
                # Basic correlation check if data available in fig
                # Note: accessing x/y might be arrays or column names depending on how fig was built
                # We'll try to use the DF if we can map it, but here we can only access fig data directly mostly
                x_data = fig.data[0].x
                y_data = fig.data[0].y
                
                # Filter Nones/NaNs for calculation
                valid_mask = [x is not None and y is not None for x, y in zip(x_data, y_data)]
                if sum(valid_mask) > 1:
                    import pandas as pd

                    # Convert to numeric if possible
                    try: 
                        s_x = pd.to_numeric(pd.Series(x_data)[valid_mask])
                        s_y = pd.to_numeric(pd.Series(y_data)[valid_mask])
                        correlation = s_x.corr(s_y)
                        
                        if abs(correlation) > 0.6:
                            direction = "positive" if correlation > 0 else "negative"
                            return f"Strong {direction} correlation (r={correlation:.2f}) observed, indicating a significant relationship between variables."
                        else:
                            return f"Weak correlation (r={correlation:.2f}) suggests distinct underlying drivers for these metrics."
                    except:
                        pass
                        
            elif chart_type == 'bar':
                # Find top category
                y_data = fig.data[0].y
                x_data = fig.data[0].x
                if y_data is not None and len(y_data) > 0:
                    try:
                         # aggregate if multiple traces? Just take first trace for simplicity in auto-insight
                         max_val = max(y_data)
                         max_idx = list(y_data).index(max_val)
                         cat = x_data[max_idx]
                         return f"Category '{cat}' dominates with highest value of {max_val:,.0f}, outperforming other segments."
                    except:
                         pass

            elif chart_type == 'box':
                # Comparative insight
                return "Distribution analysis highlights variation in median performance and identifies potential outliers across categories."
                
        except Exception as e:
            return "Visual analysis reveals distinctive patterns and performance variance across key segments."
            
        return "Data patterns indicate opportunities for strategic optimization and operational efficiency."

    def add_plot_with_caption(self, fig, title=None, caption=None, size='medium', df=None):
        """Executive Visual Analysis Card with Smart Insights"""
        
        # Auto-generate insight if missing or generic
        if not caption or "generated on" in str(caption).lower() or len(str(caption)) < 10:
             caption = self.analyze_chart_insight(fig, df)

        # Standard size configs
        sizes = {
            'small': (400, 280, 3*inch, 2*inch),
            'medium': (700, 450, 6.5*inch, 4.2*inch), # Expanded for premium look
            'large': (800, 550, 6.5*inch, 4.5*inch),
            'full': (900, 650, 7*inch, 5*inch)
        }
        px_w, px_h, disp_w, disp_h = sizes.get(size, sizes['medium'])

        # Helper: Threaded export with strict timeout
        def run_export(fig_obj):
            import plotly.io as pio
            return pio.to_image(fig_obj, format='png', width=px_w, height=px_h, scale=2.0) # Higher scale for crispness

        def export_with_timeout(fig_obj, timeout=30):
            q = queue.Queue()
            def target():
                try:
                    q.put(run_export(fig_obj))
                except Exception as e:
                    q.put(e)
            
            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            t.join(timeout)
            
            if t.is_alive():
                return TimeoutError("Rendering timed out")
            if q.empty():
                return None
            return q.get()

        # ATTEMPT 1: Threaded with Timeout
        img_bytes = None
        error_msg = "Engine Resource Timeout"
        
        try:
            res = export_with_timeout(fig, timeout=30)
            if isinstance(res, bytes) and len(res) > 100:
                img_bytes = res
            elif isinstance(res, Exception):
                error_msg = str(res)
        except Exception as e:
             error_msg = str(e)

        # RENDER CARD
        if img_bytes:
            # 1. Section Header (Business Style)
            if title:
                # Square bullet for professional feel
                self.elements.append(Paragraph(f"■ {title}", 
                    ParagraphStyle('ChartTitle', parent=self.styles['Heading2'], fontSize=13, textColor=colors.HexColor(self.colors['primary']), spaceAfter=8)))
            
            # 2. Context (Optional) - could be passed in, but we'll skip specific context for now to keep API simple
            # self.elements.append(Paragraph(f"<i>Analysis of key performance indicators...</i>", ParagraphStyle('Ctx', fontSize=9, textColor=colors.HexColor('#6B7280'), spaceAfter=6)))
            
            # 3. Chart with Border
            img = Image(BytesIO(img_bytes), width=disp_w, height=disp_h)
            
            chart_table = Table([[img]], colWidths=[disp_w + 0.2*inch])
            chart_table.setStyle(TableStyle([
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB')), # Subtle border
                ('BACKGROUND', (0,0), (-1,-1), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))
            chart_table.hAlign = 'CENTER'
            self.elements.append(chart_table)
            self.elements.append(Spacer(1, 0.1*inch))
            
            # 4. Insight Box (Teal/Premium)
            insight_html = f"<b>📊 KEY INSIGHT:</b> {caption}"
            insight_para = Paragraph(insight_html, ParagraphStyle('InsightText', parent=self.styles['Normal'], fontSize=10, textColor=colors.HexColor('#0F766E'), leading=14))
            
            insight_table = Table([[insight_para]], colWidths=[disp_w + 0.2*inch])
            insight_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0FDFA')), # Mint cream
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#14B8A6')), # Teal border
                ('TOPPADDING', (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                ('LEFTPADDING', (0,0), (-1,-1), 12),
                ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ]))
            insight_table.hAlign = 'CENTER'
            self.elements.append(insight_table)
             
        else:
            # Fallback
            ph_text = f"<font color='#9CA3AF'>Visualization Unavailable: {error_msg}</font>"
            self.elements.append(Paragraph(ph_text, self.styles['Normal']))

        self.elements.append(Spacer(1, 0.4*inch))

    def add_comprehensive_data_overview(self, df):
        """Adds a professional comprehensive data overview section."""
        self.add_colored_section("Dataset Overview")
        
        # Summary text
        start_period = df.index.min().strftime('%B %Y') if hasattr(df.index, 'min') and isinstance(df.index, pd.DatetimeIndex) else ""
        end_period = df.index.max().strftime('%B %Y') if hasattr(df.index, 'max') and isinstance(df.index, pd.DatetimeIndex) else ""
        
        period_text = f" covering {start_period} to {end_period}" if start_period and end_period else ""
        
        summary = f"""This analysis examines a dataset containing {len(df):,} records across {len(df.columns)} variables{period_text}. 
        Data quality is {(1 - df.isnull().sum().sum() / df.size) * 100:.1f}% complete. 
        The dataset provides a foundation for comprehensive analysis of performance trends and distribution patterns."""
        
        self.elements.append(Paragraph(summary, self.styles['NormalJustified']))
        self.elements.append(Spacer(1, 0.2*inch))
        
        # Key Stats Table (Styled like a data strip)
        # Using a list of (Label, Value) tuples
        stats = [
           ("Total Records", f"{len(df):,}"),
           ("Variables", f"{len(df.columns)}"),
           ("Completeness", f"{(1 - df.isnull().sum().sum() / df.size) * 100:.0f}%"),
           # Try to find a numeric column for average?
        ]
        
        # Build a horizontal stats strip
        data_row = [Paragraph(f"<b>{k}</b><br/><font size=12>{v}</font>", ParagraphStyle('Stat', alignment=TA_CENTER, textColor=colors.HexColor(self.colors['primary']))) for k,v in stats]
        
        t = Table([data_row], colWidths=[1.8*inch]*len(stats))
        t.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F9FAFB')),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        self.elements.append(t)
        self.elements.append(Spacer(1, 0.3*inch))


    def add_recommendations(self, recommendations):
        """Recommendations with priority badges"""
        
        self.add_colored_section("Strategic Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            
            # Priority styling
            priority_map = {
                'high': ('#EF4444', '🔴'),
                'medium': ('#F59E0B', '🟡'),
                'low': ('#10B981', '🟢')
            }
            
            priority = rec.get('priority', 'medium').lower()
            badge_color, badge_icon = priority_map.get(priority, priority_map['medium'])
            
            # Recommendation card
            rec_data = [
                [
                    Paragraph(f"<b>{i}. {rec['title']}</b>", self.styles['Normal']),
                    Paragraph(f"<font color='{badge_color}'><b>{badge_icon} {priority.upper()}</b></font>",
                             ParagraphStyle(f'Badge_{i}', parent=self.styles['Normal'], alignment=TA_RIGHT))
                ],
                [Paragraph(rec['description'], self.styles['NormalJustified']), '']
            ]
            
            # Reduced width to 6.2 to stay within safe printable area (A4 width ~8.27 - 1.5 margins = ~6.7, but need buffer)
            rec_table = Table(rec_data, colWidths=[4.8*inch, 1.4*inch])
            rec_table.setStyle(TableStyle([
                ('SPAN', (0,1), (1,1)),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#E5E7EB')),
                ('BACKGROUND', (0,0), (-1,-1), colors.white),
                ('TOPPADDING', (0,0), (-1,-1), 12),
                ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                ('LEFTPADDING', (0,0), (-1,-1), 15),
                ('RIGHTPADDING', (0,0), (-1,-1), 15),
            ]))
            
            # Use KeepTogether to prevent breaking inside a card
            self.elements.append(KeepTogether([
                rec_table,
                Spacer(1, 0.2*inch)
            ]))

    def add_appendix(self):
        self.elements.append(PageBreak())
        self.elements.append(Paragraph("Appendix", self.styles['SectionHeader']))

    def add_methodology_section(self, methods):
        """Adds methodology list."""
        self.elements.append(Paragraph("Methodology", self.styles['Heading2']))
        
        for m in methods:
            if m.strip():
                self.elements.append(Paragraph(f"• {m}", self.styles['Normal']))
        self.elements.append(Spacer(1, 0.1*inch))

    def _footer(self, canvas, doc):
        """Enhanced footer with branding and safe zones"""
        canvas.saveState()
        
        # Safe zone positioning
        # Stay above 0.7 inch from bottom
        footer_y = 0.5*inch 
        
        # Footer line
        canvas.setStrokeColor(colors.HexColor(self.colors['secondary']))
        canvas.setLineWidth(1)
        canvas.line(0.75*inch, footer_y + 0.15*inch, doc.pagesize[0] - 0.75*inch, footer_y + 0.15*inch)
        
        # Left: Company
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#6B7280'))
        canvas.drawString(0.75*inch, footer_y, 
                         f"{self.config.get('company', 'Plotiva')} | Confidential")
        
        # Center: Date
        canvas.drawCentredString(
            doc.pagesize[0]/2, footer_y,
            f"Generated {datetime.now().strftime('%B %d, %Y')}"
        )
        
        # Right: Page number
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawRightString(
            doc.pagesize[0] - 0.75*inch, footer_y,
            f"Page {doc.page}"
        )
        
        canvas.restoreState()
        
    def _cover_page_template(self, canvas, doc):
        """No header/footer on cover"""

    def add_text(self, text, style='BodyText'):
        """Adds a text paragraph to the report."""
        self.elements.append(Paragraph(text, self.styles[style]))
        self.elements.append(Spacer(1, 0.2*inch))

    def generate(self):
        """Builds the PDF and returns bytes."""
        # Use A4 or Letter
        page_size = A4 if self.config.get('page_size') == 'a4' else LETTER
        
        doc = SimpleDocTemplate(
            self.buffer,
            pagesize=page_size,
            rightMargin=0.75*inch, leftMargin=0.75*inch,
            topMargin=1*inch, bottomMargin=1*inch,
            title=self.config.get('title', 'Report')
        )
        
        # Use onLaterPages to add footer to every page (except cover)
        doc.build(
            self.elements, 
            onFirstPage=self._cover_page_template, 
            onLaterPages=self._footer
        )
        
        self.buffer.seek(0)
        return self.buffer.getvalue()
