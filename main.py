

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Dependency Management & Imports
# -----------------------------------------------------------------------------
# Try to import premium modules, handle failures gracefully
try:
    from dashboard_tab import render_dashboard_tab
    from premium_ab_testing_tab import render_ab_testing_tab
    from premium_analytics import PremiumAnalytics
    from premium_business_metrics import render_business_metrics_tab
    from premium_clustering_tab import render_clustering_tab
    from premium_cohort_tab import render_cohort_analysis_tab
    from premium_config import (PREMIUM_COLOR_PALETTES, PREMIUM_CSS,
                                PREMIUM_THEMES)
    from premium_dashboard_builder_tab import render_dashboard_builder_tab
    from premium_feature_engineering import render_data_processing_tab
    from premium_gallery import render_advanced_gallery
    from premium_insights import render_insights_tab
    from premium_ml_tab import render_ml_tab
    from premium_plots import PremiumPlotGenerator
    from premium_report import render_report_tab
    from premium_scenario_tab import render_scenario_tab
    # NEW MODULES
    from premium_statistics import render_statistics_tab
    from premium_time_series_tab import render_time_series_tab
    from premium_visualization_tab import render_premium_visualization_tab
    from premium_data_diagnosis import render_data_diagnosis_tab
    from utils import (calculate_data_quality_metrics, generate_sample_data,
                       get_quality_label, load_file_cached)
    
    PREMIUM_MODULES_LOADED = True
except ImportError as e:
    PREMIUM_MODULES_LOADED = False
    st.error(f"Critical Error: Failed to load premium modules. {str(e)}")

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Premium Data Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium CSS
if PREMIUM_MODULES_LOADED:
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
else:
    # Fallback CSS
    st.markdown("""
    <style>
        .stApp { background-color: #f8fafc; }
        .main-header { color: #1e293b; font-size: 2rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Application Class
# -----------------------------------------------------------------------------
class PremiumDataApp:
    """
    The Main Premium Data Analysis Application.
    Integrates all advanced features into a cohesive interface.
    """
    
    def __init__(self):
        self.initialize_session_state()
        if PREMIUM_MODULES_LOADED:
            self.plot_generator = PremiumPlotGenerator()
            self.analytics = PremiumAnalytics()
    
    def apply_theme(self):
        """Apply selected theme CSS variables"""
        if 'theme' not in st.session_state:
            st.session_state.theme = 'dark'
        
        # Get theme data
        theme_key = st.session_state.theme
        theme_config = PREMIUM_THEMES.get(theme_key, PREMIUM_THEMES['dark'])
        
        # Generating CSS variables dynamically
        css = f"""
        <style>
            :root {{
                --primary: {theme_config['primary_color']} !important;
                --primary-hover: {theme_config['primary_color']}dd !important;
                --accent: {theme_config['accent_color']} !important;
                --warm-coral: {theme_config['secondary_color']} !important;
                --bg-color: {theme_config['background']} !important;
                --card-bg: {theme_config['card_background']} !important;
                --text-main: {theme_config['text_primary']} !important;
                --text-secondary: {theme_config['text_secondary']} !important;
                --text-muted: {theme_config['text_secondary']}99 !important;
                --border: {theme_config['border_color']} !important;
                --shadow-md: {theme_config['shadow']} !important;
            }}
            
            /* Apply background to main app container */
            .stApp {{
                background-color: var(--bg-color);
                color: var(--text-main);
            }}
            
            /* Sidebar background fix */
            section[data-testid="stSidebar"] {{
                background-color: var(--bg-color);
                border-right: 1px solid var(--border);
            }}
            
            /* Input fields background fix */
            .stTextInput > div > div, .stSelectbox > div > div, .stNumberInput > div > div {{
                background-color: var(--card-bg) !important;
                color: var(--text-main) !important;
                border: 1px solid var(--border) !important;
            }}
            
            /* Text color fixes for inputs */
            input, .stSelectbox div[data-baseweb="select"] span {{
                color: var(--text-main) !important;
            }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'uploaded_file_name': None,
            'working_data': pd.DataFrame(),
            'original_data': pd.DataFrame(),
            'theme': 'dark',
            'selected_palette': 'aurora',
            'plot_history': [],
            'analytics_results': {},
            'ml_results': {},
            'dashboard_charts': {},
            'current_main_tab': 'Analyze', # Default
            'current_sub_tab': 'Overview',
            'active_filters': {}, 
            'show_filters': False,
            'data_quality': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_smart_suggestions(self):
        """Render AI-driven next step suggestions in sidebar"""
        if st.session_state.working_data.empty:
            return

        st.markdown('<div class="control-panel-card animate-enter delay-200" style="margin-bottom: 1rem; border: 1px solid var(--accent);">', unsafe_allow_html=True)
        st.markdown('<div class="filter-header" style="color: var(--accent);">💡 NEXT BEST ACTION</div>', unsafe_allow_html=True)
        
        # Heuristic Rule Engine
        df = st.session_state.working_data
        missing_count = df.isnull().sum().sum()
        
        # 1. Quality Issues
        if missing_count > 0:
            st.info(f"🚫 {missing_count} missing values detected.")
            if st.button("Repair Data", key="sugg_repair", use_container_width=True):
                st.session_state.current_main_tab = "Analyze"
                st.session_state.current_sub_tab = "Diagnosis"
                st.rerun()
        
        # 2. Empty Dashboard
        elif 'dashboard_charts' not in st.session_state or not st.session_state.dashboard_charts:
             st.success("✅ Data looks clean!")
             if st.button("Generate Dashboard", key="sugg_dash", use_container_width=True):
                 st.session_state.current_main_tab = "Analyze"
                 st.session_state.current_sub_tab = "Overview"
                 st.rerun()

        # 3. No Machine Learning Model
        elif 'ml_results' not in st.session_state or not st.session_state.ml_results.get('history'):
            st.info("🤖 No predictive models trained.")
            if st.button("Build ML Model", key="sugg_ml", use_container_width=True):
                 st.session_state.current_main_tab = "Predict"
                 st.session_state.current_sub_tab = "Intelligence"
                 st.rerun()
        
        # 4. Report
        else:
             st.success("🚀 Analysis mature.")
             if st.button("Create Report", key="sugg_report", use_container_width=True):
                 st.session_state.current_main_tab = "Report"
                 st.session_state.current_sub_tab = "Reporting"
                 st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file"""
        try:
            file_content = uploaded_file.getvalue()
            df = load_file_cached(file_content, uploaded_file.name)
            
            if df is None:
                st.error("Unsupported file format or error reading file")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def render_sidebar(self):
        """Render premium sidebar - Data Controls & Settings Only"""
        with st.sidebar:
            # 1. Branding
            st.markdown('''
<div class="sidebar-header animate-enter">
    <h2 style="font-family: 'Playfair Display', serif; font-size: 2.5rem; margin: 0; color: var(--text-main); letter-spacing: -0.02em;">Plotiva</h2>
    <div style="height: 2px; width: 40px; background: var(--accent); margin: 0.5rem auto; border-radius: 2px;"></div>
    <p style="font-family: 'Inter', sans-serif; font-size: 0.65rem; color: var(--text-secondary); letter-spacing: 0.25em; text-transform: uppercase; margin-top: 0.5rem;">Unified Intelligence Engine</p>
</div>
''', unsafe_allow_html=True)
            
            # 2. Dataset Control
            self.render_smart_suggestions() # Add suggestions here
            st.markdown('<div class="control-panel-card animate-enter delay-100">', unsafe_allow_html=True)
            st.markdown('<div class="filter-header">💾 DATA SOURCE</div>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload New Data",
                type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                if st.session_state.uploaded_file_name != uploaded_file.name:
                    with st.spinner("Ingesting data..."):
                        df = self.load_data(uploaded_file)
                        if df is not None:
                            st.session_state.uploaded_file_name = uploaded_file.name
                            st.session_state.original_data = df.copy()
                            st.session_state.working_data = df.copy()
                            st.session_state.active_filters = {} 
                            try:
                                quality_metrics = calculate_data_quality_metrics(df)
                                st.session_state.data_quality = quality_metrics
                            except Exception:
                                st.session_state.data_quality = None
                            st.toast("Dataset uploaded and processed", icon="✅")
            
            # Show current file info if listed
            if not st.session_state.working_data.empty:
                rows, cols = st.session_state.working_data.shape
                file_name = st.session_state.uploaded_file_name or "Unknown Source"
                quality_score = st.session_state.get('data_quality', {}).get('overall_score', 0) if st.session_state.get('data_quality') else 0
                quality_label = get_quality_label(quality_score)
                quality_color = "#10B981" if quality_score > 0.8 else "#F59E0B" if quality_score > 0.5 else "#EF4444"
                
                st.markdown(f'''
<div style="margin-top: 0.75rem; font-size: 0.75rem; color: var(--text-muted); display: flex; flex-direction: column; gap: 0.25rem;">
    <div style="display:flex; justify-content:space-between;"><span>📄 Source:</span> <span style="font-weight:600; color:var(--text-main);">{file_name[:18] + '...' if len(file_name) > 18 else file_name}</span></div>
    <div style="display:flex; justify-content:space-between;"><span>📊 Dimensions:</span> <span>{rows:,} rows × {cols} cols</span></div>
    <div style="display:flex; justify-content:space-between;"><span>💾 Memory:</span> <span>{st.session_state.working_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB</span></div>
    <div style="display:flex; justify-content:space-between; margin-top: 0.25rem; padding-top: 0.25rem; border-top: 1px dashed var(--border);">
        <span>🛡️ Data Quality:</span> 
        <span style="font-weight:600; color:{quality_color};">{quality_label} ({int(quality_score*100)}%)</span>
    </div>
</div>
''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
             
            # Demo Data Link
            if st.session_state.working_data.empty:
                if st.button("⚡ Load Demo Dataset", use_container_width=True, type="secondary"):
                     df = generate_sample_data('sales')
                     st.session_state.working_data = df
                     st.session_state.original_data = df.copy()
                     st.session_state.uploaded_file_name = "Sample Sales Data"
                     st.session_state.active_filters = {}
                     st.rerun()

            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            
            # 3. Data Filtering
            if not st.session_state.working_data.empty:
                st.markdown('<div class="animate-enter delay-200">', unsafe_allow_html=True)
                st.markdown('''
<div style="margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
    <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; color: var(--text-main); font-weight: 600;">Refinement Engine</div>
    <div style="background: var(--accent); width: 24px; height: 1px;"></div>
</div>
''', unsafe_allow_html=True)
                
                active_count = len(st.session_state.active_filters)
                if active_count > 0:
                    st.markdown(f'''
<div style="margin-bottom: 1.5rem; font-family: 'Inter', sans-serif; font-size: 0.8rem; color: var(--text-secondary); display: flex; align-items: center; gap: 0.5rem;">
    <span style="width: 8px; height: 8px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 8px var(--accent);"></span>
    <span>Active Context: <strong>{active_count}</strong> constraints applied</span>
</div>
''', unsafe_allow_html=True)

                df_to_filter = st.session_state.working_data
                filterable_cols = sorted(df_to_filter.columns.tolist())
                
                def on_add_constraint():
                    selected = st.session_state.get('add_constraint_selector')
                    if selected and selected != "⊕ Add Constraint...":
                        if selected not in st.session_state.active_filters:
                            current_df = st.session_state.working_data
                            if pd.api.types.is_numeric_dtype(current_df[selected]):
                                min_val = float(current_df[selected].min())
                                max_val = float(current_df[selected].max())
                                st.session_state.active_filters[selected] = (min_val, max_val)
                            else:
                                unique_vals = current_df[selected].unique().tolist()
                                st.session_state.active_filters[selected] = unique_vals
                        st.session_state.add_constraint_selector = "⊕ Add Constraint..."

                st.selectbox(
                    "Add Filter Parameter", 
                    ["⊕ Add Constraint..."] + filterable_cols, 
                    label_visibility="collapsed",
                    key="add_constraint_selector",
                    on_change=on_add_constraint
                )
                
                if st.session_state.active_filters:
                    filter_keys = list(st.session_state.active_filters.keys())
                    
                    for col in filter_keys:
                        if col not in st.session_state.active_filters: continue
                        current_val = st.session_state.active_filters[col]
                        is_numeric = isinstance(current_val, tuple)
                        icon = "🔢" if is_numeric else "🔤"
                        
                        st.markdown(f'''
<div class="control-panel-card" style="padding: 1rem; border-left: 3px solid var(--accent);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-main); display: flex; align-items: center; gap: 0.5rem;">
            <span style="opacity: 0.7;">{icon}</span> {col}
        </div>
''', unsafe_allow_html=True)
                        
                        if is_numeric:
                            min_limit = float(st.session_state.working_data[col].min())
                            max_limit = float(st.session_state.working_data[col].max())
                            if min_limit == max_limit:
                                st.caption("Invariant Feature")
                            else:
                                start, end = st.slider(f"Range for {col}", min_limit, max_limit, current_val, label_visibility="collapsed", key=f"filter_{col}")
                                if col in st.session_state.active_filters: st.session_state.active_filters[col] = (start, end)
                        else:
                            options = st.session_state.working_data[col].unique().tolist()
                            selected = st.multiselect(f"Values for {col}", options, default=current_val, label_visibility="collapsed", key=f"filter_{col}")
                            if col in st.session_state.active_filters: st.session_state.active_filters[col] = selected
                        
                        st.button("Remove Constraint", key=f"rm_{col}", use_container_width=True, on_click=lambda k=col: st.session_state.active_filters.pop(k, None))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.button("↺ Reset Canvas", type="primary", use_container_width=True, on_click=lambda: st.session_state.active_filters.clear())
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            
            # 4. Knowledge Base
            from premium_education import render_education_module
            render_education_module()

            # 5. Global Settings
            with st.expander("⚙️ SYSTEM PREFERENCES", expanded=False):
                 st.markdown("##### Visual Theme")
                 # Theme Toggle Logic
                 dark_mode = st.toggle("Dark Mode", value=(st.session_state.theme == 'dark'))
                 new_theme = 'dark' if dark_mode else 'light'
                 
                 if new_theme != st.session_state.theme:
                     st.session_state.theme = new_theme
                     st.rerun()
                
                 if PREMIUM_MODULES_LOADED:
                     st.markdown("##### Analytics Palette")
                     st.session_state.selected_palette = st.selectbox(
                        "Color Scheme",
                        options=list(PREMIUM_COLOR_PALETTES.keys()),
                        format_func=lambda x: x.replace('_', ' ').title(),
                        index=list(PREMIUM_COLOR_PALETTES.keys()).index(st.session_state.selected_palette),
                        label_visibility="collapsed"
                     )
            
            st.markdown('''
<div style="margin-top: 4rem; text-align: center;">
    <div style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 4px 12px; background: rgba(16, 185, 129, 0.1); border-radius: 100px; margin-bottom: 2rem;">
        <span class="pulse-animation" style="width: 6px; height: 6px; background: #10B981; border-radius: 50%;"></span>
        <span style="font-size: 0.7rem; color: #10B981; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Engine Active</span>
    </div>
    <div style="opacity: 0.6;">
        <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; margin-bottom: 0.2rem;">Plotiva</div>
        <div style="font-size: 0.6rem; color: var(--text-secondary); letter-spacing: 0.1em; text-transform: uppercase;">
            Refined Analytics • Privacy First<br>
            &copy; 2026 Plotiva Inc.
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply active filters to the dataframe"""
        if not st.session_state.active_filters: return df
        filtered_df = df.copy()
        for col, criterion in st.session_state.active_filters.items():
            if col not in filtered_df.columns: continue
            if isinstance(criterion, tuple):
                filtered_df = filtered_df[(filtered_df[col] >= criterion[0]) & (filtered_df[col] <= criterion[1])]
            elif isinstance(criterion, list):
                if not criterion: continue
                filtered_df = filtered_df[filtered_df[col].isin(criterion)]
        return filtered_df

    def render_top_navigation(self):
        """Render new top-level navigation bar with sub-tabs per section"""
        main_tabs = ["Analyze", "Predict", "Optimize", "Report"]
        if 'current_main_tab' not in st.session_state: st.session_state.current_main_tab = "Analyze"
        
        st.markdown('<div style="display: flex; gap: 1rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
        cols = st.columns(len(main_tabs) + 2)
        for i, tab in enumerate(main_tabs):
            is_active = (st.session_state.current_main_tab == tab)
            with cols[i]:
                 if st.button(tab, key=f"main_nav_{tab}", use_container_width=True, type="primary" if is_active else "secondary"):
                     st.session_state.current_main_tab = tab
                     st.session_state.current_sub_tab = "Default"
                     st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        current_main = st.session_state.current_main_tab
        sub_tabs = {}
        if current_main == "Analyze":
            sub_tabs = {
                "Overview": "🏠 Dashboard",
                "Diagnosis": "🔍 Quality Audit",
                "Refinement": "🛠️ Data Prep",
                "Exploration": "📈 Visualization",
                "Statistics": "🧪 Stat Lab",
                "Cohorts": "👥 Cohort Analysis"
            }
        elif current_main == "Predict":
            sub_tabs = {
                "Intelligence": "🤖 Supervised ML",
                "Clustering": "🧩 Clustering & Segments",
                "TimeSeries": "⏳ Time Series",
                "Scenario": "🎲 Sensitivity & Sim"
            }
        elif current_main == "Optimize":
            sub_tabs = {
                "Experimentation": "📐 A/B Testing",
                "Business": "📊 Business Metrics"
            }
        elif current_main == "Report":
            sub_tabs = {
                "Builder": "🔨 Dashboard Builder",
                "Reporting": "📄 Report Studio"
            }

        if sub_tabs:
            st.markdown("---")
            sub_cols = st.columns(len(sub_tabs))
            if st.session_state.current_sub_tab == "Default" or st.session_state.current_sub_tab not in sub_tabs:
                 st.session_state.current_sub_tab = list(sub_tabs.keys())[0]

            for i, (key, label) in enumerate(sub_tabs.items()):
                is_sub_active = (st.session_state.current_sub_tab == key)
                with sub_cols[i]:
                    if st.button(label, key=f"sub_nav_{key}", use_container_width=True, type="primary" if is_sub_active else "secondary"):
                        st.session_state.current_sub_tab = key
                        st.rerun()
            st.markdown("---")

    def render_data_diagnosis(self, df: pd.DataFrame):
        """Render data quality diagnosis tab using premium module"""
        render_data_diagnosis_tab(df)

    def render_main_content(self, df: pd.DataFrame):
        """Main content router with top navigation"""
        if st.session_state.working_data.empty:
            self.render_welcome_screen()
            return
            
        self.render_top_navigation()
        
        if len(df) != len(st.session_state.working_data):
            st.markdown(f"""
            <div style="background-color: #FEF3C7; color: #92400E; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #FCD34D; display: flex; align-items: center; justify-content: space-between;">
                <span>⚡ <strong>Filtered View Active</strong>: Showing {len(df):,} of {len(st.session_state.working_data):,} records.</span>
                <span style="font-size: 0.8rem;">(Analysis limited to current selection)</span>
            </div>
            """, unsafe_allow_html=True)
        
        sub_tab = st.session_state.current_sub_tab
        
        # Analyze
        if sub_tab == "Overview":
            render_dashboard_tab(df)
        elif sub_tab == "Diagnosis":
            self.render_data_diagnosis(df)
        elif sub_tab == "Refinement":
            if len(df) != len(st.session_state.working_data):
                st.info("ℹ️ Note: Data Refinement operations always apply to the FULL dataset.")
            render_data_processing_tab(st.session_state.working_data)
        elif sub_tab == "Exploration":
            exp_tabs = st.tabs(["📊 Interactive Plots", "🎨 Advanced Gallery", "🧠 AI Insights"])
            with exp_tabs[0]: render_premium_visualization_tab(df)
            with exp_tabs[1]: render_advanced_gallery(df)
            with exp_tabs[2]: render_insights_tab(df)
        elif sub_tab == "Statistics":
            render_statistics_tab(df)
        elif sub_tab == "Cohorts":
            render_cohort_analysis_tab(df)
            
        # Predict
        elif sub_tab == "Intelligence":
             render_ml_tab(df, self.analytics)
        elif sub_tab == "Clustering":
             render_clustering_tab(df)
        elif sub_tab == "TimeSeries":
             render_time_series_tab(df)
        elif sub_tab == "Scenario":
             render_scenario_tab(df)

        # Optimize
        elif sub_tab == "Experimentation":
            render_ab_testing_tab(df)
        elif sub_tab == "Business":
            render_business_metrics_tab()
            
        # Report
        elif sub_tab == "Builder":
            render_dashboard_builder_tab(df)
        elif sub_tab == "Reporting":
            render_report_tab()
            
    def load_case_study(self, case_key: str, display_name: str, icon: str):
        """Helper to load case study data"""
        with st.spinner(f"Loading {display_name}..."):
            df = generate_sample_data(case_key)
            st.session_state.working_data = df
            st.session_state.original_data = df.copy()
            st.session_state.uploaded_file_name = display_name
            st.session_state.active_filters = {}
            st.toast(f"{display_name} loaded!", icon=icon)
            st.rerun()

    def render_welcome_screen(self):
        """Render premium welcome screen"""
        st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
        st.markdown('''
<div class="hero-container animate-slide-up">
<div class="hero-tag"><span style="font-size: 1rem;">✨</span> Enterprise Analytics Suite</div>
<h1 class="hero-title" style="color: #FFFFFF !important;">Transform Raw Data.<br><span style="font-weight: 300; color: #E2E8F0 !important;">Into Competitive Strategy.</span></h1>
<p class="hero-subtitle">Secure, local-first intelligence for data-driven teams.<br>Analyze, visualize, and report with studio-grade precision.</p>
<div style="display: flex; gap: 1rem; margin-bottom: 3rem; justify-content: center; flex-wrap: wrap;">
<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; background: rgba(120, 120, 120, 0.1); border: 1px solid var(--border); border-radius: 8px;"><span style="color: var(--accent);">🔒</span> <span style="font-weight: 500; font-size: 0.9rem;">Secure & Local</span></div>
<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; background: rgba(120, 120, 120, 0.1); border: 1px solid var(--border); border-radius: 8px;"><span style="color: var(--warm-coral);">⚡</span> <span style="font-weight: 500; font-size: 0.9rem;">Instant Insights</span></div>
<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; background: rgba(120, 120, 120, 0.1); border: 1px solid var(--border); border-radius: 8px;"><span style="color: #F59E0B;">🎨</span> <span style="font-weight: 500; font-size: 0.9rem;">Production Ready</span></div>
</div></div>
''', unsafe_allow_html=True)

        st.markdown('<div class="animate-fade-in delay-200" style="max-width: 1000px; margin: 0 auto;">', unsafe_allow_html=True)
        
        # LUXURY PREVIEW CONTAINER
        st.markdown("""
<div style="background: var(--card-bg); border-radius: 16px; box-shadow: 0 20px 50px -12px rgba(0,0,0,0.5); overflow: hidden; border: 1px solid var(--border); position: relative;">
<div style="background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(12px); padding: 0.75rem 1.5rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between;">
<div style="display: flex; gap: 0.6rem;">
<div style="width: 12px; height: 12px; border-radius: 50%; background: #EF4444; box-shadow: 0 0 8px rgba(239,68,68,0.4);"></div>
<div style="width: 12px; height: 12px; border-radius: 50%; background: #F59E0B; box-shadow: 0 0 8px rgba(245,158,11,0.4);"></div>
<div style="width: 12px; height: 12px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px rgba(16,185,129,0.4);"></div>
</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-secondary); letter-spacing: 0.05em; opacity: 0.8;">
<span style="color: #3B82F6;">~/plotiva</span>/core/engine_v2
</div>
<div style="display: flex; align-items: center; gap: 0.5rem; background: rgba(16, 185, 129, 0.1); padding: 4px 12px; border-radius: 100px; border: 1px solid rgba(16, 185, 129, 0.2);">
<span class="pulse-animation" style="width: 6px; height: 6px; background: #10B981; border-radius: 50%;"></span>
<span style="font-size: 0.75rem; color: #10B981; font-weight: 600; font-family: 'Inter', sans-serif;">TURBO BOOST: ACTIVE</span>
</div>
</div>
<div style="padding: 2.5rem; background: linear-gradient(180deg, var(--card-bg) 0%, rgba(15, 23, 42, 0) 100%);">
""", unsafe_allow_html=True)
        
        c_nav, c_display = st.columns([1, 2.5], gap="large")
        
        with c_nav:
            st.markdown('<div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 1rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;">Active Module</div>', unsafe_allow_html=True)
            demo_mode = st.radio(
                "Module Selection", 
                ["📊 Live Visualization", "🧠 Predictive Engine", "🛡️ Quality Sentinel"], 
                label_visibility="collapsed"
            )
            
            st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
            
            # Technical Specs Container
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 1rem; border: 1px solid var(--border);">
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">System Metrics</div>
            """, unsafe_allow_html=True)
            
            if "Visualization" in demo_mode:
                 st.markdown("""
                 <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;"><span style="color: var(--text-secondary); font-size: 0.8rem;">Renderer:</span> <span style="color: #4ECDC4; font-family: 'JetBrains Mono'; font-size: 0.8rem;">WebGL 2.0</span></div>
                 <div style="display: flex; justify-content: space-between;"><span style="color: var(--text-secondary); font-size: 0.8rem;">Latency:</span> <span style="color: #10B981; font-family: 'JetBrains Mono'; font-size: 0.8rem;">12ms</span></div>
                 """, unsafe_allow_html=True)
            elif "Predictive" in demo_mode:
                 st.markdown("""
                 <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;"><span style="color: var(--text-secondary); font-size: 0.8rem;">Model:</span> <span style="color: #F59E0B; font-family: 'JetBrains Mono'; font-size: 0.8rem;">GBM v4</span></div>
                 <div style="display: flex; justify-content: space-between;"><span style="color: var(--text-secondary); font-size: 0.8rem;">Optimizer:</span> <span style="color: #10B981; font-family: 'JetBrains Mono'; font-size: 0.8rem;">Bayesian</span></div>
                 """, unsafe_allow_html=True)
            else:
                 st.markdown("""
                 <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;"><span style="color: var(--text-secondary); font-size: 0.8rem;">Scanner:</span> <span style="color: #3B82F6; font-family: 'JetBrains Mono'; font-size: 0.8rem;">DeepHeuristic</span></div>
                 <div style="display: flex; justify-content: space-between;"><span style="color: var(--text-secondary); font-size: 0.8rem;">Coverage:</span> <span style="color: #10B981; font-family: 'JetBrains Mono'; font-size: 0.8rem;">100%</span></div>
                 """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        with c_display:
            st.markdown('<div style="height: 100%; display: flex; flex-direction: column; justify-content: center;">', unsafe_allow_html=True)
            
            if "Visualization" in demo_mode:
                # 1. Visualization Demo (More Premium)
                df_dummy = pd.DataFrame(np.random.randn(100, 2), columns=['Sales', 'Profit'])
                fig = px.scatter(df_dummy, x='Sales', y='Profit', 
                               color_discrete_sequence=['#4ECDC4'], 
                               title="<b>Revenue Efficiency Analysis</b> <br><sup style='color:gray'>Real-time data stream simulation</sup>")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    margin=dict(l=20, r=20, t=50, b=20), 
                    height=320,
                    xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
            elif "Predictive" in demo_mode:
                # 2. ML Demo (Better Chart)
                fi_data = pd.DataFrame({'Feature': ['Customer_Age', 'Tenure', 'Account_Balance', 'Num_Products', 'Credit_Score'], 'Importance': [0.35, 0.25, 0.2, 0.15, 0.05]})
                fig = px.bar(fi_data, x='Importance', y='Feature', orientation='h', 
                           color='Importance', color_continuous_scale='Viridis', 
                           title="<b>Predictive Feature Importance</b><br><sup style='color:gray'>AutoML Model #42 (Random Forest)</sup>")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    margin=dict(l=20, r=20, t=50, b=20), 
                    height=320,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
            else:
                # 3. Data Audit (More structured)
                c1, c2, c3 = st.columns(3)
                with c1: 
                    st.markdown('<div style="text-align: center;"><div style="font-size: 2rem; font-weight: 700; color: #10B981;">98%</div><div style="font-size: 0.8rem; color: var(--text-secondary);">Health Score</div></div>', unsafe_allow_html=True)
                with c2: 
                    st.markdown('<div style="text-align: center;"><div style="font-size: 2rem; font-weight: 700; color: #F59E0B;">142</div><div style="font-size: 0.8rem; color: var(--text-secondary);">Null Values</div></div>', unsafe_allow_html=True)
                with c3: 
                    st.markdown('<div style="text-align: center;"><div style="font-size: 2rem; font-weight: 700; color: #3B82F6;">0</div><div style="font-size: 0.8rem; color: var(--text-secondary);">Duplicates</div></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("##### 🛡️ Live Audit Stream")
                st.markdown("""
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-secondary);">
                    <div style="margin-bottom: 4px;"><span style="color: #10B981;">[SUCCESS]</span> Scanned 15,000 records in 0.04s</div>
                    <div style="margin-bottom: 4px;"><span style="color: #F59E0B;">[WARN]</span> Column 'Age' contains outlier values (> 3 std dev)</div>
                    <div style="margin-bottom: 4px;"><span style="color: #3B82F6;">[INFO]</span> Imputing missing values with 'Median' strategy</div>
                    <div style="margin-bottom: 4px;"><span style="color: #10B981;">[READY]</span> Dataset normalized and ready for modeling</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True) # End Main Content
        
        # Footer of terminal
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.4); border-top: 1px solid var(--border); padding: 0.5rem 1rem; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.7rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">Status: READY</div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">v2.4.0-stable</div>
        </div>
        </div>
        """, unsafe_allow_html=True) # End Container
        
        # INITIALIZE BUTTON
        st.markdown('<div style="margin-top: 4rem; text-align: center; margin-bottom: 4rem;">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-bottom: 1rem; font-weight: 400;">Initialize Workspace</h3>', unsafe_allow_html=True)
        
        _, col_btn, _ = st.columns([1, 1.5, 1])
        with col_btn:
            if st.button("🚀 Initialize Project Workspace", type="primary", use_container_width=True):
                st.toast("Access the Sidebar on the left to upload your dataset.", icon="⬅️")
                st.balloons()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="margin-top: 6rem; margin-bottom: 2rem; text-align: center;"><span class="phase-tag" style="background: rgba(37, 99, 235, 0.1); color: #3B82F6;">INDUSTRY TEMPLATES</span><h2 style="font-size: 2.2rem; margin-top: 1rem; color: var(--text-main);">Jumpstart Your Analysis</h2><p style="color: var(--text-secondary); max-width: 600px; margin: 0.5rem auto;">Select a pre-configured workspace to explore Plotiva\'s capabilities with real-world scenarios.</p></div>', unsafe_allow_html=True)

        case_c1, case_c2, case_c3 = st.columns(3, gap="medium")
        with case_c1:
            st.markdown('<div class="premium-card" style="height: 100%;"><div style="font-size: 2rem; margin-bottom: 1rem;">🛍️</div><h3 style="font-size: 1.2rem; margin-bottom: 0.5rem;">Retail Analytics</h3><p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1.5rem;">Analyze sales performance, customer segments, and regional trends.</p></div>', unsafe_allow_html=True)
            if st.button("Load Retail Case", use_container_width=True):
                self.load_case_study('sales', "Retail Sales Case Study", "🛍️")

        with case_c2:
            st.markdown('<div class="premium-card" style="height: 100%;"><div style="font-size: 2rem; margin-bottom: 1rem;">🏥</div><h3 style="font-size: 1.2rem; margin-bottom: 0.5rem;">Health Metrics</h3><p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1.5rem;">Evaluate patient outcomes, resource utilization, and operational efficiency.</p></div>', unsafe_allow_html=True)
            if st.button("Load Health Case", use_container_width=True):
                 self.load_case_study('healthcare', "Healthcare Ops Case Study", "🏥")

        with case_c3:
            st.markdown('<div class="premium-card" style="height: 100%;"><div style="font-size: 2rem; margin-bottom: 1rem;">📉</div><h3 style="font-size: 1.2rem; margin-bottom: 0.5rem;">Market Finance</h3><p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1.5rem;">Track portfolio performance, volatility, and sector allocation.</p></div>', unsafe_allow_html=True)
            if st.button("Load Finance Case", use_container_width=True):
                self.load_case_study('financial', "Market Finance Case Study", "📉")
        
        st.markdown('<div style="margin-top: 5rem; margin-bottom: 3rem;"></div>', unsafe_allow_html=True)
        
        # New Master Documentation Section
        from premium_education import render_main_page_docs
        render_main_page_docs()

if __name__ == "__main__":
    app = PremiumDataApp()
    app.apply_theme()
    app.render_sidebar()
    if not st.session_state.working_data.empty:
        filtered_df = app.apply_filters(st.session_state.working_data)
        app.render_main_content(filtered_df)
    else:
        app.render_main_content(pd.DataFrame())