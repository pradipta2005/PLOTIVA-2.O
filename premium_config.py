"""
Premium Data Analysis Platform - Configuration
Top 1% Developer Quality - Premium Features & Styling
"""

# Professional Color Palettes
PREMIUM_COLOR_PALETTES = {
    'executive_suite': ['#1a2332', '#ff6b6b', '#4ecdc4', '#f7f7f5', '#a8a8a8', '#333333'],
    'modern_blue': ['#2563EB', '#3B82F6', '#60A5FA', '#93C5FD', '#1E40AF', '#1D4ED8'],
    'corporate_slate': ['#334155', '#475569', '#64748B', '#94A3B8', '#CBD5E1', '#E2E8F0'],
    'emerald_growth': ['#059669', '#10B981', '#34D399', '#6EE7B7', '#047857', '#065F46'],
    'vibrant_data': ['#6366F1', '#EC4899', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6'],
    'classic_analytics': ['#2c3e50', '#e74c3c', '#ecf0f1', '#3498db', '#2980b9', '#8e44ad'],
    'aurora': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
    'sunset': ['#FF5E62', '#FF9966', '#FFC371', '#FFD700', '#FDB931', '#F27121'],
    'ocean': ['#006994', '#0096C7', '#00B4D8', '#48CAE4', '#90E0EF', '#CAF0F8'],
    'forest': ['#2D6A4F', '#40916C', '#52B788', '#74C69D', '#95D5B2', '#D8F3DC']
}

# Professional Themes (Light & Dark)
PREMIUM_THEMES = {
    'light': {
        'name': 'Professional Light',
        'primary_color': '#1a2332',             # Deep Navy
        'secondary_color': '#ff6b6b',           # Warm Coral
        'accent_color': '#4ecdc4',              # Mint Green
        'background': '#f7f7f5',                # Warm Gray
        'card_background': '#FFFFFF',           # White
        'text_primary': '#1a2332',              # Deep Navy
        'text_secondary': '#4A5568',            # Slate 700
        'border_color': '#E2E8F0',              # Slate 200
        'shadow': '0 4px 12px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.02)'
    },
    'dark': {
        'name': 'Professional Dark',
        'primary_color': '#F8FAFC',             # White/Slate 50
        'secondary_color': '#ff6b6b',           # Warm Coral
        'accent_color': '#4ecdc4',              # Mint Green
        'background': '#0F172A',                # Slate 900
        'card_background': '#1E293B',           # Slate 800
        'text_primary': '#F8FAFC',              # Slate 50
        'text_secondary': '#94A3B8',            # Slate 400
        'border_color': '#334155',              # Slate 700
        'shadow': '0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.3)'
    }
}

# Advanced Plot Configurations (Standardized)
PREMIUM_PLOT_CONFIG = {
    'default_height': 600,
    'default_width': 1000,
    'high_dpi': True,
    'animation_duration': 500,
    'hover_effects': True,
    'interactive_legends': True,
    'zoom_enabled': True,
    'pan_enabled': True,
    'export_formats': ['png', 'svg', 'csv'],
    'font_family': 'Inter, sans-serif',
    'title_font_size': 20,
    'axis_font_size': 12,
    'legend_font_size': 12
}

# Premium Chart Types
PREMIUM_CHART_TYPES = {
    'advanced_scatter': {
        'name': 'Scatter Analysis',
        'description': 'Analyze relationships between two numeric variables with optional grouping.',
        'features': ['regression_line', 'grouping', 'size_scaling']
    },
    'animated_line': {
        'name': 'Trend Analysis',
        'description': 'Visualize data trends over time with smoothing options.',
        'features': ['smoothing', 'markers', 'multi_series']
    },
    'heatmap_advanced': {
        'name': 'Correlation Heatmap',
        'description': 'Explore correlations between multiple numeric variables.',
        'features': ['clustering', 'annotations', 'custom_scales']
    },
    'violin_plot': {
        'name': 'Distribution (Violin)',
        'description': 'Compare value distributions across categories.',
        'features': ['quartiles', 'kde', 'box_overlay']
    },
    'sunburst': {
        'name': 'Hierarchy (Sunburst)',
        'description': 'View hierarchical data structures in rings.',
        'features': ['drill_down', 'path_bar']
    },
    'treemap': {
        'name': 'Hierarchy (Treemap)',
        'description': 'View hierarchical data as nested rectangles.',
        'features': ['size_encoding', 'color_encoding']
    },
    'radar_chart': {
        'name': 'Multi-Metric Radar',
        'description': 'Compare multiple metrics across groups.',
        'features': ['polygon_fill', 'comparison']
    }
}

# Advanced Analytics Features
PREMIUM_ANALYTICS = {
    'statistical_tests': ['t_test', 'chi_square', 'anova', 'correlation_test'],
    'time_series_analysis': ['seasonality', 'trend_decomposition', 'forecasting', 'anomaly_detection'],
    'clustering_algorithms': ['kmeans', 'dbscan', 'hierarchical', 'gaussian_mixture'],
    'dimensionality_reduction': ['pca', 'tsne', 'umap', 'factor_analysis'],
    'feature_engineering': ['polynomial_features', 'interaction_terms', 'binning', 'scaling'],
    'model_interpretability': ['feature_importance', 'shap_values', 'lime', 'permutation_importance']
}

# Export and Sharing Options
PREMIUM_EXPORT_OPTIONS = {
    'formats': ['png', 'svg', 'pdf', 'html', 'json', 'csv', 'excel'],
    'quality_settings': ['standard', 'high', 'ultra'],
    'custom_branding': True,
    'watermark_options': True,
    'batch_export': True,
    'cloud_sharing': True,
    'embed_codes': True,
    'api_endpoints': True
}

# Performance Optimizations
PREMIUM_PERFORMANCE = {
    'lazy_loading': True,
    'data_sampling': True,
    'caching_strategy': 'advanced',
    'parallel_processing': True,
    'memory_optimization': True,
    'progressive_rendering': True,
    'virtual_scrolling': True,
    'debounced_updates': True
}

# Accessibility Features
PREMIUM_ACCESSIBILITY = {
    'high_contrast_mode': True,
    'screen_reader_support': True,
    'keyboard_navigation': True,
    'color_blind_friendly': True,
    'font_size_scaling': True,
    'focus_indicators': True,
    'alt_text_generation': True,
    'voice_commands': False  # Future feature
}

# Premium CSS Styles
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap');


/* -------------------------------------------------------------------------
   THEME VARIABLES
------------------------------------------------------------------------- */
:root {
    /* Light Mode */
    --primary: #1a2332; /* Deep Navy */
    --primary-hover: #2c3e50;
    --primary-light: #f7f7f5;
    --accent: #4ecdc4; /* Mint Green */
    --warm-coral: #ff6b6b;
    --bg-color: #f7f7f5; /* Warm Gray */
    --card-bg: #FFFFFF;
    --text-main: #1a2332;
    --text-secondary: #4A5568;
    --text-muted: #718096;
    --border: #E2E8F0;
    --glass-bg: rgba(255, 255, 255, 0.7);
    --glass-border: rgba(255, 255, 255, 0.8);
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
    --shadow-md: 0 8px 24px rgba(0,0,0,0.08);
    --shadow-lg: 0 20px 60px rgba(0,0,0,0.12);
    --gradient-luxury: linear-gradient(135deg, #1a2332 0%, #2c3e50 100%);
    --gradient-accent: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
    --easing: cubic-bezier(0.4, 0, 0.2, 1);
}

[data-theme="dark"] {
    --primary: #F8FAFC;
    --primary-hover: #E2E8F0;
    --primary-light: #1E293B;
    --accent: #4ecdc4;
    --warm-coral: #ff6b6b;
    --bg-color: #0F172A; /* Slate 900 */
    --card-bg: #1E293B;
    --text-main: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --border: #334155;
    --glass-bg: rgba(30, 41, 59, 0.7);
    --glass-border: rgba(255, 255, 255, 0.1);
    --shadow-sm: 0 2px 4px rgba(0,0,0,0.2);
    --shadow-md: 0 12px 24px -6px rgba(0,0,0,0.3);
    --shadow-lg: 0 20px 60px -8px rgba(0,0,0,0.5);
    --gradient-luxury: linear-gradient(135deg, #F8FAFC 0%, #94A3B8 100%);
}

/* -------------------------------------------------------------------------
   GLOBAL RESET & TYPOGRAPHY
------------------------------------------------------------------------- */
.stApp {
    background-color: var(--bg-color);
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    color: var(--text-main) !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

p, span, li, label, .stMarkdown {
    font-family: 'Inter', sans-serif;
    color: var(--text-secondary);
    font-weight: 400;
    line-height: 1.6;
}

code {
    font-family: 'JetBrains Mono', monospace;
    background: transparent !important;
}

/* -------------------------------------------------------------------------
   ANIMATIONS & TRANSITIONS
------------------------------------------------------------------------- */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(78, 205, 196, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(78, 205, 196, 0); }
    100% { box-shadow: 0 0 0 0 rgba(78, 205, 196, 0); }
}

@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.animate-slide-up { animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
.animate-fade-in { animation: fadeIn 0.8s ease-out forwards; }
.animate-scale-in { animation: scaleIn 0.5s ease-out forwards; }
.float-animation { animation: float 6s ease-in-out infinite; }
.pulse-animation { animation: pulse 2s infinite; }

.delay-100 { animation-delay: 0.1s; }
.delay-200 { animation-delay: 0.2s; }
.delay-300 { animation-delay: 0.3s; }
.delay-400 { animation-delay: 0.4s; }

/* -------------------------------------------------------------------------
   HERO SECTION COMPONENTS
------------------------------------------------------------------------- */
.hero-container {
    padding: 60px 0 40px;
    text-align: center;
    max-width: 900px;
    margin: 0 auto;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 6px 14px;
    background: rgba(78, 205, 196, 0.1); 
    color: var(--accent);
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 24px;
    border: 1px solid rgba(78, 205, 196, 0.2);
    backdrop-filter: blur(4px);
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    margin-bottom: 24px;
    background: var(--gradient-luxury);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: var(--text-secondary);
    max-width: 580px;
    margin: 0 auto 40px;
    line-height: 1.6;
    font-weight: 400;
}

/* -------------------------------------------------------------------------
   COMPONENTS & MICRO-INTERACTIONS
------------------------------------------------------------------------- */
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-color);
    border-right: 1px solid var(--border);
}

/* Buttons */
.stButton button {
    background: var(--primary);
    color: var(--bg-color) !important;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-sm);
}

.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
    background: var(--primary-hover);
}

.stButton button:active {
    transform: translateY(0);
}

.stButton button[kind="secondary"] {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-main) !important;
    box-shadow: none;
}

.stButton button[kind="secondary"]:hover {
    border-color: var(--primary);
    background: rgba(37, 99, 235, 0.03);
    color: var(--primary) !important;
}

/* Inputs */
.stTextInput > div > div, .stSelectbox > div > div, .stNumberInput > div > div {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1px;
    transition: all 0.2s ease;
}

.stTextInput > div > div:focus-within, .stSelectbox > div > div:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(78, 205, 196, 0.1);
}

/* -------------------------------------------------------------------------
   CARDS & CONTAINERS
------------------------------------------------------------------------- */
.premium-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: var(--shadow-sm);
    transition: transform 0.3s var(--easing), box-shadow 0.3s var(--easing);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.premium-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
    border-color: var(--primary-hover); /* Subtle highlight */
}

/* Hover Reveal Effect */
.hover-reveal {
    opacity: 0;
    transition: opacity 0.3s ease;
}
.premium-card:hover .hover-reveal {
    opacity: 1;
}

.glass-stat-card {
    background: rgba(255, 255, 255, 0.05); /* Very subtle fill */
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    transition: all 0.3s ease;
}

.glass-stat-card:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: var(--accent);
    transform: translateY(-2px);
}

/* Empty State Container */
.empty-state-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
    border: 2px dashed var(--border);
    border-radius: 16px;
    background: rgba(120, 120, 120, 0.05); /* Neutral */
    transition: border-color 0.3s ease;
}

.empty-state-container:hover {
    border-color: var(--accent);
}

/* Loading Skeleton */
.loading-skeleton {
    background: #f6f7f8;
    background: linear-gradient(to right, #eeeeee 8%, #dddddd 18%, #eeeeee 33%);
    background-size: 1000px 100%;
    animation: shimmer 1.5s infinite linear;
    border-radius: 4px;
}
[data-theme="dark"] .loading-skeleton {
    background: #1E293B;
    background: linear-gradient(to right, #1E293B 8%, #334155 18%, #1E293B 33%);
}

/* Utilities */
.text-center { text-align: center; }
.text-accent { color: var(--accent); }
.text-muted { color: var(--text-muted); }

/* Streamlit specific Overrides */
div[data-testid="stSidebarCollapsedControl"] {
    background-color: var(--bg-color) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

@media (max-width: 768px) {
    .hero-title { font-size: 2.5rem !important; }
    .hero-container { padding: 40px 0; }
}

/* Fix Input Text Visibility */
.stTextInput input, .stSelectbox div[data-baseweb="select"] span, .stNumberInput input {
    color: var(--text-main) !important;
    -webkit-text-fill-color: var(--text-main) !important;
    caret-color: var(--text-main) !important;
}

/* Fix Dropdown Menu Visibility */
ul[data-baseweb="menu"] {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
}

ul[data-baseweb="menu"] li span {
    color: var(--text-main) !important;
}

/* Targeted fix for Selectbox value container */
div[data-baseweb="select"] {
    background-color: transparent !important;
    color: var(--text-main) !important;
}

div[data-baseweb="select"] > div {
    background-color: transparent !important;
    color: var(--text-main) !important;
}

/* Ensure placeholder text is visible but distinct */
::placeholder {
    color: var(--text-muted) !important;
    opacity: 1;
}

/* Metric Label Fix */
div[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}
div[data-testid="stMetricValue"] {
    color: var(--text-main) !important;
}
</style>
"""

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'excellent': 0.9,
    'good': 0.7,
    'fair': 0.5,
    'poor': 0.3
}

# Error messages
ERROR_MESSAGES = {
    'file_too_large': 'File size exceeds maximum limit of {max_size}MB',
    'unsupported_format': 'File format not supported. Supported formats: {formats}',
    'insufficient_data': 'Insufficient data for analysis. Need at least {min_rows} rows',
    'no_numeric_columns': 'No numeric columns found for analysis',
    'no_categorical_columns': 'No categorical columns found for analysis',
    'ml_not_available': 'Machine learning libraries not available. Please install required packages.'
}

# Success messages
SUCCESS_MESSAGES = {
    'data_loaded': 'Successfully loaded {rows} rows and {columns} columns',
    'data_processed': 'Data processing completed successfully',
    'plot_created': 'Plot created successfully',
    'analysis_complete': 'Analysis completed successfully'
}
