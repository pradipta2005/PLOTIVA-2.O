# Plotiva

**Unified Intelligence Engine for Data Analysis, Machine Learning, and Executive Reporting**

---

## 1. Project Overview

Plotiva is a self-contained, browser-based data analytics platform built on Streamlit. It accepts structured tabular data from a user, processes it through a layered analysis pipeline, and returns interactive visualizations, statistical diagnostics, machine learning predictions, and exportable executive reports — all without requiring any external data service or cloud dependency.

The core problem it addresses is fragmentation. Analysts working on business datasets typically move between four or five separate tools: a spreadsheet application for initial exploration, a BI tool for visualization, a Python notebook for statistical testing or modeling, and a word processor for reporting. Plotiva consolidates this workflow into a single, coherent interface.

It is built for:
- **Data analysts and data scientists** who need a fast, interactive environment for exploratory work without managing notebook infrastructure.
- **Business analysts and operations teams** who need to explore datasets, filter subsets, identify patterns, and produce polished outputs without writing code.
- **Recruiters and hiring managers** who receive it as a demonstration of full-stack data engineering capability — from ingestion and processing through to report export.
- **Small teams and consultants** who need a local-first, privacy-respecting alternative to SaaS analytics platforms.

What makes it practically useful is its layered architecture. The user loads one dataset and can move from data quality diagnosis to visualization to supervised ML to PDF report generation without reloading or reconfiguring. All modules share a single session-state managed dataframe, so filters applied in the sidebar are reflected consistently across every tab.

---

## 2. Live Application

The application is deployed and publicly accessible at:

**[https://plotiva-v2.streamlit.app/](https://plotiva-v2.streamlit.app/)**

On the live app, users can:
- Upload their own dataset (CSV, Excel, JSON, or Parquet) directly via the sidebar.
- Alternatively, load one of three pre-built industry datasets: Retail Sales, Healthcare Operations, or Market Finance.
- Apply dynamic column-level filters from the sidebar, which propagate in real time to every analysis module.
- Navigate through the full analysis pipeline — from data quality audit to visualization, statistical testing, machine learning, and report generation.
- Download outputs in PDF, Word (DOCX), Excel (XLSX), or PowerPoint (PPTX) format.

No account creation or API key is required. All data processing occurs within the session and is not persisted on any server.

---

## 3. Core Features

**Data Ingestion**
- Accepts CSV, Excel (`.xlsx`, `.xls`), JSON, and Parquet formats.
- Automatic encoding detection for CSV files (UTF-8, Latin-1, ISO-8859-1, CP1252).
- Automatic separator detection (comma, semicolon, tab, pipe).
- Post-load column name normalization and type inference.
- In-sidebar data quality score computed on upload (completeness, consistency, validity).

**Data Quality Audit**
- Per-column missing value analysis with imputation recommendations.
- Duplicate row detection.
- Outlier flagging using IQR and z-score methods.
- Weighted quality score displayed as a labeled grade (Excellent, Good, Fair, Poor).

**Refinement Engine (Data Preparation)**
- Column selection, renaming, and type casting.
- Row and column filtering, outlier removal, and imputation strategies.
- Feature engineering utilities including binning, scaling, polynomial feature generation, and interaction terms.

**Interactive Visualization**
- Ten chart types: Scatter, Line, Bar, Histogram, Box, Heatmap, Violin, Sunburst, Treemap, Radar.
- Per-chart axis configuration, color-by column, trendline toggle, and KDE overlay for histograms.
- Ten curated color palettes: Aurora, Ocean, Forest, Sunset, Emerald Growth, and others.
- Dark and light chart templates, consistent with global theme selection.
- Charts can be saved directly to the Dashboard or added to a report queue.

**Advanced Gallery**
- Gallery of pre-configured chart compositions designed for analytical storytelling.

**AI Insights**
- Heuristic-driven insight engine that surfaces statistical patterns, correlation findings, and distribution anomalies in plain language.

**Statistical Lab**
- Descriptive statistics with skewness, kurtosis, coefficient of variation, and IQR.
- Normality testing (Shapiro-Wilk).
- Hypothesis testing engine: selects between t-test, Mann-Whitney U, one-way ANOVA, and Kruskal-Wallis automatically based on group count and normality of data.
- Correlation analysis: Pearson, Spearman, Kendall. Strong correlations flagged with threshold at 0.7.
- Effect size calculation (Cohen's d, eta-squared).

**Cohort Analysis**
- User-defined cohort grouping with retention and behavior tracking over time.

**Supervised Machine Learning**
- Manual training mode: user selects task type, target variable, feature set, algorithm, and test split.
- AutoML mode: benchmarks multiple algorithms automatically and returns the highest-performing model.
- Supported regression algorithms: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost.
- Supported classification algorithms: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, SVC, KNN.
- Optional hyperparameter tuning via GridSearchCV for compatible models.
- Post-training outputs: performance metrics, feature importance chart, actual vs. predicted scatter (regression) or confusion matrix (classification), residual distribution, and a model comparison leaderboard across the session.
- Data leakage detection: flags features with correlation > 0.95 to the target, and blocks features identical to the target.
- Class imbalance detection with F1-Score prioritization when minority class falls below 10%.

**Clustering and Segmentation**
- Algorithms: K-Means, Hierarchical (Agglomerative), DBSCAN, Gaussian Mixture.
- Elbow Method / Silhouette Score scanner to determine optimal K.
- PCA projection for 2D visualization of high-dimensional cluster results.
- Cluster profiling table with per-cluster feature means.
- Dimensionality reduction: PCA, t-SNE, UMAP with 2D and 3D output options.

**Time Series Lab**
- Seasonal decomposition (additive and multiplicative) using statsmodels.
- Stationarity testing via Augmented Dickey-Fuller.
- ACF and PACF plots.
- Forecasting models: ARIMA (with manual order parameters), Exponential Smoothing (Holt-Winters), and Facebook Prophet.
- Confidence interval bands displayed on forecast charts.

**Sensitivity and Scenario Analysis**
- What-if parameter simulation for selected variables.
- Output range visualization for user-defined input bounds.

**A/B Testing**
- Statistical significance testing between two groups.
- P-value, effect size, and confidence interval reporting.

**Business Metrics Dashboard**
- Pre-built KPI cards configurable by the user for revenue, growth, acquisition cost, and retention metrics.

**Dashboard Builder**
- Drag-and-drop style widget configuration (Charts and KPI cards) saved per named dashboard.
- Multiple dashboards storable within one session.

**Executive Reporting Suite**
- Multi-section PDF report generation using ReportLab.
- Report sections: Cover page, Executive Summary, KPI Dashboard, Visual Analysis, Strategic Insights, Recommendations, Methodology Appendix, Glossary.
- Configurable document metadata: title, subtitle, author, company, color theme, paper size.
- Export formats: PDF, Word (DOCX), Excel (XLSX), PowerPoint (PPTX).
- Auto-generated insights from the insights engine can be included without manual entry.

**Global Filtering**
- Sidebar "Refinement Engine" supports adding any column as a filter constraint.
- Numeric columns produce range sliders; categorical columns produce multi-select widgets.
- Filters propagate across all analysis modules simultaneously.
- Active filter count is displayed, and individual filters or all filters can be removed with a single click.

**Theme System**
- Dark and light mode with full CSS variable coverage.
- User-selectable color palette for all chart outputs.

---

## 4. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Application Framework | Streamlit >= 1.28.0 | Allows Python-native UI construction with reactive session state, avoiding a separate frontend build pipeline. |
| Data Layer | Pandas >= 2.0.0, NumPy >= 1.24.0 | Standard tools for tabular data manipulation. Pandas 2.0 introduced copy-on-write semantics that reduce memory overhead. |
| Primary Visualization | Plotly >= 5.15.0 | Produces fully interactive, browser-rendered charts natively compatible with Streamlit's `st.plotly_chart`. |
| Secondary Visualization | Matplotlib >= 3.7.0, Seaborn >= 0.12.0 | Used for report-embedded static charts where interactivity is not needed. |
| Machine Learning | scikit-learn >= 1.3.0 | Provides the training, evaluation, scaling, and encoding pipeline for all supervised and unsupervised models. |
| Gradient Boosting | XGBoost (optional, imported with try/except) | Strong performance on structured tabular data. Treated as an optional dependency so the app degrades gracefully if absent. |
| Statistical Testing | SciPy >= 1.11.0, statsmodels >= 0.14.0 | SciPy covers parametric and non-parametric hypothesis tests; statsmodels covers time series decomposition, ARIMA, stationarity tests, and Holt-Winters. |
| Time Series Forecasting | Prophet >= 1.1.5 | Handles trend detection and seasonality automatically with minimal configuration; imported conditionally. |
| Clustering Extensions | HDBSCAN >= 0.8.29, UMAP-learn >= 0.5.3 | UMAP provides faster and more stable dimensionality reduction than t-SNE at scale. HDBSCAN supports density-based clustering without requiring a predefined cluster count. |
| PDF Generation | ReportLab >= 4.0.0, fpdf2 >= 2.7.0 | ReportLab provides precise programmatic control over PDF layout needed for executive-grade output. |
| Word Export | python-docx >= 0.8.11 | Standard library for generating `.docx` files with styled text, tables, and embedded images. |
| PowerPoint Export | python-pptx >= 0.6.21 | `create_premium_pptx` utility builds slide decks from the session's KPI and chart data. |
| Excel Export | openpyxl >= 3.1.0 | Enables styled multi-sheet Excel export with header formatting. |
| File Parsing | xlrd >= 2.0.0, pyarrow >= 12.0.0, fastparquet >= 0.8.0 | Support legacy `.xls` files (xlrd) and the Parquet format (pyarrow/fastparquet). |
| Image Export (Charts to PDF) | Kaleido >= 0.2.1 | Required to convert Plotly figures to PNG for embedding into PDF and Word exports. |
| Image Processing | Pillow >= 10.0.0 | Used for image resizing and compression during report generation. |
| Deployment | Streamlit Community Cloud | Zero-infrastructure deployment directly from a GitHub repository. |

---

## 5. Project Architecture

### Directory Structure

```
Plotiva/
|
|-- main.py                          # Application entry point. Bootstraps all modules.
|-- premium_config.py                # Global design system: themes, CSS, palettes, constants.
|-- premium_analytics.py             # Core analytical engine: statistics, ML training, clustering.
|-- utils.py                         # Shared utilities: file loading, quality metrics, sample data.
|
|-- dashboard_tab.py                 # Auto-generated overview dashboard on data load.
|-- premium_visualization_tab.py     # Interactive chart builder with 10 chart types.
|-- premium_gallery.py               # Pre-configured chart gallery and storytelling views.
|-- premium_insights.py              # Heuristic-based insight engine.
|-- premium_statistics.py            # Statistical lab: descriptive stats, hypothesis testing.
|-- premium_data_diagnosis.py        # Data quality auditing module.
|-- premium_feature_engineering.py   # Data preparation and feature engineering tools.
|-- premium_cohort_tab.py            # Cohort analysis module.
|
|-- premium_ml_tab.py                # Supervised ML studio UI with manual and AutoML modes.
|-- premium_clustering_tab.py        # Unsupervised clustering and dimensionality reduction.
|-- premium_time_series_tab.py       # Time series decomposition and forecasting.
|-- premium_scenario_tab.py          # Sensitivity analysis and what-if simulation.
|
|-- premium_ab_testing_tab.py        # A/B testing and statistical significance module.
|-- premium_business_metrics.py      # Business KPI dashboard.
|
|-- premium_dashboard_builder_tab.py # Custom dashboard builder with saveable widget grids.
|-- premium_report.py                # Report Studio UI: assembles and triggers report generation.
|-- professional_report.py           # PDF generation engine using ReportLab.
|-- report_generator.py              # Supporting utilities for report content and layout.
|-- utils_pptx.py                    # PowerPoint generation utility.
|
|-- premium_education.py             # In-app documentation and knowledge base module.
|-- premium_plots.py                 # Internal plot generation helpers (PremiumPlotGenerator).
|
|-- requirements.txt                 # Python dependency declarations.
```

### Component Interaction

The application follows a hub-and-spoke architecture where `main.py` acts as the orchestrator.

On startup, `main.py` initializes a `PremiumDataApp` instance. The constructor calls `initialize_session_state()`, which sets up the canonical state dictionary: `working_data` (the mutable dataframe), `original_data` (the immutable backup), `active_filters`, `ml_results`, `dashboard_charts`, and theme/palette preferences.

The sidebar, rendered by `render_sidebar()`, handles all global state changes: file upload, filter application, theme toggle, and palette selection. When a file is uploaded, `load_file_cached()` from `utils.py` parses it, normalizes column names, and returns a cleaned dataframe. Quality metrics are computed immediately and stored in `st.session_state.data_quality`.

Navigation is managed through `render_top_navigation()`, which maintains `current_main_tab` and `current_sub_tab` in session state. These two keys drive the routing logic in `render_main_content()`, which calls the appropriate tab module's render function.

Before passing the dataframe to any render function, `apply_filters()` iterates over `active_filters` and returns a filtered slice of `working_data`. This filtered dataframe is what every downstream module receives. The Data Refinement tab is the exception: it always receives the full `working_data` because structural operations should not be bounded by active filters.

The ML tab (`premium_ml_tab.py`) delegates training and evaluation to `PremiumAnalytics.train_model()` in `premium_analytics.py`. Results are stored in `st.session_state.ml_results` and persist across navigation, so a model trained in the Predict section is available for inclusion in a future report.

The Report Studio (`premium_report.py`) reads from multiple session state keys — `working_data`, `ml_results`, `saved_figures`, `custom_dashboards` — to assemble a multi-section document. The actual PDF is built by `PremiumPlotivaReport` in `professional_report.py`, which programmatically draws pages using ReportLab.

### Data Flow

```
User uploads file
       |
       v
utils.load_file_cached()
  Encoding detection, separator detection, type inference, column normalization
       |
       v
st.session_state.working_data    <---  Filter Engine (apply_filters)
st.session_state.original_data           |
       |                                 |
       v                                 |
premium_analytics.PremiumAnalytics       |
       |-- statistical_summary()         |
       |-- correlation_analysis()   <----+
       |-- train_model()
       |-- automl()
       |-- advanced_clustering()
       |
       v
Tab Modules (visualization, ML, stats, time series...)
       |
       v
st.session_state.saved_figures
st.session_state.dashboard_charts
st.session_state.ml_results
       |
       v
premium_report.py --> professional_report.PremiumPlotivaReport
       |
       v
PDF / DOCX / XLSX / PPTX download
```

---

## 6. Detailed Code Implementation

### `main.py` — Application Orchestrator

The file defines `PremiumDataApp`, a class that encapsulates the full application lifecycle. The class pattern is used deliberately to keep state initialization, sidebar rendering, filter logic, and content routing separated into discrete methods rather than scattered across a flat script.

`initialize_session_state()` uses a `defaults` dictionary pattern. It only sets values that are not already present, which means navigating between pages or re-running the script does not reset existing state. This is the foundational pattern that makes Streamlit's reactive model manageable at this scale.

`apply_theme()` generates CSS variable overrides dynamically. Rather than hardcoding dark or light styles, it reads the active theme from `PREMIUM_THEMES` and injects CSS custom properties (`:root { --primary: ... }`) into the page. All component styles reference these variables, so the entire UI re-styles with a single state key change.

`render_sidebar()` implements a "Refinement Engine" filter system. When the user selects a column from the `add_constraint_selector` selectbox, the `on_add_constraint` callback fires (using Streamlit's `on_change`). It detects whether the column is numeric or categorical and initializes a sensible default filter value. Subsequent render passes show slider widgets for numeric columns and multiselect widgets for categorical ones, with live updates to `active_filters`. Remove and Reset buttons use lambda callbacks directly, avoiding the need for additional helper functions.

`render_smart_suggestions()` implements a heuristic "Next Best Action" engine. It examines session state in order of priority — unresolved data quality issues, empty dashboards, untrained models, and mature analyses — and surfaces a contextual action button. This reduces the cognitive load of deciding where to start.

`render_top_navigation()` renders a two-tier navigation structure using Streamlit columns as buttons. Active tabs are rendered with `type="primary"` and inactive with `type="secondary"`. Tab state is persisted so the user returns to the same sub-tab after interacting with the sidebar.

`render_welcome_screen()` is displayed when no data is loaded. It contains a live preview panel with three toggleable demo modes (Visualization, Predictive Engine, Quality Sentinel) showing real Plotly charts generated from synthetic data. Three industry case study cards load pre-defined sample datasets directly from `utils.generate_sample_data()`. A documentation section rendered by `premium_education.render_main_page_docs()` provides contextual guidance.

### `premium_analytics.py` — Analytics Engine

`PremiumAnalytics` is the computational core. It is instantiated once in `PremiumDataApp.__init__()` and passed to modules that need it.

`train_model()` implements the full supervised ML pipeline. Feature preparation handles three column types independently: numeric columns are imputed at the column mean; categorical columns are one-hot encoded using `pd.get_dummies`; datetime columns are converted to ordinal integers. The three resulting dataframes are concatenated before splitting. This avoids a common pipeline error where encoding changes column count partway through.

The scaler is fit only on the training split (`scaler.fit_transform(X_train)`) and applied to the test split separately (`scaler.transform(X_test)`), preventing data leakage through normalization. For classification, stratified splitting is applied when each class has at least two samples.

Data leakage detection runs before training. The function checks column-wise equality (feature identical to target) with an exact check, and flags near-perfect correlations (> 0.95) as UI warnings. A further check detects if the task type has been set incorrectly: a continuous numeric target with fewer than 20 unique values triggers a warning to switch to regression; a categorical target passed to a regression task returns an error immediately.

`automl()` loops over a predefined candidate list (`Random Forest`, `Gradient Boosting`, `Extra Trees`, and optionally `XGBoost`, plus task-appropriate linear models), calls `train_model()` for each, collects the primary metric (R2 for regression, Accuracy for classification), and returns the best result alongside a full leaderboard.

`hypothesis_testing()` uses an automatic test selection strategy. It applies Shapiro-Wilk normality tests to each group. With two groups and normal distributions, it uses the independent t-test; with non-normal distributions, Mann-Whitney U. With three or more groups, it selects between one-way ANOVA and Kruskal-Wallis on the same normality criterion. Effect size is computed as Cohen's d for two groups and eta-squared for multiple groups.

`time_series_analysis()` performs ADF stationarity testing and seasonal decomposition via statsmodels. Autocorrelation is computed for 20 lags using Pandas `Series.autocorr()`. All downstream forecasting modules build on the time series returned by this method.

### `premium_visualization_tab.py` — Interactive Chart Builder

The module exposes ten chart types managed through a button-based type selector backed by `st.session_state.selected_chart_type`. Configuration widgets are rendered dynamically based on the selected chart type in a left column, with the chart rendered in a wider right column.

Caching is applied at the chart creation level. Core chart-creation functions (`create_cached_scatter`, `create_cached_bar`, etc.) are decorated with `@st.cache_data`, which memoizes outputs based on the dataframe content and parameter signature. This avoids redundant Plotly figure construction on every widget interaction that does not change the chart inputs.

The `create_premium_chart()` function separates chart construction from layout application. The cached functions return a base figure; the caller then applies global layout overrides (font, background opacity, legend position, hover template, annotation) derived from the current theme and palette. This pattern allows the cache to remain valid even when the theme changes.

Charts can be added to the Dashboard (stored in `st.session_state.dashboard_charts` with metadata including the figure object, configuration, palette, and row count) or sent to the Report queue (`st.session_state.saved_figures`).

KDE curves on histograms are computed post-hoc using `scipy.stats.gaussian_kde`, scaled to match the histogram's bin height, and added as a separate trace.

### `premium_ml_tab.py` — Machine Learning Studio

The tab renders a two-column layout: configuration panel (left) and results console (right). The results area only renders when `st.session_state.ml_results` contains training output, showing a branded empty state otherwise.

`clear_cache()` is attached as an `on_change` callback to every configuration widget. When the user changes the target variable, feature selection, or algorithm, the cached result is immediately cleared, preventing stale output from appearing alongside new configuration.

The model information dictionary (`MODEL_INFO`) provides a structured knowledge base for each algorithm: description, ideal use case, strengths, and limitations. This is rendered as a styled card below the algorithm selector so the user can make informed choices without external documentation.

On classification tasks, a pre-flight check evaluates class balance using `value_counts(normalize=True)`. If the minority class represents less than 10% of the dataset, a warning is surfaced before training.

Result visualization uses four sub-tabs: Feature Importance (horizontal bar chart), Performance Fit (actual vs. predicted scatter for regression, confusion matrix heatmap for classification), Error Analysis (residual histogram for regression), and Explainability (a Partial Dependence Plot interface with a note that full SHAP analysis is directed to a notebook environment for performance reasons).

The Model Comparison leaderboard displays a dataframe of all models trained in the session, with a chart comparing them on any selected metric.

### `premium_clustering_tab.py` — Segmentation Module

Four clustering algorithms are supported: K-Means, Hierarchical (Agglomerative), DBSCAN, and Gaussian Mixture. Each exposes its canonical tuning parameters as slider inputs.

For K-Means, an Elbow Method scanner runs K from 2 to a user-defined maximum, collecting inertia and silhouette scores for each K value. The output is a dual-axis Plotly chart, and the K with the highest silhouette score is surfaced as a recommendation.

After clustering, when the feature space has more than two dimensions, the results are projected to 2D using PCA before visualization. The scatter plot colors points by cluster label. A cluster profiling table shows per-cluster feature means, allowing the user to characterize what each segment represents in domain terms.

The Dimensionality Reduction panel supports PCA, t-SNE, and UMAP. For t-SNE, perplexity is exposed as a slider. For UMAP, `n_neighbors` and `min_dist` are configurable. Both 2D and 3D projection outputs are supported, with `px.scatter_3d` used for the three-component case.

### `premium_time_series_tab.py` — Time Series Lab

The module auto-detects datetime columns by inspecting column types and checking column names for keywords (`date`, `time`, `year`). The selected series is resampled to the user-specified frequency and interpolated to fill gaps before any analysis runs.

Seasonal decomposition accepts both additive and multiplicative models. Stationarity is evaluated using the Augmented Dickey-Fuller test, with results displayed as metric cards and an advisory note on differencing when the series is non-stationary.

Forecasting supports three models. ARIMA uses user-specified `(p, d, q)` order and returns confidence intervals from `get_forecast()`. Exponential Smoothing (Holt-Winters) accepts configurable trend and seasonal components. Prophet is imported conditionally and wraps the series in the required `ds/y` format. Confidence intervals from ARIMA and Prophet appear as shaded bands on the forecast chart.

### `premium_report.py` and `professional_report.py` — Report Studio

`render_report_tab()` implements a two-column layout: configuration panel (left) for document metadata and visual style, and a tabbed content builder (right) for populating report sections.

Report generation follows a staged progress model. A Streamlit progress bar updates at each section addition, providing user feedback during potentially slow operations. The PDF pipeline calls `PremiumPlotivaReport` methods sequentially: `add_cover_page()`, `add_executive_summary()`, `add_executive_dashboard()`, `add_colored_section()`, `add_plot_with_caption()`, `add_recommendations()`, `add_appendix()`, and `add_methodology_section()`. Plotly figures are exported to PNG using Kaleido before embedding.

Non-PDF exports are triggered separately. The Word export builds a styled `Document` using python-docx, setting title fonts, paragraph styles, and per-run formatting. Charts are exported from Plotly to PNG bytes and embedded as images with `Inches` width. The Excel export uses `pd.ExcelWriter` with openpyxl, stylizes the header row with a teal fill, and writes the working dataset to a "Raw Data" sheet with optional dashboard metric summaries. The PPTX export delegates to `create_premium_pptx()` in `utils_pptx.py`.

### `utils.py` — Shared Utilities

`load_file_cached()` is the single file-loading function used for all upload paths. It uses `@st.cache_data(ttl=3600)` so that re-uploading the same file within an hour does not re-parse it. The CSV parsing loop tries four encodings and four separators in nested iteration, selecting the combination that produces more than one column. A final fallback uses Python's CSV engine with no separator constraint.

Post-parse, column names are stripped of whitespace, run through a regex to remove non-word characters, and spaces are replaced with underscores. Object columns that match a numeric pattern are converted with `pd.to_numeric(errors='ignore')`.

`calculate_data_quality_metrics()` computes three sub-scores: completeness (proportion of non-null cells), consistency (proportion of non-duplicate rows), and validity (penalized for infinite values and extreme outliers beyond 4 standard deviations). The overall score is a weighted average with completeness weighted at 0.4, consistency at 0.3, and validity at 0.3.

`generate_sample_data()` produces four pre-configured datasets: a 365-row retail sales dataset with seasonal signal, a 1000-row customer segmentation dataset, a 500-row financial dataset with simulated price movement, and a 300-row healthcare operations dataset. These are deterministic (`np.random.seed(42)`) and cached, so they do not vary between sessions.

### `premium_config.py` — Design System

This file centralizes all visual constants. `PREMIUM_COLOR_PALETTES` defines 10 named color sequences used by all chart modules. `PREMIUM_THEMES` defines the full token set for light and dark mode — background, card background, primary text, secondary text, border, and shadow variables. All CSS in `PREMIUM_CSS` references these tokens as CSS custom properties, making the theme system a single key change rather than a class swap.

Quality threshold constants (`QUALITY_THRESHOLDS`) are defined here so that the labeling logic in `utils.py` and the display logic in `main.py` share a single source of truth.

---

## 7. Application Workflow

**Step 1: Landing**
The user arrives at the welcome screen. A preview panel with three toggleable modes (visualization, prediction, data audit) demonstrates the platform's capabilities using synthetic data. Three industry case study buttons load curated datasets immediately. The sidebar prompts the user to upload a file or click "Load Demo Dataset."

**Step 2: Data Ingestion**
The user uploads a file or selects a case study. `load_file_cached()` parses the file, normalizes it, and stores it in `st.session_state.working_data`. A data quality score is computed and displayed in the sidebar alongside the file's dimensions and memory footprint.

**Step 3: Data Quality Audit**
Navigating to Analyze > Quality Audit opens `premium_data_diagnosis.py`. Missing values, duplicates, and outliers are reported per column. The user can decide whether to proceed directly to analysis or move to Data Prep.

**Step 4: Refinement (Optional)**
Analyze > Data Prep opens `premium_feature_engineering.py`. The user can reshape, clean, engineer, and scale the dataset. Changes are applied to `working_data`. The original data is preserved in `original_data` for reference.

**Step 5: Filtering (Ongoing)**
At any point, the user can add column-level constraints from the sidebar Refinement Engine. These filters apply immediately. A banner at the top of the main content area shows the filtered row count versus total. Filters remain active across all tabs until explicitly removed.

**Step 6: Exploration and Visualization**
Analyze > Visualization opens the chart builder. The user selects a chart type, configures axes and groupings, and the chart renders. Charts can be saved to the Dashboard or queued for inclusion in a report.

Analyze > Stat Lab handles hypothesis testing, correlation analysis, and descriptive statistics.

**Step 7: Prediction**
Predict > Supervised ML opens the ML Studio. The user selects features, a target, and an algorithm. Training triggers via button, results appear in the right panel. The user can run additional models and compare them in the leaderboard tab.

Predict > Clustering runs the segmentation module. Predict > Time Series handles forecasting. Predict > Sensitivity runs what-if simulation.

**Step 8: Optimization**
Optimize > A/B Testing runs significance tests. Optimize > Business Metrics displays KPI cards.

**Step 9: Reporting**
Report > Dashboard Builder creates custom widget grids. Report > Report Studio assembles the final document. The user configures metadata, selects which sections to include, and clicks Generate. A progress bar tracks section rendering. After completion, download buttons appear for all four export formats.

---

## 8. Design and UX Philosophy

The interface uses a CSS custom property system defined in `premium_config.PREMIUM_CSS`. All spacing, color, border radius, shadow, and animation values are derived from this token system. Switching between dark and light mode requires only overriding the root variables; no component-level style changes are needed.

Typography uses three families served via Google Fonts: Inter (body text, labels, captions), Plus Jakarta Sans (headings), and JetBrains Mono (code fragments, technical labels, status indicators). These three typefaces cover the full UI without stylistic conflict.

Animations are applied selectively. Entry animations (`slideUp`, `fadeIn`, `scaleIn`) use the class `animate-enter` applied at the container level. The pulse animation on the sidebar status indicator and the shimmer on loading skeletons are the only persistent animations, keeping the interface lively without becoming distracting.

Streamlit's native widget library is used throughout rather than custom HTML inputs, which ensures accessibility compatibility and consistent behavior under both themes. Custom HTML is used only for headers, card containers, and branded sections where Streamlit's default components are not sufficiently flexible.

Performance decisions were made deliberately. Chart creation functions are cached at the function level using `@st.cache_data`. File parsing is cached with a 3600-second TTL. Quality metric calculation is cached. Heavy operations (ML training, clustering, forecasting) are not cached because their inputs (session state, widget values) change frequently and caching would produce stale results.

The "Next Best Action" system in the sidebar reduces decision fatigue. Rather than requiring the user to navigate the full tab tree to understand what step comes next, a heuristic checks current state and surfaces the most relevant action as a labeled button.

---

## 9. Deployment Details

Plotiva is deployed on **Streamlit Community Cloud**, the managed deployment service that hosts Streamlit applications directly from a GitHub repository.

Deployment requires:
1. A public or private GitHub repository containing `main.py` and `requirements.txt`.
2. A Streamlit Community Cloud account connected to the repository.
3. Specifying `main.py` as the entry point in the deployment configuration.

Dependencies listed in `requirements.txt` are installed automatically by Streamlit Cloud during the build phase. No `Dockerfile` or infrastructure configuration is needed.

A `vercel.json` file is present in the repository but is not the active deployment target. Streamlit applications require a Python runtime and cannot be served as static assets on Vercel; this file was an earlier configuration artifact.

**Known constraints of the Streamlit Community Cloud environment:**
- Applications have a memory limit. Very large datasets (hundreds of MB) may cause the application to restart.
- There is no persistent storage. All session data is lost when the browser tab is closed or the session times out.
- Concurrent user sessions are sandboxed but share the same cold start time if the application has been inactive.
- The free tier imposes resource limits and may put idle applications to sleep. The first request after a sleep period incurs a cold start delay of 10–30 seconds.

For production enterprise deployments, the same codebase can be hosted on any environment that supports Python 3.9+ and can install the dependencies listed in `requirements.txt`. This includes Docker containers on cloud providers, private Kubernetes clusters, or Streamlit for Teams.

---

## 10. Scalability and Future Improvements

**Near-term practical extensions:**

- **Persistent sessions with a database backend.** Storing `working_data` and `ml_results` in a PostgreSQL or DuckDB instance would allow users to return to saved analyses across sessions.
- **Larger dataset support.** The current implementation loads the full dataset into memory. Introducing chunked processing using Dask or Polars for the data layer would allow analysis of datasets that exceed available RAM.
- **Authentication layer.** A simple authentication integration (e.g., Streamlit Authenticator or OAuth) would allow the platform to manage per-user workspaces.
- **Additional ML algorithms.** The model registry in `premium_analytics.py` can be extended with LightGBM, CatBoost, and linear SVM variants without changing any UI code.

**Medium-term feature additions:**

- **Automated report scheduling.** Allowing users to configure a report template once and schedule it to run against a refreshed dataset at a specified interval.
- **Dataset versioning.** Tracking the state of `working_data` after each refinement operation so the user can roll back to a previous version.
- **Connector integrations.** Native connectors to PostgreSQL, BigQuery, Snowflake, and REST APIs would replace the file upload workflow for teams with live data sources.
- **Annotation layer.** Allowing users to annotate specific data points or chart regions with comments, shareable via export.

**Enterprise-level considerations:**

- **Multi-tenancy.** Isolating session state and report storage per tenant in a hosted deployment.
- **Role-based access control.** Separating analyst, viewer, and admin roles with feature visibility controlled per role.
- **Model registry.** Persisting trained model artifacts (serialized via `joblib` or `pickle`) so models can be re-used across sessions and promoted to production prediction endpoints.
- **Audit logging.** Recording user actions, filter configurations, and report generation events for compliance-sensitive environments.

---

## 11. How to Run Locally

**Prerequisites:**
- Python 3.9 or higher
- `pip` and `venv` (standard with CPython)
- Git

**Setup:**

```bash
# 1. Clone the repository
git clone https://github.com/your-username/plotiva.git
cd plotiva

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS / Linux
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

**Note on Kaleido:** Kaleido is required for chart-to-image export during report generation. On some systems, installation requires a compatible browser runtime. If the `kaleido` install fails, run:

```bash
pip install kaleido==0.2.1
```

If you encounter issues with Prophet or UMAP on specific Python versions, install them separately before running the requirements file:

```bash
pip install prophet
pip install umap-learn
pip install hdbscan
```

**Running the application:**

```bash
streamlit run main.py
```

Streamlit will open the application at `http://localhost:8501` by default.

**Optional configuration:** To change the default port or disable the Streamlit telemetry, create a `.streamlit/config.toml` file at the project root:

```toml
[server]
port = 8502
headless = true

[browser]
gatherUsageStats = false
```

---

## 12. Contribution Guidelines

**Getting started:**

Fork the repository, create a feature branch with a descriptive name (`feature/add-lightgbm-support`, `fix/pdf-encoding-error`), make your changes, and open a Pull Request against `main`. All PRs should include a description of what changed and why.

**Code standards:**

- All Python code follows PEP 8. Variable and function names use `snake_case`; class names use `PascalCase`.
- New analytical modules should follow the existing module pattern: a single public `render_*_tab(df)` function as the entry point, with all supporting logic defined locally or delegated to `premium_analytics.py`.
- Any new algorithm added to the ML or clustering system must be added to both the model registry (in `premium_analytics.py`) and the UI selector list (in `premium_ml_tab.py` or `premium_clustering_tab.py`).
- Streamlit session state keys must be documented in a comment near their first use and initialized in `PremiumDataApp.initialize_session_state()`.
- All computationally heavy functions that accept a dataframe should use `@st.cache_data` where caching is appropriate. Functions with side effects (session state mutations) must not be cached.

**Error handling:**

- All tab render functions must wrap their core logic in a `try/except` block and display a user-facing `st.error()` message on failure. The application must never present a raw Python traceback to the user.
- All optional imports (e.g., `xgboost`, `prophet`, `umap`) must use `try/except ImportError` with a boolean flag, and downstream code must check that flag before calling the dependency.

**Testing:**

- New utility functions in `utils.py` should include inline docstrings and basic edge case handling (empty dataframe, single-column input, all-null column).
- Before opening a PR, verify that loading each of the three sample datasets (`sales`, `healthcare`, `financial`) and running through the primary analysis tabs produces no Python exceptions.

---

## 13. License and Ownership

This project is maintained by its author and made publicly available for review and collaboration.

All source code in this repository is original work unless where third-party libraries are explicitly imported. Third-party libraries are governed by their respective licenses as declared in `requirements.txt`.

For inquiries related to licensing, attribution, or commercial use, contact the repository owner directly.

---

*Plotiva v2 — Built for clarity, designed for precision.*
