"""
Premium Education & Knowledge Base Module
Provides high-level explanations for data science concepts, models, and statistics.
Serves as the central documentation hub for the platform.
"""

import streamlit as st

def render_main_page_docs():
    """
    Renders the comprehensive 'Master Documentation' section on the Welcome Screen.
    This combines the Quick Start guide with deep theoretical and feature documentation.
    """
    st.markdown("""
    <div class="animate-fade-in delay-300" style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid var(--border);">
        <div style="text-align: center; margin-bottom: 3rem;">
            <span style="background: rgba(var(--primary-rgb), 0.1); color: var(--primary); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Knowledge Center</span>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.5rem; margin-top: 1rem; color: var(--text-main);">Platform Master Guide</h2>
            <p style="color: var(--text-secondary); max-width: 700px; margin: 0.5rem auto; font-size: 1.1rem; line-height: 1.6;">
                A comprehensive resource for navigating the Unified Intelligence Engine. <br>
                Master every feature, understand the underlying science, and maximize your analytics potential.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Documentation Tabs
    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["🧭 Application Guide", "🧠 Data Science Theory", "⚡ Quick Start Workflow"])

    # -------------------------------------------------------------------------
    # TAB 1: APPLICATION GUIDE (Feature by Feature)
    # -------------------------------------------------------------------------
    with doc_tab1:
        st.markdown("### 🏛️ Module Breakdown")
        st.caption("Detailed walkthrough of every functional area within the platform.")

        with st.expander("1. Analyze: The Foundation", expanded=True):
            st.markdown("""
            The **Analyze** module is your starting point for understanding historical data.
            
            *   **🏠 Dashboard (Overview)**: 
                *   *Purpose*: Get a high-level pulse of your dataset immediately after upload.
                *   *Features*: Automatic KPI extraction, distribution snapshots, and time-series previews.
                *   *Use When*: You need an instant summary of "what happened" before diving deep.
            
            *   **🔍 Quality Audit (Diagnosis)**: 
                *   *Purpose*: Identify data health issues that could skew analysis.
                *   *Features*: Detects missing values, duplicates, and inconsistent data types. Calculates a 'Health Score'.
                *   *Use When*: Always pass your data through here first. "Garbage in, garbage out."
            
            *   **🛠️ Data Prep (Refinement)**: 
                *   *Purpose*: Fix issues found in Diagnosis and prepare features for modeling.
                *   *Features*: Imputation (mean/median/mode), Outlier removal (IQR method), One-Hot Encoding, and Column dropping.
                *   *Use When*: You need to clean data or create new calculated columns.
            
            *   **📈 Visualization (Exploration)**: 
                *   *Purpose*: Visual pattern discovery.
                *   *Features*: Drag-and-drop chart builder (Scatter, Line, Bar, Box, Violin, Heatmaps).
                *   *Use When*: You are exploring relationships between variables or looking for trends.
            
            *   **🧪 Stat Lab (Statistics)**: 
                *   *Purpose*: Rigorous hypothesis testing.
                *   *Features*: Correlation Matrix, T-Tests (Group comparison), ANOVA (Multi-group comparison).
                *   *Use When*: You want to prove that a difference between groups is statistically significant, not just random chance.
            
            *   **👥 Cohort Analysis**: 
                *   *Purpose*: Track behavior over time.
                *   *Features*: Retention matrices, churn analysis, and lifetime value tracking.
                *   *Use When*: Analyzing customer loyalty, subscription businesses, or user retention.
            """)

        with st.expander("2. Predict: Future Intelligence", expanded=False):
            st.markdown("""
            The **Predict** module leverages machine learning to forecast future outcomes.
            
            *   **🤖 Supervised ML (Intelligence)**: 
                *   *Purpose*: Train models to predict a specific target variable (e.g., Sales, Churn).
                *   *Features*: AutoML (Standard/Premium/Elite modes), Model Leaderboard, Feature Importance, Confusion Matrix.
                *   *Use When*: You have labeled historical data and want to predict future instances.
            
            *   **🧩 Clustering & Segments**: 
                *   *Purpose*: Group similar data points without predefined labels (Unsupervised Learning).
                *   *Features*: K-Means Clustering, Elbow Method Optimizer, 3D Cluster Visualization.
                *   *Use When*: You want to find customer personas or segment markets based on behavior.
            
            *   **⏳ Time Series**: 
                *   *Purpose*: Project trends into the future based on time history.
                *   *Features*: Seasonality decomposition, Trend analysis, Future forecasting.
                *   *Use When*: Your data has a date/time component and you need to know "what will sales be next month?"
            
            *   **🎲 Sensitivity & Sim (Scenario)**: 
                *   *Purpose*: "What-If" Analysis.
                *   *Features*: Adjust input drivers (e.g., Price, Marketing Spend) to see the impact on the target (e.g., Profit).
                *   *Use When*: Stress-testing your strategy against changing market conditions.
            """)

        with st.expander("3. Optimize: Strategic Decisioning", expanded=False):
            st.markdown("""
            The **Optimize** module focuses on actionable business improvements.
            
            *   **📐 A/B Testing**: 
                *   *Purpose*: Compare two versions/strategies to see which performs better.
                *   *Features*: Bayesian and Frequentist significance testing, Sample size calculators.
                *   *Use When*: Deciding between two marketing campaigns, website designs, or pricing models.
            
            *   **📊 Business Metrics**: 
                *   *Purpose*: Calculate standard ROI and Unit Economics.
                *   *Features*: CAC (Customer Acquisition Cost), LTV (Lifetime Value), ROI calculators.
                *   *Use When*: Translating data insights into financial terms for stakeholders.
            """)

        with st.expander("4. Report: Executive Delivery", expanded=False):
            st.markdown("""
            The **Report** module is for finalizing and sharing your work.
            
            *   **🔨 Dashboard Builder**: 
                *   *Purpose*: Create persistent, interactive dashboards.
                *   *Features*: Drag-and-drop widget layout, saved views.
                *   *Use When*: Building a monitoring tool for daily use.
            
            *   **📄 Report Studio**: 
                *   *Purpose*: Generate static, high-resolution documents (PDF/Word/PPT).
                *   *Features*: Auto-generated text summaries, high-DPI chart exports, custom branding.
                *   *Use When*: Preparing a slide deck or PDF for a board meeting.
            """)

    # -------------------------------------------------------------------------
    # TAB 2: DATA SCIENCE THEORY (Encyclopedia)
    # -------------------------------------------------------------------------
    with doc_tab2:
        st.markdown("### 🧬 Data Science Encyclopedia")
        st.caption("A deep dive into the algorithms, metrics, and theories power Plotiva's analytics engine.")

        # 1. THE ALGORITHMS
        with st.expander("🤖 Machine Learning Algorithms", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                st.markdown("#### Supervised Learning (Prediction)")
                st.markdown("""
                **1. Linear & Logistic Regression**
                *   *Type*: Baseline Models.
                *   *Theory*: Fits a straight line (or S-curve) to the data. 
                *   *Pros*: Highly interpretable. You know exactly how much 'X' affects 'Y'.
                *   *Cons*: Fails to capture complex, non-linear patterns.
                
                **2. Random Forest (Bagging)**
                *   *Type*: Ensemble Model.
                *   *Theory*: Builds hundreds of Decision Trees, each trained on a random subset of data. The final prediction is the *average* (Regression) or *majority vote* (Classification) of all trees.
                *   *Pros*: Extremely robust to overfitting; handles non-linear data well.
                
                **3. Gradient Boosting (Boosting)**
                *   *Type*: Ensemble Model (e.g., XGBoost, LightGBM).
                *   *Theory*: Builds trees sequentially. Each new tree tries to fix the errors (residuals) made by the previous tree.
                *   *Pros*: Often provides the highest accuracy in competitions.
                """)
            
            with cols[1]:
                st.markdown("#### Unsupervised Learning (Discovery)")
                st.markdown("""
                **1. K-Means Clustering**
                *   *Theory*: Partitions data into *K* distinct groups. It iteratively moves the center ("centroid") of each group to minimize the distance between points and their group center.
                *   *Elbow Method*: How do we choose K? We plot the "Error" vs. "Number of Clusters". The "Elbow" is the point where adding more clusters gives diminishing returns.
                
                **2. Time Series Decomposition**
                *   *Theory*: Breaks a timeline of data into three components:
                    *   **Trend**: The long-term direction (up/down).
                    *   **Seasonality**: Repeating patterns (e.g., sales spiking every December).
                    *   **Residual**: Random noise.
                """)

        # 2. METRICS
        with st.expander("📏 Performance Metrics & Evaluation", expanded=False):
            cols_met = st.columns(2)
            with cols_met[0]:
                st.markdown("#### Classification Metrics (Categories)")
                st.markdown("""
                *   **Accuracy**: % of correct predictions. *Warning: Misleading if classes are imbalanced (e.g., 99% of transactions are not fraud).*
                *   **Precision**: "Of all the ones I predicted as Fraud, how many were actually Fraud?" (Avoids False Positives).
                *   **Recall**: "Of all the actual Fraud cases, how many did I find?" (Avoids False Negatives).
                *   **F1-Score**: The harmonic mean of Precision and Recall. The best single metric for imbalanced data.
                *   **ROC / AUC**: Measures how well the model separates classes. 0.5 is random guessing; 1.0 is perfect separation.
                """)
            
            with cols_met[1]:
                st.markdown("#### Regression Metrics (Numbers)")
                st.markdown("""
                *   **MAE (Mean Absolute Error)**: The average "miss" size. If MAE is 10, the model is off by 10 units on average. Robust to outliers.
                *   **RMSE (Root Mean Square Error)**: Similar to MAE, but penalizes large errors more heavily. If being *very* wrong is distinctively bad, use RMSE.
                *   **$R^2$ (Coefficient of Determination)**: The percentage of variance in the target explained by the model. 1.0 is perfect fit.
                """)

        # 3. STATISTICAL THEORY
        with st.expander("🧪 Statistical Significance & Experimentation", expanded=False):
            st.markdown("""
            **Hypothesis Testing (The "Stat Lab")**
            *   **T-Test**: Used to compare the means of **two** groups (e.g., "Do men spend more than women?").
            *   **ANOVA (Analysis of Variance)**: Used to compare the means of **three or more** groups (e.g., "Does spend differ by Region: North vs South vs East vs West?").
            
            **A/B Testing Methodologies**
            *   **Frequentist (Classical)**: Uses P-values to reject the "Null Hypothesis" (that there is no difference). Gives you a binary answer: "Significant" or "Not Significant".
            *   **Bayesian**: Uses probability distributions. Instead of a Yes/No, it tells you: "There is a 92% probability that Variation B is better than Variation A."
            
            **Outlier Detection (IQR)**
            *   We use the **Interquartile Range** method.
            *   Data is sorted. We find the 25th percentile (Q1) and 75th percentile (Q3).
            *   IQR = Q3 - Q1.
            *   Bounds are `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`. Anything outside is flagged.
            """)
        
        st.info("💡 **Pro Tip**: Use the **'Diagnosis'** tab to ensure your data is clean before modeling. Machine Learning models are sensitive to missing values and unscaled data.")

    # -------------------------------------------------------------------------
    # TAB 3: QUICK START (Workflow)
    # -------------------------------------------------------------------------
    with doc_tab3:
        st.markdown("### ⚡ Zero to Insight in 5 Minutes")
        
        steps = [
            ("1. Ingestion", "Upload your `.csv` or `.xlsx` file via the Sidebar. Check the 'Source' metadata to confirm rows/cols."),
            ("2. Audit", "Go to **Analyze > Diagnosis**. Look for the 'Health Score'. If < 80%, use the recommended cleaning actions."),
            ("3. Exploration", "Go to **Analyze > Exploration**. Drag 'Sales' to Y-Axis and 'Date' to X-Axis to see your trend."),
            ("4. Modeling", "Go to **Predict > Intelligence**. Select your Target column (e.g., 'Churn'). Click 'Train AutoML'. Wait for the Leaderboard."),
            ("5. Report", "Go to **Report > Report Studio**. Click 'Generate PDF' to download a boardroom-ready summary.")
        ]
        
        for title, desc in steps:
            st.markdown(f"""
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem; align-items: start;">
                <div style="background: var(--accent); color: white; min-width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.8rem; margin-top: 2px;">✓</div>
                <div>
                    <div style="font-weight: 600; color: var(--text-main);">{title}</div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_education_module():
    """
    Renders the compact Help Center in the sidebar (Legacy/Compact version).
    """
    with st.expander("📚 Help Center", expanded=False):
        st.caption("Quick access to definitions and workflow tips.")
        st.markdown("[Complete Documentation is available on the Welcome Screen]")
        st.markdown("""
        **Quick Workflow:**
        1. **Upload** data in sidebar.
        2. **Clean** in *Analyze > Refinement*.
        3. **Visualise** in *Analyze > Exploration*.
        4. **Predict** in *Predict > Intelligence*.
        """)

def get_concept_tooltip(concept_name: str) -> str:
    """Returns a professional, human-readable explanation for a specific concept."""
    explanations = {
        # Statistics
        "p_value": "The probability that the observed difference occurred by random chance. Values < 0.05 suggest a real effect.",
        "confidence_interval": "The range within which we are 95% confident the true population value lies.",
        "correlation": "Measures the strength of the linear relationship between two variables, from -1 (perfect negative) to +1 (perfect positive).",
        
        # Machine Learning
        "r2_score": "R-Squared represents the proportion of variance in the dependent variable that is predictable from the independent variables.",
        "accuracy": "The ratio of correctly predicted observations to the total observations. Best used when classes are balanced.",
        "feature_importance": "A score indicating how much each factor contributed to the model's decision-making process.",
        "residuals": "The difference between the observed value and the predicted value. Patterns in residuals indicate missed information.",
        "overfitting": "When a model learns the 'noise' in the training data rather than the actual pattern, leading to poor performance on new data.",
        
        # Business
        "churn": "The rate at which customers classify as lost business over a specific period.",
        "ltv": "Projected revenue a customer will generate during their entire lifetime relationship with the company.",
        "cac": "The cost associated with convincing a customer to buy a product/service."
    }
    return explanations.get(concept_name, "Concept definition unavailable.")
