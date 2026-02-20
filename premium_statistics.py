

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st


def render_statistics_tab(df: pd.DataFrame):
    """
    Render the Premium Statistical Analysis Suite.
    """
    # Helper for Theme-Aware Plots
    def get_plot_config():
        is_dark = st.session_state.get('theme', 'dark') == 'dark'
        return {
            'template': 'plotly_dark' if is_dark else 'plotly_white',
            'font_color': '#F8FAFC' if is_dark else '#1a2332',
            'bg_color': 'rgba(0,0,0,0)'
        }
    
    theme_cfg = get_plot_config()

    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Statistical Laboratory</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Rigorous hypothesis testing and inferential statistics for data-driven decisions.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">P-Value Precision < 0.05</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Tabs for different statistical modules
    stat_tabs = st.tabs(["🧪 Hypothesis Testing", "📉 Correlation Matrix", "bell Normality Checks", "📊 Descriptive Stats"])

    # --- 1. HYPOTHESIS TESTING ---
    with stat_tabs[0]:
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        col_test_ctrl, col_test_res = st.columns([1, 2], gap="large")

        with col_test_ctrl:
            st.markdown('<div class="control-panel-card">', unsafe_allow_html=True)
            st.subheader("Test Configuration")
            
            test_type = st.selectbox(
                "Select Statistical Test",
                [
                    "One-Sample T-Test",
                    "Independent Samples T-Test",
                    "Paired Samples T-Test",
                    "Mann-Whitney U Test",
                    "One-Way ANOVA",
                    "Kruskal-Wallis Test",
                    "Chi-Square Test of Independence"
                ]
            )

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            res = None # Store results here

            if test_type == "One-Sample T-Test":
                st.caption("Compares the mean of a single group to a known value.")
                target_col = st.selectbox("Target Variable", numeric_cols)
                pop_mean = st.number_input("Hypothesized Mean", value=0.0)
                
                if st.button("Run T-Test", use_container_width=True, type="primary"):
                    try:
                        stat, p_val = stats.ttest_1samp(df[target_col].dropna(), pop_mean)
                        res = {"test": "One-Sample T-Test", "stat": stat, "p": p_val, "var": target_col}
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif test_type == "Independent Samples T-Test":
                st.caption("Compares means of two independent groups.")
                target_col = st.selectbox("Numeric Variable", numeric_cols)
                group_col = st.selectbox("Grouping Variable (Categorical)", categorical_cols)
                
                if group_col:
                    groups = df[group_col].unique()
                    if len(groups) < 2:
                        st.warning("Grouping variable must have at least 2 unique values.")
                    else:
                        g1 = st.selectbox("Group 1", groups)
                        g2 = st.selectbox("Group 2", groups, index=1 if len(groups)>1 else 0)
                        
                        if st.button("Run T-Test", use_container_width=True, type="primary"):
                            try:
                                data1 = df[df[group_col] == g1][target_col].dropna()
                                data2 = df[df[group_col] == g2][target_col].dropna()
                                stat, p_val = stats.ttest_ind(data1, data2)
                                res = {"test": f"Independent T-Test ({g1} vs {g2})", "stat": stat, "p": p_val, "var": target_col}
                            except Exception as e:
                                st.error(f"Error: {e}")

            elif test_type == "Paired Samples T-Test":
                st.caption("Compares means of two related groups (e.g., Before vs After).")
                col_sq1, col_sq2 = st.columns(2)
                with col_sq1: var1 = st.selectbox("Variable 1 (Pre)", numeric_cols)
                with col_sq2: var2 = st.selectbox("Variable 2 (Post)", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
                
                if st.button("Run Paired T-Test", use_container_width=True, type="primary"):
                    try:
                        # Ensure same length and drop NA pairwise
                        combined = df[[var1, var2]].dropna()
                        stat, p_val = stats.ttest_rel(combined[var1], combined[var2])
                        res = {"test": f"Paired T-Test ({var1} vs {var2})", "stat": stat, "p": p_val, "var": f"{var1} - {var2}"}
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif test_type == "Mann-Whitney U Test":
                st.caption("Non-parametric alternative to Independent T-Test (compares medians).")
                target_col = st.selectbox("Numeric Variable", numeric_cols)
                group_col = st.selectbox("Grouping Variable", categorical_cols)
                
                if group_col:
                    groups = df[group_col].unique()
                    if len(groups) < 2:
                        st.warning("Grouping variable must have at least 2 unique values.")
                    else:
                        g1 = st.selectbox("Group 1", groups)
                        g2 = st.selectbox("Group 2", groups, index=1 if len(groups)>1 else 0)
                        
                        if st.button("Run Mann-Whitney", use_container_width=True, type="primary"):
                            try:
                                data1 = df[df[group_col] == g1][target_col].dropna()
                                data2 = df[df[group_col] == g2][target_col].dropna()
                                stat, p_val = stats.mannwhitneyu(data1, data2)
                                res = {"test": f"Mann-Whitney U ({g1} vs {g2})", "stat": stat, "p": p_val, "var": target_col}
                            except Exception as e:
                                st.error(f"Error: {e}")

            elif test_type == "One-Way ANOVA":
                st.caption("Compares means across 3+ groups (Parametric).")
                target_col = st.selectbox("Numeric Variable", numeric_cols)
                group_col = st.selectbox("Grouping Variable", categorical_cols)
                
                if st.button("Run ANOVA", use_container_width=True, type="primary"):
                    try:
                        groups = [df[df[group_col] == g][target_col].dropna() for g in df[group_col].unique()]
                        stat, p_val = stats.f_oneway(*groups)
                        res = {"test": "One-Way ANOVA", "stat": stat, "p": p_val, "var": target_col}
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif test_type == "Kruskal-Wallis Test":
                st.caption("Non-parametric alternative to ANOVA (compares distributions).")
                target_col = st.selectbox("Numeric Variable", numeric_cols)
                group_col = st.selectbox("Grouping Variable", categorical_cols)
                
                if st.button("Run Kruskal-Wallis", use_container_width=True, type="primary"):
                    try:
                        groups = [df[df[group_col] == g][target_col].dropna() for g in df[group_col].unique()]
                        stat, p_val = stats.kruskal(*groups)
                        res = {"test": "Kruskal-Wallis Test", "stat": stat, "p": p_val, "var": target_col}
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif test_type == "Chi-Square Test of Independence":
                st.caption("Tests association between two categorical variables.")
                col1 = st.selectbox("Variable 1", categorical_cols, key="chi_c1")
                col2 = st.selectbox("Variable 2", categorical_cols, key="chi_c2")
                
                if st.button("Run Chi-Square", use_container_width=True, type="primary"):
                    try:
                        contingency = pd.crosstab(df[col1], df[col2])
                        stat, p_val, dof, expected = stats.chi2_contingency(contingency)
                        res = {"test": "Chi-Square Test", "stat": stat, "p": p_val, "var": f"{col1} vs {col2}"}
                    except Exception as e:
                        st.error(f"Error: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_test_res:
            if res:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                
                # Result Header
                sig_color = "#10B981" if res['p'] < 0.05 else "#EF4444"
                sig_text = "Statistically Significant" if res['p'] < 0.05 else "Not Significant"
                
                st.markdown(f'''
                <div style="border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 1.5rem;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-secondary);">Test Results</div>
                    <h3 style="font-size: 1.8rem; color: var(--text-main); margin: 0.5rem 0;">{res['test']}</h3>
                    <div style="display:inline-block; padding: 4px 12px; border-radius: 99px; background: {sig_color}20; color: {sig_color}; font-weight: 600; font-size: 0.9rem;">
                        {sig_text} (p < 0.05)
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Metrics
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    st.metric("Test Statistic", f"{res['stat']:.4f}")
                with m_c2:
                    st.metric("P-Value", f"{res['p']:.4e}")
                with m_c3:
                    st.metric("Variable", res['var'])
                
                st.markdown("---")
                
                # Interpretation
                if res['p'] < 0.05:
                    st.success(f"**Conclusion:** There is sufficient evidence to reject the null hypothesis. The relationship or difference observed is unlikely to be due to chance.")
                else:
                    st.info(f"**Conclusion:** There is not enough evidence to reject the null hypothesis. The observed difference could plausible be due to random chance.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="empty-state-container">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🧪</div>
                    <h3>Ready for Analysis</h3>
                    <p>Select a statistical test on the left to begin.</p>
                </div>
                ''', unsafe_allow_html=True)

    # --- 2. CORRELATION ---
    with stat_tabs[1]:
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        st.markdown("#### Multivariate Correlation Analysis")
        
        numeric_cols_corr = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols_corr) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            sel_corr_cols = st.multiselect("Select Variables", numeric_cols_corr, default=numeric_cols_corr[:min(8, len(numeric_cols_corr))])
            method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
            
            if sel_corr_cols:
                corr_matrix = df[sel_corr_cols].corr(method=method)
                
                fig = px.imshow(
                    corr_matrix, 
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title=f"{method.capitalize()} Correlation Matrix",
                    template=theme_cfg['template']
                )
                fig.update_layout(
                    height=600, 
                    plot_bgcolor=theme_cfg['bg_color'], 
                    paper_bgcolor=theme_cfg['bg_color'],
                    font_color=theme_cfg['font_color']
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- 3. NORMALITY ---
    with stat_tabs[2]:
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        col_norm_1, col_norm_2 = st.columns([1, 2])
        
        with col_norm_1:
            st.markdown('<div class="control-panel-card">', unsafe_allow_html=True)
            st.subheader("Distribution Check")
            norm_col = st.selectbox("Select Variable", numeric_cols)
            
            res_shapiro = stats.shapiro(df[norm_col].dropna().sample(min(5000, len(df)))) # Limit sample for shapiro
            stats.kstest(df[norm_col].dropna(), 'norm')
            
            st.markdown("---")
            st.caption("SHAPIRO-WILK TEST")
            st.metric("Statistic", f"{res_shapiro.statistic:.4f}")
            st.metric("P-Value", f"{res_shapiro.pvalue:.4e}")
            
            is_normal = res_shapiro.pvalue > 0.05
            st.markdown(f"**Verdict:** {'Gaussian' if is_normal else 'Non-Gaussian'}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_norm_2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.subheader(f"Distribution of {norm_col}")
            fig_hist = px.histogram(
                df, x=norm_col, marginal="box", opacity=0.7, 
                color_discrete_sequence=['#6366F1'],
                template=theme_cfg['template']
            )
            fig_hist.update_layout(
                plot_bgcolor=theme_cfg['bg_color'], 
                paper_bgcolor=theme_cfg['bg_color'],
                font_color=theme_cfg['font_color']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # --- 4. DESCRIPTIVE STATS ---
    with stat_tabs[3]:
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
