import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from statsmodels.stats.power import TTestIndPower


def render_ab_testing_tab(df: pd.DataFrame):
    """
    Render A/B Testing & Experimentation Module
    """
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Experimentation Lab</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Design, analyze, and optimize experiments with rigorous statistical frameworks.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">A/B & MVT</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    tab_design, tab_analysis = st.tabs(["📐 Experiment Design", "📊 Result Analysis"])

    # -------------------------------------------------------------------------
    # EXPERIMENT DESIGN (Sample Size, Power)
    # -------------------------------------------------------------------------
    with tab_design:
        st.subheader("Sample Size Calculator")
        
        c1, c2 = st.columns(2)
        with c1:
            exp_type = st.selectbox("Metric Type", ["Conversion Rate (Binary)", "Average Value (Continuous)"])
            alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01, help="Probability of False Positive (Type I Error)")
            power = st.slider("Statistical Power (1-β)", 0.5, 0.99, 0.8, 0.05, help="Probability of detecting an effect if it exists")
            
        with c2:
            if "Conversion" in exp_type:
                base_rate = st.number_input("Baseline Conversion Rate (%)", 0.1, 100.0, 10.0) / 100
                mde = st.number_input("Minimum Detectable Effect (Relative %)", 1.0, 100.0, 5.0) / 100
                
                if st.button("Calculate Sample Size"):
                    # Effect size for proportions (Cohen's h)
                    p1 = base_rate
                    p2 = base_rate * (1 + mde)
                    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
                    
                    analysis = TTestIndPower()
                    # Using t-test approximation for proportions usually okay for large N, but technically should use Z-test power
                    # Simplification: Use TTestIndPower for both
                    n = analysis.solve_power(effect_size=abs(h), alpha=alpha, power=power, ratio=1.0)
                    
                    st.success(f"Required Sample Size: **{int(np.ceil(n))}** per variation")
                    st.info(f"Total Visitors Needed: {int(np.ceil(n)*2)}")
                    
            else:
                st.number_input("Baseline Mean", 0.0, 100000.0, 100.0)
                std_dev = st.number_input("Standard Deviation", 0.1, 10000.0, 20.0)
                mde_abs = st.number_input("Minimum Detectable Effect (Absolute Value)", 0.1, 1000.0, 5.0)
                
                if st.button("Calculate Sample Size"):
                    effect_size = mde_abs / std_dev
                    analysis = TTestIndPower()
                    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
                    
                    st.success(f"Required Sample Size: **{int(np.ceil(n))}** per group")

    # -------------------------------------------------------------------------
    # ANALYSIS (T-Test, Chi-Square on dataset)
    # -------------------------------------------------------------------------
    with tab_analysis:
        st.subheader("Analyze Results")
        
        c1, c2 = st.columns(2)
        with c1:
            group_col = st.selectbox("Group Column (Variant)", [c for c in df.columns if df[c].nunique() < 10])
            metric_col = st.selectbox("Metric Column", df.select_dtypes(include=[np.number]).columns)
            
        with c2:
            test_method = st.selectbox("Statistical Test", ["Auto-Detect", "T-Test (Means)", "Mann-Whitney U (Medians)", "Chi-Square (Proportions)"])
            
        if group_col and metric_col:
            groups = df[group_col].dropna().unique()
            if len(groups) < 2:
                st.warning("Selected group column has fewer than 2 unique values.")
            else:
                st.markdown("---")
                # Group stats
                st.write("### 📈 Group Performance")
                stats_df = df.groupby(group_col)[metric_col].agg(['count', 'mean', 'std', 'median']).reset_index()
                st.dataframe(stats_df, use_container_width=True)
                
                base_group = st.selectbox("Select Control Group", groups, index=0)
                
                if st.button("🚀 Run Significance Test", type="primary"):
                    treatment_groups = [g for g in groups if g != base_group]
                    
                    for treat in treatment_groups:
                        st.markdown(f"#### Control ({base_group}) vs {treat}")
                        
                        ctrl_data = df[df[group_col] == base_group][metric_col].dropna()
                        trtm_data = df[df[group_col] == treat][metric_col].dropna()
                        
                        # Logic for test selection
                        p_val = None
                        stat = None
                        test_name = ""
                        
                        if test_method == "Chi-Square (Proportions)":
                            # Treat metric as binary 0/1 (sum/count = rate)
                             # Only valid if metric is binary. If not, warn user.
                             if not set(df[metric_col].unique()).issubset({0, 1}):
                                 st.error("Chi-Square requires binary metric (0/1). Using T-Test instead.")
                                 test_method = "T-Test (Means)" # Fallback
                             else:
                                 count = [trtm_data.sum(), ctrl_data.sum()]
                                 nobs = [len(trtm_data), len(ctrl_data)]
                                 # Using proportions_ztest or chi2
                                 from statsmodels.stats.proportion import \
                                     proportions_ztest
                                 try:
                                     stat, p_val = proportions_ztest(count, nobs)
                                     test_name = "Z-Test for Proportions"
                                 except:
                                     st.error("Error in proportion test.")

                        if test_method == "Mann-Whitney U (Medians)":
                            stat, p_val = stats.mannwhitneyu(ctrl_data, trtm_data)
                            test_name = "Mann-Whitney U"
                            
                        if test_method == "T-Test (Means)" or test_method == "Auto-Detect" and not test_name:
                             stat, p_val = stats.ttest_ind(ctrl_data, trtm_data, equal_var=False)
                             test_name = "Welch's T-Test"

                        # Result
                        if p_val is not None:
                            col_res, col_viz = st.columns([1, 2])
                            with col_res:
                                is_sig = p_val < 0.05
                                color = "#10B981" if is_sig else "#EF4444"
                                result_text = "SIGNIFICANT" if is_sig else "NOT SIGNIFICANT"
                                
                                st.markdown(f'''
                                <div style="padding: 1rem; border: 1px solid {color}; border-radius: 8px; background: {color}10;">
                                    <div style="color: {color}; font-weight: 700; letter-spacing: 1px;">{result_text}</div>
                                    <div style="font-size: 2rem; font-weight: 600;">p = {p_val:.4f}</div>
                                    <div style="font-size: 0.8rem; opacity: 0.8;">Test: {test_name}</div>
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                # Lift
                                lift = (trtm_data.mean() - ctrl_data.mean()) / ctrl_data.mean() * 100
                                st.metric("Observed Lift", f"{lift:+.2f}%")
                            
                            with col_viz:
                                # Distribution Plot
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(x=ctrl_data, name=f'Control ({base_group})', opacity=0.6, marker_color='#94A3B8'))
                                fig.add_trace(go.Histogram(x=trtm_data, name=f'{treat}', opacity=0.6, marker_color='#3B82F6'))
                                fig.update_layout(barmode='overlay', title="Distribution Comparison", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
