"""
Premium Data Diagnosis Tab - Interactive Data Profiling
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import calculate_data_quality_metrics

def render_data_diagnosis_tab(df: pd.DataFrame):
    """
    Render the advanced data diagnosis and profiling tab.
    """
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # Header Section
    # -------------------------------------------------------------------------
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Data Health Audit</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                AI-driven quality assessment and structural analysis.
            </p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; color: var(--text-secondary);">LAST SCAN</div>
            <div style="font-family: 'JetBrains Mono', monospace;">JUST NOW</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if df.empty:
        st.warning("No data to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # -------------------------------------------------------------------------
    # 1. Health Score & High Level Metrics
    # -------------------------------------------------------------------------
    metrics = calculate_data_quality_metrics(df)
    score = metrics.get('overall_score', 0) * 100
    
    col_score, col_stats = st.columns([1, 2], gap="large")
    
    with col_score:
        # Gauge Chart for Health Score
        fig_score = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "<b>Health Score</b>", 'font': {'size': 16, 'color': '#64748B'}},
            number = {'suffix': "%", 'font': {'color': '#10B981' if score > 80 else '#F59E0B' if score > 50 else '#EF4444'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "rgba(0,0,0,0)"}, # Invisible bar, rely on threshold/steps
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 50], 'color': "#FEF2F2"},
                    {'range': [50, 80], 'color': "#FFFBEB"},
                    {'range': [80, 100], 'color': "#ECFDF5"}
                ],
                'threshold': {
                    'line': {'color': "#10B981" if score > 80 else "#F59E0B" if score > 50 else "#EF4444", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        fig_score.update_layout(height=250, margin=dict(t=30, b=0, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_score, use_container_width=True)
        
        # Verdict text
        verdict = "Excellent" if score > 85 else "Good" if score > 70 else "Fair" if score > 50 else "Critical"
        verdict_color = "#10B981" if score > 85 else "#10B981" if score > 70 else "#F59E0B" if score > 50 else "#EF4444"
        st.markdown(f"<div style='text-align: center; color: {verdict_color}; font-weight: 600; margin-top: -1rem;'>Verdict: {verdict}</div>", unsafe_allow_html=True)

    with col_stats:
        st.markdown("##### 📊 Vital Statistics")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Completeness", f"{metrics['completeness']*100:.1f}%", help="Percentage of non-missing cells")
            st.metric("Rows", f"{len(df):,}")
        with s2:
            st.metric("Consistency", f"{metrics['consistency']*100:.1f}%", help="Percentage of unique rows")
            st.metric("Columns", f"{len(df.columns)}")
        with s3:
            st.metric("Validity", f"{metrics['validity']*100:.1f}%", help="Based on outliers and types")
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Smart Alerts / Issues List
    # -------------------------------------------------------------------------
    st.subheader("⚠️ Diagnostic Findings")
    
    issues = []
    
    # Check Missing
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        for col in missing_cols:
            pct = df[col].isnull().mean()
            if pct > 0.05:
                issues.append({"type": "warning", "msg": f"Column **{col}** has {pct:.1%} missing values. Consider imputation."})
            elif pct > 0:
                issues.append({"type": "info", "msg": f"Column **{col}** has minor missing data ({pct:.1%})."})

    # Check Duplicates
    if df.duplicated().sum() > 0:
        issues.append({"type": "warning", "msg": f"Found {df.duplicated().sum()} duplicate rows that may skew analysis."})

    # Check Constant Columns
    for col in df.columns:
        if df[col].nunique() <= 1:
             issues.append({"type": "error", "msg": f"Column **{col}** has only 1 unique value. It provides no information."})

    # Check High Correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        if to_drop:
             issues.append({"type": "warning", "msg": f"High multicollinearity detected in: {', '.join(to_drop)}. Consider feature selection."})

    if not issues:
        st.success("✨ No significant data issues detected. Your dataset is robust!")
    else:
        for issue in issues:
            if issue['type'] == 'error':
                 st.error(issue['msg'], icon="🛑")
            elif issue['type'] == 'warning':
                 st.warning(issue['msg'], icon="⚠️")
            else:
                 st.info(issue['msg'], icon="ℹ️")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. Visual Deep Dives tabs
    # -------------------------------------------------------------------------
    diag_tab1, diag_tab2, diag_tab3 = st.tabs(["🔍 Missing Matrix", "📈 Outlier Scan", "🔗 Correlation Map"])

    with diag_tab1:
        st.caption("Visual representation of missing data (Yellow = Missing)")
        # Sample if too large for heatmap
        plot_df = df if len(df) < 1000 else df.sample(1000)
        
        # Create boolean matrix for nulls
        null_matrix = plot_df.isnull()
        
        if null_matrix.sum().sum() == 0:
             st.info("No missing values to visualize.")
        else:
            fig_missing = px.imshow(
                null_matrix, 
                aspect="auto", 
                labels=dict(x="Columns", y="Rows", color="Missing"),
                color_continuous_scale=[[0, "#E2E8F0"], [1, "#F59E0B"]],
                title=f"Missing Value Patterns (Sample of {len(plot_df)} rows)"
            )
            fig_missing.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_missing, use_container_width=True)

    with diag_tab2:
        st.caption("Distribution of numeric features with outlier highlighting")
        if not numeric_df.empty:
            sel_col = st.selectbox("Select Column to Inspect", numeric_df.columns)
            c1, c2 = st.columns(2)
            with c1:
                 fig_box = px.box(df, y=sel_col, title=f"Box Plot: {sel_col}", template="plotly_white")
                 fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                 st.plotly_chart(fig_box, use_container_width=True)
            with c2:
                 fig_hist = px.histogram(df, x=sel_col, marginal="box", title=f"Distribution: {sel_col}", template="plotly_white")
                 fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                 st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No numeric columns for outlier analysis.")

    with diag_tab3:
         st.caption("Correlation Heatmap to identify relationships and redundancy")
         if not numeric_df.empty and len(numeric_df.columns) > 1:
             corr = numeric_df.corr()
             fig_corr = px.imshow(
                 corr, 
                 text_auto=".2f", 
                 aspect="auto", 
                 color_continuous_scale="RdBu_r", 
                 zmin=-1, zmax=1,
                 title="Feature Correlation Matrix"
             )
             fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
             st.plotly_chart(fig_corr, use_container_width=True)
         else:
             st.info("Insufficient numeric columns for correlation.")

    st.markdown('</div>', unsafe_allow_html=True)
