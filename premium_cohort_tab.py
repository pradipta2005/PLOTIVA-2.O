import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_cohort_analysis_tab(df: pd.DataFrame):
    """
    Render Cohort Analysis Module
    """
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Cohort Intelligence</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Track user retention, churn, and behavioral patterns over time.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">Behavioral Analytics</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Cohort Definition")
        
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not date_cols:
            st.error("Dataset must contain date/time columns for cohort analysis.")
        else:
            cohort_date = st.selectbox("Cohort Date (e.g. Signup)", date_cols, index=0)
            activity_date = st.selectbox("Activity Date (e.g. Transaction)", date_cols, index=min(1, len(date_cols)-1))
            user_id = st.selectbox("Entity ID (e.g. User ID)", [c for c in df.columns if df[c].nunique() > 10])
            
            period = st.selectbox("Time Period", ["M (Month)", "Q (Quarter)", "Y (Year)"], index=0)
            freq = period[0]
            
            op = st.selectbox("Metric", ["Retention (Count)", "Average Value", "Sum Value"])
            val_col = None
            if op != "Retention (Count)":
                val_col = st.selectbox("Value Column", df.select_dtypes(include=np.number).columns)

            run_cohort = st.button("Generate Cohort Matrix", type="primary", use_container_width=True)

    with c2:
        if run_cohort and date_cols:
            with st.spinner("Analyzing cohorts..."):
                try:
                    df_cohort = df.copy()
                    
                    # 1. Create Cohort Month
                    df_cohort['CohortMonth'] = df_cohort[cohort_date].dt.to_period(freq)
                    
                    # 2. Create Activity Month
                    df_cohort['ActivityMonth'] = df_cohort[activity_date].dt.to_period(freq)
                    
                    # 3. Calculate Cohort Index (Time diff)
                    def diff_month(x):
                        d1 = x['ActivityMonth']
                        d2 = x['CohortMonth']
                        return (d1.year - d2.year) * 12 + d1.month - d2.month
                    
                    if freq == 'M':
                        df_cohort['CohortIndex'] = (df_cohort['ActivityMonth'] - df_cohort['CohortMonth']).apply(lambda x: x.n)
                    elif freq == 'Q':
                        df_cohort['CohortIndex'] = (df_cohort['ActivityMonth'] - df_cohort['CohortMonth']).apply(lambda x: x.n)
                    elif freq == 'Y':
                        df_cohort['CohortIndex'] = (df_cohort['ActivityMonth'] - df_cohort['CohortMonth']).apply(lambda x: x.n)
                    
                    # 4. Pivot
                    if op == "Retention (Count)":
                        # Count unique users per cohort per index
                        grouping = df_cohort.groupby(['CohortMonth', 'CohortIndex'])
                        cohort_data = grouping[user_id].apply(pd.Series.nunique).reset_index()
                        cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values=user_id)
                        
                        # Retention is percentage of index 0
                        cohort_size = cohort_counts.iloc[:, 0]
                        retention = cohort_counts.divide(cohort_size, axis=0) * 100
                        title_suffix = "Retention Rate (%)"
                        data_to_plot = retention
                        fmt = ".1f"
                    elif op == "Average Value":
                        grouping = df_cohort.groupby(['CohortMonth', 'CohortIndex'])
                        cohort_data = grouping[val_col].mean().reset_index()
                        data_to_plot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values=val_col)
                        title_suffix = f"Average {val_col}"
                        fmt = ".1f"
                    
                    # Heatmap
                    fig = px.imshow(
                        data_to_plot,
                        labels=dict(x="Periods Since Acquisition", y="Cohort Group", color=title_suffix),
                        x=[str(x) for x in data_to_plot.columns],
                        y=[str(y) for y in data_to_plot.index],
                        color_continuous_scale="Viridis_r" if op == "Retention (Count)" else "Viridis",
                        text_auto=fmt,
                        title=f"Cohort Analysis: {title_suffix}"
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Line Chart for trends
                    st.subheader("Cohort Trends")
                    # Plot retention curves for each cohort
                    data_long = data_to_plot.reset_index().melt(id_vars='CohortMonth', var_name='Period', value_name='Value')
                    fig_trend = px.line(data_long, x='Period', y='Value', color=data_long['CohortMonth'].astype(str), template="plotly_dark", title="Cohort Performance Over Time")
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Cohort creation failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
