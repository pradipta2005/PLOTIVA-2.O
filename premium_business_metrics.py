
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_business_metrics_tab():
    """
    Render Premium Business Intelligence Suite.
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
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Business Intelligence Lab</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Calculate key performance indicators (KPIs) and simulate growth scenarios.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">Strategic Insight 🚀</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    metric_tabs = st.tabs(["💰 SaaS Metrics", "📈 Financial Health", "📊 Growth & Conversion", "🧮 Custom Formula"])

    # --- 1. SaaS Metrics ---
    with metric_tabs[0]:
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        col_saas_1, col_saas_2 = st.columns([1, 1], gap="large")

        with col_saas_1:
            st.markdown("#### Customer Value Engines")
            
            # CLV Calculator
            with st.expander("💎 Customer Lifetime Value (LTV)", expanded=True):
                arpu = st.number_input("Average Revenue Per User (ARPU)", value=50.0)
                churn_rate = st.slider("Churn Rate (%)", 0.1, 20.0, 5.0, 0.1)
                gross_margin = st.slider("Gross Margin (%)", 10, 100, 80)
                
                clv = (arpu * (gross_margin/100)) / (churn_rate/100)
                
                st.metric("Estimated LTV", f"${clv:,.2f}")
                st.caption(f"Formula: (ARPU * Margin) / Churn")

            # CAC Calculator
            with st.expander("💸 Customer Acquisition Cost (CAC)", expanded=False):
                marketing_spend = st.number_input("Total Marketing Spend", value=10000.0)
                sales_spend = st.number_input("Total Sales Spend", value=5000.0)
                new_customers = st.number_input("New Customers Acquired", value=100, min_value=1)
                
                cac = (marketing_spend + sales_spend) / new_customers
                
                st.metric("CAC", f"${cac:,.2f}")
                
            # LTV:CAC Ratio
            if cac > 0:
                ratio = clv / cac
                st.metric("LTV : CAC Ratio", f"{ratio:.2f}x", delta="Healthy > 3.0x" if ratio > 3 else "Needs Optimization")

        with col_saas_2:
            st.markdown("#### Retention Analysis")
            
            start_cust = st.number_input("Customers at Start of Period", value=500)
            end_cust = st.number_input("Customers at End of Period", value=530)
            new_cust_period = st.number_input("New Customers During Period", value=50)
            
            if start_cust > 0:
                churn_calc = ((start_cust - (end_cust - new_cust_period)) / start_cust) * 100
                churn_calc = max(0, churn_calc) # Prevent negative churn if input is weird
                retention_rate = 100 - churn_calc
                
                col_r1, col_r2 = st.columns(2)
                col_r1.metric("Churn Rate", f"{churn_calc:.2f}%", delta_color="inverse")
                col_r2.metric("Retention Rate", f"{retention_rate:.2f}%")
                
                # Visual
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = retention_rate,
                    title = {'text': "Retention Health", 'font': {'color': theme_cfg['font_color']}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': theme_cfg['font_color']},
                        'bar': {'color': "#10B981"},
                        'steps': [
                            {'range': [0, 80], 'color': "#FEF3C7"},
                            {'range': [80, 100], 'color': "#D1FAE5"}
                        ],
                        'bordercolor': theme_cfg['font_color']
                    },
                    number = {'font': {'color': theme_cfg['font_color']}}
                ))
                fig.update_layout(
                    height=300, 
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor=theme_cfg['bg_color'],
                    font_color=theme_cfg['font_color']
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- 2. Financial Health ---
    with metric_tabs[1]:
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        col_fin_1, col_fin_2 = st.columns(2)
        
        with col_fin_1:
            st.markdown("#### Profitability")
            revenue = st.number_input("Total Revenue", value=150000.0)
            cogs = st.number_input("Cost of Goods Sold", value=40000.0)
            opex = st.number_input("Operating Expenses", value=60000.0)
            
            gross_profit = revenue - cogs
            net_profit = gross_profit - opex
            
            gross_margin_pct = (gross_profit / revenue) * 100 if revenue else 0
            net_margin_pct = (net_profit / revenue) * 100 if revenue else 0
            
            st.metric("Gross Profit Margin", f"{gross_margin_pct:.1f}%")
            st.metric("Net Profit Margin", f"{net_margin_pct:.1f}%")

        with col_fin_2:
            st.markdown("#### Return on Investment (ROI)")
            investment = st.number_input("Invested Amount", value=50000.0)
            current_value = st.number_input("Current Value / Return", value=75000.0)
            
            roi = ((current_value - investment) / investment) * 100 if investment else 0
            
            st.metric("ROI", f"{roi:.1f}%", delta=f"${current_value - investment:,.0f} Gain")

    # --- 3. Growth ---
    with metric_tabs[2]:
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        
        st.markdown("#### Compound Annual Growth Rate (CAGR)")
        start_val = st.number_input("Beginning Value", value=10000.0)
        end_val = st.number_input("Ending Value", value=25000.0)
        years = st.number_input("Number of Years", value=3.0)
        
        if start_val > 0 and years > 0:
            cagr = ((end_val / start_val) ** (1/years) - 1) * 100
            st.metric("CAGR", f"{cagr:.2f}%")
            
            # Comparison
            st.caption("Projected Value for Next 5 Years at this Rate:")
            future_vals = [end_val * ((1 + cagr/100) ** i) for i in range(1, 6)]
            
            # Simple line chart using plotly to control theme
            df_proj = pd.DataFrame({
                'Year': [f"Year {i}" for i in range(1, 6)],
                'Value': future_vals
            })
            fig_proj = px.line(df_proj, x='Year', y='Value', template=theme_cfg['template'], markers=True)
            fig_proj.update_traces(line_color="#10B981")
            fig_proj.update_layout(
                plot_bgcolor=theme_cfg['bg_color'], 
                paper_bgcolor=theme_cfg['bg_color'],
                font_color=theme_cfg['font_color']
            )
            st.plotly_chart(fig_proj, use_container_width=True)
    
    # --- 4. Custom Formula Builder ---
    with metric_tabs[3]:
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        st.markdown("#### 🧮 Custom KPI Builder")
        st.caption("Create your own metrics by combining columns using standard math operations (+, -, *, /).")
        
        c_build, c_res = st.columns([1.5, 1])
        
        with c_build:
            # Variable Selector helper
            df = st.session_state.get('working_data', pd.DataFrame())
            if not df.empty:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                st.markdown("**Available Variables:**")
                st.code(", ".join(numeric_cols) if numeric_cols else "No numeric columns found.")
                
                KPI_name = st.text_input("KPI Name", "Profit Margin")
                formula = st.text_input("Formula", "Revenue - Cost", help="Use column names exactly as shown above.")
                
                calc_btn = st.button("Calculate Metric", type="primary")
                
        with c_res:
            st.markdown("**Preview**")
            if 'calc_btn' in locals() and calc_btn:
                if df.empty:
                    st.error("No data available.")
                else:
                    try:
                        # 1. Check if formula uses columns
                        # We use pandas eval for efficiency and safety (relative)
                        # We need to handle case where users reference columns vs scalars
                        # For a KPI, we usually want the SUM or MEAN of the result?
                        # Or is it a row-level calculation?
                        # Usually Key Metrics are aggregates.
                        # Interpretation: calculate row-wise, then sum? Or sum then calculate?
                        # Let's support both syntax: 'SUM(Revenue)' vs 'Revenue'.
                        # For simplicity: We calculate column-wise operation, then show Mean/Sum.
                        
                        result_series = df.eval(formula)
                        valid_count = result_series.count()
                        
                        avg_val = result_series.mean()
                        sum_val = result_series.sum()
                        
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: rgba(var(--accent-rgb), 0.1); border: 1px solid var(--accent); border-radius: 12px; text-align: center;">
                            <div style="font-size: 0.9rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.5rem;">{KPI_name} (Avg)</div>
                            <div style="font-size: 2rem; font-weight: 700; color: var(--text-main);">{avg_val:,.2f}</div>
                            <div style="font-size: 0.8rem; margin-top: 1rem; border-top: 1px solid var(--border); padding-top: 0.5rem; display: flex; justify-content: space-between;">
                                <span>Sum: {sum_val:,.0f}</span>
                                <span>Rows: {valid_count:,}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Histogram of distribution
                        fig = px.histogram(result_series, nbins=30, title=f"Distribution of {KPI_name}", template=theme_cfg['template'])
                        fig.update_layout(showlegend=False, height=200, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Computation Error: {str(e)}")
                        st.info("Tip: Ensure column names are correct and operations are valid for numeric data.")
            else:
                 st.info("Enter a formula to see results.")
