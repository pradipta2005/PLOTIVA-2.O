import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_scenario_tab(df: pd.DataFrame):
    """
    Render Scenario & Sensitivity Analysis Module
    """
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Scenario Simulator</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Model outcomes, test assumptions, and quantify risk with Monte Carlo simulations.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">What-If Analysis</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    tab_model, tab_sim = st.tabs(["📝 Model Builder", "🎲 Monte Carlo Simulation"])

    # -------------------------------------------------------------------------
    # MODEL BUILDER
    # -------------------------------------------------------------------------
    with tab_model:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Define Parameters")
            params = {}
            num_params = st.number_input("Number of Variables", 1, 10, 3)
            
            for i in range(num_params):
                col_p1, col_p2 = st.columns(2)
                p_name = col_p1.text_input(f"Var {i+1} Name", f"Var_{i+1}", key=f"p_name_{i}")
                p_val = col_p2.number_input(f"Base Value", value=100.0, key=f"p_val_{i}")
                params[p_name] = p_val
                
            st.subheader("Define Formula")
            formula = st.text_area("Formula (use var names)", "Var_1 * Var_2 - Var_3")
            
            try:
                # Safe eval
                result = eval(formula, {"__builtins__": None}, params)
                st.metric("Base Case Result", f"{result:,.2f}")
            except Exception as e:
                st.error("Invalid Formula or Parameters")

        with c2:
            st.subheader("Sensitivity Analysis (Tornado Chart)")
            if params and formula:
                sensitivity_data = []
                
                # Vary each parameter significantly (+/- 20%)
                for p_name, p_val in params.items():
                    # High Case
                    params_high = params.copy()
                    params_high[p_name] = p_val * 1.2
                    res_high = eval(formula, {}, params_high)
                    
                    # Low Case
                    params_low = params.copy()
                    params_low[p_name] = p_val * 0.8
                    res_low = eval(formula, {}, params_low)
                    
                    sensitivity_data.append({
                        'Parameter': p_name,
                        'Low': res_low,
                        'High': res_high,
                        'Range': abs(res_high - res_low),
                        'Base': result
                    })
                
                sens_df = pd.DataFrame(sensitivity_data).sort_values('Range', ascending=True)
                
                fig = go.Figure()
                for i, row in sens_df.iterrows():
                    fig.add_trace(go.Bar(
                        y=[row['Parameter']], x=[row['Low'] - row['Base']],
                        base=[row['Base']],
                        orientation='h',
                        name='-20%',
                        marker_color='#EF4444',
                        showlegend=False
                    ))
                    fig.add_trace(go.Bar(
                        y=[row['Parameter']], x=[row['High'] - row['Base']], 
                        base=[row['Base']], # Actually base is Base, width is High - Base
                        # Wait, stacked bar logic is width.
                        # For a Tornado, usually it's centered around Base.
                        # Let's use simple bar chart relative to base
                        orientation='h',
                        name='+20%',
                        marker_color='#10B981',
                        showlegend=False
                    ))

                # Correct Tornado Logic:
                # Bar 1: Start at Low, End at High? No.
                # It's easier to plot deviation from mean.
                
                fig_tornado = go.Figure()
                fig_tornado.add_trace(go.Bar(
                    y=sens_df['Parameter'], 
                    x=sens_df['Low'] - sens_df['Base'],
                    orientation='h', name='-20%', marker_color='#EF4444'
                ))
                fig_tornado.add_trace(go.Bar(
                    y=sens_df['Parameter'], 
                    x=sens_df['High'] - sens_df['Base'],
                    orientation='h', name='+20%', marker_color='#10B981'
                ))
                
                fig_tornado.update_layout(
                    title="Sensitivity (Impact of +/- 20% change)",
                    barmode='relative', 
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Change in Result"
                )
                st.plotly_chart(fig_tornado, use_container_width=True)

    # -------------------------------------------------------------------------
    # MONTE CARLO SIMULATION
    # -------------------------------------------------------------------------
    with tab_sim:
        st.subheader("Monte Carlo Simulation")
        iterations = st.slider("Iterations", 100, 5000, 1000)
        
        sim_params = {}
        cols = st.columns(3)
        i = 0
        for p_name, p_val in params.items():
            with cols[i % 3]:
                st.markdown(f"**{p_name}** Distribution")
                dist = st.selectbox(f"Dist for {p_name}", ["Normal", "Uniform", "Triangular"], key=f"dist_{p_name}")
                if dist == "Normal":
                    std = st.number_input(f"Std Dev ({p_name})", 0.0, p_val, p_val*0.1, key=f"std_{p_name}")
                    sim_params[p_name] = lambda n, m=p_val, s=std: np.random.normal(m, s, n)
                elif dist == "Uniform":
                    low = st.number_input(f"Min ({p_name})", 0.0, p_val, p_val*0.9, key=f"min_{p_name}")
                    high = st.number_input(f"Max ({p_name})", p_val, p_val*2, p_val*1.1, key=f"max_{p_name}")
                    sim_params[p_name] = lambda n, l=low, h=high: np.random.uniform(l, h, n)
                elif dist == "Triangular":
                    low = st.number_input(f"Min ({p_name})", 0.0, p_val, p_val*0.9, key=f"min_t_{p_name}")
                    mode = st.number_input(f"Mode ({p_name})", low, p_val*2, p_val, key=f"mode_{p_name}")
                    high = st.number_input(f"Max ({p_name})", mode, p_val*2, p_val*1.1, key=f"max_t_{p_name}")
                    sim_params[p_name] = lambda n, l=low, m=mode, h=high: np.random.triangular(l, m, h, n)
            i += 1
            
        if st.button("🎲 Run Simulation", type="primary"):
            # Generate random data
            sim_data = pd.DataFrame()
            for p_name, func in sim_params.items():
                sim_data[p_name] = func(iterations)
                
            # Calculate result
            try:
                # Vectorized evaluation
                sim_data['Result'] = sim_data.apply(lambda row: eval(formula, {}, row.to_dict()), axis=1)
                
                c_res, c_chart = st.columns([1, 2])
                with c_res:
                    st.metric("Mean Outcome", f"{sim_data['Result'].mean():,.2f}")
                    st.metric("P5 (Worst Case)", f"{sim_data['Result'].quantile(0.05):,.2f}")
                    st.metric("P95 (Best Case)", f"{sim_data['Result'].quantile(0.95):,.2f}")
                    prob_loss = (sim_data['Result'] < 0).mean() * 100
                    st.metric("Probability of Loss (< 0)", f"{prob_loss:.1f}%")
                    
                with c_chart:
                    fig_hist = px.histogram(sim_data, x="Result", nbins=50, title="Outcome Distribution", template="plotly_dark")
                    fig_hist.add_vline(x=sim_data['Result'].mean(), line_dash="dash", line_color="green", annotation_text="Mean")
                    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Simulation Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
