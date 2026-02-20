
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_dashboard_builder_tab(df: pd.DataFrame):
    """
    Render Advanced Dashboard Builder
    """
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Executive Dashboard Studio</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Compose custom views with KPIs, charts, and interactive widgets.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">Builder Mode</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if 'custom_dashboards' not in st.session_state:
        st.session_state.custom_dashboards = {}

    tab_build, tab_view = st.tabs(["🔨 Build & Edit", "👀 View Dashboard"])

    # Helper to render a single widget
    def render_widget(widget_conf, df, key_prefix):
        w_type = widget_conf.get('type')
        if w_type == 'KPI':
            col = widget_conf.get('val_col')
            if col and col in df.columns:
                val = df[col].sum()
                st.metric(widget_conf.get('label', 'KPI'), f"{val:,.0f}")
            else:
                st.warning(f"Column {col} not found")
        elif w_type == 'Chart':
            c_type = widget_conf.get('chart_type')
            x = widget_conf.get('x')
            y = widget_conf.get('y')
            color = widget_conf.get('color')
            
            if x and y and x in df.columns and y in df.columns:
                try:
                    if c_type == "Bar":
                        fig = px.bar(df, x=x, y=y, color=color, template="plotly_dark", title=f"{y} by {x}")
                    elif c_type == "Line":
                        fig = px.line(df, x=x, y=y, color=color, template="plotly_dark", title=f"{y} over {x}")
                    elif c_type == "Scatter":
                        fig = px.scatter(df, x=x, y=y, color=color, template="plotly_dark", title=f"{y} vs {x}")
                    elif c_type == "Pie":
                        fig = px.pie(df, names=x, values=y, template="plotly_dark", title=f"{y} Distribution")
                    else:
                        fig = go.Figure()
                    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")
                except Exception as e:
                    st.error(f"Chart Error: {str(e)}")
            else:
                 st.info("Configure chart axes")
        elif w_type == 'Table':
             cols = widget_conf.get('table_cols')
             if cols:
                 st.dataframe(df[cols].head(10), use_container_width=True)
             else:
                 st.dataframe(df.head(), use_container_width=True)

    with tab_build:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Layout Configuration")
            dash_name = st.text_input("Dashboard Name", "My Exec Dashboard")
            layout_type = st.selectbox("Grid Layout", ["1x1 (Single)", "1x2 (Split V)", "2x1 (Split H)", "2x2 (Grid)"])
            
            st.subheader("Widget Selection")
            
            num_slots = 1
            if layout_type == "2x2 (Grid)": num_slots = 4
            elif "2" in layout_type: num_slots = 2
            
            current_widgets = {}
            
            for i in range(1, num_slots + 1):
                with st.expander(f"Slot {i} Configuration", expanded=(i==1)):
                    w_type = st.selectbox(f"Type {i}", ["KPI", "Chart", "Table"], key=f"w{i}_type")
                    widget_conf = {'type': w_type}
                    
                    if w_type == "KPI":
                        widget_conf['label'] = st.text_input(f"Label {i}", f"Metric {i}", key=f"w{i}_l")
                        widget_conf['val_col'] = st.selectbox(f"Value Column {i}", df.select_dtypes(include=np.number).columns, key=f"w{i}_v")
                    
                    elif w_type == "Chart":
                        widget_conf['chart_type'] = st.selectbox(f"Chart Type {i}", ["Bar", "Line", "Scatter", "Pie"], key=f"w{i}_ct")
                        widget_conf['x'] = st.selectbox(f"X Axis {i}", df.columns, key=f"w{i}_x")
                        widget_conf['y'] = st.selectbox(f"Y Axis {i}", df.select_dtypes(include=np.number).columns, key=f"w{i}_y")
                        widget_conf['color'] = st.selectbox(f"Color {i}", [None] + df.columns.tolist(), key=f"w{i}_col")
                        
                    elif w_type == "Table":
                        widget_conf['table_cols'] = st.multiselect(f"Columns {i}", df.columns, default=df.columns[:3].tolist(), key=f"w{i}_tc")
                    
                    current_widgets[f'w{i}'] = widget_conf

            if st.button("💾 Save Dashboard", type="primary"):
                st.session_state.custom_dashboards[dash_name] = {
                    'layout': layout_type,
                    'widgets': current_widgets,
                    'timestamp': pd.Timestamp.now()
                }
                st.toast(f"Dashboard '{dash_name}' saved successfully!", icon="💾")

        with c2:
            st.subheader("Live Preview: " + dash_name)
            st.markdown(f"Layout: `{layout_type}`")
            
            # Preview Rendering
            slots = []
            if layout_type == "1x1 (Single)":
                slots = [st.container()]
            elif layout_type == "2x1 (Split H)":
                slots = st.columns(2)
            elif layout_type == "1x2 (Split V)":
                c = st.container()
                slots = [c.container(), c.container()] # Stacked
            elif layout_type == "2x2 (Grid)":
                r1 = st.columns(2)
                r2 = st.columns(2)
                slots = r1 + r2
            
            for i, slot in enumerate(slots):
                with slot:
                    w_key = f'w{i+1}'
                    if w_key in current_widgets:
                        render_widget(current_widgets[w_key], df, f"prev_{i}")
                    else:
                        st.info(f"Slot {i+1} Empty")

    with tab_view:
        if not st.session_state.custom_dashboards:
             st.info("No custom dashboards saved yet. Switch to the 'Build & Edit' tab to create one.")
        else:
            c_sel, c_del = st.columns([3, 1])
            with c_sel:
                current_dash = st.selectbox("Select Dashboard", list(st.session_state.custom_dashboards.keys()), label_visibility="collapsed")
            with c_del:
                if st.button("Delete"):
                    del st.session_state.custom_dashboards[current_dash]
                    st.rerun()

            if current_dash in st.session_state.custom_dashboards:
                dash_data = st.session_state.custom_dashboards[current_dash]
                l_type = dash_data['layout']
                w_data = dash_data['widgets']
                
                st.caption(f"Viewing: {current_dash} • {l_type}")
                st.markdown("---")
                
                # View Rendering (Same logic as preview)
                v_slots = []
                if l_type == "1x1 (Single)":
                    v_slots = [st.container()]
                elif l_type == "2x1 (Split H)":
                    v_slots = st.columns(2)
                elif l_type == "1x2 (Split V)":
                    vc = st.container()
                    v_slots = [vc.container(), vc.container()]
                elif l_type == "2x2 (Grid)":
                    vr1 = st.columns(2)
                    vr2 = st.columns(2)
                    v_slots = vr1 + vr2
                
                for i, slot in enumerate(v_slots):
                    with slot:
                        w_key = f'w{i+1}'
                        if w_key in w_data:
                            render_widget(w_data[w_key], df, f"view_{current_dash}_{i}")
                        else:
                            st.empty()

    st.markdown('</div>', unsafe_allow_html=True)
