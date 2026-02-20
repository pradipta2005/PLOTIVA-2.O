"""
Dashboard Tab - Interactive Charts Dashboard
"""

import pandas as pd
import streamlit as st


def render_dashboard_tab(df=None):
    """Render the main dashboard tab with saved charts"""
    
    # Minimal Luxury Header
    st.markdown("""
    <div class="animate-enter" style="text-align: center; margin-bottom: 4rem;">
        <h1 style="font-family: 'Playfair Display', serif; font-size: 3.5rem; margin-bottom: 1rem; color: var(--text-main);">Executive Dashboard</h1>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: var(--text-secondary); letter-spacing: 0.05em; font-weight: 300;">
            A curated collection of your most critical data narratives.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Premium data status with auto-update indicator
    current_df = df if df is not None else st.session_state.get('working_data', pd.DataFrame())
    
    if current_df is not None and not current_df.empty:
        # Check if length matches original data
        original_len = len(st.session_state.get('working_data', []))
        is_filtered = len(current_df) != original_len
        
        status_html = ""
        if is_filtered:
            status_html = f"""
            <div class="animate-enter delay-100" style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 3rem;">
                <span class="phase-tag" style="border-color: var(--accent); color: var(--accent);">LIVE SYNC</span>
                <span style="font-family: 'Inter'; font-size: 0.9rem; color: var(--text-secondary);">
                    Viewing <strong>{len(current_df):,}</strong> filtered records. Analysis updates in real-time.
                </span>
            </div>
            """
        else:
             status_html = f"""
            <div class="animate-enter delay-100" style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 3rem;">
                <span class="phase-tag">COMPLETE</span>
                <span style="font-family: 'Inter'; font-size: 0.9rem; color: var(--text-secondary);">
                    Analyzing full dataset of <strong>{len(current_df):,}</strong> records.
                </span>
            </div>
            """
        st.markdown(status_html, unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'dashboard_charts') or not st.session_state.dashboard_charts:
        # Empty dashboard state
        st.markdown("""
        <div class="empty-state-container animate-enter delay-200">
            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.3;">📊</div>
            <h2 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.8rem; margin-bottom: 0.5rem; color: var(--text-main);">The Canvas is Blank</h2>
            <p style="font-size: 1rem; color: var(--text-secondary); margin-bottom: 2rem; max-width: 400px; line-height: 1.6;">
                Your executive insights will appear here. Start by exploring your data and pinning key visualizations.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center; width: 100%;">
                <div style="text-align: center;">
                 <!-- Streamlit button hack for centering -->
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_alt1, c_alt2, c_alt3 = st.columns([1, 2, 1])
        with c_alt2:
             if st.button("✨ Auto-Generate Dashboard", use_container_width=True, type="primary"):
                 with st.spinner("Analyzing data structure and generating insights..."):
                     generate_smart_dashboard(current_df)
                     st.rerun()
        return
    
    # Dashboard controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 📊 **{len(st.session_state.dashboard_charts)}** Active Insights")
    
    with col2:
        layout_style = st.selectbox("Layout", ["Grid", "Single Column"], index=0, label_visibility="collapsed")
    
    with col3:
        if st.button("🗑️ Clear Canvas", help="Remove all charts"):
            st.session_state.dashboard_charts = {}
            st.rerun()
    
    st.markdown("---")
    
    # Display charts based on layout
    if layout_style == "Grid":
        # Grid layout (2 columns)
        dashboard_cols = st.columns(2)
        for i, (key, plot_data) in enumerate(st.session_state.dashboard_charts.items()):
            with dashboard_cols[i % 2]:
                render_dashboard_chart(key, plot_data, i, current_df)
    else:
        # Single column layout
        for i, (key, plot_data) in enumerate(st.session_state.dashboard_charts.items()):
            render_dashboard_chart(key, plot_data, i, current_df)

def render_dashboard_chart(key, plot_data, index, df=None):
    """Render individual chart in dashboard with current filtered data"""
    
    # Update chart logic... (Using existing recreation logic)
    current_df = df if df is not None else st.session_state.get('working_data', None)
    if current_df is not None and not current_df.empty:
        updated_fig = recreate_chart_with_current_data(current_df, plot_data)
        if updated_fig:
            plot_data['fig'] = updated_fig
            plot_data['data_rows'] = len(current_df)
            plot_data['last_updated'] = pd.Timestamp.now()
    
    # Container structure
    st.markdown(f'<div class="premium-card animate-slide-up" style="animation-delay: {index * 0.1}s; padding: 1.5rem;">', unsafe_allow_html=True)
    
    # Header
    cols = st.columns([1, 20]) # Spacer for icon if needed, currently just title
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
        <div>
            <h3 style="margin: 0; font-size: 1.25rem; color: var(--text-main);">{plot_data['title']}</h3>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem;">
                Based on {plot_data.get('data_rows', 'N/A'):,} records • Updated {plot_data.get('last_updated', pd.Timestamp.now()).strftime('%H:%M')}
            </div>
        </div>
        <div style="background: rgba(var(--accent), 0.1); color: var(--accent); padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
            LIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    with st.container():
        st.plotly_chart(plot_data['fig'], use_container_width=True, config={'displayModeBar': "hover"})

    # Footer Actions
    st.markdown('<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border); display: flex; justify-content: flex-end; gap: 0.5rem;">', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1,1,1]) # Use columns for buttons to keep them small/aligned? No, markdown + buttons is tricky.
    # Better to use st.columns for the button row
    b_col1, b_col2, b_col3, spacer = st.columns([1, 1, 1, 3])
    with b_col1:
         if st.button("📤", key=f"exp_{key}", help="Export"): st.toast("Export ready")
    with b_col2:
         if st.button("️🗑️", key=f"del_{key}", help="Remove"): 
             del st.session_state.dashboard_charts[key]
             st.rerun()
             
    st.markdown('</div></div>', unsafe_allow_html=True)

def recreate_chart_with_current_data(df, plot_data):
    """Recreate chart with current filtered data"""
    try:
        from premium_visualization_tab import create_premium_chart
        
        chart_type = plot_data.get('type', 'scatter')
        config = plot_data.get('config', {})
        colors = plot_data.get('colors', ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        # Recreate the chart with current data
        updated_fig = create_premium_chart(df, chart_type, config, colors, 'modern')
        
        if updated_fig:
            # Apply the same styling as the original
            updated_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors[0],
                title_font_color=colors[0],
                title_font_size=18,
                showlegend=True,
                margin=dict(t=50, b=50, l=50, r=50),
                height=500
            )
            
            return updated_fig
    except Exception:
        pass
    
    
    # Return original figure if recreation fails
    return plot_data.get('fig')

def generate_smart_dashboard(df):
    """Automatically generate a dashboard based on data characteristics"""
    
    if df is None or df.empty:
        return

    import numpy as np
    import time
    from premium_visualization_tab import create_premium_chart, get_premium_colors

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    charts_to_create = []
    
    # Smart Chart Logic
    
    # 1. Timeline (if dates exist)
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        val_col = numeric_cols[0]
        charts_to_create.append({
            'type': 'line',
            'config': {'x_col': date_col, 'y_col': val_col, 'color_col': 'None', 'smooth': True, 'add_markers': False},
            'title': f"{val_col} Trend",
            'desc': 'Temporal Analysis'
        })
    elif numeric_cols:
        # Fallback if no dates: Line chart by index or first numeric
        charts_to_create.append({
            'type': 'line',
            'config': {'x_col': df.index.name or 'index', 'y_col': numeric_cols[0], 'color_col': 'None', 'smooth': True},
            'title': f"{numeric_cols[0]} Overview",
             'desc': 'Sequential Overview'
        })

    # 2. Distribution (Histogram of key metric)
    if numeric_cols:
        target_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        charts_to_create.append({
            'type': 'histogram',
            'config': {'column': target_col, 'bins': 30, 'show_kde': True},
            'title': f"Distribution of {target_col}",
             'desc': 'Distribution Analysis'
        })

    # 3. Categorical Breakdown (Bar)
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        val_col = numeric_cols[0]
        charts_to_create.append({
            'type': 'bar',
            'config': {'x_col': cat_col, 'y_col': val_col, 'color_col': cat_col, 'orientation': 'vertical'},
            'title': f"{val_col} by {cat_col}",
             'desc': 'Category Comparison'
        })
    elif len(categorical_cols) >= 1:
        # Count plot if no numeric
        # We need a way to do counts. premium_visualization_tab bar chart expects numeric y.
        # We can simulate it by aggregating
        pass 

    # 4. Correlation/Relationship (Scatter)
    if len(numeric_cols) >= 2:
        charts_to_create.append({
            'type': 'scatter',
            'config': {
                'x_col': numeric_cols[0], 
                'y_col': numeric_cols[1], 
                'color_col': categorical_cols[0] if categorical_cols else 'None',
                'size_col': numeric_cols[2] if len(numeric_cols) > 2 else 'None',
                'add_trendline': True
            },
            'title': f"{numeric_cols[1]} vs {numeric_cols[0]}",
             'desc': 'Correlation Analysis'
        })

    # Generate the charts
    if 'dashboard_charts' not in st.session_state:
        st.session_state.dashboard_charts = {}

    colors = get_premium_colors(st.session_state.get('selected_palette', 'aurora'))

    for i, chart_def in enumerate(charts_to_create[:4]): # Limit to 4 smart charts
        try:
            fig = create_premium_chart(df, chart_def['type'], chart_def['config'], colors, 'modern')
            if fig:
                 key = f"auto_{chart_def['type']}_{int(time.time())}_{i}"
                 st.session_state.dashboard_charts[key] = {
                    'fig': fig,
                    'type': chart_def['type'],
                    'config': chart_def['config'],
                    'palette': st.session_state.get('selected_palette', 'aurora'),
                    'title': chart_def['title'],
                    'colors': colors,
                    'last_updated': pd.Timestamp.now(),
                    'data_rows': len(df)
                }
        except Exception:
            continue
    
    if st.session_state.dashboard_charts:
        st.success("✨ Dashboard generated with AI-suggested insights!")
    else:
        st.warning("Could not auto-generate charts. Data might be insufficient.")