
import pandas as pd
import plotly.express as px
import streamlit as st


def render_advanced_gallery(df: pd.DataFrame, plot_generator=None):
    """
    Render advanced plots gallery.
    
    Args:
        df: The dataframe to visualize
        plot_generator: Optional instance of PremiumPlotGenerator
    """
    # Try to import generator if not provided
    if plot_generator is None:
        try:
            from premium_plots import PremiumPlotGenerator
            plot_generator = PremiumPlotGenerator()
        except ImportError:
            st.error("Premium Plot Generator not available.")
            return

    from premium_config import PREMIUM_COLOR_PALETTES

    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem;">
        <div>
            <h3 style="font-family: 'Playfair Display', serif; font-size: 1.8rem; margin: 0; color: var(--text-main);">Premium Chart Gallery</h3>
            <p style="color: var(--text-secondary); margin-top: 0.25rem;">Specialized visualizations for complex data relationships.</p>
        </div>
        <div style="font-size: 2rem;">🎨</div>
    </div>
    """, unsafe_allow_html=True)
    
    chart_options = {
        'violin_plot': '🎻 Violin Distribution',
        'sunburst': '☀️ Sunburst Hierarchy',
        'treemap': '🌳 Treemap Composition',
        'radar_chart': '📡 Radar / Spider Chart',
        'waterfall': '💧 Waterfall Analysis',
        '3d_scatter': '🧊 3D Multidimensional',
        'animated_scatter': '🎬 Animated Time-Series'
    }
    
    # Custom stylized selector
    col_sel, col_desc = st.columns([1, 2])
    with col_sel:
        st.markdown('<div class="control-panel-card" style="padding: 1rem;">', unsafe_allow_html=True)
        st.caption("SELECT VISUALIZATION")
        selected_chart = st.radio(
            "Select Premium Chart", 
            options=list(chart_options.keys()),
            format_func=lambda x: chart_options[x],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_desc:
        # Show description based on selection
        descriptions = {
            'violin_plot': "Combines a box plot with a kernel density plot (violin shape). Ideal for comparing distribution shapes across categories.",
            'sunburst': "Visualizes hierarchical data structure with concentric rings. Great for exploring multi-level categorical breakdowns.",
            'treemap': "Displays hierarchical data using nested rectangles. Efficient for comparing proportions within categories.",
            'radar_chart': "Compares multiple quantitative variables on a polar grid. Useful for profiling or comparing performance metrics.",
            'waterfall': "Shows the cumulative effect of sequentially introduced values. Perfect for financial or inventory analysis.",
            '3d_scatter': "Adds a third dimension (Z-axis) to standard scatter plots. Helps identify clusters in complex datasets.",
            'animated_scatter': "incorporates a time or sequence dimension to show how data evolves. Powerful for storytelling."
        }
        st.info(f"💡 **{chart_options[selected_chart]}**: {descriptions.get(selected_chart, '')}")

    st.markdown("---")
    
    col_info = st.session_state.get('column_info')
    if not col_info:
        from utils import get_column_info
        col_info = get_column_info(df)
        
    numeric_cols = col_info['numeric_columns']
    categorical_cols = col_info['categorical_columns']
    
    current_palette = st.session_state.get('selected_palette', 'aurora')
    
    # 3D Scatter
    if selected_chart == '3d_scatter':
        if len(numeric_cols) >= 3:
            c1, c2, c3, c4 = st.columns(4)
            with c1: x = st.selectbox("X Axis", numeric_cols)
            with c2: y = st.selectbox("Y Axis", [c for c in numeric_cols if c != x])
            with c3: z = st.selectbox("Z Axis", [c for c in numeric_cols if c not in [x, y]])
            with c4: color = st.selectbox("Color By", ['None'] + categorical_cols + numeric_cols)
            
            if st.button("Generate 3D Model", type="primary", use_container_width=True):
                fig = px.scatter_3d(df, x=x, y=y, z=z, 
                                  color=color if color != 'None' else None,
                                  color_discrete_sequence=PREMIUM_COLOR_PALETTES.get(current_palette))
                fig.update_layout(height=600, margin=dict(l=0, r=0, b=0, t=0), scene=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 numeric columns for 3D Scatter.")

    # Animated Scatter
    elif selected_chart == 'animated_scatter':
        if len(numeric_cols) >= 2 and len(categorical_cols) > 0:
            c1, c2, c3, c4 = st.columns(4)
            with c1: x = st.selectbox("X Axis", numeric_cols)
            with c2: y = st.selectbox("Y Axis", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
            with c3: anim = st.selectbox("Animation Frame", categorical_cols)
            with c4: color = st.selectbox("Color By", ['None'] + categorical_cols)
            
            if st.button("Generate Animation", type="primary", use_container_width=True):
                fig = px.scatter(df, x=x, y=y, animation_frame=anim,
                               color=color if color != 'None' else None,
                               color_discrete_sequence=PREMIUM_COLOR_PALETTES.get(current_palette),
                               range_x=[df[x].min(), df[x].max()],
                               range_y=[df[y].min(), df[y].max()])
                fig.update_layout(height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need categorical column for animation frame.")

    # Other Premium Charts from Generator
    elif selected_chart == 'violin_plot':
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            with col1: x_col = st.selectbox("Category Column", categorical_cols)
            with col2: y_col = st.selectbox("Value Column", numeric_cols)
            if st.button("Render Violin Plot", type="primary", use_container_width=True):
                fig = plot_generator.violin_plot(df, x_col, y_col, palette=current_palette)
                st.plotly_chart(fig, use_container_width=True)
        else:
             st.warning("Need categorical and numeric columns.")
    
    elif selected_chart == 'radar_chart':
        if len(numeric_cols) >= 3:
            categories = st.multiselect("Select Metrics (3+)", numeric_cols, default=numeric_cols[:5])
            group_col = st.selectbox("Group By (Optional)", ['None'] + categorical_cols)
            if st.button("Render Radar Chart", type="primary", use_container_width=True): 
                if len(categories) >= 3:
                    fig = plot_generator.radar_chart(df, categories, group_col=group_col if group_col != 'None' else None, palette=current_palette)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Please select at least 3 metrics.")
        else:
            st.warning("Not enough numeric columns available.")
    
    elif selected_chart == 'waterfall':
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            with col1: x_col = st.selectbox("Category Column", df.columns.tolist())
            with col2: y_col = st.selectbox("Value Column", numeric_cols)
            if st.button("Render Waterfall Chart", type="primary", use_container_width=True):
                fig = plot_generator.waterfall_chart(df, x_col, y_col, palette=current_palette)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available.")

    elif selected_chart in ['sunburst', 'treemap']:
        if len(categorical_cols) >= 2 and len(numeric_cols) > 0:
             hierarchy = st.multiselect("Hierarchy Layers (Order matters)", categorical_cols, default=categorical_cols[:2])
             val_col = st.selectbox("Size/Value Column", numeric_cols)
             if st.button(f"Render {chart_options[selected_chart]}", type="primary", use_container_width=True):
                 if hierarchy:
                     if selected_chart == 'sunburst':
                         fig = plot_generator.sunburst_chart(df, hierarchy, val_col, palette=current_palette)
                     else:
                         fig = plot_generator.treemap_chart(df, hierarchy, val_col, palette=current_palette)
                     st.plotly_chart(fig, use_container_width=True)
                 else:
                     st.error("Please select hierarchy columns.")
        else:
             st.warning("Need at least 2 categorical columns and 1 numeric column.")

    
    st.markdown('</div>', unsafe_allow_html=True)
