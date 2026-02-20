
import math

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_insights_tab(df: pd.DataFrame, dataset_summary: dict = None, color_palette: str = 'aurora'):
    """
    Render insights and recommendations tab with Data Storytelling.
    
    Args:
        df: The dataframe to analyze
        dataset_summary: Optional pre-computed summary
        color_palette: Current color palette name
    """
    from premium_config import PREMIUM_COLOR_PALETTES
    
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
        <div style="background: linear-gradient(135deg, #6366F1, #8B5CF6); width: 64px; height: 64px; border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);">
            🧠
        </div>
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2rem; margin: 0; color: var(--text-main);">Deep Insights</h2>
            <p style="color: var(--text-secondary); margin-top: 0.25rem;">Automated narrative generation and pattern discovery.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # We need to get summary if not provided
    if dataset_summary is None:
        from utils import get_data_summary
        dataset_summary = get_data_summary(df)
        
    if not dataset_summary:
        st.error("Could not generate summary.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # 1. The Data Story
    st.markdown("### 📖 The Data Narrative")
    
    row_count = len(df)
    col_count = len(df.columns)
    numeric_cols = dataset_summary['column_info']['numeric_columns']
    categorical_cols = dataset_summary['column_info']['categorical_columns']
    
    # Narrative Generation
    completeness = "pristine" if dataset_summary['missing_data']['missing_percentage'] == 0 else \
                  "mostly complete" if dataset_summary['missing_data']['missing_percentage'] < 5 else "gapped"
                  
    narrative = f"""
    <div style="font-family: 'Inter', sans-serif; font-size: 1.05rem; line-height: 1.7; color: var(--text-secondary); background: rgba(var(--primary-rgb), 0.03); padding: 1.5rem; border-left: 4px solid var(--accent); border-radius: 0 8px 8px 0;">
        You are analyzing a dataset of <strong>{row_count:,} records</strong> and <strong>{col_count} features</strong>. 
        The architecture consists of <strong>{len(numeric_cols)} numeric variables</strong> and <strong>{len(categorical_cols)} categorical variables</strong>.
        The data integrity appears <strong>{completeness}</strong>, with {dataset_summary['missing_data']['missing_percentage']:.1f}% missing values overall.
    </div>
    """
    st.markdown(narrative, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Key Drivers & Correlations
    st.markdown("### 🔗 Key Relationships")
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Find strongest correlations
            stacked_corr = corr_matrix.stack()
            # Remove self-correlation and duplicates
            stacked_corr = stacked_corr[stacked_corr.index.get_level_values(0) != stacked_corr.index.get_level_values(1)]
            strongest_pairs = stacked_corr.abs().sort_values(ascending=False).head(5)
            
            col_insight, col_graph = st.columns([1, 1.5], gap="large")
            
            with col_insight:
                 if not strongest_pairs.empty:
                    st.markdown('<div class="control-panel-card">', unsafe_allow_html=True)
                    st.caption("STRONGEST DRIVERS")
                    for idx, val in strongest_pairs.items():
                        col1, col2 = idx
                        val_raw = corr_matrix.loc[col1, col2]
                        strength = "Strong" if abs(val_raw) > 0.7 else "Moderate"
                        direction = "Positive" if val_raw > 0 else "Negative"
                        color = "#10B981" if val_raw > 0 else "#EF4444"
                        
                        # Only show every pair once (A-B, not B-A)
                        if list(df.columns).index(col1) < list(df.columns).index(col2):
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem; border-bottom: 1px dashed var(--border); padding-bottom: 0.5rem;">
                                <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.9rem;">
                                    <span>{col1} &leftrightarrow; {col2}</span>
                                    <span style="color: {color};">{strength} {direction}</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.25rem;">
                                    <div style="flex-grow: 1; height: 6px; background: rgba(0,0,0,0.1); border-radius: 3px; position: relative;">
                                        <div style="position: absolute; left: 50%; height: 100%; width: {abs(val_raw)*50}%; background: {color}; border-radius: 3px; transform: translateX({0 if val_raw > 0 else '-100%'});"></div>
                                    </div>
                                    <span style="font-size: 0.8rem; font-family: monospace;">{val_raw:.2f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col_graph:
                # Network Graph of Correlations
                # Simple circular layout
                n_nodes = len(numeric_cols[:10]) # Limit to top 10 numeric vars for clarity
                radius = 1
                nodes = numeric_cols[:10]
                node_x = []
                node_y = []
                node_text = [] 
                coords = {}
                
                for i, node in enumerate(nodes):
                    angle = 2 * math.pi * i / n_nodes
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    coords[node] = (x, y)
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                
                edge_x = []
                edge_y = []
                
                for i, node1 in enumerate(nodes):
                    for j, node2 in enumerate(nodes):
                        if i < j:
                            corr = corr_matrix.loc[node1, node2]
                            if abs(corr) > 0.4: # Threshold
                                x0, y0 = coords[node1]
                                x1, y1 = coords[node2]
                                edge_x.append(x0)
                                edge_x.append(x1)
                                edge_x.append(None)
                                edge_y.append(y0)
                                edge_y.append(y1)
                                edge_y.append(None)
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        size=25,
                        color=PREMIUM_COLOR_PALETTES.get(color_palette, ['#6366F1'])[0],
                        line=dict(width=2, color='white')
                    ),
                    textfont=dict(size=10)
                )
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='rgba(150, 150, 150, 0.5)'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                fig_net = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=20,r=20,t=20),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                )
                st.markdown('<div style="text-align: center; font-size: 0.9rem; margin-bottom: 0.5rem; color: var(--text-secondary);">Correlation Topology</div>', unsafe_allow_html=True)
                st.plotly_chart(fig_net, use_container_width=True)
                        
        except Exception as e:
            st.info(f"Analysis limited: {str(e)}")
    else:
        st.info("Insufficient numeric data for correlation analysis.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. Distribution & Outliers
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("#### 📊 Distribution Profile")
        
        skewed_cols = []
        for col in numeric_cols:
            try:
                skew = df[col].skew()
                if abs(skew) > 1:
                    skewed_cols.append((col, skew))
            except: pass
        
        if skewed_cols:
            st.write("The following variables exhibit significant skew, indicating potential outliers or non-normal distribution:")
            for col, skew in skewed_cols[:3]:
                direction = "Right-skewed (Tail points positive)" if skew > 0 else "Left-skewed (Tail points negative)"
                st.markdown(f"""
                <div style="background: rgba(var(--bg-color-rgb), 0.5); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid var(--border);">
                    <div style="font-weight: 600;">{col}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">{direction}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 150px; text-align: center; color: var(--text-muted);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚖️</div>
                <div>Balanced Distributions<br><span style="font-size: 0.8rem;">Most numeric variables appear normally distributed.</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Complexity Analysis")
        
        high_card_cols = [c for c in categorical_cols if df[c].nunique() > 50]
        if high_card_cols:
            st.write(f"Detected {len(high_card_cols)} high-cardinality categorical features:")
            st.markdown(f"""
            <div style="background: #FFFBEB; border: 1px solid #FCD34D; color: #92400E; padding: 1rem; border-radius: 8px;">
                <strong>Features:</strong> {', '.join(high_card_cols[:5])}
                <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                    Recommendation: Consider grouping rare categories or using Target Encoding in the Refinement tab.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 150px; text-align: center; color: var(--text-muted);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">✅</div>
                <div>Optimal Cardinality<br><span style="font-size: 0.8rem;">Categorical variables are well-structured.</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)


def generate_executive_insights(df: pd.DataFrame) -> list:
    """
    Generate high-level strategic insights for executive reports.
    Returns: List of dicts with 'text' and 'priority'
    """
    insights = []
    
    # 1. Volume & Growth
    row_count = len(df)
    insights.append({
        'text': f"Analysis based on a robust dataset of {row_count:,} records.", 
        'priority': 'low'
    })
    
    try:
        # 2. Correlations
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            # Create upper triangle mask
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            
            # Find max correlation
            max_corr = 0
            best_pair = None
            
            # Iterate upper triangle
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if val > max_corr:
                        max_corr = val
                        best_pair = (corr_matrix.columns[i], corr_matrix.columns[j])

            if max_corr > 0.7 and best_pair:
                insights.append({
                    'text': f"Strong predictive relationship detected between {best_pair[0]} and {best_pair[1]} (Correlation: {max_corr:.2f}).",
                    'priority': 'high'
                })
        
        # 3. Categorical Dominance
        cat_df = df.select_dtypes(include=['object', 'category'])
        for col in cat_df.columns[:3]: # Check top 3 text cols
            try:
                if df[col].nunique() < 50:
                    top_val = df[col].mode()[0]
                    freq = df[col].value_counts().iloc[0]
                    pct = freq / row_count
                    if pct > 0.6:
                         insights.append({
                            'text': f"Variable '{col}' is dominated by segment '{top_val}' ({pct:.1%} of total).",
                            'priority': 'medium'
                         })
            except: pass
            
        # 4. Outliers / Variance
        for col in numeric_df.columns[:5]:
            try:
                std = df[col].std()
                mean = df[col].mean()
                if mean > 0 and (std / mean) > 1.5:
                     insights.append({
                         'text': f"High variance observed in '{col}' (CV > 1.5), indicating volatile or diverse data points.",
                         'priority': 'medium'
                     })
            except: pass
            
    except Exception as e:
        insights.append({'text': f"Automated insight generation limited: {str(e)}", 'priority': 'low'})

    return insights
