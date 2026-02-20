"""
Premium Plot Generation Module
Advanced, Interactive, and Customizable Visualizations
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from premium_config import (PREMIUM_COLOR_PALETTES, PREMIUM_PLOT_CONFIG)


class PremiumPlotGenerator:
    """Premium plot generator with advanced customization options"""
    
    def __init__(self):
        self.default_config = PREMIUM_PLOT_CONFIG
        self.color_palettes = PREMIUM_COLOR_PALETTES
        
    def get_color_palette(self, palette_name: str = 'aurora') -> List[str]:
        """Get color palette by name"""
        default_palette = list(self.color_palettes.values())[0] if self.color_palettes else ['#6366F1', '#EC4899', '#F59E0B']
        return self.color_palettes.get(palette_name, self.color_palettes.get('aurora', default_palette))
    
    def create_premium_layout(self, title: str, palette: str = 'aurora') -> Dict:
        """Create premium layout configuration"""
        colors = self.get_color_palette(palette)
        
        # Determine theme colors
        current_theme = getattr(st.session_state, 'selected_theme', 'light')
        is_dark = current_theme == 'dark'
        
        text_color = '#F8FAFC' if is_dark else '#1E293B'
        secondary_text = '#CBD5E1' if is_dark else '#64748B'
        grid_color = 'rgba(255, 255, 255, 0.1)' if is_dark else 'rgba(0, 0, 0, 0.05)'
        line_color = 'rgba(255, 255, 255, 0.2)' if is_dark else 'rgba(0, 0, 0, 0.1)'
        
        return {
            'title': {
                'text': title,
                'x': 0.05,
                'xanchor': 'left',
                'font': {
                    'family': self.default_config['font_family'],
                    'size': self.default_config['title_font_size'],
                    'color': colors[0]
                }
            },
            'font': {
                'family': self.default_config['font_family'],
                'size': self.default_config['axis_font_size'],
                'color': secondary_text
            },
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1,
                'font': {'size': self.default_config['legend_font_size'], 'color': text_color}
            },
            'margin': {'l': 60, 'r': 40, 't': 80, 'b': 60},
            'height': self.default_config['default_height'],
            'width': self.default_config['default_width'],
            'hoverlabel': {
                'bgcolor': '#1E293B' if is_dark else 'white',
                'font_size': 14,
                'font_family': self.default_config['font_family'],
                'font_color': '#F8FAFC' if is_dark else '#1E293B'
            },
            'xaxis': {
                'showgrid': True,
                'gridcolor': grid_color,
                'zeroline': False,
                'showline': True,
                'linecolor': line_color,
                'tickfont': {'color': secondary_text},
                'title_font': {'color': text_color}
            },
            'yaxis': {
                'showgrid': True,
                'gridcolor': grid_color,
                'zeroline': False,
                'showline': True,
                'linecolor': line_color,
                'tickfont': {'color': secondary_text},
                'title_font': {'color': text_color}
            }
        }
    
    def advanced_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                            color_col: Optional[str] = None, size_col: Optional[str] = None,
                            palette: str = 'aurora', add_regression: bool = True,
                            add_marginals: bool = True) -> go.Figure:
        """Create advanced scatter plot with regression and marginals"""
        
        colors = self.get_color_palette(palette)
        
        if add_marginals:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                           marginal_x="histogram", marginal_y="histogram",
                           color_discrete_sequence=colors,
                           hover_data=df.columns.tolist())
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                           color_discrete_sequence=colors,
                           hover_data=df.columns.tolist())
        
        if add_regression and color_col is None:
            # Add regression line
            z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
            p = np.poly1d(z)
            x_reg = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            y_reg = p(x_reg)
            
            fig.add_trace(go.Scatter(
                x=x_reg, y=y_reg,
                mode='lines',
                name='Regression Line',
                line=dict(color=colors[1], width=3, dash='dash')
            ))
        
        layout = self.create_premium_layout(f'Advanced Scatter: {x_col} vs {y_col}', palette)
        fig.update_layout(**layout)
        
        # Add hover effects
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         f'{x_col}: %{{x}}<br>' +
                         f'{y_col}: %{{y}}<br>' +
                         '<extra></extra>',
            hoverlabel=dict(bgcolor="white", font_size=12, font_family=self.default_config['font_family'])
        )
        
        return fig
    
    def animated_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str,
                          color_col: Optional[str] = None, palette: str = 'aurora',
                          add_trend: bool = True, smooth: bool = True) -> go.Figure:
        """Create animated line chart with trend analysis"""
        
        colors = self.get_color_palette(palette)
        
        if color_col and color_col in df.columns:
            fig = px.line(df, x=x_col, y=y_col, color=color_col,
                         color_discrete_sequence=colors,
                         line_shape='spline' if smooth else 'linear')
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(color=colors[0], width=3, shape='spline' if smooth else 'linear'),
                marker=dict(size=6, color=colors[1], line=dict(width=2, color='white'))
            ))
        
        if add_trend:
            # Add trend line
            if pd.api.types.is_numeric_dtype(df[x_col]):
                z = np.polyfit(range(len(df)), df[y_col].fillna(df[y_col].mean()), 1)
                p = np.poly1d(z)
                trend_y = p(range(len(df)))
                
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=trend_y,
                    mode='lines',
                    name='Trend',
                    line=dict(color=colors[2], width=2, dash='dot'),
                    opacity=0.7
                ))
        
        layout = self.create_premium_layout(f'Time Series: {y_col} over {x_col}', palette)
        layout['xaxis'] = {'showgrid': True, 'gridcolor': 'rgba(128,128,128,0.2)'}
        layout['yaxis'] = {'showgrid': True, 'gridcolor': 'rgba(128,128,128,0.2)'}
        
        fig.update_layout(**layout)
        
        # Add animations
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                      'fromcurrent': True, 'transition': {'duration': 300}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                        'mode': 'immediate', 'transition': {'duration': 0}}]
                    }
                ]
            }]
        )
        
        return fig
    
    def advanced_heatmap(self, df: pd.DataFrame, palette: str = 'aurora',
                        cluster: bool = True, annotate: bool = True) -> go.Figure:
        """Create advanced heatmap with clustering"""
        
        colors = self.get_color_palette(palette)
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.error("No numeric columns found for heatmap")
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        if cluster:
            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import squareform

            # Convert correlation to distance
            distance_matrix = 1 - np.abs(corr_matrix)
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='average')
            
            # Get dendrogram order
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro['leaves']
            
            # Reorder correlation matrix
            ordered_columns = corr_matrix.columns[cluster_order]
            corr_matrix = corr_matrix.loc[ordered_columns, ordered_columns]
        
        # Create custom colorscale
        colorscale = [
            [0.0, colors[0]],
            [0.25, colors[1]],
            [0.5, colors[2]],
            [0.75, colors[3]],
            [1.0, colors[4]]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=colorscale,
            text=corr_matrix.round(2).values if annotate else None,
            texttemplate='%{text}' if annotate else None,
            textfont={'size': 10},
            hoverongaps=False,
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        layout = self.create_premium_layout('Advanced Correlation Heatmap', palette)
        layout['xaxis'] = {'side': 'bottom'}
        layout['yaxis'] = {'side': 'left'}
        
        fig.update_layout(**layout)
        
        return fig
    
    def violin_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                   palette: str = 'aurora', add_box: bool = True) -> go.Figure:
        """Create violin plot with box plot overlay"""
        
        colors = self.get_color_palette(palette)
        
        fig = go.Figure()
        
        categories = df[x_col].unique()
        
        for i, category in enumerate(categories):
            data = df.loc[df[x_col] == category, y_col].dropna()
            
            fig.add_trace(go.Violin(
                y=data,
                name=str(category),
                box_visible=add_box,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.7,
                line_color=colors[i % len(colors)]
            ))
        
        layout = self.create_premium_layout(f'Distribution of {y_col} by {x_col}', palette)
        fig.update_layout(**layout)
        
        return fig
    
    def sunburst_chart(self, df: pd.DataFrame, hierarchy_cols: List[str],
                      value_col: str, palette: str = 'aurora') -> go.Figure:
        """Create sunburst chart for hierarchical data"""
        
        self.get_color_palette(palette)
        
        # Prepare data for sunburst
        df_grouped = df.groupby(hierarchy_cols)[value_col].sum().reset_index()
        
        # Create labels and parents for sunburst
        labels = []
        parents = []
        values = []
        
        # Add root
        labels.append("Total")
        parents.append("")
        values.append(df_grouped[value_col].sum())
        
        # Add first level
        for val in df[hierarchy_cols[0]].unique():
            labels.append(str(val))
            parents.append("Total")
            values.append(df[df[hierarchy_cols[0]] == val][value_col].sum())
        
        # Add subsequent levels
        for i in range(1, len(hierarchy_cols)):
            for combo in df[hierarchy_cols[:i+1]].drop_duplicates().values:
                label = " - ".join([str(x) for x in combo])
                parent = " - ".join([str(x) for x in combo[:-1]]) if i > 1 else str(combo[0])
                
                mask = True
                for j, col in enumerate(hierarchy_cols[:i+1]):
                    mask = mask & (df[col] == combo[j])
                
                value = df[mask][value_col].sum()
                
                labels.append(label)
                parents.append(parent)
                values.append(value)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percentParent}<extra></extra>',
            maxdepth=len(hierarchy_cols) + 1,
        ))
        
        layout = self.create_premium_layout(f'Hierarchical View: {" → ".join(hierarchy_cols)}', palette)
        fig.update_layout(**layout)
        
        return fig
    
    def treemap_chart(self, df: pd.DataFrame, hierarchy_cols: List[str],
                     value_col: str, palette: str = 'aurora') -> go.Figure:
        """Create treemap chart"""
        
        colors = self.get_color_palette(palette)
        
        fig = px.treemap(df, path=hierarchy_cols, values=value_col,
                        color_discrete_sequence=colors,
                        hover_data={value_col: ':,.0f'})
        
        layout = self.create_premium_layout(f'Treemap: {" → ".join(hierarchy_cols)}', palette)
        fig.update_layout(**layout)
        
        return fig
    
    def radar_chart(self, df: pd.DataFrame, categories: List[str],
                   group_col: Optional[str] = None, palette: str = 'aurora') -> go.Figure:
        """Create radar chart for multi-dimensional comparison"""
        
        colors = self.get_color_palette(palette)
        
        fig = go.Figure()
        
        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            for i, group in enumerate(groups):
                group_data = df[df[group_col] == group]
                values = [group_data[cat].mean() for cat in categories]
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=str(group),
                    fillcolor=colors[i % len(colors)],
                    line_color=colors[i % len(colors)],
                    opacity=0.6
                ))
        else:
            values = [df[cat].mean() for cat in categories]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Average',
                fillcolor=colors[0],
                line_color=colors[0],
                opacity=0.6
            ))
        
        layout = self.create_premium_layout('Multi-Dimensional Comparison', palette)
        layout['polar'] = dict(
            radialaxis=dict(visible=True, range=[0, max([df[cat].max() for cat in categories])]),
            angularaxis=dict(tickfont_size=12)
        )
        
        fig.update_layout(**layout)
        
        return fig
    
    def waterfall_chart(self, df: pd.DataFrame, x_col: str, y_col: str,
                       palette: str = 'aurora') -> go.Figure:
        """Create waterfall chart for sequential changes"""
        
        colors = self.get_color_palette(palette)
        
        # Calculate cumulative values
        values = df[y_col].values
        cumulative = np.cumsum(values)
        
        fig = go.Figure()
        
        # Add bars
        for i, (x_val, y_val) in enumerate(zip(df[x_col], values)):
            color = colors[0] if y_val >= 0 else colors[1]
            
            fig.add_trace(go.Bar(
                x=[x_val],
                y=[y_val],
                name=str(x_val),
                marker_color=color,
                text=f'{y_val:+.1f}',
                textposition='outside',
                showlegend=False
            ))
        
        # Add cumulative line
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=cumulative,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color=colors[2], width=3),
            marker=dict(size=8, color=colors[2])
        ))
        
        layout = self.create_premium_layout(f'Waterfall Chart: {y_col}', palette)
        fig.update_layout(**layout)
        
        return fig
    
    def candlestick_chart(self, df: pd.DataFrame, date_col: str,
                         open_col: str, high_col: str, low_col: str, close_col: str,
                         volume_col: Optional[str] = None, palette: str = 'aurora') -> go.Figure:
        """Create candlestick chart for financial data"""
        
        colors = self.get_color_palette(palette)
        
        if volume_col:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.1, row_heights=[0.7, 0.3])
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df[date_col],
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                increasing_line_color=colors[0],
                decreasing_line_color=colors[1],
                name='OHLC'
            ), row=1, col=1)
            
            # Volume chart
            fig.add_trace(go.Bar(
                x=df[date_col],
                y=df[volume_col],
                marker_color=colors[2],
                name='Volume',
                opacity=0.7
            ), row=2, col=1)
            
        else:
            fig = go.Figure(data=[go.Candlestick(
                x=df[date_col],
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                increasing_line_color=colors[0],
                decreasing_line_color=colors[1]
            )])
        
        layout = self.create_premium_layout('Financial Candlestick Chart', palette)
        fig.update_layout(**layout)
        
        return fig
    
    def sankey_diagram(self, df: pd.DataFrame, source_col: str, target_col: str,
                      value_col: str, palette: str = 'aurora') -> go.Figure:
        """Create Sankey diagram for flow visualization"""
        
        colors = self.get_color_palette(palette)
        
        # Prepare data
        sources = df[source_col].unique()
        targets = df[target_col].unique()
        all_nodes = list(sources) + list(targets)
        
        # Create node indices
        node_dict = {node: i for i, node in enumerate(all_nodes)}
        
        # Prepare links
        source_indices = [node_dict[source] for source in df[source_col]]
        target_indices = [node_dict[target] for target in df[target_col]]
        values = df[value_col].tolist()
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=colors[:len(all_nodes)]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=[colors[i % len(colors)] + '40' for i in range(len(values))]  # Add transparency
            )
        )])
        
        layout = self.create_premium_layout(f'Flow Diagram: {source_col} → {target_col}', palette)
        fig.update_layout(**layout)
        
        return fig

def create_premium_plot(plot_type: str, df: pd.DataFrame, **kwargs) -> go.Figure:
    """Factory function to create premium plots"""
    generator = PremiumPlotGenerator()
    
    if plot_type == 'advanced_scatter':
        return generator.advanced_scatter_plot(df, **kwargs)
    elif plot_type == 'animated_line':
        return generator.animated_line_chart(df, **kwargs)
    elif plot_type == 'heatmap_advanced':
        return generator.advanced_heatmap(df, **kwargs)
    elif plot_type == 'violin_plot':
        return generator.violin_plot(df, **kwargs)
    elif plot_type == 'sunburst':
        return generator.sunburst_chart(df, **kwargs)
    elif plot_type == 'treemap':
        return generator.treemap_chart(df, **kwargs)
    elif plot_type == 'radar_chart':
        return generator.radar_chart(df, **kwargs)
    elif plot_type == 'waterfall':
        return generator.waterfall_chart(df, **kwargs)
    elif plot_type == 'candlestick':
        return generator.candlestick_chart(df, **kwargs)
    elif plot_type == 'sankey':
        return generator.sankey_diagram(df, **kwargs)
    else:
        st.error(f"Unknown plot type: {plot_type}")
        return go.Figure()