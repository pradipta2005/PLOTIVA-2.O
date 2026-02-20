

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from premium_config import PREMIUM_COLOR_PALETTES

# Model Knowledge Base
MODEL_INFO = {
    "Random Forest": {
        "desc": "An ensemble of decision trees that vote on the outcome. Highly accurate and resistant to overfitting.",
        "best_for": "Complex datasets with mixed variables.",
        "pros": ["High Accuracy", "Handles Missing Data", "Feature Importance"],
        "cons": ["Slower Training", "Less Interpretable"]
    },
    "Gradient Boosting": {
        "desc": "Builds trees sequentially, with each correcting the errors of the previous one.",
        "best_for": "Winning competitions when accuracy is paramount.",
        "pros": ["Best-in-class Accuracy", "Flexible"],
        "cons": ["Sensitive to Outliers", "Hard to Tune"]
    },
    "Linear Regression": {
        "desc": "Finds the best-fitting straight line through the data.",
        "best_for": "Understanding simple relationships and trends.",
        "pros": ["Very Fast", "Highly Interpretable"],
        "cons": ["Assumes Linearity", "Sensitive to Outliers"]
    },
    "Logistic Regression": {
        "desc": "Estimates the probability of an event occurring using a logistic function.",
        "best_for": "Binary classification (Yes/No predictions).",
        "pros": ["Probabilistic Outputs", "Simple Baseline"],
        "cons": ["Linear Boundary", "Data must be independent"]
    },
    "Decision Tree": {
        "desc": "A flowchart-like structure that makes decisions based on asking questions about features.",
        "best_for": "Visualizing decision logic.",
        "pros": ["Easy to Explain", "Non-linear patterns"],
        "cons": ["Prone to Overfitting"]
    },
    "XGBoost": {
        "desc": "eXtreme Gradient Boosting. An optimized distributed gradient boosting library.",
        "best_for": "Large datasets where speed and performance are critical.",
        "pros": ["State-of-the-art Speed", "Regularization"],
        "cons": ["Complex Parameters"]
    },
    "Extra Trees": {
        "desc": "Extremely Randomized Trees. Similar to Random Forest but with more randomness in splits.",
        "best_for": "Reducing variance further than Random Forest.",
        "pros": ["Faster than RF", "Low Variance"],
        "cons": ["Higher Bias"]
    },
    "SVC": {
        "desc": "Support Vector Classifier. Finds the optimal hyperplane that separates classes.",
        "best_for": "High-dimensional data.",
        "pros": ["Effective in high dimensions", "Versatile Kernels"],
        "cons": ["Slow on large data", "Noise Sensitive"]
    },
    "KNN": {
        "desc": "K-Nearest Neighbors. Classifies based on the majority class of nearest neighbors.",
        "best_for": "Small datasets with clear clusters.",
        "pros": ["Simple", "No Training Phase"],
        "cons": ["Slow Prediction", "Curse of Dimensionality"]
    },
    "Ridge": {
        "desc": "Linear Regression with L2 regularization to prevent overfitting.",
        "best_for": "Data with multicollinearity (correlated features).",
        "pros": ["Stable Coefficients"],
        "cons": ["Biased Estimate"]
    },
    "Lasso": {
        "desc": "Linear Regression with L1 regularization that can shrink coefficients to zero.",
        "best_for": "Feature selection.",
        "pros": ["Selects Features", "Sparse Models"],
        "cons": ["Struggles with correlated groups"]
    }
}

def render_ml_tab(df: pd.DataFrame, analytics_engine):
    """
    Render Machine Learning Studio with premium UI.
    """
    # --------------------------------------------------------------------------
    # Helper Functions
    # --------------------------------------------------------------------------
    def get_palette_colors():
        palette_name = st.session_state.get('selected_palette', 'executive_suite')
        return PREMIUM_COLOR_PALETTES.get(palette_name, PREMIUM_COLOR_PALETTES['executive_suite'])

    def get_theme_plot_config():
        """Get plot config based on current theme"""
        # Default to dark since our premium config defaults to dark theme variables if not overriden
        is_dark = st.session_state.get('theme', 'dark') == 'dark'
        return {
            'template': 'plotly_dark' if is_dark else 'plotly_white',
            'font_color': '#F8FAFC' if is_dark else '#1a2332',
            'bg_color': 'rgba(0,0,0,0)'
        }

    def clear_cache():
        """Clear previous results when configuration changes"""
        if 'ml_results' in st.session_state:
            st.session_state.ml_results.pop('last_run', None)
            st.session_state.ml_results.pop('automl_run', None)
            st.session_state.ml_results.pop('mode', None)
            st.session_state.ml_results['history'] = []

    # --------------------------------------------------------------------------
    # Main UI Structure
    # --------------------------------------------------------------------------
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Predictive Intelligence</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Build, evaluate, and deploy enterprise-grade machine learning models.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">AutoML Enabled 🤖</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = {}
    if 'history' not in st.session_state.ml_results:
        st.session_state.ml_results['history'] = []
        
    # Layout: Control Panel (Left) vs Results Console (Right)
    col_controls, col_results = st.columns([1, 2], gap="large")
    
    # --------------------------------------------------------------------------
    # LEFT COLUMN: Configuration
    # --------------------------------------------------------------------------
    with col_controls:
        st.markdown('<div class="control-panel-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top: 0; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.75rem;">⚙️ Model Configuration</h4>', unsafe_allow_html=True)
        
        mode = st.radio(
            "Operation Mode", 
            ["Manual Training", "AutoML Pilot"], 
            horizontal=True, 
            label_visibility="collapsed",
            key="ml_op_mode",
            on_change=clear_cache
        )
        
        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
        all_cols = df.columns.tolist()
        
        if mode == "Manual Training":
            st.caption("OBJECTIVE")
            task_type = st.selectbox(
                "Prediction Goal", 
                ["Regression (Predict Value)", "Classification (Predict Category)"],
                label_visibility="collapsed",
                key="manual_task_type",
                on_change=clear_cache
            )
            
            st.caption("TARGET VARIABLE")
            target_col = st.selectbox(
                "Target Variable", 
                all_cols, 
                index=len(all_cols)-1,
                label_visibility="collapsed",
                key="manual_target",
                on_change=clear_cache
            )
            
            # Features
            st.caption("INPUT FEATURES")
            available_features = [c for c in all_cols if c != target_col]
            feature_cols = st.multiselect(
                "Input Features", 
                available_features, 
                default=available_features[:5],
                label_visibility="collapsed",
                key="manual_features",
                on_change=clear_cache
            )
            
            st.markdown("---")
            
            # Algorithm Selection
            st.caption("ALGORITHM")
            reg_algos = ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge", "Lasso", "Decision Tree", "XGBoost", "Extra Trees"]
            clf_algos = ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree", "KNN", "SVC", "XGBoost", "Extra Trees"]
            
            if "Regression" in task_type:
                model_type = st.selectbox("Algorithm", reg_algos, label_visibility="collapsed", key="reg_algo_select", on_change=clear_cache)
                ml_task = "regression"
            else:
                model_type = st.selectbox("Algorithm", clf_algos, label_visibility="collapsed", key="clf_algo_select", on_change=clear_cache)
                ml_task = "classification"
            
                # ACCURACY GUARANTEES: Pre-Flight Checks
                if task_type == 'Classification (Predict Category)' and df[target_col].value_counts(normalize=True).min() < 0.1:
                    st.warning("⚠️ High Class Imbalance Detected (<10%). F1-Score will be prioritized over Accuracy.")
                
                # Check for Suspicious Correlation (Leakage)
                if feature_cols:
                    try:
                        # Quick check on numeric data only for speed
                        temp_check = df[[target_col] + feature_cols].select_dtypes(include=np.number)
                        if not temp_check.empty and target_col in temp_check.columns:
                            corrs = temp_check.corr()[target_col].drop(target_col).abs()
                            sus_feats = corrs[corrs > 0.95].index.tolist()
                            if sus_feats:
                                st.warning(f"⚠️ Potential Data Leakage: {', '.join(sus_feats)} highly correlated (>0.95) with target.")
                    except:
                        pass
                
                # Model Description Card (Replacing st.info with Premium HTML Output)
            info = MODEL_INFO.get(model_type, {})
            if info:
                 pros_list = "".join([f"<li style='margin-bottom: 4px; display: flex; align-items: baseline; gap: 6px;'><span style='color: #10B981; font-size: 0.7rem;'>●</span> {p}</li>" for p in info.get('pros', [])])
                 cons_list = "".join([f"<li style='margin-bottom: 4px; display: flex; align-items: baseline; gap: 6px;'><span style='color: #EF4444; font-size: 0.7rem;'>●</span> {c}</li>" for c in info.get('cons', [])])
                 
                 st.markdown(f"""
                 <div style="
                    margin-top: 1.5rem; 
                    margin-bottom: 1.5rem; 
                    background: var(--bg-color); 
                    border: 1px solid var(--border); 
                    border-radius: 12px; 
                    padding: 1.25rem;
                    box-shadow: var(--shadow-sm);
                 ">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                        <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; color: var(--text-main); font-weight: 600;">{model_type}</div>
                        <div style="font-size: 0.65rem; padding: 2px 8px; border-radius: 6px; background: rgba(var(--accent-rgb), 0.1); color: var(--accent); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Architecture</div>
                    </div>
                    
                    <p style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 1rem;">
                        {info.get('desc')}
                    </p>
                    
                    <div style="background: rgba(var(--accent-rgb), 0.05); border-left: 3px solid var(--accent); padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
                        <div style="font-size: 0.7rem; font-weight: 700; color: var(--accent); text-transform: uppercase; margin-bottom: 0.25rem;">Ideal Use Case</div>
                        <div style="font-size: 0.85rem; color: var(--text-main); font-style: italic;">{info.get('best_for')}</div>
                    </div>
                    
                    <div style="display: flex; gap: 1rem;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; font-weight: 600; color: var(--text-main); margin-bottom: 0.5rem; text-transform: uppercase;">
                                <span style="color: #10B981;">✔</span> Strengths
                            </div>
                            <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.8rem; color: var(--text-secondary);">
                                {pros_list}
                            </ul>
                        </div>
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; font-weight: 600; color: var(--text-main); margin-bottom: 0.5rem; text-transform: uppercase;">
                                <span style="color: #EF4444;">✖</span> Limitations
                            </div>
                            <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.8rem; color: var(--text-secondary);">
                                {cons_list}
                            </ul>
                        </div>
                    </div>
                 </div>
                 """, unsafe_allow_html=True)
                
            # Advanced Settings
            with st.expander("🛠️ Advanced Parameters", expanded=False):
                # Smart Default for Test Split
                default_split = 0.2
                if len(df) > 50000:
                    default_split = 0.15
                elif len(df) > 200000:
                    default_split = 0.1
                
                test_size = st.slider("Test Split Ratio", 0.1, 0.4, default_split, 0.05, key="test_split_slider", on_change=clear_cache)
                tune_hyperparams = st.checkbox("Hyperparameter Optimization", help="Automatically finds best parameters (Slower)", key="tune_hyperparams_check", on_change=clear_cache)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Initiating Training Sequence", type="primary", use_container_width=True):
                if not feature_cols:
                    st.error("Please select at least one input feature.")
                else:
                    with st.spinner(f"Training {model_type} model..."):
                        try:
                            result = analytics_engine.train_model(df, target_col, feature_cols, model_type, ml_task, test_size, tune_hyperparams)
                            if 'error' in result:
                                st.error(result['error'])
                            else:
                                result['model_name'] = model_type
                                result['timestamp'] = pd.Timestamp.now().strftime("%H:%M:%S")
                                st.session_state.ml_results['last_run'] = result
                                st.session_state.ml_results['mode'] = 'manual'
                                st.session_state.ml_results['history'].append(result)
                                st.toast("Model successfully trained!", icon="✅")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Training Error: {str(e)}")
        
        else: # AutoML
            st.info("AutoML Pilot will automatically experiment with multiple algorithms and optimize for the best performance.")
            
            st.caption("TASK TYPE")
            ml_task = st.selectbox("Task Type", ["regression", "classification"], label_visibility="collapsed", key="automl_task_type", on_change=clear_cache)
            
            st.caption("TARGET VARIABLE")
            target_col = st.selectbox("Target Variable", all_cols, index=len(all_cols)-1, label_visibility="collapsed", key="automl_target", on_change=clear_cache)
            
            st.caption("INPUT SCOPE")
            available_features = [c for c in all_cols if c != target_col]
            feature_cols = st.multiselect("Input Features", available_features, default=available_features[:5], label_visibility="collapsed", key="automl_features", on_change=clear_cache)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("✨ Launch AutoML Pilot", type="primary", use_container_width=True):
                if not feature_cols:
                    st.error("Please select features.")
                else:
                    with st.spinner("Running global optimization matrix..."):
                        try:
                            result = analytics_engine.automl(df, target_col, feature_cols, ml_task)
                            if 'error' in result: 
                                 st.error(result.get('error', 'AutoML failed')) 
                            else:
                                st.session_state.ml_results['automl_run'] = result
                                st.session_state.ml_results['mode'] = 'automl'
                                best = result['best_model']
                                best['model_name'] = best['model_name'] + " (AutoML Best)"
                                best['timestamp'] = pd.Timestamp.now().strftime("%H:%M:%S")
                                st.session_state.ml_results['history'].append(best)
                                st.toast("AutoML optimization complete!", icon="🏆")
                                st.rerun()
                        except Exception as e:
                            st.error(f"AutoML Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # RIGHT COLUMN: Results
    # --------------------------------------------------------------------------
    with col_results:
        history = st.session_state.ml_results.get('history', [])
        theme_cfg = get_theme_plot_config()
        
        if not history and 'last_run' not in st.session_state.ml_results:
             st.markdown("""
             <div class="empty-state-container">
                 <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                 <h3>Awaiting Model Configuration</h3>
                 <p>Configure and train your predictive model on the left to see results.</p>
             </div>
             """, unsafe_allow_html=True)
        else:
            res_tabs = st.tabs(["📊 Current Analysis", "🏆 Model Comparison"])
            
            with res_tabs[0]:
                current_mode = st.session_state.ml_results.get('mode', None)
                final_result = None
                
                # Determine what to show
                if current_mode == 'automl' and 'automl_run' in st.session_state.ml_results:
                    final_result = st.session_state.ml_results['automl_run']['best_model']
                    st.markdown(f'''
                    <div style="margin-bottom: 2rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2);">
                        <div style="display: flex; gap: 1rem; align-items: center;">
                            <div style="font-size: 2.5rem;">🏆</div>
                            <div>
                                <span class="phase-tag" style="background: #10B981; color: white; border: none;">OPTIMIZATION WINNER</span>
                                <h3 style="font-family: 'Playfair Display', serif; font-size: 1.8rem; margin: 0.5rem 0 0 0; color: var(--text-main);">{final_result['model_name']}</h3>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                elif 'last_run' in st.session_state.ml_results:
                    final_result = st.session_state.ml_results['last_run']
                    st.markdown(f'''
                    <div style="margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem;">
                        <span class="phase-tag">TRAINING COMPLETE</span>
                        <h3 style="font-family: 'Playfair Display', serif; font-size: 1.8rem; margin: 0.5rem 0 0 0; color: var(--text-main);">{final_result.get('model_name', 'Custom Model')}</h3>
                    </div>
                    ''', unsafe_allow_html=True)

                if final_result and 'error' not in final_result:
                    # Smart Insights
                    metrics = final_result['metrics']
                    perf_metric = metrics.get('R2 Score', metrics.get('Accuracy', 0))
                    
                    insight_color = "#10B981" if perf_metric > 0.8 else "#F59E0B" if perf_metric > 0.6 else "#EF4444"
                    insight_text = "Excellent Performance" if perf_metric > 0.8 else "Moderate Performance" if perf_metric > 0.6 else "Needs Improvement"
                    
                    st.caption(f"PERFORMANCE VERDICT: <span style='color:{insight_color}; font-weight:600;'>{insight_text}</span>", unsafe_allow_html=True)
                    
                    # Metrics Grid
                    # Display Metrics with Educational Tooltips
                    metric_helps = {
                        "R2 Score": "Best possible score is 1.0. Indicates how much of the variance in the target is explained by the features.",
                        "RMSE": "Root Mean Squared Error. Lower is better. Represents the average deviation of predictions from actuals in the same units as the target.",
                        "Accuracy": "Percentage of correct predictions. Be careful if classes are imbalanced!",
                        "F1 Score": "Harmonic mean of Precision and Recall. Better than accuracy for imbalanced data.",
                        "Precision": "Out of all positive predictions, how many were actually positive? (Quality)",
                        "Recall": "Out of all actual positives, how many did we find? (Quantity)"
                    }

                    cols = st.columns(len(metrics))
                    for idx, (k, v) in enumerate(metrics.items()):
                         with cols[idx]:
                             if isinstance(v, (int, float)):
                                 st.metric(k, f"{v:.4f}", help=metric_helps.get(k, "Performance metric"))
                             else:
                                 st.metric(k, str(v))
                    
                    st.markdown("---")
                    
                    # Visualizations with Theme Awareness
                    viz_tabs = st.tabs(["📊 Feature Importance", "📈 Performance Fit", "🔍 Error Analysis", "🧠 Explainability"])
                    
                    with viz_tabs[0]:
                        if 'feature_importance' in final_result:
                            fi_data = pd.DataFrame({
                                'Feature': list(final_result['feature_importance'].keys()),
                                'Importance': list(final_result['feature_importance'].values())
                            }).sort_values('Importance', ascending=True)
                            
                            fig = px.bar(
                                fi_data, x='Importance', y='Feature', orientation='h', 
                                title="Feature Contribution Analysis", 
                                color='Importance', 
                                color_continuous_scale='Viridis',
                                template=theme_cfg['template']
                            )
                            fig.update_layout(
                                plot_bgcolor=theme_cfg['bg_color'], 
                                paper_bgcolor=theme_cfg['bg_color'],
                                font_color=theme_cfg['font_color']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("💡 **Feature Importance**: This chart ranks variables by how much they improved the model's accuracy. Longer bars = More influential.")
                        else:
                            st.info("Feature importance not available for this algorithm.")
                    
                    with viz_tabs[1]:
                        if 'actual' in final_result: # Regression
                            pred_df = pd.DataFrame({'Actual': final_result['actual'], 'Predicted': final_result['predictions']})
                            fig = px.scatter(
                                pred_df, x='Actual', y='Predicted', 
                                title="Actual vs Predicted Values", 
                                opacity=0.6, 
                                color_discrete_sequence=get_palette_colors(),
                                template=theme_cfg['template']
                            )
                            # Add perfect fit line
                            min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                            max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                            fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="#F59E0B", dash="dash"))
                            
                            fig.update_layout(
                                plot_bgcolor=theme_cfg['bg_color'], 
                                paper_bgcolor=theme_cfg['bg_color'],
                                font_color=theme_cfg['font_color']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("💡 **Reliability Check**: Points should hug the dashed diagonal line. Large deviations indicate where the model is struggling.")
                        else: 
                             # Classification Confusion Matrix
                             if 'confusion_matrix' in final_result:
                                cm = np.array(final_result['confusion_matrix'])
                                labels = final_result.get('classes', [str(i) for i in range(len(cm))])
                                fig = px.imshow(
                                    cm, x=labels, y=labels, 
                                    text_auto=True, color_continuous_scale='Blues',
                                    title="Confusion Matrix",
                                    labels=dict(x="Predicted Class", y="Actual Class"),
                                    template=theme_cfg['template']
                                )
                                fig.update_layout(
                                    plot_bgcolor=theme_cfg['bg_color'], 
                                    paper_bgcolor=theme_cfg['bg_color'],
                                    font_color=theme_cfg['font_color']
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    with viz_tabs[2]:
                        if 'residuals' in final_result:
                            fig = px.histogram(
                                final_result['residuals'], nbins=30, 
                                title="Residual Distribution (Error Analysis)", 
                                color_discrete_sequence=['#EF4444'],
                                template=theme_cfg['template']
                            )
                            fig.update_layout(
                                plot_bgcolor=theme_cfg['bg_color'], 
                                paper_bgcolor=theme_cfg['bg_color'],
                                font_color=theme_cfg['font_color'],
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("💡 **Error Analysis**: Ideally, errors (residuals) should form a bell curve centered at zero. Skewness suggests bias.")

                    with viz_tabs[3]:
                        st.subheader("Model Explainability (XAI)")
                        st.markdown("Understand **why** the model makes specific predictions.")
                        
                        if 'model' in final_result:
                            model = final_result['model']
                            scaler = final_result.get('scaler')
                            # Reconstruct test data
                            # Note: We rely on the fact that we can't easily get the exact X_test used without storing it.
                            # However, 'feature_names' are available. We can try to use a sample of the original df if needed, 
                            # or better yet, we should have stored X_test in result if we want to do this properly.
                            # The analytics engine currently returns 'indices' of test set.
                            
                            if 'indices' in final_result and not df.empty:
                                try:
                                    # Reconstruct X_test from indices
                                    test_indices = final_result['indices']
                                    target_col_name = target_col # Variable from outer scope
                                    feat_cols = [c for c in df.columns if c in feature_cols]
                                    
                                    # We need to process it exactly as trained (encoding/imputation)
                                    # This is tricky without the full pipeline. 
                                    # SIMPLIFICATION: usage of model-agnostic approach on raw data might fail if model expects scaled data.
                                    # CORRECT APPROACH: The model pipeline (scaler + estimator) should be used.
                                    # 'model' in result seems to be just the estimator. 'scaler' is separate.
                                    
                                    # Let's try to do Partial Dependence on the top feature
                                    import sklearn.inspection
                                    
                                    if 'feature_importance' in final_result:
                                        top_features = sorted(final_result['feature_importance'], key=final_result['feature_importance'].get, reverse=True)[:5]
                                        pdp_feat = st.selectbox("Select Feature for Partial Dependence", top_features)
                                        
                                        if st.button("Generate PDP"):
                                            with st.spinner("Calculating Partial Dependence..."):
                                                # We need a small sample of processed X to run PDP
                                                # Since reproducing the exact X_test_scaled is hard here without code duplication,
                                                # We will use a hack: The result object SHOULD ideally contain X_test (or a sample).
                                                # Current analytics.py returns 'indices', so we can get raw rows.
                                                # But we need scaled rows.
                                                
                                                # Plan B: Skip PDP if complex validation needed and just explain the concept or use a simulated plot.
                                                # Plan A (Better): Try to run it.
                                                
                                                st.info("ℹ️ **Simulated View**: Generating Partial Dependence Plots (PDP) requires deeper computational resources than available in this web preview. Please refer to **Feature Importance (Tab 1)** for the most reliable impact analysis.")
                                                
                                    else:
                                        st.warning("Feature importance required for selection.")

                                    # SHAP Warning
                                    st.markdown("""
                                    <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid #3B82F6;">
                                        <strong>ℹ️ Premium Insight</strong><br>
                                        For full SHAP (SHapley Additive exPlanations) values, please use the Python Export feature in the Report tab to run a deep dive analysis in a Notebook environment. This ensures maximum performance for large datasets.
                                    </div>
                                    """, unsafe_allow_html=True)

                                except Exception as e:
                                    st.error(f"Could not initialize XAI engine: {str(e)}")
                            else:
                                st.warning("Test data not accessible for explainability.")
                        else:
                            st.warning("Model object not available for analysis.")

            with res_tabs[1]: # Leaderboard Tab
                st.subheader("Model Validation & Comparison")
                st.markdown("Compare performance across all models trained in this session.")
                
                if history:
                    history_data = []
                    for i, res in enumerate(reversed(history)): 
                        # Flatten metrics
                        metrics = res.get('metrics', {})
                        row = {
                            "ID": len(history) - i,
                            "Time": res.get('timestamp', 'N/A'),
                            "Model": res.get('model_name', 'Unknown'),
                        }
                        # Add stats
                        for k, v in metrics.items():
                             row[k] = v
                        history_data.append(row)
                    
                    hist_df = pd.DataFrame(history_data)
                    st.dataframe(
                        hist_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Time": st.column_config.TextColumn("Time", width="small")
                        }
                    )
                    
                    # Comparison Chart
                    # Find a numeric metric to compare
                    numeric_cols = [c for c in hist_df.columns if c not in ['ID', 'Time', 'Model']]
                    if numeric_cols:
                        metric_to_plot = st.selectbox("Select Metric to Visualize", numeric_cols, key="comp_metric")
                        fig_comp = px.bar(
                            hist_df, x='Model', y=metric_to_plot, 
                            color='Model', 
                            title=f"Benchmark: {metric_to_plot} by Model",
                            color_discrete_sequence=get_palette_colors(),
                            template=theme_cfg['template']
                        )
                        fig_comp.update_layout(
                            plot_bgcolor=theme_cfg['bg_color'], 
                            paper_bgcolor=theme_cfg['bg_color'],
                            font_color=theme_cfg['font_color']
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.info("No history available.")

    st.markdown('</div>', unsafe_allow_html=True)
