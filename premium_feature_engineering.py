"""
Premium Feature Engineering Module
Advanced data transformation and feature creation tools
"""

import numpy as np
import pandas as pd
import streamlit as st


def update_working_data(df, message, icon):
    """Update working data with history tracking"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Save current state to history before updating (limit to last 10 steps)
    st.session_state.history.append(st.session_state.working_data.copy())
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)
        
    st.session_state.working_data = df
    st.toast(message, icon=icon)
    st.rerun()

def render_data_processing_tab(df: pd.DataFrame):
    """Render premium data processing interface"""
    
    # Premium Header
    st.markdown("""
    <div class="animate-enter" style="margin-bottom: 2rem;">
        <h1 style="font-family: 'Playfair Display', serif; font-size: 3rem; color: var(--text-main);">Refinement Studio</h1>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: var(--text-secondary); max-width: 700px;">
            Engineer enterprise-grade features with precision tools. Clean, transform, and enrich your dataset for superior model performance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get current data
    if 'working_data' not in st.session_state or st.session_state.working_data.empty:
        st.markdown("""
        <div class="premium-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">📂</div>
            <h3 style="margin-bottom: 1rem;">No Active Dataset</h3>
            <p style="color: var(--text-secondary);">Initialize a project in the sidebar to access the Refinement Studio.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.working_data.copy()
    
    # Custom Tabs styling through CSS is handled globally, so we use standard tabs but with better labels
    fe_tabs = st.tabs([
        "🛡️ Quality Control",
        "⚡ Feature Engineering", 
        "🧮 Calculator", 
        "📅 Temporal Logic",
        "🏷️ Encoders",
        "📈 Rolling Stats"
    ])
    
    with fe_tabs[0]: # Quality Control
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        
        col_qc_1, col_qc_2 = st.columns([1, 1], gap="medium")
        
        with col_qc_1:
            st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("#### 🧩 Missing Value Strategy")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Diagnose and repair data gaps using statistical imputation.</div>", unsafe_allow_html=True)
            
            # Identify missing columns
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if missing_cols:
                target_col = st.selectbox("Select Target Field", missing_cols)
                imputation_method = st.selectbox(
                    "Repair Strategy",
                    ["Drop Rows", "Mean Imputation", "Median Imputation", "Mode Imputation", "Constant Value", "Forward Fill", "Backward Fill"]
                )
                
                constant_val = None
                if imputation_method == "Constant Value":
                    constant_val = st.text_input("Value")
                
                if st.button("Execute Repair Sequence", type="primary", use_container_width=True):
                    try:
                        if imputation_method == "Drop Rows":
                            df = df.dropna(subset=[target_col])
                            action = "Pruned incomplete records"
                        elif "Mean" in imputation_method:
                            df[target_col] = df[target_col].fillna(df[target_col].mean())
                            action = "Imputed with mean"
                        elif "Median" in imputation_method:
                            df[target_col] = df[target_col].fillna(df[target_col].median())
                            action = "Imputed with median"
                        elif "Mode" in imputation_method:
                            df[target_col] = df[target_col].fillna(df[target_col].mode()[0])
                            action = "Imputed with mode"
                        elif "Constant" in imputation_method:
                            df[target_col] = df[target_col].fillna(constant_val)
                            action = f"Filled with constant '{constant_val}'"
                        elif "Forward" in imputation_method:
                            df[target_col] = df[target_col].ffill()
                            action = "Forward propagation applied"
                        elif "Backward" in imputation_method:
                            df[target_col] = df[target_col].bfill()
                            action = "Backward propagation applied"
                            
                        update_working_data(df, f"✅ Success: {action} for '{target_col}'", "🛡️")
                    except Exception as e:
                        st.error(f"Operation Failed: {str(e)}")
            else:
                st.markdown("""
                <div class="empty-state-container" style="padding: 2rem; border-color: rgba(16, 185, 129, 0.2); background: rgba(16, 185, 129, 0.02);">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem; animation: pulse 3s infinite;">✨</div>
                    <div style="font-weight: 600; color: #10B981; font-size: 1.1rem; margin-bottom: 0.25rem;">Perfect Integrity</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">No missing values detected. Your dataset is pristine.</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_qc_2:
            st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("#### 🏗️ Schema Management")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Audit structure and type consistency.</div>", unsafe_allow_html=True)
            
            # Duplicate check
            duplicates = df.duplicated().sum()
            sub_c1, sub_c2 = st.columns([2, 1])
            with sub_c1:
                st.markdown(f"**Duplicate Records:** <span style='color: {'#EF4444' if duplicates > 0 else '#10B981'}; font-weight: 600;'>{duplicates}</span>", unsafe_allow_html=True)
            with sub_c2:
                if duplicates > 0:
                    if st.button("Prune", type="secondary", use_container_width=True):
                        df = df.drop_duplicates()
                        update_working_data(df, f"Removed {duplicates} duplicate records", "🗑️")
            
            st.markdown("---")
            
            # Drop logic
            st.caption("FIELD PRUNING")
            cols_to_drop = st.multiselect("Select Fields to Remove", df.columns, label_visibility="collapsed", placeholder="Choose columns...")
            if cols_to_drop:
                if st.button(f"Drop {len(cols_to_drop)} Columns", type="secondary", use_container_width=True):
                    df = df.drop(columns=cols_to_drop)
                    update_working_data(df, f"Pruned {len(cols_to_drop)} columns", "🗑️")
            
            st.markdown("---")
            
            # Type Conversion
            st.caption("TYPE CASTING")
            type_c1, type_c2 = st.columns([1.5, 1])
            with type_c1:
                conv_col = st.selectbox("Field", df.columns, label_visibility="collapsed")
            with type_c2:
                target_type = st.selectbox("Type", ["Numeric", "String", "Date", "Category"], label_visibility="collapsed")
            
            if st.button("Cast Type", use_container_width=True):
                try:
                    if target_type == "Numeric":
                        df[conv_col] = pd.to_numeric(df[conv_col], errors='coerce')
                    elif target_type == "String":
                        df[conv_col] = df[conv_col].astype(str)
                    elif target_type == "Date":
                        df[conv_col] = pd.to_datetime(df[conv_col], errors='coerce')
                    elif target_type == "Category":
                        df[conv_col] = df[conv_col].astype('category')
                        
                    update_working_data(df, f"Converted '{conv_col}' to {target_type}", "🔄")
                except Exception as e:
                    st.error(f"Cast failed: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

    with fe_tabs[1]:  # Feature Engineering
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        
        col_fe_1, col_fe_2 = st.columns([1, 1.5], gap="large")
        
        with col_fe_1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### ⚡ Feature Synthesis")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Derive new attributes using mathematical and logical operations.</div>", unsafe_allow_html=True)
            
            # Column selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_type = st.radio(
                "Operation Type",
                ["Mathematical", "Conditional", "Binning", "Interaction"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            if feature_type == "Mathematical":
                st.caption("ARITHMETIC OPERATIONS")
                col1_select = st.selectbox("Field A", numeric_cols, key="math_col1")
                operation = st.selectbox("Operator", ["+", "-", "*", "/", "**", "log", "sqrt"])
                
                if operation in ["+", "-", "*", "/", "**"]:
                    col2_select = st.selectbox("Field B", numeric_cols, key="math_col2")
                    default_name = f"{col1_select}_{operation}_{col2_select}"
                else:
                    default_name = f"{operation}_{col1_select}"
                
                new_col_name = st.text_input("Resulting Field Name", default_name)
                
                if st.button("Generate Feature", type="primary", use_container_width=True):
                    try:
                        if operation == "+":
                            df[new_col_name] = df[col1_select] + df[col2_select]
                        elif operation == "-":
                            df[new_col_name] = df[col1_select] - df[col2_select]
                        elif operation == "*":
                            df[new_col_name] = df[col1_select] * df[col2_select]
                        elif operation == "/":
                            df[new_col_name] = np.where(df[col2_select] != 0, df[col1_select] / df[col2_select], 0)
                        elif operation == "**":
                            df[new_col_name] = df[col1_select] ** df[col2_select]
                        elif operation == "log":
                            df[new_col_name] = np.log(df[col1_select].clip(lower=1e-10))
                        elif operation == "sqrt":
                            df[new_col_name] = np.sqrt(df[col1_select].clip(lower=0))
                        
                        update_working_data(df, f"✅ Created feature: {new_col_name}", "⚡")
                    except Exception as e:
                        st.error(f"Computation Error: {str(e)}")
            
            elif feature_type == "Conditional":
                st.caption("LOGICAL BRANCHING")
                condition_col = st.selectbox("Target Field", df.columns.tolist())
                c_c1, c_c2 = st.columns([1, 1])
                with c_c1:
                    condition_type = st.selectbox("Operator", ["Greater than", "Less than", "Equal to", "Contains"])
                with c_c2:
                    if "Contains" in condition_type:
                        threshold = st.text_input("Value")
                    else:
                        threshold = st.number_input("Threshold") if df[condition_col].dtype != 'object' else st.text_input("Value")
                
                v_c1, v_c2 = st.columns(2)
                with v_c1:
                    true_value = st.text_input("If True", "Yes")
                with v_c2:
                    false_value = st.text_input("If False", "No")
                    
                new_col_name = st.text_input("Result Name", f"{condition_col}_flag")
                
                if st.button("Apply Logic", type="primary", use_container_width=True):
                    try:
                        if condition_type == "Greater than":
                            df[new_col_name] = np.where(df[condition_col] > threshold, true_value, false_value)
                        elif condition_type == "Less than":
                            df[new_col_name] = np.where(df[condition_col] < threshold, true_value, false_value)
                        elif condition_type == "Equal to":
                            df[new_col_name] = np.where(df[condition_col] == threshold, true_value, false_value)
                        else:
                            df[new_col_name] = np.where(df[condition_col].astype(str).str.contains(str(threshold), na=False), true_value, false_value)
                        
                        update_working_data(df, f"✅ Logic applied: {new_col_name}", "⚡")
                    except Exception as e:
                        st.error(f"Logic Error: {str(e)}")

            elif feature_type == "Binning":
                st.caption("DISCRETIZATION STRATEGY")
                if len(numeric_cols) > 0:
                    bin_col = st.selectbox("Target Field", numeric_cols, key="bin_col_synth")
                    n_bins = st.slider("Bin Count", 2, 20, 5, key="n_bins_synth")
                    bin_method = st.selectbox("Strategy", ["Equal Width", "Equal Frequency (Quantile)"], key="bin_method_synth")
                    
                    if st.button("Generate Bins", type="primary", use_container_width=True, key="btn_bin_synth"):
                        try:
                            new_col_name = f"{bin_col}_bins"
                            if "Width" in bin_method:
                                df[new_col_name] = pd.cut(df[bin_col], bins=n_bins, labels=False)
                            else:
                                df[new_col_name] = pd.qcut(df[bin_col], q=n_bins, labels=False, duplicates='drop')
                            
                            update_working_data(df, f"✅ Discretized '{bin_col}' into {n_bins} bins", "📊")
                        except Exception as e:
                            st.error(f"Binning failed: {str(e)}")
                else:
                    st.info("No numeric fields available for binning")

            elif feature_type == "Interaction":
                st.caption("CROSS-FEATURE SYNTHESIS")
                if len(numeric_cols) >= 2:
                    interact_col1 = st.selectbox("Field A", numeric_cols, key="interact1_synth")
                    interact_col2 = st.selectbox("Field B", [col for col in numeric_cols if col != interact_col1], key="interact2_synth")
                    interaction_op = st.selectbox("Interaction Type", ["Multiply (A × B)", "Divide (A ÷ B)"], key="interact_op_synth")
                    
                    if st.button("Generate Interaction", type="primary", use_container_width=True, key="btn_interact_synth"):
                        try:
                            if "Multiply" in interaction_op:
                                new_col_name = f"{interact_col1}_x_{interact_col2}"
                                df[new_col_name] = df[interact_col1] * df[interact_col2]
                                icon = "✖️"
                            else:
                                new_col_name = f"{interact_col1}_div_{interact_col2}"
                                # Handle division by zero
                                df[new_col_name] = np.where(df[interact_col2] != 0, df[interact_col1] / df[interact_col2], 0)
                                icon = "➗"
                            
                            update_working_data(df, f"✅ Created interaction: {new_col_name}", icon)
                        except Exception as e:
                            st.error(f"Interaction failed: {str(e)}")
                else:
                    st.warning("Requires at least two numeric fields.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_fe_2:
             # Real-time preview of the dataframe
            st.markdown("#### 👁️ Live Data Preview")
            if not df.empty:
                # Highlight potentially new columns (last 5)
                cols_to_show = df.columns[-5:].tolist() if len(df.columns) > 5 else df.columns.tolist()
                
                st.dataframe(
                    df[cols_to_show].head(10),
                    use_container_width=True,
                    hide_index=True
                )
                
                if len(cols_to_show) > 0:
                    st.markdown("#### 📊 Distribution Analysis")
                    stats_df = df[cols_to_show].describe()
                    st.dataframe(stats_df, use_container_width=True)
            else:
                 st.info("Upload data to see preview")
    
    with fe_tabs[2]:  # Calculator
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        
        col_calc_1, col_calc_2 = st.columns(2, gap="large")
        
        with col_calc_1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### ⚖️ Scale & Distribution")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Normalize numeric ranges for machine learning compatibility.</div>", unsafe_allow_html=True)
            
            if len(numeric_cols) > 0:
                norm_col = st.selectbox("Target Field", numeric_cols, key="norm_col")
                norm_method = st.selectbox("Methodology", ["Min-Max Scaling", "Z-Score Standardization", "Robust Scaling (IQR)"])
                
                if st.button("Apply Scaling", use_container_width=True):
                    try:
                        new_col_name = f"{norm_col}_{norm_method.lower().split(' ')[0]}"
                        
                        if "Min-Max" in norm_method:
                            min_val = df[norm_col].min()
                            max_val = df[norm_col].max()
                            df[new_col_name] = (df[norm_col] - min_val) / (max_val - min_val)
                        elif "Z-Score" in norm_method:
                            mean_val = df[norm_col].mean()
                            std_val = df[norm_col].std()
                            df[new_col_name] = (df[norm_col] - mean_val) / std_val
                        else:  # Robust
                            median_val = df[norm_col].median()
                            q75 = df[norm_col].quantile(0.75)
                            q25 = df[norm_col].quantile(0.25)
                            iqr = q75 - q25
                            df[new_col_name] = (df[norm_col] - median_val) / iqr
                        
                        update_working_data(df, f"✅ Scaled '{norm_col}' via {norm_method}", "⚖️")
                    except Exception as e:
                        st.error(f"Scaling failed: {str(e)}")
            else:
                st.info("No numeric fields available for scaling")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_calc_2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Discretization Strategy")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Convert continuous variables into categorical bins.</div>", unsafe_allow_html=True)
            
            if len(numeric_cols) > 0:
                bin_col = st.selectbox("Target Field", numeric_cols, key="bin_col")
                n_bins = st.slider("Bin Count", 2, 20, 5)
                bin_method = st.selectbox("Strategy", ["Equal Width", "Equal Frequency (Quantile)"])
                
                if st.button("Generate Bins", use_container_width=True):
                    try:
                        new_col_name = f"{bin_col}_bins"
                        
                        if "Width" in bin_method:
                            df[new_col_name] = pd.cut(df[bin_col], bins=n_bins, labels=False)
                        else:
                            df[new_col_name] = pd.qcut(df[bin_col], q=n_bins, labels=False, duplicates='drop')
                        
                        update_working_data(df, f"✅ Discretized '{bin_col}' into {n_bins} bins", "📊")
                    except Exception as e:
                        st.error(f"Binning failed: {str(e)}")
            else:
                st.info("No numeric fields available for binning")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with fe_tabs[3]:  # Temporal Logic
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        
        # Find datetime columns (including object-type dates)
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        potential_date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Check first non-null value
                    val = df[col].dropna().iloc[0]
                    if len(str(val)) > 6 and ('-' in str(val) or '/' in str(val)):
                        potential_date_cols.append(col)
                except:
                    pass
        
        all_date_cols = list(set(datetime_cols + potential_date_cols))
        
        col_time_1, col_time_2 = st.columns(2, gap="large")
        
        with col_time_1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### 📅 Feature Extraction")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Derive cyclical components from timestamps.</div>", unsafe_allow_html=True)
            
            if len(all_date_cols) > 0:
                date_col = st.selectbox("Source Timestamp", all_date_cols)
                
                components = st.multiselect(
                    "Components",
                    ["Year", "Month", "Day", "Weekday", "Quarter", "Week of Year"],
                    default=["Year", "Month", "Day"],
                    label_visibility="collapsed"
                )
                
                if st.button("Extract Components", use_container_width=True):
                    try:
                        # Ensure datetime type
                        if df[date_col].dtype != 'datetime64[ns]':
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        
                        extracted = []
                        for component in components:
                            if component == "Year":
                                df[f"{date_col}_year"] = df[date_col].dt.year
                                extracted.append("Year")
                            elif component == "Month":
                                df[f"{date_col}_month"] = df[date_col].dt.month
                                extracted.append("Month")
                            elif component == "Day":
                                df[f"{date_col}_day"] = df[date_col].dt.day
                                extracted.append("Day")
                            elif component == "Weekday":
                                df[f"{date_col}_weekday"] = df[date_col].dt.dayofweek
                                extracted.append("Weekday")
                            elif component == "Quarter":
                                df[f"{date_col}_quarter"] = df[date_col].dt.quarter
                                extracted.append("Quarter")
                            elif component == "Week of Year":
                                df[f"{date_col}_week"] = df[date_col].dt.isocalendar().week
                                extracted.append("Week")
                        
                        update_working_data(df, f"✅ Extracted: {', '.join(extracted)}", "📅")
                    except Exception as e:
                        st.error(f"Extraction failed: {str(e)}")
            else:
                 st.info("No datetime fields detected.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_time_2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### ⏳ Temporal Deltas")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Calculate duration between two milestones.</div>", unsafe_allow_html=True)
            
            if len(all_date_cols) >= 2:
                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    date_col1 = st.selectbox("Start Event", all_date_cols, key="date1")
                with d_c2:
                    date_col2 = st.selectbox("End Event", all_date_cols, key="date2")
                
                unit = st.selectbox("Output Unit", ["Days", "Weeks", "Months", "Years"])
                
                if st.button("Calculate Delta", use_container_width=True):
                    try:
                        # Ensure datetime type
                        for d_col in [date_col1, date_col2]:
                            if df[d_col].dtype != 'datetime64[ns]':
                                df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
                        
                        diff = df[date_col2] - df[date_col1]
                        col_name = f"{date_col1}_to_{date_col2}_{unit.lower()}"
                        
                        if unit == "Days":
                            df[col_name] = diff.dt.days
                        elif unit == "Weeks":
                            df[col_name] = diff.dt.days / 7
                        elif unit == "Months":
                            df[col_name] = diff.dt.days / 30.44
                        elif unit == "Years":
                            df[col_name] = diff.dt.days / 365.25
                        
                        update_working_data(df, f"✅ Calculated duration in {unit.lower()}", "⏳")
                    except Exception as e:
                        st.error(f"Calculation failed: {str(e)}")
            else:
                st.info("Requires at least two date fields.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with fe_tabs[4]:  # Encoders
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        
        col_enc_1, col_enc_2 = st.columns([1.2, 1], gap="large")
        
        with col_enc_1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### 🏷️ Categorical Transformation")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Convert high-cardinality nominal data relevant for modeling.</div>", unsafe_allow_html=True)
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_cols) > 0:
                cat_col = st.selectbox("Target Category", categorical_cols)
                
                # Check cardinality
                cardinality = df[cat_col].nunique()
                
                c_e1, c_e2 = st.columns(2)
                with c_e1:
                    st.metric("Unique Values", cardinality)
                with c_e2:
                    if cardinality > 50:
                         st.warning("High Cardinality Detected")
                
                st.markdown("---")
                
                encoding_method = st.selectbox(
                    "Encoding Strategy",
                    ["One-Hot Encoding", "Label Encoding", "Frequency Encoding", "Target Encoding"]
                )
                
                if encoding_method == "Target Encoding":
                    if len(numeric_cols) > 0:
                         target_col = st.selectbox("Target Field (for Mean)", numeric_cols)
                    else:
                        st.error("Target encoding requires a numeric target field.")
                        target_col = None
                
                if st.button("Apply Transformation", use_container_width=True):
                    try:
                        if encoding_method == "One-Hot Encoding":
                            # Create dummy variables
                            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
                            df = pd.concat([df, dummies], axis=1)
                            action = f"One-Hot Encoded '{cat_col}'"
                            
                        elif encoding_method == "Label Encoding":
                            # Simple label encoding
                            unique_vals = df[cat_col].unique()
                            label_map = {val: i for i, val in enumerate(unique_vals)}
                            df[f"{cat_col}_encoded"] = df[cat_col].map(label_map)
                            action = f"Label Encoded '{cat_col}'"
                            
                        elif encoding_method == "Frequency Encoding":
                            # Encode by frequency
                            freq_map = df[cat_col].value_counts().to_dict()
                            df[f"{cat_col}_frequency"] = df[cat_col].map(freq_map)
                            action = f"Frequency Encoded '{cat_col}'"
                            
                        elif encoding_method == "Target Encoding" and target_col:
                            # Encode by target mean
                            target_map = df.groupby(cat_col)[target_col].mean().to_dict()
                            df[f"{cat_col}_target_encoded"] = df[cat_col].map(target_map)
                            action = f"Target Encoded '{cat_col}' against '{target_col}'"
                        
                        update_working_data(df, f"✅ Success: {action}", "🏷️")
                    except Exception as e:
                        st.error(f"Encoding Error: {str(e)}")
            else:
                st.info("No categorical fields detected for encoding.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_enc_2:
             st.markdown('<div class="premium-card">', unsafe_allow_html=True)
             st.markdown("#### 👁️ Schema Preview")
             st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Inspect category distribution.</div>", unsafe_allow_html=True)
             
             if len(categorical_cols) > 0 and 'cat_col' in locals():
                 if cat_col in df.columns:
                     val_counts = df[cat_col].value_counts().head(10)
                     st.bar_chart(val_counts, color="#10B981")
                     st.caption("Top 10 High-Frequency Labels")
             else:
                 st.info("Select a category to view distribution.")
             st.markdown('</div>', unsafe_allow_html=True)
    
    with fe_tabs[5]:  # Rolling Stats
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        
        col_stat_1, col_stat_2 = st.columns(2, gap="large")
        
        with col_stat_1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Window Functions")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Compute moving statistics over a defined window.</div>", unsafe_allow_html=True)
            
            if len(numeric_cols) > 0:
                stat_col = st.selectbox("Time-Series Field", numeric_cols, key="stat_col")
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    window_size = st.number_input("Window Size", min_value=2, value=5, step=1)
                with s_c2:
                    stat_type = st.selectbox("Metric", ["Mean", "Median", "Std Dev", "Min", "Max"])
                
                if st.button("Compute Rolling Stat", use_container_width=True):
                    try:
                        new_col_name = f"{stat_col}_rolling_{stat_type.lower().split(' ')[0]}_{window_size}"
                        
                        if "Mean" in stat_type:
                            df[new_col_name] = df[stat_col].rolling(window=window_size).mean()
                        elif "Median" in stat_type:
                            df[new_col_name] = df[stat_col].rolling(window=window_size).median()
                        elif "Std" in stat_type:
                            df[new_col_name] = df[stat_col].rolling(window=window_size).std()
                        elif "Min" in stat_type:
                            df[new_col_name] = df[stat_col].rolling(window=window_size).min()
                        elif "Max" in stat_type:
                            df[new_col_name] = df[stat_col].rolling(window=window_size).max()
                        
                        update_working_data(df, f"✅ Created rolling feature: {new_col_name}", "📈")
                    except Exception as e:
                        st.error(f"Computation failed: {str(e)}")
            else:
                st.info("No numeric fields available.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_stat_2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### ✖️ Cross-Feature Interactions")
            st.markdown("<div style='margin-bottom: 1.5rem; color: var(--text-secondary); font-size: 0.9rem;'>Synthesize features via multiplicative interactions.</div>", unsafe_allow_html=True)
            
            if len(numeric_cols) >= 2:
                interact_col1 = st.selectbox("Field A", numeric_cols, key="interact1")
                interact_col2 = st.selectbox("Field B", [col for col in numeric_cols if col != interact_col1], key="interact2")
                
                if st.button("Generate Interactions", use_container_width=True):
                    try:
                        new_col_name = f"{interact_col1}_x_{interact_col2}"
                        df[new_col_name] = df[interact_col1] * df[interact_col2]
                        
                        # Also create ratio if no zeros
                        msg = f"Created interaction: {new_col_name}"
                        if (df[interact_col2] != 0).all():
                            ratio_col_name = f"{interact_col1}_div_{interact_col2}"
                            df[ratio_col_name] = df[interact_col1] / df[interact_col2]
                            msg += f" & ratio: {ratio_col_name}"
                        
                        update_working_data(df, f"✅ {msg}", "✖️")
                    except Exception as e:
                        st.error(f"Interaction failed: {str(e)}")
            else:
                st.info("Requires at least two numeric fields.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature summary
    st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="premium-card">
        <h3 style="margin-bottom: 1rem;">📊 Engineering Summary</h3>
        <div style="display: flex; gap: 2rem; justify-content: space-between; align-items: center;">
    """, unsafe_allow_html=True)
    
    sum_col1, sum_col2, sum_col3 = st.columns([1, 1, 1])
    
    with sum_col1:
        st.metric("Total Features", len(df.columns))
    
    with sum_col2:
        original_cols = len(st.session_state.get('original_data', df).columns)
        new_features = len(df.columns) - original_cols
        st.metric("Engineered Features", max(0, new_features), delta=new_features)
    
    with sum_col3:
        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
        if st.button("🔄 Reset Data", use_container_width=True, type="secondary", key="btn_reset_data"):
            if 'original_data' in st.session_state:
                # Save current to history before reset
                if 'history' not in st.session_state: st.session_state.history = []
                st.session_state.history.append(st.session_state.working_data.copy())
                
                st.session_state.working_data = st.session_state.original_data.copy()
                st.toast("Reverted dataset to initial state.", icon="🔄")
                st.rerun()
                
    with sum_col3:
        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
        # Undo Button
        has_history = 'history' in st.session_state and len(st.session_state.history) > 0
        
        if st.button("↩️ Undo Last Change", use_container_width=True, type="primary", disabled=not has_history):
            if has_history:
                last_state = st.session_state.history.pop()
                st.session_state.working_data = last_state
                st.toast("Restored previous state", icon="↩️")
                st.rerun()
                
    st.markdown("</div>", unsafe_allow_html=True)

def apply_ai_suggestion(df, suggestion_type, **kwargs):
    """Apply AI-suggested feature engineering"""
    try:
        if suggestion_type == "create_ratio":
            col1, col2 = kwargs['col1'], kwargs['col2']
            new_col = f"{col1}_to_{col2}_ratio"
            df[new_col] = df[col1] / df[col2].replace(0, np.nan)
            
        elif suggestion_type == "create_interaction":
            col1, col2 = kwargs['col1'], kwargs['col2']
            new_col = f"{col1}_x_{col2}"
            df[new_col] = df[col1] * df[col2]
            
        elif suggestion_type == "normalize":
            col = kwargs['col']
            new_col = f"{col}_normalized"
            df[new_col] = (df[col] - df[col].mean()) / df[col].std()
        
        return df, f"✅ Applied AI suggestion: {suggestion_type}"
        
    except Exception as e:
        return df, f"❌ Error applying suggestion: {str(e)}"