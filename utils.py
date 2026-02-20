"""
Utility functions for the Advanced Data Analysis Platform
"""

import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from premium_config import QUALITY_THRESHOLDS


@st.cache_data(show_spinner=False)
def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> Tuple[bool, str]:
    """
    Validate if dataframe meets minimum requirements
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"Insufficient data. Need at least {min_rows} rows, got {len(df)}"
    
    return True, "DataFrame is valid"

@st.cache_data(show_spinner=False)
def get_column_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about DataFrame columns
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with column information
    """
    if df.empty:
        return {}
    
    info = {
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'boolean_columns': df.select_dtypes(include=['bool']).columns.tolist(),
        'column_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }
    
    return info

@st.cache_data(show_spinner=False)
def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive data quality metrics
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality metrics
    """
    if df.empty:
        return {'overall_score': 0.0, 'completeness': 0.0, 'consistency': 0.0, 'validity': 0.0}
    
    # Completeness: percentage of non-null values
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = total_cells - df.isnull().sum().sum()
    completeness = non_null_cells / total_cells if total_cells > 0 else 0
    
    # Consistency: percentage of non-duplicate rows
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    consistency = unique_rows / total_rows if total_rows > 0 else 0
    
    # Validity: check for reasonable data ranges and types
    validity_score = 1.0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        try:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validity_score *= (1 - inf_count / len(df))
            
            # Check for extreme outliers (beyond 4 standard deviations)
            if len(df[col].dropna()) > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-8))
                extreme_outliers = (z_scores > 4).sum()
                if extreme_outliers > 0:
                    validity_score *= (1 - extreme_outliers / len(df))
        except:
            pass
    
    # Overall score (weighted average)
    overall_score = (completeness * 0.4 + consistency * 0.3 + validity_score * 0.3)
    
    return {
        'overall_score': overall_score,
        'completeness': completeness,
        'consistency': consistency,
        'validity': validity_score
    }

@st.cache_data(show_spinner=False)
def get_quality_label(score: float) -> str:
    """
    Get quality label based on score
    
    Args:
        score: Quality score between 0 and 1
    
    Returns:
        Quality label string
    """
    if score >= QUALITY_THRESHOLDS['excellent']:
        return "Excellent"
    elif score >= QUALITY_THRESHOLDS['good']:
        return "Good"
    elif score >= QUALITY_THRESHOLDS['fair']:
        return "Fair"
    else:
        return "Poor"

@st.cache_data(ttl=3600, show_spinner=False)
def load_file_cached(file_content: bytes, file_name: str) -> Optional[pd.DataFrame]:
    """
    Load data from file content with caching and robust error handling.
    Supports CSV, Excel, JSON, Parquet with international format detection.
    """
    try:
        file_buffer = io.BytesIO(file_content)
        df = None
        
        if file_name.endswith('.csv') or file_name.endswith('.txt'):
            # Try multiple encodings and separators
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            separators = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        file_buffer.seek(0)
                        df = pd.read_csv(file_buffer, sep=sep, encoding=encoding, low_memory=False)
                        
                        # Heuristic: If we read a file with 1 column but it should have more, the separator is likely wrong.
                        # Unless it's really a single column file.
                        if df.shape[1] > 1:
                            break
                        else:
                            # If only 1 column, maybe retry unless it's the last separator
                             if sep == separators[-1]:
                                 break
                    except Exception:
                        continue
                if df is not None and df.shape[1] > 1:
                    break
            
            # Final fallback if loop finished but results are poor
            if df is None:
                 file_buffer.seek(0)
                 df = pd.read_csv(file_buffer, engine='python')

        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_buffer)
        elif file_name.endswith('.json'):
            df = pd.read_json(file_buffer)
        elif file_name.endswith('.parquet'):
            df = pd.read_parquet(file_buffer)
        else:
            return None
            
        if df is not None:
            # 1. Clean Column Names
            df.columns = df.columns.astype(str).str.strip().str.replace(r'\s+', '_', regex=True).str.replace(r'[^\w\s]', '', regex=True)
            
            # 2. Robust Type Inference
            # Attempt to convert object columns to numeric (handling European decimals like '1.000,00')
            # This is complex to do universally without risking data loss, so we stick to safe conversions.
            
            # Convert 'object' that look like numbers
            for col in df.select_dtypes(include=['object']):
                try:
                    # Check if it looks numeric but maybe with commas
                    if df[col].astype(str).str.match(r'^-?[\d,.]+$').all():
                         # Try standard conversion first
                         df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
            
            # 3. Optimize Types (downcast)
            # df = df.convert_dtypes() # Can be slow on large data, skipping for responsiveness

            return df
            
        return None
        
    except Exception as e:
        # st.error(f"Debug Load Error: {str(e)}") # Optional debug
        return None



@st.cache_data(show_spinner=False)
def generate_sample_data(data_type: str = 'sales') -> pd.DataFrame:
    """
    Generate sample data for demonstration
    
    Args:
        data_type: Type of sample data to generate
    
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    if data_type == 'sales':
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        data = {
            'Date': dates,
            'Sales': np.random.normal(1000, 200, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 100,
            'Customers': np.random.poisson(50, 365),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 365),
            'Product': np.random.choice(['Product A', 'Product B', 'Product C'], 365),
            'Revenue': np.random.normal(5000, 1000, 365),
            'Profit_Margin': np.random.uniform(0.1, 0.3, 365)
        }
    
    elif data_type == 'customer':
        data = {
            'Customer_ID': range(1, 1001),
            'Age': np.random.normal(35, 12, 1000).astype(int),
            'Income': np.random.normal(50000, 15000, 1000),
            'Spending_Score': np.random.randint(1, 101, 1000),
            'Gender': np.random.choice(['Male', 'Female'], 1000),
            'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 1000),
            'Satisfaction': np.random.randint(1, 6, 1000)
        }
    
    elif data_type == 'financial':
        dates = pd.date_range(start='2024-01-01', periods=500)
        df = pd.DataFrame({
            'Date': dates,
            'Ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], 500),
            'Volume': np.random.randint(100000, 1000000, 500),
            'Sector': 'Technology'
        })
        # Simulate price movement
        df['Close_Price'] = np.random.normal(150, 30, 500).cumsum().round(2)
        df['Close_Price'] = df['Close_Price'].abs() + 100
        return df

    elif data_type == 'healthcare':
        dates = pd.date_range(start='2024-01-01', periods=300)
        df = pd.DataFrame({
            'Admission_Date': dates,
            'Patient_Age': np.random.randint(18, 90, 300),
            'Department': np.random.choice(['Cardiology', 'Neurology', 'Orthopedics', 'General'], 300),
            'Length_of_Stay': np.random.gamma(2, 2, 300),
            'Bill_Amount': np.random.normal(5000, 2000, 300).round(2),
            'Outcome': np.random.choice(['Discharged', 'Transferred', 'Readmitted'], 300, p=[0.8, 0.1, 0.1])
        })
        return df
    
    else:  # Default to simple numeric data
        data = {
            'X': np.random.normal(0, 1, 100),
            'Y': np.random.normal(0, 1, 100),
            'Z': np.random.normal(0, 1, 100),
            'Category': np.random.choice(['A', 'B', 'C'], 100)
        }
    
    return pd.DataFrame(data)


@st.cache_data(show_spinner=False)
def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get memory usage information for DataFrame
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with memory usage info
    """
    if df.empty:
        return {'total': '0 B', 'per_column': {}}
    
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"
    
    return {
        'total': format_bytes(total_memory),
        'per_column': {col: format_bytes(usage) for col, usage in memory_usage.items()}
    }


@st.cache_data(show_spinner=False)
def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive data summary
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        Dictionary with summary information
    """
    if df.empty:
        return {}
    
    col_info = get_column_info(df)
    quality_metrics = calculate_data_quality_metrics(df)
    memory_info = get_memory_usage(df)
    
    summary = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': memory_info['total'],
            'file_size_estimate': memory_info['total']
        },
        'column_info': col_info,
        'quality_metrics': quality_metrics,
        'data_types': {
            'numeric': len(col_info['numeric_columns']),
            'categorical': len(col_info['categorical_columns']),
            'datetime': len(col_info['datetime_columns']),
            'boolean': len(col_info['boolean_columns'])
        },
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'columns_with_missing': len([col for col in df.columns if df[col].isnull().any()]),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        },
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    }
    
    return summary