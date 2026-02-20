import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf

# Try importing Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Try importing pmdarima for AutoARIMA
try:
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False

warnings.filterwarnings("ignore")

def render_time_series_tab(df: pd.DataFrame):
    """
    Render Time Series Analysis & Forecasting Module
    """
    st.markdown('<div class="premium-card animate-enter">', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h2 style="font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0; color: var(--text-main);">Time Series Lab</h2>
            <p style="color: var(--text-secondary); margin-top: 0.5rem; font-family: 'Inter', sans-serif;">
                Analyze trends, seasonality, and forecast future values with advanced models.
            </p>
        </div>
        <div style="background: rgba(var(--accent-rgb), 0.1); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--accent);">
            <span style="color: var(--accent); font-weight: 600; font-size: 0.9rem;">Temporal Analytics</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # DATA SELECTION
    # -------------------------------------------------------------------------
    cols = df.columns.tolist()
    date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not date_cols:
        # Try to infer
        for c in cols:
            if "date" in c.lower() or "time" in c.lower() or "year" in c.lower():
                date_cols.append(c)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox("Date Column", date_cols if date_cols else cols, index=0)
    with c2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_col = st.selectbox("Value Column (Target)", numeric_cols, index=0)
    with c3:
        freq = st.selectbox("Frequency", ["D (Daily)", "W (Weekly)", "M (Monthly)", "Q (Quarterly)", "Y (Yearly)"], index=0)
        freq_code = freq.split(" ")[0]

    # Prepare Data
    try:
        ts_df = df[[date_col, value_col]].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
        ts_df = ts_df.dropna().sort_values(date_col).set_index(date_col)
        # Resample if needed to ensure regular frequency
        ts_df = ts_df.resample(freq_code).mean().interpolate()
        
        series = ts_df[value_col]
        
        # Plot Original Series
        fig = px.line(ts_df, y=value_col, title=f"Time Series: {value_col}", template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing time series data: {str(e)}")
        return

    # -------------------------------------------------------------------------
    # ANALYSIS TABS
    # -------------------------------------------------------------------------
    tab_decomp, tab_stat, tab_forecast = st.tabs(["🔍 Decomposition & ACF", "🧪 Stationarity Tests", "🔮 Forecasting"])

    with tab_decomp:
        st.subheader("Seasonal Decomposition")
        model_type = st.radio("Decomposition Type", ["additive", "multiplicative"], horizontal=True)
        
        try:
            decomp = seasonal_decompose(series, model=model_type, period=None) # Let statsmodels infer period or use freq
            
            fig_trend = px.line(decomp.trend, title="Trend Component", template="plotly_dark")
            fig_seas = px.line(decomp.seasonal, title="Seasonal Component", template="plotly_dark")
            fig_resid = px.scatter(decomp.resid, title="Residuals", template="plotly_dark")
            
            # Use smaller charts
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_trend, use_container_width=True)
            c2.plotly_chart(fig_seas, use_container_width=True)
            st.plotly_chart(fig_resid, use_container_width=True)
            
        except Exception as e:
            st.error(f"Decomposition failed: {str(e)}")

        st.subheader("Autocorrelation (ACF) & Partial Autocorrelation (PACF)")
        try:
            lag_acf = acf(series.dropna(), nlags=20)
            lag_pacf = pacf(series.dropna(), nlags=20)
            
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(x=list(range(len(lag_acf))), y=lag_acf, name='ACF'))
            fig_acf.update_layout(title="Autocorrelation Function (ACF)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            
            fig_pacf = go.Figure()
            fig_pacf.add_trace(go.Bar(x=list(range(len(lag_pacf))), y=lag_pacf, name='PACF'))
            fig_pacf.update_layout(title="Partial Autocorrelation Function (PACF)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_acf, use_container_width=True)
            c2.plotly_chart(fig_pacf, use_container_width=True)
        except Exception as e:
            st.warning("Could not calculate ACF/PACF")

    with tab_stat:
        st.subheader("Augmented Dickey-Fuller Test")
        st.markdown("Tests whether a time series is stationary (no trend/seasonality over time).")
        
        result = adfuller(series.dropna())
        statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        c1, c2 = st.columns(2)
        c1.metric("ADF Statistic", f"{statistic:.4f}")
        c2.metric("P-Value", f"{p_value:.4f}", delta="Stationary" if p_value < 0.05 else "Non-Stationary", delta_color="inverse")
        
        st.write("Critical Values:")
        st.json(critical_values)
        
        if p_value > 0.05:
            st.info("The series is likely non-stationary. Consider differencing (d=1) for ARIMA models.")
        else:
            st.success("The series is likely stationary.")

    with tab_forecast:
        st.subheader("Forecast Future Values")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            model_choice = st.selectbox("Model", ["ARIMA", "Exponential Smoothing", "Prophet"])
            horizon = st.slider("Forecast Horizon (periods)", 1, 30, 12)
            
            model_params = {}
            if model_choice == "ARIMA":
                p = st.number_input("p (AR)", 0, 10, 1)
                d = st.number_input("d (Diff)", 0, 2, 1)
                q = st.number_input("q (MA)", 0, 10, 1)
                model_params = {'order': (p, d, q)}
            elif model_choice == "Exponential Smoothing":
                trend = st.selectbox("Trend", ["add", "mul", None], index=0)
                seasonal = st.selectbox("Seasonal", ["add", "mul", None], index=0)
                model_params = {'trend': trend, 'seasonal': seasonal}
            elif model_choice == "Prophet":
                if not PROPHET_AVAILABLE:
                    st.error("Prophet is not installed.")
                else:
                    st.caption("Prophet automatically handles trend and seasonality.")
            
            train_btn = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)

        with c2:
            if train_btn:
                with st.spinner(f"Training {model_choice}..."):
                    try:
                        forecast_df = None
                        conf_int = None
                        
                        # Train/Test Split logic (optional, here we train on full data for future forecast)
                        
                        model_fit = None
                        
                        if model_choice == "ARIMA":
                            model = ARIMA(series, order=model_params['order'])
                            model_fit = model.fit()
                            forecast_res = model_fit.get_forecast(steps=horizon)
                            forecast_values = forecast_res.predicted_mean
                            conf_int = forecast_res.conf_int()
                            
                            # Create DF
                            last_date = series.index[-1]
                            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq_code)[1:]
                            forecast_df = pd.DataFrame({'Forecast': forecast_values.values}, index=future_dates)
                            
                        elif model_choice == "Exponential Smoothing":
                            model = ExponentialSmoothing(series, trend=model_params['trend'], seasonal=model_params['seasonal'])
                            model_fit = model.fit()
                            forecast_values = model_fit.forecast(horizon)
                            
                            last_date = series.index[-1]
                            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq_code)[1:]
                            forecast_df = pd.DataFrame({'Forecast': forecast_values.values}, index=future_dates)
                            
                        elif model_choice == "Prophet" and PROPHET_AVAILABLE:
                            df_prophet = ts_df.reset_index().rename(columns={date_col: 'ds', value_col: 'y'})
                            m = Prophet()
                            m.fit(df_prophet)
                            future = m.make_future_dataframe(periods=horizon, freq=freq_code)
                            forecast = m.predict(future)
                            
                            forecast_df = forecast[['ds', 'yhat']].tail(horizon).set_index('ds')
                            forecast_df.columns = ['Forecast']
                            conf_int = forecast[['ds', 'yhat_lower', 'yhat_upper']].tail(horizon).set_index('ds')

                        # Plotting
                        fig_fc = go.Figure()
                        
                        # Historical
                        fig_fc.add_trace(go.Scatter(x=series.index, y=series.values, name='Historical', line=dict(color='white')))
                        
                        # Forecast
                        fig_fc.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], name='Forecast', line=dict(color='#10B981', width=3)))
                        
                        # Confidence Intervals
                        if conf_int is not None:
                            if model_choice == "ARIMA":
                                lower = conf_int.iloc[:, 0]
                                upper = conf_int.iloc[:, 1]
                            elif model_choice == "Prophet":
                                lower = conf_int['yhat_lower']
                                upper = conf_int['yhat_upper']
                            
                            fig_fc.add_trace(go.Scatter(
                                x=forecast_df.index, y=upper,
                                mode='lines', line=dict(width=0),
                                showlegend=False
                            ))
                            fig_fc.add_trace(go.Scatter(
                                x=forecast_df.index, y=lower,
                                mode='lines', line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(16, 185, 129, 0.2)',
                                name='95% Confidence'
                            ))
                            
                        fig_fc.update_layout(title=f"{model_choice} Forecast ({horizon} periods)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_fc, use_container_width=True)
                        
                        # Metrics (In-sample if possible, or just show model summary)
                        if model_choice == "ARIMA":
                            st.text("Model Summary:")
                            st.text(model_fit.summary().tables[1])

                    except Exception as e:
                        st.error(f"Forecasting failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
