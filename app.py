import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Render
import matplotlib.pyplot as plt
import shap
import os

DATA_PATH = 'data/macro_financial_data.csv'
MODEL_PATH = 'model/xgb_model.joblib'
SHAP_PLOT_PATH = 'model/shap_summary.png'

@st.cache_data
def load_data():
    """Load processed macro-financial time series."""
    try:
        return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Run data_ingestion.py first.")
        return None

@st.cache_resource
def load_model():
    """Load trained XGBoost model."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model not found at {MODEL_PATH}. Run ml_pipeline.py first.")
        return None

def get_risk_level(probability):
    """Map probability to risk category and color."""
    if probability >= 0.75:
        return "HIGH RISK", "red"
    elif probability >= 0.5:
        return "MEDIUM RISK", "orange"
    return "LOW RISK", "green"

def feature_engineer_single_point(df, latest_data):
    """
    Generate features for a single prediction point.
    
    In production, this would hit a feature store. For now, we replicate
    the training pipeline logic on-the-fly.
    """
    df_temp = pd.concat([df.drop(columns=['risk_event']), latest_data.to_frame().T])
    
    base_features = ['CPI', 'UNRATE', 'FEDFUNDS', 'T10Y2Y', 'VIX', 'SP500']
    
    # Rolling stats: 1mo, 3mo, 1yr windows
    for feature in base_features:
        for window in [4, 12, 52]:
            df_temp[f'{feature}_MA_{window}W'] = df_temp[feature].rolling(window=window).mean()
            df_temp[f'{feature}_VOL_{window}W'] = df_temp[feature].rolling(window=window).std()
    
    # Lag features: 1wk, 1mo, 3mo
    for feature in base_features + ['HY_SPREAD']:
        for lag in [1, 4, 12]:
            df_temp[f'{feature}_LAG_{lag}W'] = df_temp[feature].shift(lag)
    
    # VIX normalized to its 1yr rolling mean
    df_temp['VIX_NORM'] = df_temp['VIX'] / df_temp['VIX'].rolling(window=52).mean()
    
    feature_vector = df_temp.iloc[-1].drop(labels=['HY_SPREAD'])
    
    # Edge case: insufficient history for rolling features
    if feature_vector.isnull().any():
        st.warning("Insufficient historical data for full feature set. Using last valid point.")
        return df_temp.dropna().drop(columns=['HY_SPREAD']).iloc[-1]
    
    return feature_vector

def main():
    st.set_page_config(layout="wide", page_title="Financial Risk EWS Dashboard")
    st.title("Financial Risk Early-Warning System")
    st.markdown("Predicting periods of elevated financial stress using macro-financial indicators")

    df = load_data()
    model = load_model()

    if df is None or model is None:
        st.stop()

    st.sidebar.header("Controls")
    
    latest_date = df.index[-1]
    selected_date = st.sidebar.date_input(
        "Prediction Date", 
        latest_date, 
        min_value=df.index[0].date(), 
        max_value=latest_date.date()
    )
    
    selected_date_dt = pd.to_datetime(selected_date)
    
    # Handle non-trading days
    if selected_date_dt not in df.index:
        closest_date = df.index[df.index.get_loc(selected_date_dt, method='nearest')]
        st.sidebar.info(f"Using closest available date: **{closest_date.strftime('%Y-%m-%d')}**")
        selected_date_dt = closest_date

    st.header("Model Prediction")
    
    latest_data = df.loc[selected_date_dt]
    feature_vector = feature_engineer_single_point(df, latest_data)
    
    X_pred = feature_vector.to_frame().T
    y_proba = model.predict_proba(X_pred)[0][1]
    risk_label, risk_color = get_risk_level(y_proba)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Date", selected_date_dt.strftime('%Y-%m-%d'))
    with col2:
        st.metric("Risk Probability", f"{y_proba:.2%}")
    with col3:
        st.markdown("### Risk Level")
        st.markdown(f"<h1 style='color:{risk_color};'>{risk_label}</h1>", unsafe_allow_html=True)

    st.header("Economic Indicators")
    
    indicator_options = ['HY_SPREAD', 'CPI', 'UNRATE', 'FEDFUNDS', 'T10Y2Y', 'VIX', 'SP500']
    selected_indicator = st.selectbox("Select Indicator", indicator_options)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[selected_indicator], label=selected_indicator, linewidth=1.5)
    ax.axvline(selected_date_dt, color='red', linestyle='--', alpha=0.7, label='Prediction Date')
    ax.set_title(f'{selected_indicator} Over Time')
    ax.set_xlabel("Date")
    ax.set_ylabel(selected_indicator)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.header("Model Explainability")
    st.markdown("SHAP values show feature contributions to the risk prediction.")
    
    if os.path.exists(SHAP_PLOT_PATH):
        st.image(SHAP_PLOT_PATH, caption="SHAP Feature Importance")
    else:
        st.warning("SHAP plot not found. Run ml_pipeline.py to generate.")
        
    st.header("Recent Data")
    st.dataframe(df.tail(10))

if __name__ == "__main__":
    main()
