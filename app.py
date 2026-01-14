import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

# --- FIX: Import the feature_engineering function directly ---
from ml_pipeline import feature_engineering 

# --- Configuration ---
DATA_PATH = 'data/macro_financial_data.csv'
MODEL_PATH = 'model/xgb_model.joblib'
SHAP_PLOT_PATH = 'model/shap_summary.png'

# --- Helper Functions ---

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Please run data_ingestion.py first.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please run ml_pipeline.py first.")
        return None

def get_risk_level(probability):
    if probability >= 0.75:
        return "HIGH RISK", "red"
    elif probability >= 0.5:
        return "MEDIUM RISK", "orange"
    else:
        return "LOW RISK", "green"

def feature_engineer_single_point(df, latest_data_series):
    # 1. Create a temporary DataFrame with the historical data and the new point
    df_historical = df.drop(columns=['risk_event'], errors='ignore')
    latest_data_df = latest_data_series.to_frame().T
    df_temp = pd.concat([df_historical, latest_data_df])
    
    # 2. Apply the feature engineering function
    df_featured = feature_engineering(df_temp.copy())
    
    # 3. Extract the feature vector for the latest point
    feature_vector = df_featured.iloc[-1]
    
    # 4. Clean up the feature vector
    feature_vector = feature_vector.drop(labels=['HY_SPREAD', 'risk_event'], errors='ignore')
    
    return feature_vector

# --- Streamlit App ---

def main():
    st.set_page_config(layout="wide", page_title="Financial Risk EWS Dashboard")
    st.title("Financial Risk Early-Warning System (EWS)")
    st.markdown("A demonstration of an end-to-end ML pipeline for financial risk prediction.")

    df = load_data()
    model = load_model()

    if df is None or model is None:
        st.stop()

    # --- Sidebar for Navigation ---
    st.sidebar.header("Dashboard Controls")
    latest_date = df.index[-1]
    selected_date = st.sidebar.date_input("Select Date for Prediction", latest_date, min_value=df.index[0].date(), max_value=latest_date.date())
    selected_date_dt = pd.to_datetime(selected_date)
    
    if selected_date_dt not in df.index:
        selected_date_dt = df.index[df.index.get_loc(selected_date_dt, method='nearest')]
        st.sidebar.info(f"Using closest available data point: **{selected_date_dt.strftime('%Y-%m-%d')}**")

    # --- Prediction Section ---
    st.header("1. Model Prediction")
    latest_data_series = df.loc[selected_date_dt]
    feature_vector = feature_engineer_single_point(df, latest_data_series)
    
    # --- FIX: Ensure X_pred is a DataFrame with correct feature names ---
    X_pred = pd.DataFrame([feature_vector])
    
    # Align columns with what the model expects
    expected_features = model.get_booster().feature_names
    X_pred = X_pred.reindex(columns=expected_features, fill_value=0)
    
    # Make prediction
    y_proba = model.predict_proba(X_pred)[0][1]
    risk_label, risk_color = get_risk_level(y_proba)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction Date", selected_date_dt.strftime('%Y-%m-%d'))
    with col2:
        st.metric("Risk Probability (P=1)", f"{y_proba:.2%}")
    with col3:
        st.markdown(f"### Risk Level")
        st.markdown(f"<h1 style='color:{risk_color};'>{risk_label}</h1>", unsafe_allow_html=True)

    # --- Economic Indicators Over Time ---
    st.header("2. Economic Indicators Over Time")
    indicator_options = ['HY_SPREAD', 'CPI', 'UNRATE', 'FEDFUNDS', 'T10Y2Y', 'VIX', 'SP500']
    selected_indicator = st.selectbox("Select Indicator to Visualize", indicator_options)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[selected_indicator], label=selected_indicator)
    ax.axvline(selected_date_dt, color='red', linestyle='--', label='Prediction Date')
    ax.set_title(f'{selected_indicator} Over Time')
    ax.set_xlabel("Date")
    ax.set_ylabel(selected_indicator)
    ax.legend()
    st.pyplot(fig)

    # --- Model Explainability ---
    st.header("3. Model Explainability (SHAP)")
    if os.path.exists(SHAP_PLOT_PATH):
        st.image(SHAP_PLOT_PATH, caption="SHAP Feature Importance Summary Plot")
    else:
        st.warning("SHAP plot not found.")
        
    # --- Data Table ---
    st.header("4. Raw Data Snapshot")
    st.dataframe(df.tail(10))

if __name__ == "__main__":
    main()
