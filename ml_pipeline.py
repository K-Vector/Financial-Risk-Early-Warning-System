import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import shap

DATA_PATH = 'data/macro_financial_data.csv'
MODEL_PATH = 'model/xgb_model.joblib'
METRICS_PATH = 'model/metrics.txt'
SHAP_PLOT_PATH = 'model/shap_summary.png'

os.makedirs('model', exist_ok=True)

def feature_engineering(df):
    """Build features from raw macro indicators.
    
    Creates rolling stats, lags, and normalized indicators.
    Standard approach for time-series financial data.
    """
    base_features = ['CPI', 'UNRATE', 'FEDFUNDS', 'T10Y2Y', 'VIX', 'SP500']
    
    # Rolling means and volatility: 1mo, 3mo, 1yr windows
    for feature in base_features:
        for window in [4, 12, 52]:
            df[f'{feature}_MA_{window}W'] = df[feature].rolling(window=window).mean()
            df[f'{feature}_VOL_{window}W'] = df[feature].rolling(window=window).std()
    
    # Lag features: 1wk, 1mo, 3mo
    for feature in base_features + ['HY_SPREAD']:
        for lag in [1, 4, 12]:
            df[f'{feature}_LAG_{lag}W'] = df[feature].shift(lag)
    
    # VIX normalized to its 1yr rolling mean (captures relative stress)
    df['VIX_NORM'] = df['VIX'] / df['VIX'].rolling(window=52).mean()
    
    df_clean = df.dropna()
    print(f"Features created: {df_clean.shape[1]} features, {df_clean.shape[0]} samples")
    
    return df_clean

def explain_model(model, X_test):
    """Generate SHAP explanations for model interpretability."""
    print("\nComputing SHAP values...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle binary classification output format
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    
    feature_importance = pd.Series(
        np.abs(shap_values).mean(axis=0), 
        index=X_test.columns
    ).sort_values(ascending=False)
    
    print("Top 5 features by SHAP importance:")
    for feat, imp in feature_importance.head().items():
        print(f"  {feat}: {imp:.4f}")
    
    return feature_importance

def train_and_evaluate_model(df):
    """Train XGBoost classifier and evaluate on held-out test set."""
    X = df.drop(columns=['HY_SPREAD', 'risk_event'])
    y = df['risk_event']
    
    # Time-based split (80/20) - critical for time series
    split_point = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"\nTrain: {len(X_train)} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
    print(f"Test:  {len(X_test)} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")
    
    model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Handle edge case: single class in predictions
    try:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = np.nan
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    
    with open(METRICS_PATH, 'w') as f:
        f.write("Financial Risk EWS Model Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: XGBoost Classifier\n")
        f.write(f"Training: {X_train.index.min().date()} to {X_train.index.max().date()}\n")
        f.write(f"Testing:  {X_test.index.min().date()} to {X_test.index.max().date()}\n\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))
    
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    
    return model, X_test, y_test

def main():
    try:
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Run data_ingestion.py first.")
        return
    
    df_features = feature_engineering(df)
    model, X_test, y_test = train_and_evaluate_model(df_features)
    explain_model(model, X_test)

if __name__ == "__main__":
    main()
