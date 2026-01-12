# Model Card: Financial Risk Early-Warning System

**Version:** 1.0  
**Date:** December 1, 2025

## Purpose

This model predicts periods of elevated financial stress in the macro-economy using standard macro-financial indicators. It's designed as a pre-emptive signal for risk managers, allowing them to adjust exposure and hedging strategies before conditions deteriorate.

The model outputs a binary classification: **Risk Event (1)** or **Normal Market Condition (0)**, based on whether the High Yield Spread exceeds its 75th percentile threshold.

## Training Data

**Target Variable:** Binary indicator (1 = Risk Event, 0 = Normal) based on ICE BofA US High Yield Spread (HY_SPREAD) exceeding its 75th percentile. Source: FRED series BAMLH0A0HYM2.

**Input Features:** CPI, Unemployment Rate, Fed Funds Rate, 10Y-2Y Yield Curve Spread, VIX, S&P 500. All sourced from FRED and resampled to weekly frequency (Friday close).

**Feature Engineering:** 
- Rolling means and volatility: 4, 12, 52 week windows
- Lag features: 1, 4, 12 week lags
- VIX normalized to its 1-year rolling mean

**Data Period:** 2000-01-01 to present (or synthetic data if FRED API key unavailable)

**Train/Test Split:** 80/20 time-based split (no random shuffling)

## Assumptions

1. **Stable relationships:** The relationship between macro indicators and financial stress remains consistent over time (reasonable for US markets post-2000).
2. **Predictive lags:** Past values of economic indicators contain information about future stress periods.
3. **HY Spread as proxy:** High Yield Spread is a reasonable proxy for systemic financial stress (widely used in practice).
4. **Data quality:** FRED data is reliable. Synthetic data is for development only and not suitable for production.

## Limitations & Risks

**Look-Ahead Bias:** Rolling features use data up to the current time step. We use lags to mitigate this, but care is needed in production to ensure no future data leaks into predictions.

**Model Drift:** Trained on historical data, so it may miss novel crisis patterns (e.g., COVID-19 style shocks). Monitor performance metrics and retrain quarterly.

**Class Imbalance:** Financial crises are rare (~25% of periods), which can bias predictions toward the normal class. We use ROC-AUC and recall metrics that handle imbalance better than accuracy.

**Synthetic Data:** If no FRED API key is available, the model uses synthetic data. This is fine for development/testing but **not suitable for production**. Always use real FRED data in production.

## Ethical & Security Considerations

**Geographic Bias:** Model is trained on US data only. Don't apply to other economies without retraining and validation.

**Security:** Model artifacts and training data are sensitive. In production, store in encrypted storage with RBAC. Never commit API keys or model files to version control.

**Compliance:** For regulated environments (banks, etc.), ensure audit trails, data lineage tracking, and proper documentation. This model card is a start but may need additional compliance documentation.

## Versioning

- **Model:** `xgb_model.joblib` (v1.0) - XGBoost Classifier
- **Features:** `ml_pipeline.py` (v1.0) - Feature engineering pipeline
- **Data:** `macro_financial_data.csv` (v1.0) - FRED data, weekly frequency
- **Code:** Python 3.x, pandas, xgboost, streamlit

## Interpretability

We use **SHAP values** to explain predictions. The SHAP summary plot (`model/shap_summary.png`) shows which features drive risk predictions.

**Key Insights (from synthetic data - validate with real data):**

Top drivers tend to be:
- CPI level (inflation pressure)
- VIX 52-week volatility (sustained market fear)
- Fed Funds Rate volatility (policy uncertainty)

This makes economic sense: high inflation combined with sustained volatility in fear and policy rates typically precedes stress periods. The model isn't a black box - it's learning interpretable relationships.

---

*Update this card when retraining or changing the pipeline.*
