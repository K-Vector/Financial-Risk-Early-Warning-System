# Financial Risk Early-Warning System

A machine learning system that predicts periods of elevated financial stress using macro-financial indicators.

## Overview

This project uses XGBoost to classify periods of financial stress based on macroeconomic indicators (CPI, unemployment, Fed Funds Rate, yield curve, VIX, S&P 500). The model outputs a binary risk classification and provides SHAP-based explanations.

## Project Structure

```
.
├── app.py                 # Streamlit dashboard
├── data_ingestion.py      # Fetch and process FRED data
├── ml_pipeline.py         # Train model and generate SHAP plots
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment config
├── data/                  # Processed data (CSV)
└── model/                 # Trained model and artifacts
    ├── xgb_model.joblib
    ├── metrics.txt
    └── shap_summary.png
```

## Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set FRED API key (optional):**
   ```bash
   export FRED_API_KEY='your_key_here'
   ```
   Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

3. **Generate data:**
   ```bash
   python data_ingestion.py
   ```

4. **Train model:**
   ```bash
   python ml_pipeline.py
   ```

5. **Run dashboard:**
   ```bash
   streamlit run app.py
   ```

## Deployment on Render

### Prerequisites

- GitHub repository with this code
- FRED API key (optional, will use synthetic data if not provided)

### Steps

1. **Push code to GitHub** (ensure `data/` and `model/` directories exist with required files)

2. **Create new Web Service on Render:**
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml` configuration
   - Or manually configure:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
     - **Environment:** Python 3

3. **Set Environment Variables:**
   - `FRED_API_KEY` (optional): Your FRED API key for real data

4. **Deploy**

### Important Notes

- **Data and Model Files:** Ensure `data/macro_financial_data.csv` and `model/xgb_model.joblib` are committed to your repository, or they will be generated on first run (which may take time).
- **Build Time:** First deployment may take 5-10 minutes to install dependencies (especially XGBoost and SHAP).
- **Memory:** XGBoost and SHAP can be memory-intensive. Consider upgrading to a larger instance if you encounter issues.

## Model Details

- **Algorithm:** XGBoost Classifier
- **Target:** Binary classification (Risk Event vs Normal)
- **Features:** Rolling statistics, lag features, and normalized indicators
- **Evaluation:** Precision, Recall, ROC-AUC
- **Interpretability:** SHAP values for feature importance

See `model_card.md` for detailed model documentation.

## License

MIT
