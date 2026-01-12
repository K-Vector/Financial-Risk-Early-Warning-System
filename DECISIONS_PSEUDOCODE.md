# Decision Pseudocode: Financial Risk Early-Warning System

This document outlines the key architectural and implementation decisions made in this project, along with the reasoning behind each choice.

## 1. SYSTEM ARCHITECTURE DECISIONS

```
DECISION: Three-stage pipeline (Data → Training → Serving)
WHY:
    - Separation of concerns: Each stage has distinct responsibilities
    - Enables independent testing and debugging
    - Allows for different deployment strategies (batch training vs real-time serving)
    - Standard MLOps pattern that scales well

ALTERNATIVE CONSIDERED: Monolithic script
REJECTED BECAUSE: Harder to maintain, test, and deploy incrementally
```

```
DECISION: Use Streamlit for dashboard
WHY:
    - Rapid prototyping and deployment
    - Built-in caching (@st.cache_data, @st.cache_resource) reduces computation
    - No frontend framework knowledge required
    - Good for internal tools and demos

ALTERNATIVE CONSIDERED: Flask/FastAPI + React
REJECTED BECAUSE: Overkill for this use case, adds complexity and deployment overhead
```

```
DECISION: Store data/model files in repository
WHY:
    - Simplifies deployment (no need for external storage)
    - Ensures reproducibility (same data/model across environments)
    - Faster cold starts (no download step)
    - Files are relatively small (< 50MB total)

ALTERNATIVE CONSIDERED: Cloud storage (S3, GCS)
REJECTED BECAUSE: Adds complexity, requires credentials, slower initial load
```

## 2. DATA PIPELINE DECISIONS

```
DECISION: Use FRED API as primary data source
WHY:
    - Authoritative source for US macroeconomic data
    - Free API with reasonable rate limits
    - Standard in financial analysis
    - Comprehensive historical coverage (2000+)

ALTERNATIVE CONSIDERED: Yahoo Finance, Alpha Vantage
REJECTED BECAUSE: Less reliable, inconsistent data quality, rate limits
```

```
DECISION: Resample all series to weekly frequency (W-FRI)
WHY:
    - Aligns different frequencies (daily VIX, monthly CPI, etc.)
    - Weekly is standard for macro analysis
    - Reduces noise while preserving signal
    - Friday close captures end-of-week sentiment

ALTERNATIVE CONSIDERED: Daily frequency
REJECTED BECAUSE: Too noisy, some series not available daily
ALTERNATIVE CONSIDERED: Monthly frequency
REJECTED BECAUSE: Too coarse, loses important short-term dynamics
```

```
DECISION: Use HY_SPREAD 75th percentile as risk threshold
WHY:
    - High yield spread is a proven stress indicator
    - 75th percentile captures ~25% of periods as "stress" (reasonable imbalance)
    - Relative threshold adapts to changing market conditions
    - Standard approach in financial risk literature

ALTERNATIVE CONSIDERED: Fixed threshold (e.g., 500 bps)
REJECTED BECAUSE: Doesn't adapt to market regime changes
ALTERNATIVE CONSIDERED: 90th percentile
REJECTED BECAUSE: Too rare, severe class imbalance
```

```
DECISION: Generate synthetic data when FRED_API_KEY missing
WHY:
    - Allows development/testing without API access
    - Enables quick demos and prototyping
    - Maintains pipeline structure even without real data
    - Clearly documented as not suitable for production

ALTERNATIVE CONSIDERED: Fail fast if no API key
REJECTED BECAUSE: Blocks development workflow, harder to onboard new developers
```

## 3. FEATURE ENGINEERING DECISIONS

```
DECISION: Create rolling statistics (mean, volatility) for multiple windows
WHY:
    - Captures trends and volatility at different time horizons
    - 4-week: Short-term dynamics
    - 12-week: Medium-term trends (quarterly)
    - 52-week: Long-term context (annual)
    - Standard approach in time-series financial modeling

ALTERNATIVE CONSIDERED: Single window size
REJECTED BECAUSE: Loses multi-scale information
ALTERNATIVE CONSIDERED: More windows (8, 16, 24, etc.)
REJECTED BECAUSE: Diminishing returns, increases feature space unnecessarily
```

```
DECISION: Include lag features (1, 4, 12 weeks)
WHY:
    - Captures temporal dependencies
    - 1-week: Immediate past
    - 4-week: Monthly patterns
    - 12-week: Quarterly cycles
    - Critical for time-series prediction

ALTERNATIVE CONSIDERED: No lags
REJECTED BECAUSE: Model would only use current values, missing temporal patterns
ALTERNATIVE CONSIDERED: More lags (up to 26 weeks)
REJECTED BECAUSE: Diminishing returns, risk of overfitting
```

```
DECISION: Normalize VIX by its rolling mean
WHY:
    - VIX absolute level varies by market regime
    - Normalized VIX captures "relative fear" vs historical context
    - More stable feature than raw VIX
    - Standard normalization technique

ALTERNATIVE CONSIDERED: Use raw VIX
REJECTED BECAUSE: Less interpretable, harder for model to learn relative patterns
ALTERNATIVE CONSIDERED: Z-score normalization
REJECTED BECAUSE: Rolling mean normalization is more intuitive for this use case
```

```
DECISION: Drop rows with NaN after feature engineering
WHY:
    - Rolling/lag features create NaNs at start of series
    - XGBoost can't handle NaN values directly
    - Better to drop than impute (preserves data integrity)
    - Loss of initial ~52 weeks is acceptable given long history

ALTERNATIVE CONSIDERED: Forward fill or interpolation
REJECTED BECAUSE: Introduces artificial patterns, especially for lag features
ALTERNATIVE CONSIDERED: Keep NaNs and let XGBoost handle
REJECTED BECAUSE: XGBoost doesn't natively handle NaNs well
```

## 4. MODEL SELECTION DECISIONS

```
DECISION: Use XGBoost Classifier
WHY:
    - Handles non-linear relationships well
    - Built-in feature importance
    - Robust to outliers and missing values (after preprocessing)
    - Fast training and prediction
    - Industry standard for tabular financial data
    - Good performance on imbalanced datasets

ALTERNATIVE CONSIDERED: Random Forest
REJECTED BECAUSE: Generally slower, less performant than XGBoost
ALTERNATIVE CONSIDERED: Neural Networks
REJECTED BECAUSE: Overkill for tabular data, harder to interpret, requires more data
ALTERNATIVE CONSIDERED: Logistic Regression
REJECTED BECAUSE: Too simple, can't capture non-linear relationships
```

```
DECISION: Use time-based train/test split (80/20)
WHY:
    - Preserves temporal order (critical for time series)
    - Simulates real-world deployment (train on past, predict future)
    - Prevents data leakage from future to past
    - Standard practice for time-series ML

ALTERNATIVE CONSIDERED: Random split
REJECTED BECAUSE: Violates temporal structure, causes data leakage
ALTERNATIVE CONSIDERED: Walk-forward validation
REJECTED BECAUSE: More complex, 80/20 split sufficient for initial model
```

```
DECISION: Use binary:logistic objective with logloss
WHY:
    - Standard for binary classification
    - Logloss is proper scoring rule (penalizes confident wrong predictions)
    - Works well with probability outputs needed for risk assessment

ALTERNATIVE CONSIDERED: Other objectives
REJECTED BECAUSE: binary:logistic is standard and well-tested
```

```
DECISION: Use default XGBoost hyperparameters
WHY:
    - Defaults are well-tuned for general use cases
    - Avoids overfitting from excessive tuning
    - Faster iteration cycle
    - Good baseline performance

ALTERNATIVE CONSIDERED: Hyperparameter tuning (GridSearch, Optuna)
REJECTED BECAUSE: Risk of overfitting to validation set, adds complexity
FUTURE IMPROVEMENT: Add hyperparameter tuning with proper time-series CV
```

## 5. EVALUATION METRICS DECISIONS

```
DECISION: Use Precision, Recall, ROC-AUC
WHY:
    - Precision: Important when false positives are costly (unnecessary risk actions)
    - Recall: Important when false negatives are costly (missing crises)
    - ROC-AUC: Handles class imbalance well, threshold-independent
    - Standard metrics for binary classification

ALTERNATIVE CONSIDERED: Accuracy only
REJECTED BECAUSE: Misleading with imbalanced data (would be high even if predicting all zeros)
ALTERNATIVE CONSIDERED: F1-score only
REJECTED BECAUSE: Single metric doesn't capture full picture
```

```
DECISION: Use SHAP for model interpretability
WHY:
    - Provides feature-level explanations
    - Model-agnostic (though we use TreeExplainer for efficiency)
    - Standard in ML interpretability
    - Helps validate model learns sensible relationships

ALTERNATIVE CONSIDERED: Feature importance only
REJECTED BECAUSE: Less informative, doesn't show direction of impact
ALTERNATIVE CONSIDERED: LIME
REJECTED BECAUSE: SHAP has better theoretical foundation (Shapley values)
```

## 6. DEPLOYMENT DECISIONS

```
DECISION: Deploy on Render
WHY:
    - Simple deployment process
    - Good free tier for demos
    - Automatic HTTPS
    - Easy environment variable management
    - Good for Streamlit apps

ALTERNATIVE CONSIDERED: AWS/GCP/Azure
REJECTED BECAUSE: More complex setup, overkill for this project
ALTERNATIVE CONSIDERED: Heroku
REJECTED BECAUSE: More expensive, less modern platform
```

```
DECISION: Use matplotlib 'Agg' backend
WHY:
    - Required for headless servers (no display)
    - Render and most cloud platforms don't have GUI
    - Non-interactive backend is standard for production

ALTERNATIVE CONSIDERED: Default backend
REJECTED BECAUSE: Would fail on headless servers
```

```
DECISION: Use relative file paths
WHY:
    - Works across different environments
    - No hardcoded paths that break on different machines
    - Standard practice for portable code

ALTERNATIVE CONSIDERED: Absolute paths
REJECTED BECAUSE: Breaks when deployed to different environments
```

```
DECISION: Cache data and model loading in Streamlit
WHY:
    - Data/model files don't change during session
    - Avoids reloading on every interaction
    - Significantly improves performance
    - Reduces memory usage

ALTERNATIVE CONSIDERED: Load on every request
REJECTED BECAUSE: Slow, inefficient, poor user experience
```

## 7. ERROR HANDLING DECISIONS

```
DECISION: Graceful degradation when files missing
WHY:
    - Better UX than crashing
    - Shows helpful error messages
    - Allows partial functionality if possible
    - Standard practice for production apps

ALTERNATIVE CONSIDERED: Fail fast with exceptions
REJECTED BECAUSE: Poor user experience, harder to debug
```

```
DECISION: Handle single-class predictions edge case
WHY:
    - Can happen with imbalanced data or small test sets
    - ROC-AUC undefined when only one class present
    - Better to handle gracefully than crash

ALTERNATIVE CONSIDERED: Assume both classes always present
REJECTED BECAUSE: Unrealistic assumption, would crash in edge cases
```

```
DECISION: Use try/except for file operations
WHY:
    - Files may not exist in all environments
    - Network issues with FRED API
    - More robust than assuming files always exist
    - Standard defensive programming

ALTERNATIVE CONSIDERED: Assume files always exist
REJECTED BECAUSE: Unrealistic, breaks in many scenarios
```

## 8. CODE ORGANIZATION DECISIONS

```
DECISION: Separate functions for each major step
WHY:
    - Single Responsibility Principle
    - Easier to test individual components
    - Easier to modify without breaking other parts
    - Better code readability

ALTERNATIVE CONSIDERED: Monolithic functions
REJECTED BECAUSE: Harder to test, maintain, and debug
```

```
DECISION: Use descriptive variable names
WHY:
    - Self-documenting code
    - Easier for others to understand
    - Reduces need for excessive comments
    - Standard best practice

ALTERNATIVE CONSIDERED: Short variable names
REJECTED BECAUSE: Reduces readability, increases cognitive load
```

```
DECISION: Minimal but meaningful comments
WHY:
    - Code should be self-explanatory where possible
    - Comments explain "why" not "what"
    - Avoids comment rot (outdated comments)
    - Cleaner, more professional code

ALTERNATIVE CONSIDERED: Extensive comments
REJECTED BECAUSE: Can become outdated, clutters code
ALTERNATIVE CONSIDERED: No comments
REJECTED BECAUSE: Some decisions need explanation
```

## 9. DATA STORAGE DECISIONS

```
DECISION: Use CSV for data storage
WHY:
    - Simple, human-readable format
    - Easy to inspect and debug
    - Works well with pandas
    - No external dependencies

ALTERNATIVE CONSIDERED: Parquet, HDF5
REJECTED BECAUSE: Adds complexity, CSV sufficient for this use case
ALTERNATIVE CONSIDERED: Database (PostgreSQL, etc.)
REJECTED BECAUSE: Overkill, adds infrastructure complexity
```

```
DECISION: Use joblib for model serialization
WHY:
    - Standard for scikit-learn/XGBoost models
    - Efficient for NumPy arrays
    - Handles large models well
    - Simple API

ALTERNATIVE CONSIDERED: pickle
REJECTED BECAUSE: Less efficient, security concerns
ALTERNATIVE CONSIDERED: ONNX
REJECTED BECAUSE: Adds complexity, joblib sufficient
```

## 10. FUTURE IMPROVEMENTS (Not Implemented But Considered)

```
DECISION: Defer hyperparameter tuning
WHY:
    - Baseline model sufficient for initial deployment
    - Can be added later without breaking changes
    - Requires proper time-series cross-validation
    - Premature optimization avoided

FUTURE: Implement walk-forward validation for tuning
```

```
DECISION: Defer automated retraining
WHY:
    - Manual retraining acceptable for initial version
    - Can be added as cron job or scheduled task later
    - Requires monitoring infrastructure
    - Out of scope for MVP

FUTURE: Add scheduled data refresh and model retraining
```

```
DECISION: Defer feature store implementation
WHY:
    - On-the-fly feature engineering sufficient for now
    - Feature store adds significant infrastructure
    - Can be refactored later
    - Overkill for current scale

FUTURE: Implement proper feature store for production
```

```
DECISION: Defer model monitoring/alerting
WHY:
    - Initial deployment focuses on core functionality
    - Monitoring can be added incrementally
    - Requires additional infrastructure
    - Out of scope for MVP

FUTURE: Add model performance monitoring and drift detection
```

---

## SUMMARY OF KEY PRINCIPLES

1. **Simplicity over complexity**: Choose simpler solutions that work well
2. **Standard practices**: Use established patterns and tools
3. **Graceful degradation**: Handle errors gracefully, don't crash
4. **Separation of concerns**: Each component has clear responsibility
5. **Reproducibility**: Ensure consistent results across environments
6. **Maintainability**: Code should be easy to understand and modify
7. **Practical over perfect**: Ship working solution, improve iteratively
