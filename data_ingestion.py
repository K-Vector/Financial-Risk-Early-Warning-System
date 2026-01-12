import os
import pandas as pd
import numpy as np
from fredapi import Fred

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

FRED_API_KEY = os.environ.get('FRED_API_KEY')

# FRED series IDs for macro-financial indicators
SERIES_IDS = {
    'HY_SPREAD': 'BAMLH0A0HYM2',  # High Yield Spread - our stress proxy
    'CPI': 'CPIAUCSL',
    'UNRATE': 'UNRATE',
    'FEDFUNDS': 'FEDFUNDS',
    'T10Y2Y': 'T10Y2Y',  # Yield curve spread
    'VIX': 'VIXCLS',
    'SP500': 'SP500'
}

OUTPUT_FILE = 'data/macro_financial_data.csv'
START_DATE = '2000-01-01'

def generate_dummy_data(output_path):
    """Generate synthetic data when FRED API key isn't available.
    
    Useful for local dev/testing. Not suitable for production.
    """
    print("No FRED_API_KEY found. Generating synthetic data...")
    
    dates = pd.date_range(start='2020-01-01', periods=52 * 5, freq='W-FRI')
    df = pd.DataFrame(index=dates)
    
    np.random.seed(42)
    
    # Generate realistic-looking time series with some autocorrelation
    df['HY_SPREAD'] = (np.random.normal(loc=400, scale=100, size=len(dates)).cumsum() / 100 + 300) * \
                      (1 + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.5)
    df['CPI'] = np.random.normal(loc=2.5, scale=0.5, size=len(dates)).cumsum() / 50 + 250
    df['UNRATE'] = np.random.normal(loc=4.0, scale=1.0, size=len(dates)).cumsum() / 100 + 3.5
    df['FEDFUNDS'] = np.random.normal(loc=1.0, scale=0.5, size=len(dates)).cumsum() / 200 + 0.25
    df['T10Y2Y'] = np.random.normal(loc=0.5, scale=0.2, size=len(dates)).cumsum() / 500 + 0.5
    df['VIX'] = np.random.normal(loc=15, scale=5, size=len(dates)).cumsum() / 100 + 12
    df['SP500'] = np.random.normal(loc=0.05, scale=0.5, size=len(dates)).cumsum() / 10 + 3000
    
    # Smooth out noise
    for col in df.columns:
        df[col] = df[col].rolling(window=4, min_periods=1).mean()
    
    # Define risk events as HY_SPREAD above rolling 75th percentile
    rolling_threshold = df['HY_SPREAD'].rolling(window=52, min_periods=1).quantile(0.75)
    df['risk_event'] = (df['HY_SPREAD'] > rolling_threshold).astype(int)
    
    # Handle initial NaNs
    overall_threshold = df['HY_SPREAD'].quantile(0.75)
    df['risk_event'] = df['risk_event'].fillna(
        (df['HY_SPREAD'] > overall_threshold).astype(int)
    )
    
    df.to_csv(output_path)
    print(f"Synthetic data saved to {output_path}")
    print(f"Risk event rate: {df['risk_event'].mean():.2%}")
    
    return df

def fetch_and_clean_data():
    """Fetch FRED data, align frequencies, and create target variable."""
    if not FRED_API_KEY:
        return generate_dummy_data(OUTPUT_FILE)

    fred = Fred(api_key=FRED_API_KEY)
    all_data = {}
    
    print(f"Fetching {len(SERIES_IDS)} series from FRED...")
    for name, series_id in SERIES_IDS.items():
        try:
            data = fred.get_series(series_id, observation_start=START_DATE)
            data.name = name
            all_data[name] = data
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue

    if not all_data:
        print("No data fetched. Falling back to synthetic data.")
        return generate_dummy_data(OUTPUT_FILE)

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Resample to weekly (Friday close) - standard for macro data
    df = df.resample('W-FRI').last().ffill()
    df = df.dropna()
    
    print(f"Data shape: {df.shape[0]} weeks, {df.shape[1]} series")
    
    # Risk event = HY_SPREAD above 75th percentile
    risk_threshold = df['HY_SPREAD'].quantile(0.75)
    df['risk_event'] = (df['HY_SPREAD'] > risk_threshold).astype(int)
    
    print(f"Risk threshold: {risk_threshold:.2f} bps")
    print(f"Risk event rate: {df['risk_event'].mean():.2%}")
    
    df.to_csv(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
    
    return df

if __name__ == "__main__":
    fetch_and_clean_data()
