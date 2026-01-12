import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Configuration ---
# NOTE: You must set your FRED API key as an environment variable.
# Get a key here: https://fred.stlouisfed.org/docs/api/api_key.html
# export FRED_API_KEY=\'YOUR_KEY_HERE\'
FRED_API_KEY = os.environ.get(\'FRED_API_KEY\' )

# Selected FRED Series IDs
SERIES_IDS = {
    \'HY_SPREAD\': \'BAMLH0A0HYM2\', # High Yield Spread (Target Proxy)
    \'CPI\': \'CPIAUCSL\',
    \'UNRATE\': \'UNRATE\',
    \'FEDFUNDS\': \'FEDFUNDS\',
    \'T10Y2Y\': \'T10Y2Y\', # 10Y-2Y Yield Curve Spread
    \'VIX\': \'VIXCLS\',
    \'SP500\': \'SP500\'
}

OUTPUT_FILE = \'data/macro_financial_data.csv\'
START_DATE = \'2000-01-01\' # Updated Start Date
END_DATE = (date.today() - timedelta(days=1)).strftime(\'%Y-%m-%d\') # End Date is Yesterday

def generate_dummy_data(output_path):
    """Generates a dummy dataset for development when FRED API key is missing."""
    print("FRED_API_KEY not found. Generating dummy data for development purposes.")
    
    # Create a time index (Daily data from START_DATE to END_DATE)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq=\'D\')
    df = pd.DataFrame(index=dates)
    
    # Generate synthetic data for each series
    np.random.seed(42)
    # Create a more volatile HY_SPREAD to ensure stress events occur
    df[\'HY_SPREAD\'] = (np.random.normal(loc=400, scale=100, size=len(dates)).cumsum() / 100 + 300) * (1 + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.5)
    df[\'CPI\'] = np.random.normal(loc=2.5, scale=0.5, size=len(dates)).cumsum() / 50 + 250
    df[\'UNRATE\'] = np.random.normal(loc=4.0, scale=1.0, size=len(dates)).cumsum() / 100 + 3.5
    df[\'FEDFUNDS\'] = np.random.normal(loc=1.0, scale=0.5, size=len(dates)).cumsum() / 200 + 0.25
    df[\'T10Y2Y\'] = np.random.normal(loc=0.5, scale=0.2, size=len(dates)).cumsum() / 500 + 0.5
    df[\'VIX\'] = np.random.normal(loc=15, scale=5, size=len(dates)).cumsum() / 100 + 12
    df[\'SP500\'] = np.random.normal(loc=0.05, scale=0.5, size=len(dates)).cumsum() / 10 + 3000
    
    # Smooth the data a bit
    for col in df.columns:
        df[col] = df[col].rolling(window=7, min_periods=1).mean() # Use 7-day rolling for daily data
        
    # --- Target Variable Creation (Revised) ---
    print("Creating \'risk_event\' target variable...")
    
    # Use a long rolling window (e.g., 1 year) to define the "normal" condition
    rolling_threshold = df[\'HY_SPREAD\'].rolling(window=252, min_periods=1).quantile(0.75) # 252 trading days in a year
    
    # 1 = FSI above 75th percentile (stress event)
    df[\'risk_event\'] = (df[\'HY_SPREAD\'] > rolling_threshold).astype(int)
    
    # Fill the initial NaNs from the rolling window with the overall 75th percentile
    overall_threshold = df[\'HY_SPREAD\'].quantile(0.75)
    df[\'risk_event\'] = df[\'risk_event\'].fillna((df[\'HY_SPREAD\'] > overall_threshold).astype(int))
    
    # Save to CSV
    df.to_csv(output_path)
    print(f"Dummy data saved to {output_path}. Please set FRED_API_KEY to use real data.")
    
    # Print a summary of the target variable distribution
    print("\nTarget Variable Distribution:")
    print(df[\'risk_event\'].value_counts(normalize=True))
    
    return df

def fetch_and_clean_data():
    """
    Fetches time-series data from FRED, cleans, aligns, and saves it.
    """
    # Import fredapi here to ensure it's only imported if FRED_API_KEY is present
    # This helps avoid ModuleNotFoundError if fredapi installation is problematic
    if not FRED_API_KEY:
        return generate_dummy_data(OUTPUT_FILE)
    
    try:
        from fredapi import Fred
    except ImportError:
        print("Error: fredapi not found. Please ensure it is installed or set FRED_API_KEY to generate dummy data.")
        return generate_dummy_data(OUTPUT_FILE)

    print("Initializing FRED API client...")
    fred = Fred(api_key=FRED_API_KEY)
    
    all_data = {}
    print(f"Fetching {len(SERIES_IDS)} series from FRED from {START_DATE} to {END_DATE}...")
    for name, series_id in SERIES_IDS.items():
        print(f"  -> Fetching {name} ({series_id})...")
        try:
            # Fetch data up to yesterday
            data = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            data.name = name
            all_data[name] = data
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            continue

    # Combine all series into a single DataFrame
    df = pd.DataFrame(all_data)
    
    # --- Data Cleaning and Alignment ---
    print("\nCleaning and aligning data...")
    
    # 1. Convert index to datetime and ensure all data is numeric
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors=\'coerce\')
    
    # 2. Resample to a common frequency (Daily, filling forward)
    # This aligns all data to a daily frequency, which is suitable for market open days.
    # We use \'ffill\' (forward-fill) to carry the last known value forward.
    df_resampled = df.resample(\'D\').last().ffill()
    
    # 3. Drop rows where all values are NaN (e.g., weekends/holidays where no data was published)
    # Then, only keep rows that correspond to a market open day (i.e., where SP500 or VIX has a value)
    df_cleaned = df_resampled.dropna(how=\'all\').dropna(subset=[\'SP500\', \'VIX\'])
    
    print(f"Original data shape: {df.shape}")
    print(f"Cleaned and aligned data shape: {df_cleaned.shape}")
    
    # --- Target Variable Creation ---
    print("Creating \'risk_event\' target variable...")
    
    # Calculate the 75th percentile threshold based on a rolling 1-year window (approx 252 trading days)
    rolling_threshold = df_cleaned[\'HY_SPREAD\'].rolling(window=252, min_periods=1).quantile(0.75)
    
    # 1 = FSI above 75th percentile (stress event)
    df_cleaned[\'risk_event\'] = (df_cleaned[\'HY_SPREAD\'] > rolling_threshold).astype(int)
    
    # --- Save Data ---
    print(f"\nSaving cleaned data to {OUTPUT_FILE}...")
    df_cleaned.to_csv(OUTPUT_FILE)
    print("Data ingestion complete.")
    
    return df_cleaned

if __name__ == "__main__":
    fetch_and_clean_data()
