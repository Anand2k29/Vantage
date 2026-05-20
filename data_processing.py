"""
Vantage Data Engineering Pipeline
Handles loading, cleaning, normalization, missing data interpolation, and operational signal engineering.
"""

import pandas as pd
import numpy as np
import os


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw HHS UAC program data, normalizes date-time columns, 
    and handles missing values using time-based linear interpolation.
    
    Parameters:
        filepath (str): Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: Cleaned time-series DataFrame with a regular daily index.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
        
    print(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Date normalization
    # Find a column containing 'date' (case insensitive) and parse it
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break
            
    if date_col is None:
        # Fallback to the first column if no date column is named explicitly
        date_col = df.columns[0]
        print(f"Warning: No explicit 'date' column found. Using '{date_col}' as the date field.")
        
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'Date'})
    
    # Sort by date and set as index
    df = df.sort_values('Date').set_index('Date')
    
    # 2. Standardize column names (mapping to standard terms: Intake, Discharges, Occupancy, Capacity)
    # First check for exact matches with the HHS Cleaned Data schema
    hhs_schema_mapping = {
        'Children apprehended and placed in CBP custody*': 'CBP_Apprehensions',
        'Children in CBP custody': 'CBP_Occupancy',
        'Children transferred out of CBP custody': 'Intake',
        'Children in HHS Care': 'Occupancy',
        'Children discharged from HHS Care': 'Discharges'
    }
    
    # Check if we have overlap with the HHS schema
    has_hhs_columns = any(col in df.columns for col in hhs_schema_mapping)
    
    col_mapping = {}
    if has_hhs_columns:
        print("Detected HHS UAC standard dataset schema. Applying precise column mapping...")
        for col in df.columns:
            if col in hhs_schema_mapping:
                col_mapping[col] = hhs_schema_mapping[col]
    else:
        print("Applying generic keyword-based column mapping...")
        for col in df.columns:
            col_lower = col.lower()
            if 'discharge' in col_lower or 'release' in col_lower or 'outflow' in col_lower:
                col_mapping[col] = 'Discharges'
            elif 'intake' in col_lower or 'referral' in col_lower or 'transfer' in col_lower or 'inflow' in col_lower:
                col_mapping[col] = 'Intake'
            elif 'occupancy' in col_lower or 'care_load' in col_lower or 'in_care' in col_lower or 'hhs care' in col_lower or 'active' in col_lower:
                col_mapping[col] = 'Occupancy'
            elif 'capacity' in col_lower or 'beds' in col_lower:
                col_mapping[col] = 'Capacity'
            
    df = df.rename(columns=col_mapping)
    
    # Validate essential columns
    required_cols = ['Intake', 'Discharges', 'Occupancy']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        print(f"Warning: Missing expected columns: {missing_required}. Existing columns: {list(df.columns)}")
        # We will create placeholders or try to infer if they are missing
        for col in missing_required:
            df[col] = np.nan
            
    if 'Capacity' not in df.columns:
        # If Capacity is missing, set a default operational baseline (e.g. max occupancy or 10000)
        max_occ = df['Occupancy'].max()
        default_cap = max_occ * 1.2 if not pd.isna(max_occ) else 10000
        df['Capacity'] = default_cap
        print(f"Capacity column not found. Created baseline capacity of {default_cap:.0f}")

    # 3. Handle data continuity (ensure daily frequency)
    df = df.asfreq('D')
    
    # 4. Impute missing values via time-based linear interpolation
    # Time-based interpolation fills gaps based on the index (which is a DatetimeIndex)
    df = df.interpolate(method='time')
    
    # Forward-fill / Backward-fill remaining edge cases (e.g., at boundaries)
    df = df.ffill().bfill()
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers operational signals to capture trends, temporal dependencies, and system pressure.
    
    Features created:
        - Net_Pressure: (Intake - Discharges)
        - 7-Day Rolling Averages of Intake, Discharges, and Occupancy
        - Lag variables (t-1, t-2, t-3, t-7, t-14) for occupancy and intake
        - Calendar features (Day of week, Month, Day, is_weekend)
    """
    df = df.copy()
    
    # 1. Operational signals
    df['Net_Pressure'] = df['Intake'] - df['Discharges']
    
    # 2. Rolling Averages (smoothing weekly noise)
    df['Intake_7D_Avg'] = df['Intake'].rolling(window=7, min_periods=1).mean()
    df['Discharges_7D_Avg'] = df['Discharges'].rolling(window=7, min_periods=1).mean()
    df['Occupancy_7D_Avg'] = df['Occupancy'].shift(1).rolling(window=7, min_periods=1).mean()
    
    # 3. Lag Variables
    for lag in [1, 2, 3, 7, 14]:
        df[f'Occupancy_Lag_{lag}'] = df['Occupancy'].shift(lag)
        df[f'Intake_Lag_{lag}'] = df['Intake'].shift(lag)
        df[f'Discharges_Lag_{lag}'] = df['Discharges'].shift(lag)
        if 'CBP_Apprehensions' in df.columns:
            df[f'CBP_Apprehensions_Lag_{lag}'] = df['CBP_Apprehensions'].shift(lag)
        if 'CBP_Occupancy' in df.columns:
            df[f'CBP_Occupancy_Lag_{lag}'] = df['CBP_Occupancy'].shift(lag)
            
    # Also create rolling averages for CBP features if present
    if 'CBP_Apprehensions' in df.columns:
        df['CBP_Apprehensions_7D_Avg'] = df['CBP_Apprehensions'].rolling(window=7, min_periods=1).mean()
    if 'CBP_Occupancy' in df.columns:
        df['CBP_Occupancy_7D_Avg'] = df['CBP_Occupancy'].rolling(window=7, min_periods=1).mean()
        
    # 4. Calendar Features
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Drop rows with NaN (introduced by lags) to ensure modeling integrity
    # (Typically shift(14) creates 14 NaNs at the beginning of the dataset)
    df = df.dropna()
    
    return df


if __name__ == '__main__':
    # Simple verification when executed directly
    print("Vantage Data Engineering Module loaded successfully.")
    try:
        raw_path = 'data/HHS_Cleaned_Data.csv'
        if os.path.exists(raw_path):
            cleaned = load_and_clean_data(raw_path)
            print("Successfully cleaned data. Shape:", cleaned.shape)
            print("Columns after cleaning:", list(cleaned.columns))
            
            engineered = engineer_features(cleaned)
            print("Successfully engineered features. Shape:", engineered.shape)
            print("Columns after feature engineering:", list(engineered.columns))
            print("First few rows:\n", engineered.head(3))
        else:
            print(f"HHS_Cleaned_Data.csv not found at {raw_path}. Skipping local data run.")
    except Exception as e:
        print("Error during verification:", e)

