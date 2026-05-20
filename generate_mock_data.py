"""
Vantage Mock Data Generator
Generates a realistic, high-fidelity synthetic dataset for HHS UAC program tracking.
Includes seasonal trends, weekly patterns, capacity adjustments, and random missing data points.
"""

import pandas as pd
import numpy as np
import os


def generate_synthetic_uac_data(filepath: str, start_date="2021-01-01", end_date="2026-05-15", seed=42):
    """
    Generates a synthetic HHS UAC daily time-series dataset.
    """
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # 1. Base time trend and seasonal cycles (Spring surges)
    # Day count
    t = np.arange(n_days)
    
    # Annual cycle: peak in Spring (around March - June, day of year 60 to 180)
    day_of_year = date_range.dayofyear
    annual_cycle = np.sin(2 * np.pi * (day_of_year - 40) / 365.25)
    # Add a slight upward long-term trend
    long_trend = 1.5 * t / 365.25 
    
    # 2. Intake: weekly seasonality (lower on weekends) + annual cycle + random spikes + noise
    weekly_cycle = np.array([1.2, 1.3, 1.1, 1.0, 0.9, 0.4, 0.5]) # Mon-Sun scaling
    day_of_week = date_range.dayofweek
    intake_base = 250 + 150 * annual_cycle + long_trend * 10
    
    # Apply weekly scaling
    intake = intake_base * weekly_cycle[day_of_week]
    # Add random spikes (surges)
    surge_chance = np.random.binomial(1, 0.02, n_days)
    surges = surge_chance * np.random.exponential(300, n_days)
    intake = intake + surges
    # Add random noise
    intake = intake + np.random.normal(0, 30, n_days)
    intake = np.clip(intake, 10, None) # Min 10 intakes per day
    
    # 3. Discharges: lags intake by ~30 days on average, weekly pattern (very low on weekends)
    # We will simulate discharge as a rolling average of intake with a lag, plus seasonal efficiency bottlenecks
    discharges = np.zeros(n_days)
    initial_occupancy = 6500
    
    # Basic discharge rate: matches intake with some lag, plus weekend drop-off
    discharge_weekly_cycle = np.array([1.3, 1.4, 1.2, 1.1, 0.9, 0.1, 0.2]) # Mon-Sun
    
    for i in range(n_days):
        if i < 30:
            # Warm-up period
            discharges[i] = intake[i] * 0.9 + np.random.normal(0, 15)
        else:
            # Lags intake by 30 days on average
            past_intake = np.mean(intake[max(0, i-45):max(1, i-15)])
            discharges[i] = past_intake * 0.95 + np.random.normal(0, 20)
            
        discharges[i] = discharges[i] * discharge_weekly_cycle[day_of_week[i]]
        discharges[i] = np.clip(discharges[i], 5, None)
        
    # 4. Occupancy: Cumulative sum of Intake - Discharges
    occupancy = np.zeros(n_days)
    occupancy[0] = initial_occupancy
    for i in range(1, n_days):
        net = intake[i] - discharges[i]
        # Prevent extreme drift, bound it between 3000 and 14000
        occupancy[i] = occupancy[i-1] + net
        if occupancy[i] < 3000:
            # Boost intake or reduce discharge to stabilize mock dynamics
            discharges[i] = max(5, discharges[i] - 100)
            occupancy[i] = occupancy[i-1] + intake[i] - discharges[i]
        elif occupancy[i] > 14000:
            # Boost discharge or reduce intake to stabilize
            discharges[i] = discharges[i] + 150
            occupancy[i] = occupancy[i-1] + intake[i] - discharges[i]
            
    # Round to integers
    intake = np.round(intake).astype(int)
    discharges = np.round(discharges).astype(int)
    occupancy = np.round(occupancy).astype(int)
    
    # 5. Capacity: Stepwise adjustments.
    # Base capacity is 10,000. If occupancy > 8,500, we expand capacity to 12,500.
    # If occupancy > 11,000, we expand to 15,000.
    capacity = np.zeros(n_days)
    current_cap = 10000
    for i in range(n_days):
        # We check occupancy and adjust capacity with some lag/inertia (e.g., stay expanded for a bit)
        if occupancy[i] > 11000:
            current_cap = 15000
        elif occupancy[i] > 8500:
            # If we were at 15000, don't drop down immediately
            if current_cap < 12500:
                current_cap = 12500
        else:
            # Bring capacity back down when occupancy is low
            if occupancy[i] < 7000:
                current_cap = 10000
        capacity[i] = current_cap
        
    df = pd.DataFrame({
        'Date': date_range,
        'Intake_Referrals': intake,
        'Discharges_Releases': discharges,
        'Shelter_Occupancy': occupancy,
        'Operational_Capacity': capacity
    })
    
    # 6. Inject random missingness to test the interpolation pipeline (e.g., 5% random NaNs)
    # We don't remove Date, but we introduce NaNs in some rows for Intake, Discharges, and Occupancy
    mask_intake = np.random.rand(n_days) < 0.05
    mask_discharges = np.random.rand(n_days) < 0.05
    mask_occupancy = np.random.rand(n_days) < 0.05
    
    df.loc[mask_intake, 'Intake_Referrals'] = np.nan
    df.loc[mask_discharges, 'Discharges_Releases'] = np.nan
    df.loc[mask_occupancy, 'Shelter_Occupancy'] = np.nan
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic HHS UAC dataset with {n_days} rows at: {filepath}")
    return df


if __name__ == '__main__':
    generate_synthetic_uac_data('p:\\Intenship\\Vantage\\data\\mock_uac_data.csv')
