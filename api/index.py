import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Resolve parent directory path for imports in Vercel Serverless environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import load_and_clean_data, engineer_features
from modeling import MachineLearningForecaster, train_discharge_model, forecast_exogenous, forecast_ml_recursive

app = FastAPI(
    title="Vantage Predictive API",
    description="Serverless Forecasting & Surge Simulation API for UAC Operational Planning",
    version="1.0.0"
)

# Enable CORS for frontend hosting on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data path helper
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'HHS_Cleaned_Data.csv')

def get_processed_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    cleaned = load_and_clean_data(DATA_PATH)
    engineered = engineer_features(cleaned)
    return cleaned, engineered

@app.get("/api/status")
def status():
    """Health check endpoint."""
    data_exists = os.path.exists(DATA_PATH)
    return {
        "status": "healthy",
        "dataset_loaded": data_exists,
        "environment": "Vercel Serverless"
    }

@app.get("/api/forecast")
def get_forecast(days: int = Query(30, ge=1, le=90)):
    """
    Generates a baseline 30-day (up to 90 days) forecast for Occupancy,
    Intake, and Discharges using the trained XGBoost ML model.
    """
    try:
        cleaned, engineered = get_processed_data()
        
        # Train ML models on full historical data
        feature_cols = [c for c in engineered.columns if c not in ['Occupancy']]
        X = engineered[feature_cols]
        y = engineered['Occupancy']
        
        xgb_model = MachineLearningForecaster(model_type='xgboost')
        xgb_model.fit(X, y)
        
        # Forecast future exogenous variables
        exo_cols = ['Intake', 'Discharges', 'CBP_Apprehensions', 'CBP_Occupancy']
        exo_cols_present = [c for c in exo_cols if c in engineered.columns]
        future_exo = forecast_exogenous(engineered, exo_cols_present, steps=days)
        
        # Recursive predict occupancy
        preds_occ = forecast_ml_recursive(xgb_model, engineered, future_exo, steps=days)
        
        # Format response
        dates = [d.strftime('%Y-%m-%d') for d in future_exo.index]
        return {
            "success": True,
            "dates": dates,
            "forecasts": {
                "occupancy": [float(v) for v in preds_occ.values],
                "intake": [float(v) for v in future_exo['Intake'].values],
                "discharges": [float(v) for v in future_exo['Discharges'].values]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/simulate")
def simulate_surge(
    surge_pct: float = Query(0.5, description="Intake surge scale (e.g. 0.5 = +50%)"),
    duration: int = Query(14, ge=1, le=60, description="Duration of surge in days"),
    delay: int = Query(3, ge=0, le=30, description="Delay in days before surge starts"),
    steps: int = Query(30, ge=1, le=90)
):
    """
    Simulates a border surge by altering future intake variables and propagating the 
    effects through the trained discharge prediction feedback loop.
    """
    try:
        cleaned, engineered = get_processed_data()
        
        # Train discharge model and ML model
        discharge_model = train_discharge_model(engineered)
        
        # Create baseline future exo
        exo_cols = ['Intake', 'Discharges', 'CBP_Apprehensions', 'CBP_Occupancy']
        exo_cols_present = [c for c in exo_cols if c in engineered.columns]
        future_exo = forecast_exogenous(engineered, exo_cols_present, steps=steps)
        
        # Build simulated future exo copy
        future_sim = future_exo.copy()
        for i in range(steps):
            if delay <= i < (delay + duration):
                future_sim.iloc[i, future_sim.columns.get_loc('Intake')] *= (1 + surge_pct)
                if 'CBP_Apprehensions' in future_sim.columns:
                    future_sim.iloc[i, future_sim.columns.get_loc('CBP_Apprehensions')] *= (1 + surge_pct)
                    
        # Simulate step-by-step
        df_work = engineered.copy()
        sim_occ = []
        sim_disc = []
        
        # Exogenous variables we need to keep updated
        exo_cols = ['Intake', 'Discharges', 'CBP_Apprehensions', 'CBP_Occupancy', 'Capacity']
        
        for i in range(steps):
            target_date = future_sim.index[i]
            new_row = pd.Series(index=df_work.columns, dtype=float, name=target_date)
            
            # Exogenous
            for col in exo_cols:
                if col in future_sim.columns:
                    new_row[col] = future_sim.loc[target_date, col]
                elif col in df_work.columns:
                    new_row[col] = df_work[col].iloc[-1]
                    
            # Calendar
            new_row['DayOfWeek'] = float(target_date.dayofweek)
            new_row['Month'] = float(target_date.month)
            new_row['Day'] = float(target_date.day)
            new_row['IsWeekend'] = float(1.0 if target_date.dayofweek in [5, 6] else 0.0)
            
            # Lags
            for lag in [1, 2, 3, 7, 14]:
                new_row[f'Occupancy_Lag_{lag}'] = df_work['Occupancy'].iloc[-lag]
                new_row[f'Intake_Lag_{lag}'] = df_work['Intake'].iloc[-lag]
                new_row[f'Discharges_Lag_{lag}'] = df_work['Discharges'].iloc[-lag]
                if 'CBP_Apprehensions' in df_work.columns:
                    new_row[f'CBP_Apprehensions_Lag_{lag}'] = df_work['CBP_Apprehensions'].iloc[-lag]
                if 'CBP_Occupancy' in df_work.columns:
                    new_row[f'CBP_Occupancy_Lag_{lag}'] = df_work['CBP_Occupancy'].iloc[-lag]
                    
            # Rolling
            intake_vals = df_work['Intake'].iloc[-6:].tolist() + [new_row['Intake']]
            new_row['Intake_7D_Avg'] = np.mean(intake_vals)
            
            occupancy_past = df_work['Occupancy'].iloc[-7:].tolist()
            new_row['Occupancy_7D_Avg'] = np.mean(occupancy_past)
            
            if 'CBP_Apprehensions' in df_work.columns:
                cbp_app_past = df_work['CBP_Apprehensions'].iloc[-6:].tolist() + [new_row['CBP_Apprehensions']]
                new_row['CBP_Apprehensions_7D_Avg'] = np.mean(cbp_app_past)
            if 'CBP_Occupancy' in df_work.columns:
                cbp_occ_past = df_work['CBP_Occupancy'].iloc[-6:].tolist() + [new_row['CBP_Occupancy']]
                new_row['CBP_Occupancy_7D_Avg'] = np.mean(cbp_occ_past)
                
            # Predict Discharges
            feat_cols = discharge_model.features
            X_step = pd.DataFrame([new_row[feat_cols]], columns=feat_cols)
            pred_discharge = max(0.0, float(discharge_model.predict(X_step)[0]))
            new_row['Discharges'] = pred_discharge
            
            # Now update discharges rolling average
            discharge_vals = df_work['Discharges'].iloc[-6:].tolist() + [new_row['Discharges']]
            new_row['Discharges_7D_Avg'] = np.mean(discharge_vals)
            
            # Physical system update
            prev_occ = df_work['Occupancy'].iloc[-1]
            pred_occ = max(0.0, prev_occ + new_row['Intake'] - new_row['Discharges'])
            new_row['Occupancy'] = pred_occ
            new_row['Net_Pressure'] = new_row['Intake'] - new_row['Discharges']
            
            df_work = pd.concat([df_work, pd.DataFrame([new_row], index=[target_date])])
            sim_occ.append(pred_occ)
            sim_disc.append(pred_discharge)
            
        dates = [d.strftime('%Y-%m-%d') for d in future_sim.index]
        return {
            "success": True,
            "dates": dates,
            "simulated": {
                "occupancy": [float(v) for v in sim_occ],
                "discharges": [float(v) for v in sim_disc],
                "intake": [float(v) for v in future_sim['Intake'].values]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
