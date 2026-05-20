"""
Vantage Predictive Modeling Module
Implements Baseline (Naive Persistence), SARIMA, XGBoost, and Random Forest models.
Features Walk-Forward Validation, feature importance, and recursive time-series forecasting.
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


class NaivePersistenceModel:
    """
    Naive Persistence Model: Predicts t + h = t.
    """
    def __init__(self):
        self.features = []
        
    def fit(self, X, y):
        self.features = list(X.columns) if isinstance(X, pd.DataFrame) else []
        
    def predict(self, history, steps=30):
        # Predicts the last observed value for all future steps
        last_val = history.iloc[-1] if hasattr(history, 'iloc') else history[-1]
        return np.full(steps, last_val)


class SarimaForecaster:
    """
    SARIMA Model for trend and seasonal forecasting.
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
        
    def fit(self, y):
        # Fit SARIMAX
        model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        self.model_fit = model.fit(disp=False)
        
    def predict(self, steps=30):
        if self.model_fit is None:
            raise ValueError("Model has not been fitted yet.")
        forecast_res = self.model_fit.get_forecast(steps=steps)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.10)  # 90% CI
        return forecast_mean.values, conf_int


class MachineLearningForecaster:
    """
    XGBoost or Random Forest regression-based time-series forecasting.
    Uses lags and rolling features.
    """
    def __init__(self, model_type='xgboost', **kwargs):
        self.model_type = model_type
        if model_type == 'xgboost':
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.08,
                'random_state': 42
            }
            params.update(kwargs)
            self.model = xgb.XGBRegressor(**params)
        elif model_type == 'random_forest':
            params = {
                'n_estimators': 100,
                'max_depth': 12,
                'random_state': 42
            }
            params.update(kwargs)
            self.model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.features = []
        
    def fit(self, X, y):
        self.features = list(X.columns)
        self.model.fit(X, y)
        
    def predict(self, X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame([X])
        return self.model.predict(X)
        
    def get_feature_importance(self):
        """Returns feature importance dataframe."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return pd.DataFrame({
                'Feature': self.features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        return None

    def get_shap_explanations(self, X):
        """Generates SHAP explainer and SHAP values for given features."""
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer(X)
            return explainer, shap_values
        except Exception as e:
            print("Error generating SHAP explanations:", e)
            # Fallback explainer if TreeExplainer fails
            try:
                explainer = shap.Explainer(self.model, X)
                shap_values = explainer(X)
                return explainer, shap_values
            except Exception as e2:
                print("Fallback SHAP explainer also failed:", e2)
                return None, None


def forecast_exogenous(df_history: pd.DataFrame, columns: list, steps: int = 30) -> pd.DataFrame:
    """
    Forecasts exogenous variables (e.g., Intake, Discharges, CBP custody) for the future window
    using simple univariate SARIMA models.
    """
    future_index = pd.date_range(start=df_history.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    future_df = pd.DataFrame(index=future_index)
    
    for col in columns:
        if col in df_history.columns:
            try:
                # Fit a simple SARIMA(1, 1, 1)x(1, 0, 0)7 model for the exogenous variable
                model = SARIMAX(df_history[col], order=(1, 1, 1), seasonal_order=(1, 0, 0, 7),
                                enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                future_df[col] = res.forecast(steps=steps).values
            except Exception as e:
                # Fallback to persistence + weekly average offset
                print(f"SARIMA forecast failed for exogenous '{col}' ({e}). Falling back to persistence.")
                last_val = df_history[col].iloc[-1]
                # Calculate mean weekly profile to add variation
                try:
                    weekly_profile = df_history.groupby(df_history.index.dayofweek)[col].mean()
                    weekly_profile_diff = weekly_profile - weekly_profile.mean()
                    
                    forecast_vals = []
                    for d in future_index:
                        day_idx = d.dayofweek
                        forecast_vals.append(max(0.0, last_val + weekly_profile_diff[day_idx]))
                    future_df[col] = forecast_vals
                except:
                    future_df[col] = np.full(steps, last_val)
                    
    # Capacity is assumed constant by default unless specified
    if 'Capacity' in df_history.columns:
        future_df['Capacity'] = df_history['Capacity'].iloc[-1]
        
    return future_df


def forecast_ml_recursive(model: MachineLearningForecaster, df_history: pd.DataFrame, 
                          future_exo: pd.DataFrame, steps: int = 30) -> pd.Series:
    """
    Recursively forecasts Occupancy for h-steps using a trained machine learning model.
    It updates lags and rolling averages dynamically at each step.
    
    Parameters:
        model: Trained MachineLearningForecaster.
        df_history: Cleaned and engineered historical data.
        future_exo: Forecasted exogenous variables for the forecasting horizon.
        steps: Number of days forward to forecast.
    """
    df_work = df_history.copy()
    
    exo_cols = ['Intake', 'Discharges', 'CBP_Apprehensions', 'CBP_Occupancy', 'Capacity']
    
    for i in range(steps):
        target_date = future_exo.index[i]
        
        # Initialize a new row
        new_row = pd.Series(index=df_work.columns, dtype=float, name=target_date)
        
        # 1. Fill exogenous variables
        for col in exo_cols:
            if col in future_exo.columns:
                new_row[col] = future_exo.loc[target_date, col]
            elif col in df_work.columns:
                new_row[col] = df_work[col].iloc[-1]
                
        # 2. Fill calendar features
        new_row['DayOfWeek'] = float(target_date.dayofweek)
        new_row['Month'] = float(target_date.month)
        new_row['Day'] = float(target_date.day)
        new_row['IsWeekend'] = float(1.0 if target_date.dayofweek in [5, 6] else 0.0)
        
        # 3. Calculate Net Pressure
        new_row['Net_Pressure'] = new_row['Intake'] - new_row['Discharges']
        
        # 4. Fill lag variables
        for lag in [1, 2, 3, 7, 14]:
            new_row[f'Occupancy_Lag_{lag}'] = df_work['Occupancy'].iloc[-lag]
            new_row[f'Intake_Lag_{lag}'] = df_work['Intake'].iloc[-lag]
            new_row[f'Discharges_Lag_{lag}'] = df_work['Discharges'].iloc[-lag]
            if 'CBP_Apprehensions' in df_work.columns:
                new_row[f'CBP_Apprehensions_Lag_{lag}'] = df_work['CBP_Apprehensions'].iloc[-lag]
            if 'CBP_Occupancy' in df_work.columns:
                new_row[f'CBP_Occupancy_Lag_{lag}'] = df_work['CBP_Occupancy'].iloc[-lag]
                
        # 5. Fill rolling features
        # For Intake, Discharges, CBP: includes current step's value
        intake_vals = df_work['Intake'].iloc[-6:].tolist() + [new_row['Intake']]
        new_row['Intake_7D_Avg'] = np.mean(intake_vals)
        
        discharge_vals = df_work['Discharges'].iloc[-6:].tolist() + [new_row['Discharges']]
        new_row['Discharges_7D_Avg'] = np.mean(discharge_vals)
        
        # Occupancy rolling average is shifted by 1 (does NOT include current occupancy, preventing leak)
        occupancy_past = df_work['Occupancy'].iloc[-7:].tolist()
        new_row['Occupancy_7D_Avg'] = np.mean(occupancy_past)
        
        if 'CBP_Apprehensions' in df_work.columns:
            cbp_app_past = df_work['CBP_Apprehensions'].iloc[-6:].tolist() + [new_row['CBP_Apprehensions']]
            new_row['CBP_Apprehensions_7D_Avg'] = np.mean(cbp_app_past)
        if 'CBP_Occupancy' in df_work.columns:
            cbp_occ_past = df_work['CBP_Occupancy'].iloc[-6:].tolist() + [new_row['CBP_Occupancy']]
            new_row['CBP_Occupancy_7D_Avg'] = np.mean(cbp_occ_past)
            
        # 6. Extract features and make prediction
        feat_cols = model.features
        X_step = pd.DataFrame([new_row[feat_cols]], columns=feat_cols)
        pred_occ = model.predict(X_step)[0]
        
        # Cap occupancy at 0
        pred_occ = max(0.0, pred_occ)
        new_row['Occupancy'] = pred_occ
        
        # Append to df_work so that next step can read lags of this prediction
        df_work = pd.concat([df_work, pd.DataFrame([new_row], index=[target_date])])
        
    return df_work['Occupancy'].iloc[-steps:]


def walk_forward_validation(df: pd.DataFrame, initial_train_size: int, 
                            forecast_horizon: int = 30, step_size: int = 30, 
                            model_type: str = 'xgboost') -> dict:
    """
    Performs expanding-window Walk-Forward Validation.
    """
    total_len = len(df)
    maes = []
    rmses = []
    mapes = []
    
    train_idx = initial_train_size
    window_count = 0
    
    print(f"Starting Walk-Forward Validation for {model_type}...")
    
    while train_idx + forecast_horizon <= total_len:
        window_count += 1
        df_train = df.iloc[:train_idx]
        df_test = df.iloc[train_idx:train_idx + forecast_horizon]
        
        # Train and forecast
        if model_type == 'persistence':
            model = NaivePersistenceModel()
            y_pred = model.predict(df_train['Occupancy'], steps=forecast_horizon)
        elif model_type == 'sarima':
            model = SarimaForecaster()
            model.fit(df_train['Occupancy'])
            y_pred, _ = model.predict(steps=forecast_horizon)
        else:
            # Machine learning
            # 1. Forecast future exogenous variables
            exo_cols = ['Intake', 'Discharges', 'CBP_Apprehensions', 'CBP_Occupancy']
            # Only pass columns that exist
            exo_cols_present = [c for c in exo_cols if c in df_train.columns]
            future_exo = forecast_exogenous(df_train, exo_cols_present, steps=forecast_horizon)
            
            # 2. Fit ML model on training features
            feature_cols = [c for c in df_train.columns if c not in ['Occupancy']]
            X_train = df_train[feature_cols]
            y_train = df_train['Occupancy']
            
            model = MachineLearningForecaster(model_type=model_type)
            model.fit(X_train, y_train)
            
            # 3. Recursive predict
            y_pred = forecast_ml_recursive(model, df_train, future_exo, steps=forecast_horizon).values
            
        y_true = df_test['Occupancy'].values
        
        # Calculate metrics for window
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = calculate_mape(y_true, y_pred)
        
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        
        # Expand window
        train_idx += step_size
        
    print(f"Completed {window_count} validation windows.")
    return {
        'MAE': np.mean(maes) if maes else 0.0,
        'RMSE': np.mean(rmses) if rmses else 0.0,
        'MAPE': np.mean(mapes) if mapes else 0.0
    }


def train_discharge_model(df_engineered: pd.DataFrame) -> MachineLearningForecaster:
    """
    Trains a specialized model to forecast Discharges based on lagged Intake and Occupancy.
    This is used by the simulation engine to simulate how a surge in Intake translates
    to changes in Discharges and subsequent Occupancy.
    """
    print("Training specialized Discharge model for the simulation engine...")
    df = df_engineered.copy()
    
    # We want to predict Discharges
    # Features will be: lags of Occupancy, lags of Intake, lags of Discharges, and Calendar features
    feature_cols = [c for c in df.columns if c not in ['Discharges', 'Net_Pressure', 'Discharges_7D_Avg', 'Capacity']]
    X = df[feature_cols]
    y = df['Discharges']
    
    model = MachineLearningForecaster(model_type='xgboost', max_depth=5, learning_rate=0.08)
    model.fit(X, y)
    return model


if __name__ == '__main__':
    print("Vantage Modeling Module loaded successfully.")
    
    # Test pipeline if file is present
    from data_processing import load_and_clean_data, engineer_features
    raw_path = 'data/HHS_Cleaned_Data.csv'
    
    if os.path.exists(raw_path):
        try:
            cleaned = load_and_clean_data(raw_path)
            engineered = engineer_features(cleaned)
            
            # Use last 90 days as test validation holdout, so initial train is total_len - 90
            initial_train = len(engineered) - 90
            
            print(f"\n--- Running Walk-Forward Benchmarking (Holdout: 90 days) ---")
            
            # Persistence
            res_p = walk_forward_validation(engineered, initial_train, forecast_horizon=30, step_size=30, model_type='persistence')
            print(f"Naive Persistence -> MAE: {res_p['MAE']:.2f}, RMSE: {res_p['RMSE']:.2f}, MAPE: {res_p['MAPE']:.2f}%")
            
            # SARIMA
            res_s = walk_forward_validation(engineered, initial_train, forecast_horizon=30, step_size=30, model_type='sarima')
            print(f"SARIMA            -> MAE: {res_s['MAE']:.2f}, RMSE: {res_s['RMSE']:.2f}, MAPE: {res_s['MAPE']:.2f}%")
            
            # Random Forest
            res_rf = walk_forward_validation(engineered, initial_train, forecast_horizon=30, step_size=30, model_type='random_forest')
            print(f"Random Forest     -> MAE: {res_rf['MAE']:.2f}, RMSE: {res_rf['RMSE']:.2f}, MAPE: {res_rf['MAPE']:.2f}%")
            
            # XGBoost
            res_xgb = walk_forward_validation(engineered, initial_train, forecast_horizon=30, step_size=30, model_type='xgboost')
            print(f"XGBoost           -> MAE: {res_xgb['MAE']:.2f}, RMSE: {res_xgb['RMSE']:.2f}, MAPE: {res_xgb['MAPE']:.2f}%")
            
        except Exception as e:
            print("Error during test modeling run:", e)
    else:
        print(f"HHS_Cleaned_Data.csv not found. Skipping test modeling.")
