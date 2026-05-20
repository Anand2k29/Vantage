# Vantage: Predictive Capacity & Resource Forecasting for Humanitarian Systems
## Executive Summary & Briefing Document

### 1. Executive Briefing
The Department of Health and Human Services (HHS) Unaccompanied Children (UAC) program operates in a highly volatile humanitarian environment. Historically, system capacity planning and staffing decisions have been reactive—driven by crisis management protocols triggered after surges occur. 

**Vantage** represents a paradigm shift. By implementing an enterprise-grade predictive analytics pipeline, Vantage transforms raw, volatile, and incomplete UAC data into proactive operational intelligence. This system enables leadership to forecast shelter occupancy, simulate intake surges, anticipate release bottlenecks, and coordinate resources up to 30 days in advance.

---

### 2. The Operational Challenge: Reactive vs. Proactive Capacity
Under the historical model, shelter expansions and staffing deployments are triggered by current occupancy levels:
- **Reactive Model:** If occupancy exceeds 85%, emergency influx shelters are activated. This results in premium staffing costs, rushed logistics, and operational strain.
- **Predictive Model (Vantage):** By forecasting that occupancy will exceed 85% in 14 days, the program can activate resources incrementally, negotiate standard-rate contracts, and prepare case managers, saving millions in operational costs and ensuring better care quality.

---

### 3. Data Engineering & Integrity
Real-world UAC data contains significant challenges:
- **Non-standard formatting:** Volatile naming conventions across agencies (e.g., Referrals vs. Intakes, Releases vs. Discharges).
- **Time-series gaps:** Missing reporting periods, weekends, and holidays.
- **Data Continuity:** The need for daily granularity to capture fast-moving surges.

**Vantage's pipeline resolves these by:**
1. **Dynamic Normalization:** Auto-mapping columns based on semantic keywords (e.g., mapping `Shelter_Occupancy` to `Occupancy`).
2. **Time-Based Linear Interpolation:** Reconstructing missing reporting periods to maintain mathematical continuity for advanced forecasting models.
3. **Feature Engineering:** Creating high-signal operational indicators:
   - `Net_Pressure = Intake - Discharges` (quantifies the daily surplus/deficit in care load).
   - `7-Day Rolling Averages` (removes weekly administrative noise).
   - `Lags (t-1, t-7, t-14)` (captures short-term and weekly auto-regressive patterns).

---

### 4. Predictive Modeling Framework
Vantage evaluates four distinct modeling approaches to balance simplicity, seasonality, and non-linear patterns:

1. **Naïve Persistence Model (Baseline):**
   - *Logic:* $y_{t+h} = y_t$.
   - *Purpose:* Serves as the bottom-line benchmark. Any ML model must outperform this to justify implementation.
2. **SARIMA (Statistical):**
   - *Logic:* Autoregressive Integrated Moving Average with Seasonality.
   - *Purpose:* Captures the strong weekly administration cycles (low weekend discharges/intakes) and annual spring-summer cycles.
3. **Random Forest & XGBoost (Machine Learning):**
   - *Logic:* Non-linear ensemble methods using engineered features (lags, rolling averages, calendar markers).
   - *Purpose:* Models complex, non-linear relationships, such as sudden surges and interactive bottlenecks.

#### Validation & Robustness
Models are evaluated using **Walk-Forward Validation (Expanding Window)**. Instead of a single train-test split, the models are iteratively trained on an expanding history and tested on a rolling forward horizon (e.g., 30 days). This mimics real-world deployment and prevents lookahead bias.
Metrics tracked include:
- **Mean Absolute Error (MAE):** Average absolute deviation in bed counts.
- **Root Mean Squared Error (RMSE):** Penalizes larger forecasting errors (critical for capacity planning).
- **Mean Absolute Percentage Error (MAPE):** Relative error scale.

---

### 5. Operations Dashboard & Simulation Engine
The Vantage dashboard is built in Streamlit, serving as the user-facing command center:
1. **Forecast Module:** Interactive visualizations showing projected occupancy with a 90% confidence interval.
2. **Demand Panel:** Early-warning indicator highlighting when occupancy is projected to cross the critical **85% capacity threshold**.
3. **Simulation Engine:** A "What-If" sandboxed environment. Planners can toggle intake increases (e.g., +40% surge) or discharge restrictions to simulate shelter stress and resource exhaustion under stress-test scenarios.
4. **Governance & Explainability:** Integrates SHAP (SHapley Additive exPlanations) and Feature Importance, showing stakeholders exactly *why* a model is predicting a surge (e.g., identifying whether the forecast is driven by an inflow spike or a discharge backlog).

---

### 6. Implementation Checklist
- [x] Create project file structure and configure environment (`requirements.txt`).
- [x] Implement synthetic dataset generator for operational validation.
- [ ] Implement `data_processing.py` to normalize and clean raw UAC files.
- [ ] Implement `modeling.py` with SARIMA, Random Forest, and XGBoost.
- [ ] Integrate components into the user-facing `app.py` Streamlit dashboard.
- [ ] Execute walk-forward benchmarking and record performance metrics.
