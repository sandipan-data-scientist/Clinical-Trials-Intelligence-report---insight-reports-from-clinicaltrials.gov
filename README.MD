# Clinical Trial Sponsor Intelligence Engine

End-to-end data science platform built on ClinicalTrials.gov data (2005-2025).
Covers sponsor-level R&D investment signals, trial failure pattern analysis, drug development bottlenecks, competitive landscaping, ML-based trial failure prediction and a 20-year strategic forecast, all packaged into an interactive Streamlit UI and a production-ready REST API, for delivering one click reusable sponsor dashboards.


Prepared by Sandipan Acharjee


## Live Demo

## [Click Here to check out the Platform](https://clinical-trials-intelligence-report.streamlit.app/)

## Project Structure

    clinical-trials-intelligence/
    ├── app.py                     Streamlit application (main entry point)
    ├── requirements.txt           Python dependencies for Streamlit Cloud
    ├── .streamlit/
    │   └── config.toml            Theme and server configuration
    ├── src/
    │   ├── data_loader.py         DataLoader class
    │   ├── preprocessor.py        DataPreprocessor class
    │   ├── feature_engineer.py    FeatureEngineer class
    │   ├── model_trainer.py       ModelTrainer class
    │   ├── visualizer.py          Visualizer class (all EDA charts)
    │   └── sponsor_dashboard.py   SponsorDashboard class
    ├── api/
    │   ├── app.py                 FastAPI REST inference endpoint
    │   └── Dockerfile             Docker build for API
    ├── data/
    │   └── raw_ct_data.csv        Place dataset here
    ├── artifacts/                 Model pkl files land here after notebook run
    └── IPYNB trials/              Prior Trials and Analysis framework
        └── Deliverable Reports/
            └── Project_Clinical_trials_demo_deliverable_e2e_report_v2.pdf 
        └── Figure outputs/
            ├── fig_overview.png
            ├── fig_enrollment.png
            ├── fig_timeseries.png
            ├── fig_forecast.png
            ├── fig_sponsor_comparison.png
            ├── fig_correlation.png
            ├── fig_model_performance.png
            ├── fig_feature_importance.png
            ├── fig_rnd_signals.png
            ├── fig_hidden_patterns.png
            ├── fig_competitive_dashboard.png
            └── fig_sponsor_report_*.png
        └── jupyter Notebooks/
            ├── CTgov_trial_analysis.ipynb
            └── Sponsor_report.ipynb
        └── Rootfiles/
            ├── mysql_queries.sql  
            └── Root.py
        └── Sponsor dashboards/
            └── Pfizer/Merck/GSK/....850+ Sponsors reports 


## Quickstart (Local)

    git clone https://github.com/YOUR_USERNAME/clinical-trials-intelligence.git
    cd clinical-trials-intelligence
    pip install -r requirements.txt
    # place raw_ct_data.csv in data/
    streamlit run app.py


## Running the FastAPI Service

Without Docker:

    pip install fastapi uvicorn
    # run the notebook to generate pkl files in artifacts/ first
    uvicorn api.app:app --host 0.0.0.0 --port 8000

With Docker:

    docker build -t ct-predictor .
    docker run -d -p 8000:8000 --name ct-api ct-predictor

API docs available at http://localhost:8000/docs


## Example Prediction Request

    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "phase_rank": 3,
        "log_enrollment": 5.5,
        "duration_months": 36,
        "sponsor_size": 150,
        "sponsor_hist_fail_rate": 0.12,
        "funder_rank": 3,
        "masking_rank": 4,
        "is_randomized": 1,
        "intervention_type": "DRUG",
        "medical_domain": "Oncology",
        "years_since_2005": 18
      }'


## Module Design (OOP)

Each class has a single responsibility:

    DataLoader          Reads raw CSV
    DataPreprocessor    Cleans dates, derives duration, encodes phases and domains
    FeatureEngineer     Adds sponsor-level aggregates, encodes categoricals for ML
    ModelTrainer        Trains, evaluates, and persists three classifiers
    Visualizer          All EDA and model evaluation charts, returns matplotlib Figure objects
    SponsorDashboard    Builds the full sponsor-specific 14-panel dashboard figure


## Tech Stack

| Layer | Tools |
|-------|-------|
| Data handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Time series | Statsmodels (Holt-Winters Exponential Smoothing) |
| Machine learning | Scikit-learn, XGBoost |
| Model persistence | Joblib |
| API framework | FastAPI, Pydantic, Uvicorn |
| Containerisation | Docker, docker-compose |
| Notebook | Jupyter |

---

## Machine Learning Model

The goal is to predict at trial registration time whether a trial will be terminated early, using only information available when the trial is first registered.

### Target Variable

`is_terminated` — binary. 1 if `Study Status == TERMINATED`, 0 if `Study Status == COMPLETED`. Trials with other statuses (ACTIVE, UNKNOWN) are excluded from the classification task.

### Features

| Feature | Description |
|---------|-------------|
| log_enrollment | log10 of planned enrollment (normalises extreme skew) |
| log_duration_days | log10 of expected trial duration |
| phase_num | Ordinal phase encoding: Phase 1=1, Phase 2=2, Phase 3=3, Phase 4=4 |
| funder_enc | Label-encoded funder type (INDUSTRY, NIH, FED, etc.) |
| ta_enc | Label-encoded therapeutic area inferred from condition text |
| interv_enc | Label-encoded intervention type (Drug, Biological, Device, etc.) |
| purpose_enc | Label-encoded primary purpose (TREATMENT, PREVENTION, etc.) |
| start_year_feat | Trial start year (captures regulatory era) |
| sponsor_hist_term_rate | Sponsor's historical termination rate computed from training data only |
| log_sponsor_volume | log10 of sponsor's total trial count (size proxy) |

### Results

| Model | Test AUC | Accuracy | F1 Score |
|-------|----------|----------|----------|
| Logistic Regression | ~0.76 | ~0.76 | ~0.74 |
| Random Forest | ~0.84 | ~0.80 | ~0.79 |
| XGBoost | 0.87 | ~0.82 | ~0.81 |

XGBoost was selected as the production model. AUC was chosen as the primary metric rather than accuracy because the dataset is imbalanced (roughly 80% completed trials, 20% terminated). A model that always predicts "completed" achieves 80% accuracy but has zero practical value — AUC penalises this correctly.

The two strongest predictors were `sponsor_hist_term_rate` (the sponsor's track record matters most) and `log_enrollment` (larger trials are harder to terminate once started).

---

## Key Insights from the Analysis

**Therapeutic area bloom signal:** Infectious Disease trials surged significantly in 2020-2024 driven by COVID-19. Oncology remains the single largest area by trial count. Metabolic and Neurology/Psychiatry are showing steady compound growth and are projected to increase their share through 2044.

**Where trials fail:** Phase 2 carries the highest absolute number of terminations. Phase 3 failures are fewer but far more costly given the larger enrolled populations and longer durations. Phase 1 has the lowest termination rate, which makes sense as early-phase trials often have exploratory objectives.

**Sponsor performance signals:** Trial volume alone does not indicate R&D efficiency. Sponsors with moderate portfolios but high completion rates and large median enrollments consistently outperform high-volume sponsors on the ROI signal metric.

**20-year forecast:** The Holt-Winters model projects continued growth through 2044. Prediction intervals widen substantially beyond 2030, reflecting genuine uncertainty. The forecast should be treated as a directional signal for strategic planning, not a point estimate.

---

## Dataset

Source: ClinicalTrials.gov public registry, extracted January 2025.

The dataset covers clinical trials registered between 2005 and 2025. Key columns include NCT Number, Study Status, Phases, Sponsor, Funder Type, Enrollment, Start Date, Completion Date, Conditions, Interventions, and Study Design. After cleaning, over 18,000 records with valid sponsor and phase information were retained. The raw file is included in the repository as `raw_ct_data.csv`.

---

## Limitations

This model captures historical patterns in public registration data. It does not have access to internal sponsor financials, regulatory feedback, interim efficacy or safety results, or competitive changes that occur after the training cutoff. Predictions are decision-support signals and should be reviewed alongside domain expertise before any resource allocation decisions are made.

---

## License

MIT License. Free to use, modify, and distribute with attribution