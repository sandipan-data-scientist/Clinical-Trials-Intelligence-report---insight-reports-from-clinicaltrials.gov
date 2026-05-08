# Clinical Trials Strategic Intelligence Platform (2005-2025)

End-to-end healthcare data science project analysing 18,000+ clinical trials from ClinicalTrials.gov. Covers sponsor-level R&D investment signals, trial failure pattern analysis, drug development bottlenecks, competitive landscaping, and a 20-year strategic forecast, all packaged into an interactive notebook and a production-ready REST API.

---

## What This Project Does

Raw clinical trial registration data from ClinicalTrials.gov is messy, incomplete, and spread across 30+ columns. This project turns it into actionable intelligence across three layers:

**Analysis layer (Jupyter notebook):** A 16-section, fully narrated pipeline from raw CSV to published charts. Every statistical and machine learning decision is explained in plain language with mathematical justification, written to be readable by a fresher data scientist and interpretable by a non-technical business audience at the same time.

**Insight layer (dashboards and reports):** A single-figure competitive intelligence dashboard summarising the entire 20-year dataset, plus a reusable sponsor-level report function that regenerates nine charts for any named sponsor by changing one variable.

**Deployment layer (FastAPI + Docker):** The trained XGBoost model is served via a containerised REST API with interactive Swagger documentation. Any team member — technical or not — can query trial termination risk from a browser or a curl command.

---

## Project Structure

```
clinical-trials-intelligence/
│
├── clinical_trials_analysis.ipynb      Main notebook, run top to bottom
├── raw_ct_data.csv                     Source dataset (ClinicalTrials.gov)
├── requirements.txt                    Full project Python dependencies
│
├── api/
│   ├── main.py                         FastAPI application (5 endpoints)
│   ├── Dockerfile                      Container build instructions
│   ├── docker-compose.yml              One-command local deployment
│   ├── requirements.txt                API-only pinned dependencies
│   ├── README.md                       Full API usage guide
│   └── model_artifacts/
│       ├── trial_outcome_model.joblib  Trained XGBoost classifier
│       ├── scaler.joblib               StandardScaler (LR fallback)
│       ├── le_funder.joblib            LabelEncoder: funder type
│       ├── le_ta.joblib                LabelEncoder: therapeutic area
│       ├── le_int.joblib               LabelEncoder: intervention type
│       ├── le_pp.joblib                LabelEncoder: primary purpose
│       ├── sponsor_summary.csv         Aggregated sponsor statistics
│       └── metadata.json              Model metadata and feature list
│
└── fig_*.png                           All generated chart exports
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
```

---

## Quick Start

### Requirements

- Python 3.10 or higher
- Docker Desktop (Windows / Mac) or Docker Engine (Linux) — only needed for the API
- 8 GB RAM recommended for full notebook execution

### 1. Clone the repository

```bash
git clone https://github.com/your-username/clinical-trials-intelligence.git
cd clinical-trials-intelligence
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook clinical_trials_analysis.ipynb
```

Open the notebook in your browser, then run all cells from top to bottom (Kernel > Restart and Run All). Execution takes roughly 3 to 5 minutes end to end.

### 4. Start the API with Docker

```bash
cd api
docker-compose up -d
```

Verify it is running:

```bash
curl http://localhost:8000/health
```

You should see:

```json
{
  "status": "ok",
  "model": "XGBoost",
  "test_auc": 0.8711,
  "feature_count": 10
}
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI where you can test every endpoint from the browser.

---

## Notebook Sections at a Glance

| Section | What It Does |
|---------|-------------|
| 1. Environment | Imports, global plot styling, random seed |
| 2. Data Loading | CSV ingestion, shape and type inspection |
| 3. Data Cleaning | Missing values, date parsing, duration flags, phase standardisation |
| 4. EDA | Volume trends, enrollment distributions, sponsor comparison, therapeutic area breakdown |
| 5. Time Series | Per-area trend lines 2005-2024, phase progression, 20-year Holt-Winters forecast |
| 6. Feature Engineering | 10 features including log-transforms, ordinal encoding, sponsor track record |
| 7. Train-Test Split | 80/20 stratified split, StandardScaler fitted on training data only |
| 8. Model Training | Logistic Regression, Random Forest, XGBoost; 5-fold CV + test evaluation |
| 9. R&D Signals | Bloom signal by therapeutic area, ROI proxy, phase failure rates |
| 10. Competitive Dashboard | Single 24x20 inch figure summarising all major dataset trends |
| 11. Sponsor Report | Reusable `generate_sponsor_report(sponsor_name)` function |
| 12. Hidden Patterns | Pipeline funnel, enrollment quartile vs success, trial start seasonality |
| 13. Key Findings | Plain-language summary of all major insights |
| 14. Model Export | Saves all artifacts to `model_artifacts/` for API use |

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

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check and model metadata |
| GET | /model/info | Full feature list, model name, test AUC |
| POST | /predict/termination | Predict termination probability for a trial |
| GET | /sponsor/{name} | Aggregated stats for a named sponsor |
| GET | /sponsors/list | All known sponsors with optional min_trials filter |
| GET | /therapeutic-areas | Recognised condition categories used for inference |

Full request schema, response schema, example curl calls, and a Python client snippet are in `api/README.md`.

---

## Reusing the Sponsor Report

The `generate_sponsor_report()` function in Section 11 of the notebook produces a full nine-panel sponsor brief. To generate a report for any sponsor:

```python
# change this one line and re-run the cell
SPONSOR_NAME = "Boehringer Ingelheim"
generate_sponsor_report(SPONSOR_NAME)
```

The function checks whether the name exists in the dataset and prints the available top sponsors if it does not find a match. The chart is automatically saved as `fig_sponsor_report_<sponsor_name>.png`.

---

## Updating the Model

If you retrain on new data or add features, the notebook saves updated artifacts automatically in Section 14. To redeploy:

```bash
cp -r model_artifacts api/model_artifacts
cd api
docker-compose down
docker-compose up -d --build
```

No changes to `main.py` are needed as long as the feature list in `metadata.json` stays the same.

---

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

## Dataset

Source: ClinicalTrials.gov public registry, extracted January 2025.

The dataset covers clinical trials registered between 2005 and 2025. Key columns include NCT Number, Study Status, Phases, Sponsor, Funder Type, Enrollment, Start Date, Completion Date, Conditions, Interventions, and Study Design. After cleaning, over 18,000 records with valid sponsor and phase information were retained. The raw file is included in the repository as `raw_ct_data.csv`.

---

## Limitations

This model captures historical patterns in public registration data. It does not have access to internal sponsor financials, regulatory feedback, interim efficacy or safety results, or competitive changes that occur after the training cutoff. Predictions are decision-support signals and should be reviewed alongside domain expertise before any resource allocation decisions are made.

---

## License

MIT License. Free to use, modify, and distribute with attribution.
