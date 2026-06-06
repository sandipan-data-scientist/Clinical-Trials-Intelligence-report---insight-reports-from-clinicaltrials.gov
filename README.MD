# Clinical Trial Sponsor Intelligence Engine

End-to-end data science platform built on ClinicalTrials.gov data (2005-2025).
Covers data cleaning, EDA, feature engineering, ML-based trial failure prediction,
20-year domain forecasting, and reusable sponsor dashboards.
Built in Python with scikit-learn, Pandas, and Matplotlib.

Prepared by Sandipan Acharjee


## Live Demo

[Click Here to check out the Platform](https://clinical-trials-intelligence-report.streamlit.app/)

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
    └── notebooks/
        └── CTgov_trial_analysis.ipynb


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

    DataLoader          Reads raw CSV, nothing else
    DataPreprocessor    Cleans dates, derives duration, encodes phases and domains
    FeatureEngineer     Adds sponsor-level aggregates, encodes categoricals for ML
    ModelTrainer        Trains, evaluates, and persists three classifiers
    Visualizer          All EDA and model evaluation charts, returns matplotlib Figure objects
    SponsorDashboard    Builds the full sponsor-specific 14-panel dashboard figure