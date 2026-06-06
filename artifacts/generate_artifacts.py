# generate_artifacts.py
# run this once from your project root to produce all pkl files in artifacts/
# usage: python generate_artifacts.py

import os
import joblib
import pandas as pd
import numpy as np
from src.data_loader      import DataLoader
from src.preprocessor     import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer    import ModelTrainer

print("Step 1: Loading raw data...")
loader = DataLoader()
df_raw = loader.load()
print(f"  Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

print("Step 2: Cleaning and preprocessing...")
preprocessor = DataPreprocessor(df_raw)
df_clean = preprocessor.run()
print(f"  Clean dataset: {df_clean.shape[0]} rows")

print("Step 3: Feature engineering...")
engineer = FeatureEngineer(df_clean)
df_feat  = engineer.run()
le_int, le_dom = engineer.get_encoders()
feature_cols   = engineer.get_feature_cols()
print(f"  Features: {feature_cols}")

print("Step 4: Training models (this takes about 30-60 seconds)...")
trainer = ModelTrainer(df_feat, feature_cols)
results = trainer.train_all()

print("\nModel results:")
for name, metrics in results.items():
    print(f"  {name}: AUC={metrics['AUC']:.3f}, F1={metrics['F1']:.3f}")

print(f"\nBest model: {trainer.best_model_name}")

print("\nStep 5: Saving artifacts...")
trainer.save_artifacts(le_int, le_dom)

# verify all files exist
expected_files = [
    "artifacts/model_gb.pkl",
    "artifacts/scaler.pkl",
    "artifacts/le_intervention.pkl",
    "artifacts/le_domain.pkl",
    "artifacts/feature_cols.pkl",
]
print("\nVerification:")
all_ok = True
for fpath in expected_files:
    exists = os.path.exists(fpath)
    size   = os.path.getsize(fpath) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {fpath}  ({size:,} bytes)")
    if not exists:
        all_ok = False

if all_ok:
    print("\nAll artifacts saved successfully. You are ready to run the app.")
else:
    print("\nSome files are missing. Check the error messages above.")