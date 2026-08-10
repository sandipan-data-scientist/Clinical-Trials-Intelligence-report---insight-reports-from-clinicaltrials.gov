"""
ModelTrainer
Handles training, evaluation, and persistence of the trial
failure prediction model. Uses GradientBoostingClassifier as
the primary model per findings from the notebook.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, confusion_matrix
)
from typing import Dict, Tuple


ARTIFACT_DIR = "artifacts"


class ModelTrainer:
    """
    Trains and evaluates multiple classifiers on the trial failure
    prediction task. Exposes the best model and scoring utilities.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str = "Is_Failed"):
        self.feature_cols = feature_cols
        self.target_col   = target_col
        self.scaler       = StandardScaler()

        # build model-ready dataset: only rows with known outcome
        df_ml = df[df[target_col].notna()][feature_cols + [target_col]].dropna()
        self.X = df_ml[feature_cols].values
        self.y = df_ml[target_col].values.astype(int)

        # train-test split with stratification to preserve class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.20, random_state=42, stratify=self.y
        )
        self.X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_test_sc  = self.scaler.transform(self.X_test)

        self.models  = {}
        self.results = {}
        self.best_model_name = None
        self.best_model      = None

    def train_all(self) -> Dict:
        """Trains Logistic Regression, Random Forest, and Gradient Boosting."""
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(self.X_train_sc, self.y_train)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)

        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        gb.fit(self.X_train, self.y_train)

        self.models = {
            "Logistic Regression": (lr, self.X_test_sc),
            "Random Forest":       (rf, self.X_test),
            "Gradient Boosting":   (gb, self.X_test),
        }

        for name, (model, X_eval) in self.models.items():
            y_pred  = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1]
            auc     = roc_auc_score(self.y_test, y_proba)
            report  = classification_report(self.y_test, y_pred, output_dict=True)
            self.results[name] = {
                "AUC":       round(auc, 4),
                "Accuracy":  round(report["accuracy"], 4),
                "Precision": round(report["1"]["precision"], 4),
                "Recall":    round(report["1"]["recall"], 4),
                "F1":        round(report["1"]["f1-score"], 4),
            }

        # select best model by AUC
        self.best_model_name = max(self.results, key=lambda k: self.results[k]["AUC"])
        self.best_model      = self.models[self.best_model_name][0]
        return self.results

    # def score_full_dataset(self, X_all: np.ndarray) -> np.ndarray:
    #     """Returns predicted failure probabilities for every trial in the full dataset."""
    #     if self.best_model is None:
    #         raise RuntimeError("Call train_all() before scoring.")
    #     return self.best_model.predict_proba(X_all)[:, 1]

    def score_full_dataset(self, X_all):
        if self.best_model_name == "Logistic Regression":
            X_scoring = self.scaler.transform(X_all)   # LR needs scaled input
        else:
            X_scoring = X_all                           # RF and GB are scale-invariant
        return self.best_model.predict_proba(X_scoring)[:, 1]

    def get_roc_data(self) -> Dict:
        """Returns FPR/TPR arrays for each model for ROC curve plotting."""
        roc_data = {}
        for name, (model, X_eval) in self.models.items():
            y_proba = model.predict_proba(X_eval)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": self.results[name]["AUC"]}
        return roc_data

    def get_confusion_matrix(self) -> np.ndarray:
        if self.best_model is None:
            raise RuntimeError("Call train_all() before getting confusion matrix.")
        model, X_eval = self.models[self.best_model_name]
        return confusion_matrix(self.y_test, model.predict(X_eval))

    def get_feature_importances(self) -> pd.Series:
        """Returns feature importances from the best tree-based model."""
        gb_model, _ = self.models.get("Gradient Boosting", (None, None))
        if gb_model is None:
            raise RuntimeError("Gradient Boosting model not trained.")
        return pd.Series(gb_model.feature_importances_, index=self.feature_cols)

    def save_artifacts(self, le_intervention, le_domain):
        """Persists model and encoders to the artifacts/ directory."""
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        joblib.dump(self.best_model,   os.path.join(ARTIFACT_DIR, "model_gb.pkl"))
        joblib.dump(self.scaler,       os.path.join(ARTIFACT_DIR, "scaler.pkl"))
        joblib.dump(le_intervention,   os.path.join(ARTIFACT_DIR, "le_intervention.pkl"))
        joblib.dump(le_domain,         os.path.join(ARTIFACT_DIR, "le_domain.pkl"))
        joblib.dump(self.feature_cols, os.path.join(ARTIFACT_DIR, "feature_cols.pkl"))

    @staticmethod
    def load_artifacts():
        """Loads saved artifacts. Returns (model, scaler, le_intervention, le_domain, feature_cols)."""
        base = ARTIFACT_DIR
        return (
            joblib.load(os.path.join(base, "model_gb.pkl")),
            joblib.load(os.path.join(base, "scaler.pkl")),
            joblib.load(os.path.join(base, "le_intervention.pkl")),
            joblib.load(os.path.join(base, "le_domain.pkl")),
            joblib.load(os.path.join(base, "feature_cols.pkl")),
        )