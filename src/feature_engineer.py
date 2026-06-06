"""
FeatureEngineer
Adds sponsor-level aggregate signals and encodes categorical features.
Designed to be run after DataPreprocessor.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """
    Computes sponsor-level aggregates and encodes categoricals
    so the dataframe is ready for machine learning.
    """

    FEATURE_COLS = [
        "Phase_Rank", "Log_Enrollment", "Duration_Months",
        "Sponsor_Size", "Sponsor_Hist_Fail_Rate", "Funder_Rank",
        "Masking_Rank", "Is_Randomized", "Intervention_Code",
        "Domain_Code", "Years_Since_2005",
    ]
    TARGET_COL = "Is_Failed"

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self.le_intervention = LabelEncoder()
        self.le_domain       = LabelEncoder()

    def run(self) -> pd.DataFrame:
        self._add_sponsor_size()
        self._add_sponsor_fail_rate()
        self._encode_intervention()
        self._encode_domain()
        return self._df.copy()

    def get_encoders(self):
        return self.le_intervention, self.le_domain

    def get_feature_cols(self):
        return self.FEATURE_COLS

    # private steps

    def _add_sponsor_size(self):
        counts = self._df["Sponsor"].value_counts()
        self._df["Sponsor_Size"] = self._df["Sponsor"].map(counts)

    def _add_sponsor_fail_rate(self):
        fail_rate = (
            self._df[self._df["Is_Failed"].notna()]
            .groupby("Sponsor")["Is_Failed"].mean()
        )
        global_mean = self._df["Is_Failed"].mean()
        self._df["Sponsor_Hist_Fail_Rate"] = (
            self._df["Sponsor"].map(fail_rate).fillna(global_mean)
        )

    def _encode_intervention(self):
        values = self._df["Intervention_Type"].fillna("OTHER")
        self._df["Intervention_Code"] = self.le_intervention.fit_transform(values)

    def _encode_domain(self):
        values = self._df["Medical_Domain"].fillna("Other")
        self._df["Domain_Code"] = self.le_domain.fit_transform(values)