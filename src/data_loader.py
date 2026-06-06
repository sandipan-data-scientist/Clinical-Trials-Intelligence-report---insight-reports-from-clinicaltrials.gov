"""
DataLoader
Responsible solely for reading the raw CSV from disk and
returning a consistent raw dataframe. No transformation here.
OOP pattern: single-responsibility class with a cached load method.
"""

import pandas as pd
import os


class DataLoader:
    """
    Loads the raw ClinicalTrials.gov dataset from a given path.
    Keeps the raw data intact; all cleaning is delegated to DataPreprocessor.
    """

    DEFAULT_PATH = os.path.join("data", "raw_ct_data.csv")

    def __init__(self, filepath: str = None):
        self.filepath = filepath or self.DEFAULT_PATH
        self._df_raw = None

    def load(self) -> pd.DataFrame:
        """
        Reads the CSV and returns a copy of the raw dataframe.
        Raises FileNotFoundError with a clear message if the file is missing.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset not found at '{self.filepath}'. "
                "Place raw_ct_data.csv inside the data/ folder."
            )
        self._df_raw = pd.read_csv(self.filepath, low_memory=False)
        return self._df_raw.copy()

    @property
    def shape(self):
        if self._df_raw is None:
            raise RuntimeError("Call load() before accessing shape.")
        return self._df_raw.shape

    @property
    def columns(self):
        if self._df_raw is None:
            raise RuntimeError("Call load() before accessing columns.")
        return self._df_raw.columns.tolist()