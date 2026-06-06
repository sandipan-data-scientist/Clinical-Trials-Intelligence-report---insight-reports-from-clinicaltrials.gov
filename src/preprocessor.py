"""
DataPreprocessor
Handles all data cleaning, type conversion, and derived column creation.
Keeps each step as a named method so individual steps are testable
and the Streamlit app can report progress per step.
"""

import pandas as pd
import numpy as np
from typing import Tuple


DOMAIN_MAP = {
    "Oncology": [
        "cancer", "tumor", "carcinoma", "lymphoma", "leukemia",
        "melanoma", "glioma", "sarcoma", "myeloma", "neoplasm"
    ],
    "Infectious": [
        "infection", "hiv", "covid", "influenza", "hepatitis",
        "malaria", "tuberculosis", "bacterial", "viral", "candida"
    ],
    "Cardiovascular": [
        "heart", "cardiac", "hypertension", "coronary", "arrhythmia",
        "stroke", "vascular", "atherosclerosis"
    ],
    "Neurology": [
        "alzheimer", "parkinson", "epilepsy", "multiple sclerosis",
        "neurological", "dementia", "migraine", "autism"
    ],
    "Immunology": [
        "autoimmune", "lupus", "rheumatoid", "crohn", "psoriasis",
        "inflammatory bowel", "allergy", "immunology"
    ],
    "Endocrinology": [
        "diabetes", "obesity", "thyroid", "metabolic", "insulin", "endocrine"
    ],
    "Respiratory": [
        "asthma", "copd", "lung", "pulmonary", "respiratory", "bronchitis"
    ],
    "Healthy": ["healthy", "volunteer"],
}

PHASE_MAP = {
    "PHASE1":        "Phase 1",
    "EARLY_PHASE1":  "Phase 1",
    "PHASE1|PHASE2": "Phase 1/2",
    "PHASE2":        "Phase 2",
    "PHASE2|PHASE3": "Phase 2/3",
    "PHASE3":        "Phase 3",
    "PHASE4":        "Phase 4",
}

STATUS_MAP = {
    "COMPLETED":              "Completed",
    "TERMINATED":             "Terminated",
    "ACTIVE_NOT_RECRUITING":  "Active",
    "UNKNOWN":                "Unknown",
}

MASKING_RANK = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
FUNDER_RANK  = {
    "INDUSTRY": 3, "NIH": 2, "FED": 2, "OTHER_GOV": 2,
    "NETWORK": 1, "OTHER": 1, "UNKNOWN": 0, "INDIV": 0,
}


class DataPreprocessor:
    """
    Cleans raw ClinicalTrials.gov data and adds structured derived columns.
    Call run() to execute the full pipeline and retrieve a clean dataframe.
    """

    def __init__(self, df_raw: pd.DataFrame):
        self._df = df_raw.copy()

    # public API
    def run(self) -> pd.DataFrame:
        self._fill_collaborators()
        self._drop_incomplete_core()
        self._parse_dates()
        self._compute_duration()
        self._add_year()
        self._clean_phases()
        self._clean_status()
        self._build_failure_flag()
        self._extract_intervention_type()
        self._extract_masking()
        self._extract_allocation()
        self._classify_domain()
        self._parse_age_groups()
        self._parse_sex()
        self._log_enrollment()
        return self._df.copy()

    # private steps

    def _fill_collaborators(self):
        self._df["Collaborators"] = self._df["Collaborators"].fillna("None")

    def _drop_incomplete_core(self):
        required = [
            "Sponsor", "Phases", "Start Date", "Completion Date",
            "Enrollment", "Funder Type", "Study Type"
        ]
        self._df = self._df.dropna(subset=required)

    def _parse_dates(self):
        for col in ("Start Date", "Completion Date"):
            self._df[col] = pd.to_datetime(
                self._df[col], dayfirst=True, errors="coerce"
            )

    def _compute_duration(self):
        self._df["Duration_Months"] = (
            (self._df["Completion Date"] - self._df["Start Date"]).dt.days / 30.44
        ).round(1)
        self._df = self._df[self._df["Duration_Months"] > 0]

    def _add_year(self):
        self._df["Start_Year"] = self._df["Start Date"].dt.year
        self._df["Years_Since_2005"] = self._df["Start_Year"] - 2005

    def _clean_phases(self):
        self._df["Phase_Clean"] = self._df["Phases"].map(PHASE_MAP).fillna("Unknown")
        phase_rank = {
            "Phase 1": 1, "Phase 1/2": 1.5, "Phase 2": 2,
            "Phase 2/3": 2.5, "Phase 3": 3, "Phase 4": 4, "Unknown": 0,
        }
        self._df["Phase_Rank"] = self._df["Phase_Clean"].map(phase_rank)

    def _clean_status(self):
        self._df["Status_Clean"] = self._df["Study Status"].map(STATUS_MAP)

    def _build_failure_flag(self):
        self._df["Is_Failed"] = self._df["Status_Clean"].map(
            {"Terminated": 1, "Completed": 0}
        )

    def _extract_intervention_type(self):
        def _parse(text):
            if pd.isna(text):
                return "OTHER"
            text = str(text).upper()
            for t in ["BIOLOGICAL", "DRUG", "DEVICE", "PROCEDURE", "BEHAVIORAL", "DIETARY"]:
                if t in text:
                    return t
            return "OTHER"
        self._df["Intervention_Type"] = self._df["Interventions"].apply(_parse)

    def _extract_masking(self):
        def _parse(design):
            if pd.isna(design):
                return "NONE"
            d = str(design).upper()
            for level in ["QUADRUPLE", "TRIPLE", "DOUBLE", "SINGLE"]:
                if level in d:
                    return level
            return "NONE"
        self._df["Masking_Type"]  = self._df["Study Design"].apply(_parse)
        self._df["Masking_Rank"]  = self._df["Masking_Type"].map(MASKING_RANK).fillna(0)

    def _extract_allocation(self):
        def _parse(design):
            if pd.isna(design):
                return "UNKNOWN"
            d = str(design).upper()
            if "NON_RANDOMIZED" in d:
                return "NON_RANDOMIZED"
            elif "RANDOMIZED" in d:
                return "RANDOMIZED"
            return "NA"
        self._df["Allocation_Type"] = self._df["Study Design"].apply(_parse)
        self._df["Is_Randomized"]   = (self._df["Allocation_Type"] == "RANDOMIZED").astype(int)

    def _classify_domain(self):
        def _parse(text):
            if pd.isna(text):
                return "Other"
            text = str(text).lower()
            for domain, keywords in DOMAIN_MAP.items():
                for kw in keywords:
                    if kw in text:
                        return domain
            return "Other"
        self._df["Medical_Domain"] = self._df["Conditions"].apply(_parse)

    def _parse_age_groups(self):
        """
        Age column contains pipe-separated values like ADULT|OLDER_ADULT.
        We create a human-readable label and individual boolean columns.
        """
        def _label(text):
            if pd.isna(text):
                return "Not Specified"
            parts = [p.strip() for p in str(text).split("|")]
            readable = []
            for p in parts:
                if p == "CHILD":
                    readable.append("Child (<18)")
                elif p == "ADULT":
                    readable.append("Adult (18-64)")
                elif p == "OLDER_ADULT":
                    readable.append("Older Adult (65+)")
            return " + ".join(readable) if readable else "Not Specified"

        self._df["Age_Group_Label"] = self._df["Age"].apply(_label)
        self._df["Has_Child"]        = self._df["Age"].fillna("").str.contains("CHILD").astype(int)
        self._df["Has_Adult"]        = self._df["Age"].fillna("").str.contains("ADULT").astype(int)
        self._df["Has_OlderAdult"]   = self._df["Age"].fillna("").str.contains("OLDER_ADULT").astype(int)

    def _parse_sex(self):
        """
        Sex column: ALL, MALE, FEMALE. Map to clean labels.
        """
        sex_map = {"ALL": "All Sexes", "MALE": "Male Only", "FEMALE": "Female Only"}
        self._df["Sex_Clean"] = self._df["Sex"].map(sex_map).fillna("Not Specified")

    def _log_enrollment(self):
        self._df["Funder_Rank"]    = self._df["Funder Type"].map(FUNDER_RANK).fillna(0)
        self._df["Log_Enrollment"] = np.log1p(self._df["Enrollment"])