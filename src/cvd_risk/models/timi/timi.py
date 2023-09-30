"""
TIMI (Thrombolysis In Myocardial Infarction) Risk Score for UA/NSTEMI.

The TIMI risk score is a clinical prediction tool used to stratify risk
in patients with unstable angina (UA) and non-ST elevation myocardial
infarction (NSTEMI).

Reference:
    Antman EM, Cohen M, Bernink PJ, et al. The TIMI risk score for unstable
    angina/non-ST elevation MI: A method for prognostication and therapeutic
    decision making. JAMA. 2000;284(7):835-842.
    DOI: 10.1001/jama.284.7.835
"""

import logging
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# TIMI risk score categories
_TIMI_CATEGORIES = {
    "very_low": "Very low risk",
    "low": "Low risk",
    "moderate": "Moderate risk",
    "high": "High risk"
}


class TIMI(RiskModel):
    """
    TIMI Risk Score for UA/NSTEMI patients.

    The TIMI risk score ranges from 0-7 points and stratifies patients into
    four risk categories for mortality and ischemic events.

    Valid for patients presenting with unstable angina or NSTEMI.
    """

    model_name = "TIMI"
    model_version = "2000"
    supported_regions = None  # TIMI is not region-specific

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)

        # TIMI-specific validation
        required_fields = [
            "diabetes", "hypertension", "hyperlipidaemia", "family_history",
            "previous_pci", "previous_cabg", "aspirin_use", "angina_episodes_24h",
            "ecg_st_depression", "troponin_level"
        ]

        missing_fields = []
        for field in required_fields:
            if getattr(patient, field, None) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"TIMI model requires the following fields: {missing_fields}. "
                f"Patient is missing: {missing_fields}"
            )

        # Validate angina episodes
        if patient.angina_episodes_24h is not None and patient.angina_episodes_24h < 0:
            raise ValueError("Number of angina episodes cannot be negative")

        # Validate troponin level
        if patient.troponin_level is not None and patient.troponin_level < 0:
            raise ValueError("Troponin level cannot be negative")

    def _calculate_age_score(self, age: int) -> int:
        """Calculate age component of TIMI score."""
        if age < 65:
            return 0
        elif 65 <= age <= 74:
            return 2
        else:  # age >= 75
            return 3

    def _calculate_risk_factors_score(self, patient: PatientData) -> int:
        """Calculate risk factors component (≥3 risk factors = 1 point)."""
        risk_factors = [
            patient.diabetes,
            patient.smoking,  # Note: PatientData uses 'smoking', R code uses 'smoker'
            patient.hypertension,
            patient.hyperlipidaemia,
            patient.family_history
        ]

        # Count True values (None values are treated as False)
        risk_factor_count = sum(1 for rf in risk_factors if rf is True)

        return 1 if risk_factor_count >= 3 else 0

    def _calculate_cad_history_score(self, patient: PatientData) -> int:
        """Calculate CAD history component (previous PCI or CABG = 1 point)."""
        has_previous_pci = patient.previous_pci is True
        has_previous_cabg = patient.previous_cabg is True

        return 1 if (has_previous_pci or has_previous_cabg) else 0

    def _calculate_aspirin_score(self, patient: PatientData) -> int:
        """Calculate aspirin use component."""
        return 1 if patient.aspirin_use is True else 0

    def _calculate_angina_score(self, patient: PatientData) -> int:
        """Calculate severe angina component (≥2 episodes in 24h = 1 point)."""
        episodes = patient.angina_episodes_24h or 0
        return 1 if episodes >= 2 else 0

    def _calculate_ecg_score(self, patient: PatientData) -> int:
        """Calculate ECG ST depression component."""
        return 1 if patient.ecg_st_depression is True else 0

    def _calculate_marker_score(self, patient: PatientData) -> int:
        """Calculate cardiac marker component based on gender and troponin."""
        troponin = patient.troponin_level or 0

        if patient.sex == "male":
            return 1 if troponin >= 34 else 0
        else:  # female
            return 1 if troponin >= 16 else 0

    def _get_risk_category(self, score: int) -> str:
        """Get risk category based on TIMI score."""
        if score == 0:
            return "very_low"
        elif 1 <= score <= 2:
            return "low"
        elif 3 <= score <= 4:
            return "moderate"
        else:  # score >= 5
            return "high"

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # Calculate individual components
        age_score = self._calculate_age_score(patient.age)
        risk_factors_score = self._calculate_risk_factors_score(patient)
        cad_history_score = self._calculate_cad_history_score(patient)
        aspirin_score = self._calculate_aspirin_score(patient)
        angina_score = self._calculate_angina_score(patient)
        ecg_score = self._calculate_ecg_score(patient)
        marker_score = self._calculate_marker_score(patient)

        # Total TIMI score
        total_score = (
            age_score + risk_factors_score + cad_history_score +
            aspirin_score + angina_score + ecg_score + marker_score
        )

        # Ensure score is within valid range
        total_score = max(0, min(7, total_score))

        # Get risk category
        risk_category = self._get_risk_category(total_score)

        # Create metadata with component scores
        metadata = {
            "age_score": age_score,
            "risk_factors_score": risk_factors_score,
            "cad_history_score": cad_history_score,
            "aspirin_score": aspirin_score,
            "angina_score": angina_score,
            "ecg_score": ecg_score,
            "marker_score": marker_score,
            "total_score": total_score
        }

        return RiskResult(
            risk_score=float(total_score),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        """
        required = [
            "age", "sex", "diabetes", "smoking", "hypertension", "hyperlipidaemia",
            "family_history", "previous_pci", "previous_cabg", "aspirin_use",
            "angina_episodes_24h", "ecg_st_depression", "troponin_level"
        ]

        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Vectorized age scoring
        age_conditions = [
            results["age"] < 65,
            (results["age"] >= 65) & (results["age"] <= 74),
            results["age"] >= 75
        ]
        age_choices = [0, 2, 3]
        age_scores = np.select(age_conditions, age_choices, default=0)

        # Vectorized risk factors scoring (count >= 3)
        risk_factor_cols = ["diabetes", "smoking", "hypertension", "hyperlipidaemia", "family_history"]
        risk_factor_sum = results[risk_factor_cols].sum(axis=1)
        risk_factors_scores = (risk_factor_sum >= 3).astype(int)

        # Vectorized CAD history scoring (PCI or CABG)
        cad_history_scores = ((results["previous_pci"] == True) | (results["previous_cabg"] == True)).astype(int)

        # Vectorized aspirin scoring
        aspirin_scores = (results["aspirin_use"] == True).astype(int)

        # Vectorized angina scoring (>=2 episodes)
        angina_scores = (results["angina_episodes_24h"].fillna(0) >= 2).astype(int)

        # Vectorized ECG scoring
        ecg_scores = (results["ecg_st_depression"] == True).astype(int)

        # Vectorized cardiac marker scoring (gender-specific troponin thresholds)
        troponin = results["troponin_level"].fillna(0)
        male_mask = results["sex"].str.lower() == "male"
        female_mask = results["sex"].str.lower() == "female"

        marker_scores = np.zeros(len(results), dtype=int)
        marker_scores[male_mask] = (troponin[male_mask] >= 34).astype(int)
        marker_scores[female_mask] = (troponin[female_mask] >= 16).astype(int)

        # Calculate total scores
        total_scores = (
            age_scores + risk_factors_scores + cad_history_scores +
            aspirin_scores + angina_scores + ecg_scores + marker_scores
        )

        # Ensure scores are within valid range
        total_scores = np.clip(total_scores, 0, 7)

        # Vectorized categorization
        category_conditions = [
            total_scores == 0,
            (total_scores >= 1) & (total_scores <= 2),
            (total_scores >= 3) & (total_scores <= 4),
            total_scores >= 5
        ]
        category_choices = ["very_low", "low", "moderate", "high"]

        results["risk_score"] = total_scores
        results["risk_category"] = np.select(category_conditions, category_choices, default="unknown")

        return results
