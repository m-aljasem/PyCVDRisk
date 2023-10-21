"""
CHADS2 Score for Atrial Fibrillation Stroke Risk.

The CHADS2 score is a clinical prediction tool used to estimate the risk
of stroke in patients with non-valvular atrial fibrillation.

Reference:
    Gage BF, Waterman AD, Shannon W, et al. Validation of clinical
    classification schemes for predicting stroke: results from the
    National Registry of Atrial Fibrillation. JAMA. 2001;285(22):2864-2870.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)


class CHADS2(RiskModel):
    """
    CHADS2 Score for atrial fibrillation stroke risk assessment.

    The CHADS2 score ranges from 0-6 points and stratifies patients into
    low, moderate, and high risk categories for ischemic stroke.

    Scoring:
    - Congestive heart failure: 1 point
    - Hypertension: 1 point
    - Age ≥75 years: 1 point
    - Diabetes mellitus: 1 point
    - Prior stroke or TIA: 2 points

    Risk categories:
    - 0 points: Low risk (<1.9% per year)
    - 1 point: Moderate risk (2.8% per year)
    - ≥2 points: High risk (≥4% per year)

    Parameters
    ----------
    age : int
        Age in years.
    congestive_heart_failure : bool
        History of congestive heart failure.
    hypertension : bool
        History of hypertension.
    diabetes : bool
        Diabetes mellitus.
    prior_stroke_tia : bool
        Prior stroke or transient ischemic attack.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: CHADS2 score (0-6)
        - risk_category: Risk classification ("low", "moderate", "high")
        - model_name: "CHADS2"
        - model_version: "2001"
    """

    model_name = "CHADS2"
    model_version = "2001"
    supported_regions = None  # Global applicability for AF patients

    def __init__(self) -> None:
        """Initialize CHADS2 model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for CHADS2 requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for CHADS2 calculation.
        """
        super().validate_input(patient)

        # CHADS2 requires specific fields
        required_fields = [
            "congestive_heart_failure", "hypertension", "diabetes", "prior_stroke_tia"
        ]

        missing_fields = []
        for field in required_fields:
            if getattr(patient, field, None) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"CHADS2 model requires the following fields: {missing_fields}. "
                f"Patient is missing: {missing_fields}"
            )

    def _calculate_age_score(self, age: int) -> int:
        """Calculate age component (≥75 years = 1 point)."""
        return 1 if age >= 75 else 0

    def _calculate_chf_score(self, congestive_heart_failure: bool) -> int:
        """Calculate congestive heart failure component."""
        return 1 if congestive_heart_failure else 0

    def _calculate_hypertension_score(self, hypertension: bool) -> int:
        """Calculate hypertension component."""
        return 1 if hypertension else 0

    def _calculate_diabetes_score(self, diabetes: bool) -> int:
        """Calculate diabetes component."""
        return 1 if diabetes else 0

    def _calculate_stroke_score(self, prior_stroke_tia: bool) -> int:
        """Calculate prior stroke/TIA component (2 points)."""
        return 2 if prior_stroke_tia else 0

    def _get_risk_category(self, score: int) -> str:
        """Get risk category based on CHADS2 score."""
        if score == 0:
            return "low"
        elif score == 1:
            return "moderate"
        else:  # score >= 2
            return "high"

    def _get_annual_stroke_risk(self, score: int) -> float:
        """Get estimated annual stroke risk percentage."""
        risk_table = {
            0: 1.9,
            1: 2.8,
            2: 4.0,
            3: 5.9,
            4: 8.5,
            5: 12.5,
            6: 18.2
        }
        return risk_table.get(score, 18.2)  # Default to highest risk for scores > 6

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate CHADS2 score for atrial fibrillation stroke risk.

        Parameters
        ----------
        patient : PatientData
            Validated patient data.

        Returns
        -------
        RiskResult
            Risk calculation result with metadata.
        """
        self.validate_input(patient)

        # Calculate individual components
        age_score = self._calculate_age_score(patient.age)
        chf_score = self._calculate_chf_score(patient.congestive_heart_failure)
        hypertension_score = self._calculate_hypertension_score(patient.hypertension)
        diabetes_score = self._calculate_diabetes_score(patient.diabetes)
        stroke_score = self._calculate_stroke_score(patient.prior_stroke_tia)

        # Total CHADS2 score
        total_score = (
            age_score + chf_score + hypertension_score +
            diabetes_score + stroke_score
        )

        # Ensure score is within valid range
        total_score = max(0, min(6, total_score))

        # Get risk category and annual risk
        risk_category = self._get_risk_category(total_score)
        annual_risk = self._get_annual_stroke_risk(total_score)

        # Create metadata with component scores
        metadata = {
            "age_score": age_score,
            "chf_score": chf_score,
            "hypertension_score": hypertension_score,
            "diabetes_score": diabetes_score,
            "stroke_score": stroke_score,
            "total_score": total_score,
            "annual_stroke_risk_percent": annual_risk
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
            "age", "congestive_heart_failure", "hypertension",
            "diabetes", "prior_stroke_tia"
        ]

        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Vectorized calculations
        age_scores = (results["age"] >= 75).astype(int)
        chf_scores = results["congestive_heart_failure"].astype(int)
        hypertension_scores = results["hypertension"].astype(int)
        diabetes_scores = results["diabetes"].astype(int)
        stroke_scores = results["prior_stroke_tia"].astype(int) * 2

        # Calculate total scores
        total_scores = (
            age_scores + chf_scores + hypertension_scores +
            diabetes_scores + stroke_scores
        )

        # Ensure scores are within valid range
        total_scores = np.clip(total_scores, 0, 6)

        # Vectorized categorization
        category_conditions = [
            total_scores == 0,
            total_scores == 1,
            total_scores >= 2
        ]
        category_choices = ["low", "moderate", "high"]

        results["risk_score"] = total_scores
        results["risk_category"] = np.select(category_conditions, category_choices, default="high")

        return results
