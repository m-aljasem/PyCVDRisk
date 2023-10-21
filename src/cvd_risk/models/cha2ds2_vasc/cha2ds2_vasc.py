"""
CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk.

The CHA2DS2-VASc score is an enhanced version of CHADS2 that incorporates
additional stroke risk factors for a more comprehensive risk assessment
in patients with atrial fibrillation.

Reference:
    Lip GY, Nieuwlaat R, Pisters R, Lane DA, Crijns HJ. Refining clinical
    risk stratification for predicting stroke and thromboembolism in atrial
    fibrillation using a novel risk factor-based approach: the euro heart
    survey on atrial fibrillation. Chest. 2010;137(2):263-272.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)


class CHA2DS2_VASc(RiskModel):
    """
    CHA2DS2-VASc Score for atrial fibrillation stroke risk assessment.

    The CHA2DS2-VASc score ranges from 0-9 points and provides a more
    comprehensive assessment than CHADS2 by including additional risk factors.

    Scoring:
    - Congestive heart failure/LV dysfunction: 1 point
    - Hypertension: 1 point
    - Age ≥75 years: 2 points
    - Diabetes mellitus: 1 point
    - Prior stroke/TIA/thromboembolism: 2 points
    - Vascular disease (prior MI, PAD, aortic plaque): 1 point
    - Age 65-74 years: 1 point
    - Sex category (female): 1 point

    Risk categories:
    - 0 points: Low risk (0% per year, anticoagulation may not be needed)
    - 1 point: Moderate risk (1.3% per year)
    - ≥2 points: High risk (≥2.2% per year, anticoagulation recommended)

    Parameters
    ----------
    age : int
        Age in years.
    sex : Literal["male", "female"]
        Biological sex (females get 1 point).
    congestive_heart_failure : bool
        History of congestive heart failure or LV dysfunction.
    hypertension : bool
        History of hypertension.
    diabetes : bool
        Diabetes mellitus.
    prior_stroke_tia : bool
        Prior stroke, TIA, or thromboembolism.
    vascular_disease : bool
        Vascular disease (prior MI, PAD, or aortic plaque).

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: CHA2DS2-VASc score (0-9)
        - risk_category: Risk classification ("low", "moderate", "high")
        - model_name: "CHA2DS2-VASc"
        - model_version: "2010"
    """

    model_name = "CHA2DS2-VASc"
    model_version = "2010"
    supported_regions = None  # Global applicability for AF patients

    def __init__(self) -> None:
        """Initialize CHA2DS2-VASc model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for CHA2DS2-VASc requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for CHA2DS2-VASc calculation.
        """
        super().validate_input(patient)

        # CHA2DS2-VASc requires specific fields
        required_fields = [
            "congestive_heart_failure", "hypertension", "diabetes",
            "prior_stroke_tia", "vascular_disease"
        ]

        missing_fields = []
        for field in required_fields:
            if getattr(patient, field, None) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"CHA2DS2-VASc model requires the following fields: {missing_fields}. "
                f"Patient is missing: {missing_fields}"
            )

    def _calculate_congestive_heart_failure_score(self, congestive_heart_failure: bool) -> int:
        """Calculate congestive heart failure component."""
        return 1 if congestive_heart_failure else 0

    def _calculate_hypertension_score(self, hypertension: bool) -> int:
        """Calculate hypertension component."""
        return 1 if hypertension else 0

    def _calculate_age_score(self, age: int) -> int:
        """Calculate age components (65-74 = 1 point, ≥75 = 2 points)."""
        if age >= 75:
            return 2
        elif age >= 65:
            return 1
        else:
            return 0

    def _calculate_diabetes_score(self, diabetes: bool) -> int:
        """Calculate diabetes component."""
        return 1 if diabetes else 0

    def _calculate_stroke_score(self, prior_stroke_tia: bool) -> int:
        """Calculate prior stroke/TIA component (2 points)."""
        return 2 if prior_stroke_tia else 0

    def _calculate_vascular_disease_score(self, vascular_disease: bool) -> int:
        """Calculate vascular disease component."""
        return 1 if vascular_disease else 0

    def _calculate_sex_score(self, sex: str) -> int:
        """Calculate sex category component (female = 1 point)."""
        return 1 if sex.lower() == "female" else 0

    def _get_risk_category(self, score: int) -> str:
        """Get risk category based on CHA2DS2-VASc score."""
        if score == 0:
            return "low"
        elif score == 1:
            return "moderate"
        else:  # score >= 2
            return "high"

    def _get_annual_stroke_risk(self, score: int) -> float:
        """Get estimated annual stroke risk percentage."""
        risk_table = {
            0: 0.0,
            1: 1.3,
            2: 2.2,
            3: 3.2,
            4: 4.0,
            5: 6.7,
            6: 9.8,
            7: 9.6,
            8: 6.7,
            9: 15.2
        }
        return risk_table.get(score, 15.2)  # Default to highest risk

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate CHA2DS2-VASc score for atrial fibrillation stroke risk.

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
        chf_score = self._calculate_congestive_heart_failure_score(patient.congestive_heart_failure)
        hypertension_score = self._calculate_hypertension_score(patient.hypertension)
        age_score = self._calculate_age_score(patient.age)
        diabetes_score = self._calculate_diabetes_score(patient.diabetes)
        stroke_score = self._calculate_stroke_score(patient.prior_stroke_tia)
        vascular_score = self._calculate_vascular_disease_score(patient.vascular_disease)
        sex_score = self._calculate_sex_score(patient.sex)

        # Total CHA2DS2-VASc score
        total_score = (
            chf_score + hypertension_score + age_score +
            diabetes_score + stroke_score + vascular_score + sex_score
        )

        # Ensure score is within valid range
        total_score = max(0, min(9, total_score))

        # Get risk category and annual risk
        risk_category = self._get_risk_category(total_score)
        annual_risk = self._get_annual_stroke_risk(total_score)

        # Create metadata with component scores
        metadata = {
            "chf_score": chf_score,
            "hypertension_score": hypertension_score,
            "age_score": age_score,
            "diabetes_score": diabetes_score,
            "stroke_score": stroke_score,
            "vascular_score": vascular_score,
            "sex_score": sex_score,
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
            "age", "sex", "congestive_heart_failure", "hypertension",
            "diabetes", "prior_stroke_tia", "vascular_disease"
        ]

        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Vectorized calculations
        chf_scores = results["congestive_heart_failure"].astype(int)
        hypertension_scores = results["hypertension"].astype(int)

        # Age scoring (65-74 = 1, ≥75 = 2)
        age_scores = np.zeros(len(results), dtype=int)
        age_scores[(results["age"] >= 65) & (results["age"] < 75)] = 1
        age_scores[results["age"] >= 75] = 2

        diabetes_scores = results["diabetes"].astype(int)
        stroke_scores = results["prior_stroke_tia"].astype(int) * 2
        vascular_scores = results["vascular_disease"].astype(int)
        sex_scores = (results["sex"].str.lower() == "female").astype(int)

        # Calculate total scores
        total_scores = (
            chf_scores + hypertension_scores + age_scores +
            diabetes_scores + stroke_scores + vascular_scores + sex_scores
        )

        # Ensure scores are within valid range
        total_scores = np.clip(total_scores, 0, 9)

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
