"""
HAS-BLED Score for Anticoagulation Bleeding Risk.

The HAS-BLED score is a clinical prediction tool used to assess the risk
of major bleeding in patients receiving anticoagulation therapy, particularly
for atrial fibrillation.

Reference:
    Pisters R, Lane DA, Nieuwlaat R, et al. A novel user-friendly score
    (HAS-BLED) to assess 1-year risk of major bleeding in patients with
    atrial fibrillation: the Euro Heart Survey. Chest. 2010;138(5):1093-1100.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)


class HAS_BLED(RiskModel):
    """
    HAS-BLED Score for anticoagulation bleeding risk assessment.

    The HAS-BLED score ranges from 0-9 points and helps clinicians make
    informed decisions about anticoagulation therapy by assessing bleeding risk.

    Scoring:
    - Hypertension (uncontrolled): 1 point
    - Abnormal renal/liver function: 1 point each (max 2)
    - Stroke history: 1 point
    - Bleeding history or predisposition: 1 point
    - Labile INR: 1 point
    - Elderly (age ≥65): 1 point
    - Drugs/alcohol: 1 point each (max 2)

    Risk categories:
    - 0-1 points: Low risk (1.1% per year)
    - 2 points: Moderate risk (1.9% per year)
    - ≥3 points: High risk (3.7-8.7% per year)

    Parameters
    ----------
    age : int
        Age in years.
    hypertension : bool
        Uncontrolled hypertension (SBP >160 mmHg).
    abnormal_renal_function : bool
        Abnormal renal function (Cr >2.26 mg/dL or dialysis).
    abnormal_liver_function : bool
        Abnormal liver function (cirrhosis, bilirubin >2x ULN, etc.).
    stroke_history : bool
        History of stroke.
    bleeding_history : bool
        History of major bleeding or bleeding predisposition.
    labile_inr : bool
        Labile INR (unstable, high INR, or poor TTR <60%).
    antiplatelet_drugs : bool
        Concomitant use of antiplatelet drugs.
    alcohol_abuse : bool
        Alcohol abuse (>8 drinks/week).

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: HAS-BLED score (0-9)
        - risk_category: Risk classification ("low", "moderate", "high")
        - model_name: "HAS-BLED"
        - model_version: "2010"
    """

    model_name = "HAS-BLED"
    model_version = "2010"
    supported_regions = None  # Global applicability for anticoagulated patients

    def __init__(self) -> None:
        """Initialize HAS-BLED model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for HAS-BLED requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for HAS-BLED calculation.
        """
        super().validate_input(patient)

        # HAS-BLED requires specific fields that may not be in standard PatientData
        # We'll make most fields optional and provide defaults
        logger.info("HAS-BLED validation: Some fields may use default values if not provided")

    def _calculate_hypertension_score(self, hypertension: Optional[bool]) -> int:
        """Calculate uncontrolled hypertension component."""
        return 1 if hypertension else 0

    def _calculate_renal_liver_score(self, abnormal_renal: Optional[bool],
                                   abnormal_liver: Optional[bool]) -> int:
        """Calculate abnormal renal/liver function component (max 2 points)."""
        renal_score = 1 if abnormal_renal else 0
        liver_score = 1 if abnormal_liver else 0
        return min(renal_score + liver_score, 2)  # Max 2 points

    def _calculate_stroke_score(self, prior_stroke_tia: Optional[bool]) -> int:
        """Calculate stroke history component."""
        return 1 if prior_stroke_tia else 0

    def _calculate_bleeding_score(self, bleeding_history: Optional[bool]) -> int:
        """Calculate bleeding history/predisposition component."""
        return 1 if bleeding_history else 0

    def _calculate_labile_inr_score(self, labile_inr: Optional[bool]) -> int:
        """Calculate labile INR component."""
        return 1 if labile_inr else 0

    def _calculate_elderly_score(self, age: int) -> int:
        """Calculate elderly component (age ≥65)."""
        return 1 if age >= 65 else 0

    def _calculate_drugs_alcohol_score(self, antiplatelet_drugs: Optional[bool],
                                      alcohol_abuse: Optional[bool]) -> int:
        """Calculate drugs/alcohol component (max 2 points)."""
        drugs_score = 1 if antiplatelet_drugs else 0
        alcohol_score = 1 if alcohol_abuse else 0
        return min(drugs_score + alcohol_score, 2)  # Max 2 points

    def _get_risk_category(self, score: int) -> str:
        """Get risk category based on HAS-BLED score."""
        if score <= 1:
            return "low"
        elif score == 2:
            return "moderate"
        else:  # score >= 3
            return "high"

    def _get_annual_bleeding_risk(self, score: int) -> float:
        """Get estimated annual major bleeding risk percentage."""
        risk_table = {
            0: 0.9,
            1: 1.1,
            2: 1.9,
            3: 3.7,
            4: 6.1,
            5: 8.7,
            6: 8.7,  # Same as 5
            7: 8.7,  # Same as 5
            8: 8.7,  # Same as 5
            9: 8.7   # Same as 5
        }
        return risk_table.get(score, 8.7)  # Default to highest risk

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate HAS-BLED score for anticoagulation bleeding risk.

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
        # Note: Some fields may not exist in PatientData, so we use getattr with defaults
        hypertension_score = self._calculate_hypertension_score(patient.hypertension)
        renal_liver_score = self._calculate_renal_liver_score(
            getattr(patient, 'abnormal_renal_function', None),
            getattr(patient, 'abnormal_liver_function', None)
        )
        stroke_score = self._calculate_stroke_score(patient.prior_stroke_tia)
        bleeding_score = self._calculate_bleeding_score(
            getattr(patient, 'bleeding_history', None)
        )
        labile_inr_score = self._calculate_labile_inr_score(
            getattr(patient, 'labile_inr', None)
        )
        elderly_score = self._calculate_elderly_score(patient.age)
        drugs_alcohol_score = self._calculate_drugs_alcohol_score(
            getattr(patient, 'antiplatelet_drugs', None),
            getattr(patient, 'alcohol_abuse', None)
        )

        # Total HAS-BLED score
        total_score = (
            hypertension_score + renal_liver_score + stroke_score +
            bleeding_score + labile_inr_score + elderly_score + drugs_alcohol_score
        )

        # Ensure score is within valid range
        total_score = max(0, min(9, total_score))

        # Get risk category and annual risk
        risk_category = self._get_risk_category(total_score)
        annual_risk = self._get_annual_bleeding_risk(total_score)

        # Create metadata with component scores
        metadata = {
            "hypertension_score": hypertension_score,
            "renal_liver_score": renal_liver_score,
            "stroke_score": stroke_score,
            "bleeding_score": bleeding_score,
            "labile_inr_score": labile_inr_score,
            "elderly_score": elderly_score,
            "drugs_alcohol_score": drugs_alcohol_score,
            "total_score": total_score,
            "annual_bleeding_risk_percent": annual_risk
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
        # HAS-BLED doesn't have strict required columns, uses defaults for missing ones
        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Fill missing optional columns with defaults
        defaults = {
            "hypertension": False,
            "abnormal_renal_function": False,
            "abnormal_liver_function": False,
            "prior_stroke_tia": False,
            "bleeding_history": False,
            "labile_inr": False,
            "antiplatelet_drugs": False,
            "alcohol_abuse": False
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Vectorized calculations
        hypertension_scores = results["hypertension"].astype(int)

        # Renal/liver (max 2 points total)
        renal_scores = results["abnormal_renal_function"].astype(int)
        liver_scores = results["abnormal_liver_function"].astype(int)
        renal_liver_scores = np.minimum(renal_scores + liver_scores, 2)

        stroke_scores = results["prior_stroke_tia"].astype(int)
        bleeding_scores = results["bleeding_history"].astype(int)
        labile_inr_scores = results["labile_inr"].astype(int)
        elderly_scores = (results["age"] >= 65).astype(int)

        # Drugs/alcohol (max 2 points total)
        antiplatelet_scores = results["antiplatelet_drugs"].astype(int)
        alcohol_scores = results["alcohol_abuse"].astype(int)
        drugs_alcohol_scores = np.minimum(antiplatelet_scores + alcohol_scores, 2)

        # Calculate total scores
        total_scores = (
            hypertension_scores + renal_liver_scores + stroke_scores +
            bleeding_scores + labile_inr_scores + elderly_scores + drugs_alcohol_scores
        )

        # Ensure scores are within valid range
        total_scores = np.clip(total_scores, 0, 9)

        # Vectorized categorization
        category_conditions = [
            total_scores <= 1,
            total_scores == 2,
            total_scores >= 3
        ]
        category_choices = ["low", "moderate", "high"]

        results["risk_score"] = total_scores
        results["risk_category"] = np.select(category_conditions, category_choices, default="high")

        return results
