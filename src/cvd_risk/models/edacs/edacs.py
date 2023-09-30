"""
Emergency Department Assessment of Chest Pain Score (EDACS) model.

EDACS is a clinical decision tool for risk stratification of patients
presenting to the emergency department with chest pain. It helps identify
patients who are at low risk for major adverse cardiac events.

Reference:
    Than, M., Cullen, L., Aldous, S., et al. (2014).
    2-Hour accelerated diagnostic protocol to assess patients with chest pain
    symptoms using contemporary troponins as the only biomarker.
    Journal of the American College of Cardiology, 64(3), 256-266.
    DOI: 10.1016/j.jacc.2014.05.016
"""

import logging
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)


class EDACS(RiskModel):
    """
    Emergency Department Assessment of Chest Pain Score (EDACS).

    EDACS is a clinical decision tool for risk stratification of chest pain
    patients in the emergency department. It combines patient demographics,
    risk factors, symptoms, and cardiac biomarkers to identify low-risk patients
    who may be suitable for early discharge.

    Valid for ages 18+.
    """

    model_name = "EDACS"
    model_version = "2014"

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)

        # Check required EDACS fields
        required_fields = [
            "diabetes", "smoking", "hypertension", "hyperlipidaemia", "family_history",
            "sweating", "pain_radiation", "pleuritic", "palpation",
            "ecg_st_depression", "ecg_twi", "presentation_hstni", "second_hstni"
        ]

        missing_fields = []
        for field in required_fields:
            if getattr(patient, field) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"EDACS requires the following fields: {missing_fields}")

    def _get_age_score(self, age: int) -> int:
        """
        Calculate age component of EDACS score.

        Parameters
        ----------
        age : int
            Patient age in years

        Returns
        -------
        int
            Age score component
        """
        if age < 18:
            return 0
        elif 18 <= age < 46:
            return 2
        elif 46 <= age < 51:
            return 4
        elif 51 <= age < 56:
            return 6
        elif 56 <= age < 61:
            return 8
        elif 61 <= age < 66:
            return 10
        elif 66 <= age < 71:
            return 12
        elif 71 <= age < 76:
            return 14
        elif 76 <= age < 81:
            return 16
        elif 81 <= age < 86:
            return 18
        else:  # age >= 86
            return 20

    def _get_sex_score(self, sex: str) -> int:
        """
        Calculate sex component of EDACS score.

        Parameters
        ----------
        sex : str
            Patient sex ('male' or 'female')

        Returns
        -------
        int
            Sex score component
        """
        return 6 if sex.lower() == "male" else 0

    def _get_risk_factors_score(self, age: int, diabetes: bool, smoking: bool,
                               hypertension: bool, hyperlipidaemia: bool,
                               family_history: bool) -> int:
        """
        Calculate risk factors component of EDACS score.

        For patients aged 18-50, score = 4 if >=3 risk factors present,
        otherwise 0. For patients >50, score = 0.

        Parameters
        ----------
        age : int
            Patient age
        diabetes : bool
            Diabetes status
        smoking : bool
            Smoking status
        hypertension : bool
            Hypertension status
        hyperlipidaemia : bool
            Hyperlipidaemia status
        family_history : bool
            Family history of CVD

        Returns
        -------
        int
            Risk factors score component
        """
        if age < 18 or age > 50:
            return 0

        # Count risk factors for ages 18-50
        risk_factors = sum([diabetes, smoking, hypertension, hyperlipidaemia, family_history])

        return 4 if risk_factors >= 3 else 0

    def _get_symptoms_score(self, sweating: bool, pain_radiation: bool,
                           pleuritic: bool, palpation: bool) -> int:
        """
        Calculate symptoms component of EDACS score.

        Parameters
        ----------
        sweating : bool
            Diaphoresis present
        pain_radiation : bool
            Pain radiates to arm/shoulder/neck/jaw
        pleuritic : bool
            Pain worsens with inspiration
        palpation : bool
            Pain reproduced by palpation

        Returns
        -------
        int
            Symptoms score component
        """
        score = 0
        score += 3 if sweating else 0
        score += 5 if pain_radiation else 0
        score -= 4 if pleuritic else 0
        score -= 6 if palpation else 0
        return score

    def _get_troponin_threshold(self, sex: str) -> float:
        """
        Get troponin threshold for risk stratification.

        Parameters
        ----------
        sex : str
            Patient sex

        Returns
        -------
        float
            Troponin threshold (ng/L)
        """
        return 34.0 if sex.lower() == "male" else 16.0

    def _stratify_risk(self, edacs_score: int, sex: str, ecg_st_depression: bool,
                      ecg_twi: bool, presentation_hstni: float,
                      second_hstni: float) -> str:
        """
        Stratify patient into risk categories based on EDACS score,
        ECG findings, and troponin levels.

        Low risk criteria:
        - EDACS < 16
        - No ECG ischemia (no ST depression or TWI)
        - Both troponins < sex-specific threshold

        Parameters
        ----------
        edacs_score : int
            EDACS score
        sex : str
            Patient sex
        ecg_st_depression : bool
            ST depression present
        ecg_twi : bool
            T-wave inversion present
        presentation_hstni : float
            Presentation troponin level
        second_hstni : float
            Second troponin level

        Returns
        -------
        str
            Risk category ('low_risk' or 'not_low_risk')
        """
        troponin_threshold = self._get_troponin_threshold(sex)
        ecg_ischemia = ecg_st_depression or ecg_twi

        # Check low risk criteria
        if (edacs_score < 16 and
            not ecg_ischemia and
            presentation_hstni < troponin_threshold and
            second_hstni < troponin_threshold):
            return "low_risk"
        else:
            return "not_low_risk"

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # Calculate score components
        age_score = self._get_age_score(patient.age)
        sex_score = self._get_sex_score(patient.sex)
        risk_score = self._get_risk_factors_score(
            patient.age, patient.diabetes, patient.smoking,
            patient.hypertension, patient.hyperlipidaemia, patient.family_history
        )
        symptoms_score = self._get_symptoms_score(
            patient.sweating, patient.pain_radiation,
            patient.pleuritic, patient.palpation
        )

        # Total EDACS score
        edacs_score = age_score + sex_score + risk_score + symptoms_score

        # Risk stratification
        risk_category = self._stratify_risk(
            edacs_score, patient.sex, patient.ecg_st_depression,
            patient.ecg_twi, patient.presentation_hstni, patient.second_hstni
        )

        return RiskResult(
            risk_score=float(edacs_score),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "age_score": age_score,
                "sex_score": sex_score,
                "risk_factors_score": risk_score,
                "symptoms_score": symptoms_score,
                "troponin_threshold": self._get_troponin_threshold(patient.sex)
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        """
        required = [
            "age", "sex", "diabetes", "smoking", "hypertension", "hyperlipidaemia",
            "family_history", "sweating", "pain_radiation", "pleuritic", "palpation",
            "ecg_st_depression", "ecg_twi", "presentation_hstni", "second_hstni"
        ]

        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Vectorized age scoring
        age_bins = [0, 18, 46, 51, 56, 61, 66, 71, 76, 81, 86, float('inf')]
        age_scores = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        results["age_score"] = pd.cut(
            results["age"], bins=age_bins, labels=age_scores, right=False
        ).astype(int)

        # Vectorized sex scoring
        results["sex_score"] = np.where(results["sex"].str.lower() == "male", 6, 0)

        # Vectorized risk factors scoring (only for ages 18-50)
        risk_factors_cols = ["diabetes", "smoking", "hypertension", "hyperlipidaemia", "family_history"]
        results["risk_factors_count"] = results[risk_factors_cols].sum(axis=1)
        results["risk_factors_score"] = np.where(
            (results["age"] >= 18) & (results["age"] <= 50) & (results["risk_factors_count"] >= 3),
            4, 0
        )

        # Vectorized symptoms scoring
        symptoms_scores = (
            (results["sweating"].astype(int) * 3) +
            (results["pain_radiation"].astype(int) * 5) +
            (results["pleuritic"].astype(int) * (-4)) +
            (results["palpation"].astype(int) * (-6))
        )
        results["symptoms_score"] = symptoms_scores

        # Total EDACS score
        results["risk_score"] = (
            results["age_score"] +
            results["sex_score"] +
            results["risk_factors_score"] +
            results["symptoms_score"]
        )

        # Vectorized risk stratification
        troponin_thresholds = np.where(results["sex"].str.lower() == "male", 34.0, 16.0)
        ecg_ischemia = results["ecg_st_depression"] | results["ecg_twi"]

        low_risk_conditions = (
            (results["risk_score"] < 16) &
            (~ecg_ischemia) &
            (results["presentation_hstni"] < troponin_thresholds) &
            (results["second_hstni"] < troponin_thresholds)
        )

        results["risk_category"] = np.where(low_risk_conditions, "low_risk", "not_low_risk")

        # Clean up intermediate columns
        results = results.drop(columns=["age_score", "sex_score", "risk_factors_count",
                                      "risk_factors_score", "symptoms_score"])

        return results
