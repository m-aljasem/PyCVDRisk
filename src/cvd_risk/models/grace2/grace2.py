"""
GRACE2 (Global Registry of Acute Coronary Events version 2.0) risk model.

GRACE2 is a clinical prediction tool for estimating the risk of death
from admission to 6 months in patients presenting with acute coronary syndromes.

Reference:
    Granger CB, Goldberg RJ, Dabbous O, et al. Predictors of hospital mortality
    in the global registry of acute coronary events. Arch Intern Med.
    2003;163(19):2345-2353.
    DOI: 10.1001/archinte.163.19.2345
"""

import logging
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# GRACE2 risk score categories
_GRACE2_CATEGORIES = {
    "low": "Low risk",
    "moderate": "Moderate risk",
    "high": "High risk"
}


class GRACE2(RiskModel):
    """
    GRACE2 risk prediction model for acute coronary syndromes.

    The GRACE2 score ranges from 1-263 points and stratifies patients into
    three risk categories for 6-month mortality following ACS presentation.

    Valid for patients presenting with acute coronary syndromes (ACS).
    """

    model_name = "GRACE2"
    model_version = "2.0"
    supported_regions = None  # GRACE2 is not region-specific

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)

        # GRACE2-specific validation
        required_fields = [
            "heart_rate", "creatinine", "killip_class", "cardiac_arrest",
            "ecg_st_depression", "troponin_level"
        ]

        missing_fields = []
        for field in required_fields:
            if getattr(patient, field, None) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"GRACE2 model requires the following fields: {missing_fields}. "
                f"Patient is missing: {missing_fields}"
            )

        # Validate killip class
        if patient.killip_class not in [1, 2, 3, 4]:
            raise ValueError("Killip class must be 1, 2, 3, or 4")

        # Validate heart rate
        if patient.heart_rate < 30 or patient.heart_rate > 300:
            raise ValueError("Heart rate must be between 30-300 bpm")

        # Validate creatinine
        if patient.creatinine < 0.1 or patient.creatinine > 20.0:
            raise ValueError("Creatinine must be between 0.1-20.0 mg/dL")

    def _calculate_killip_points(self, killip_class: int) -> int:
        """Calculate points for Killip classification."""
        killip_points = {1: 0, 2: 15, 3: 29, 4: 44}
        return killip_points.get(killip_class, 0)

    def _calculate_systolic_bp_points(self, systolic_bp: float) -> float:
        """Calculate points for systolic blood pressure."""
        if systolic_bp < 80:
            return 40.0
        elif 80 <= systolic_bp < 100:
            return 40 - (systolic_bp - 80) * 0.3
        elif 100 <= systolic_bp < 110:
            return 34 - (systolic_bp - 100) * 0.3
        elif 110 <= systolic_bp < 120:
            return 31 - (systolic_bp - 110) * 0.4
        elif 120 <= systolic_bp < 130:
            return 27 - (systolic_bp - 120) * 0.3
        elif 130 <= systolic_bp < 140:
            return 24 - (systolic_bp - 130) * 0.3
        elif 140 <= systolic_bp < 150:
            return 20 - (systolic_bp - 140) * 0.4
        elif 150 <= systolic_bp < 160:
            return 17 - (systolic_bp - 150) * 0.3
        elif 160 <= systolic_bp < 180:
            return 14 - (systolic_bp - 160) * 0.3
        elif 180 <= systolic_bp < 200:
            return 8 - (systolic_bp - 180) * 0.4
        else:  # >= 200
            return 0.0

    def _calculate_heart_rate_points(self, heart_rate: float) -> float:
        """Calculate points for heart rate."""
        if heart_rate < 70:
            return 0.0
        elif 70 <= heart_rate < 80:
            return 0 + (heart_rate - 70) * 0.3
        elif 80 <= heart_rate < 90:
            return 3 + (heart_rate - 80) * 0.2
        elif 90 <= heart_rate < 100:
            return 5 + (heart_rate - 90) * 0.3
        elif 100 <= heart_rate < 110:
            return 8 + (heart_rate - 100) * 0.2
        elif 110 <= heart_rate < 150:
            return 10 + (heart_rate - 110) * 0.3
        elif 150 <= heart_rate < 200:
            return 22 + (heart_rate - 150) * 0.3
        else:  # >= 200
            return 34.0

    def _calculate_age_points(self, age: int) -> float:
        """Calculate points for age."""
        if age < 35:
            return 0.0
        elif 35 <= age < 45:
            return 0 + (age - 35) * 1.8
        elif 45 <= age < 55:
            return 18 + (age - 45) * 1.8
        elif 55 <= age < 65:
            return 36 + (age - 55) * 1.8
        elif 65 <= age < 75:
            return 54 + (age - 65) * 1.9
        elif 75 <= age < 85:
            return 73 + (age - 75) * 1.8
        elif 85 <= age < 90:
            return 91 + (age - 85) * 1.8
        else:  # >= 90
            return 100.0

    def _calculate_creatinine_points(self, creatinine: float) -> float:
        """Calculate points for creatinine (converted from mg/dL to µmol/L)."""
        # Convert creatinine from mg/dL to µmol/L
        creatinine_umol = creatinine / 0.0884

        if creatinine_umol < 0.2:
            return 0 + (creatinine_umol - 0) * (1 / 0.2)
        elif 0.2 <= creatinine_umol < 0.4:
            return 1 + (creatinine_umol - 0.2) * (2 / 0.2)
        elif 0.4 <= creatinine_umol < 0.6:
            return 3 + (creatinine_umol - 0.4) * (1 / 0.2)
        elif 0.6 <= creatinine_umol < 0.8:
            return 4 + (creatinine_umol - 0.6) * (2 / 0.2)
        elif 0.8 <= creatinine_umol < 1.0:
            return 6 + (creatinine_umol - 0.8) * (1 / 0.2)
        elif 1.0 <= creatinine_umol < 1.2:
            return 7 + (creatinine_umol - 1.0) * (1 / 0.2)
        elif 1.2 <= creatinine_umol < 1.4:
            return 8 + (creatinine_umol - 1.2) * (2 / 0.2)
        elif 1.4 <= creatinine_umol < 1.6:
            return 10 + (creatinine_umol - 1.4) * (1 / 0.2)
        elif 1.6 <= creatinine_umol < 1.8:
            return 11 + (creatinine_umol - 1.6) * (2 / 0.2)
        elif 1.8 <= creatinine_umol < 2.0:
            return 13 + (creatinine_umol - 1.8) * (1 / 0.2)
        elif 2.0 <= creatinine_umol < 3.0:
            return 14 + (creatinine_umol - 2.0) * (7 / 1.0)
        elif 3.0 <= creatinine_umol < 4.0:
            return 21 + (creatinine_umol - 3.0) * (7 / 1.0)
        else:  # >= 4.0
            return 28.0

    def _calculate_troponin_points(self, troponin_level: float, sex: str) -> int:
        """Calculate points for troponin level."""
        if sex.lower() == "male":
            return 13 if troponin_level >= 34 else 0
        else:  # female
            return 13 if troponin_level >= 16 else 0

    def _calculate_cardiac_arrest_points(self, cardiac_arrest: bool) -> int:
        """Calculate points for cardiac arrest."""
        return 30 if cardiac_arrest else 0

    def _calculate_st_depression_points(self, ecg_st_depression: bool) -> int:
        """Calculate points for ST segment depression."""
        return 17 if ecg_st_depression else 0

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # Calculate points for each component
        killip_points = self._calculate_killip_points(patient.killip_class)
        sbp_points = self._calculate_systolic_bp_points(patient.systolic_bp)
        hr_points = self._calculate_heart_rate_points(patient.heart_rate)
        age_points = self._calculate_age_points(patient.age)
        creat_points = self._calculate_creatinine_points(patient.creatinine)
        troponin_points = self._calculate_troponin_points(patient.troponin_level, patient.sex)
        arrest_points = self._calculate_cardiac_arrest_points(patient.cardiac_arrest)
        st_points = self._calculate_st_depression_points(patient.ecg_st_depression)

        # Sum all points for total GRACE2 score
        total_score = (
            killip_points + sbp_points + hr_points + age_points +
            creat_points + troponin_points + arrest_points + st_points
        )

        # Determine risk category
        if total_score <= 88:
            category = "low"
        elif 89 <= total_score <= 118:
            category = "moderate"
        else:  # > 118
            category = "high"

        return RiskResult(
            risk_score=float(total_score),
            risk_category=category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "killip_points": killip_points,
                "sbp_points": sbp_points,
                "hr_points": hr_points,
                "age_points": age_points,
                "creat_points": creat_points,
                "troponin_points": troponin_points,
                "arrest_points": arrest_points,
                "st_points": st_points
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        """
        required = [
            "age", "sex", "systolic_bp", "heart_rate", "creatinine",
            "killip_class", "cardiac_arrest", "ecg_st_depression", "troponin_level"
        ]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Vectorized calculations
        results["killip_points"] = results["killip_class"].map({1: 0, 2: 15, 3: 29, 4: 44})

        # Systolic BP points (vectorized)
        sbp = results["systolic_bp"]
        results["sbp_points"] = np.select(
            [
                sbp < 80,
                (sbp >= 80) & (sbp < 100),
                (sbp >= 100) & (sbp < 110),
                (sbp >= 110) & (sbp < 120),
                (sbp >= 120) & (sbp < 130),
                (sbp >= 130) & (sbp < 140),
                (sbp >= 140) & (sbp < 150),
                (sbp >= 150) & (sbp < 160),
                (sbp >= 160) & (sbp < 180),
                (sbp >= 180) & (sbp < 200),
                sbp >= 200
            ],
            [
                40.0,
                40 - (sbp - 80) * 0.3,
                34 - (sbp - 100) * 0.3,
                31 - (sbp - 110) * 0.4,
                27 - (sbp - 120) * 0.3,
                24 - (sbp - 130) * 0.3,
                20 - (sbp - 140) * 0.4,
                17 - (sbp - 150) * 0.3,
                14 - (sbp - 160) * 0.3,
                8 - (sbp - 180) * 0.4,
                0.0
            ]
        )

        # Heart rate points (vectorized)
        hr = results["heart_rate"]
        results["hr_points"] = np.select(
            [
                hr < 70,
                (hr >= 70) & (hr < 80),
                (hr >= 80) & (hr < 90),
                (hr >= 90) & (hr < 100),
                (hr >= 100) & (hr < 110),
                (hr >= 110) & (hr < 150),
                (hr >= 150) & (hr < 200),
                hr >= 200
            ],
            [
                0.0,
                0 + (hr - 70) * 0.3,
                3 + (hr - 80) * 0.2,
                5 + (hr - 90) * 0.3,
                8 + (hr - 100) * 0.2,
                10 + (hr - 110) * 0.3,
                22 + (hr - 150) * 0.3,
                34.0
            ]
        )

        # Age points (vectorized)
        age = results["age"]
        results["age_points"] = np.select(
            [
                age < 35,
                (age >= 35) & (age < 45),
                (age >= 45) & (age < 55),
                (age >= 55) & (age < 65),
                (age >= 65) & (age < 75),
                (age >= 75) & (age < 85),
                (age >= 85) & (age < 90),
                age >= 90
            ],
            [
                0.0,
                0 + (age - 35) * 1.8,
                18 + (age - 45) * 1.8,
                36 + (age - 55) * 1.8,
                54 + (age - 65) * 1.9,
                73 + (age - 75) * 1.8,
                91 + (age - 85) * 1.8,
                100.0
            ]
        )

        # Creatinine points (vectorized) - convert to µmol/L first
        creat_umol = results["creatinine"] / 0.0884
        results["creat_points"] = np.select(
            [
                creat_umol < 0.2,
                (creat_umol >= 0.2) & (creat_umol < 0.4),
                (creat_umol >= 0.4) & (creat_umol < 0.6),
                (creat_umol >= 0.6) & (creat_umol < 0.8),
                (creat_umol >= 0.8) & (creat_umol < 1.0),
                (creat_umol >= 1.0) & (creat_umol < 1.2),
                (creat_umol >= 1.2) & (creat_umol < 1.4),
                (creat_umol >= 1.4) & (creat_umol < 1.6),
                (creat_umol >= 1.6) & (creat_umol < 1.8),
                (creat_umol >= 1.8) & (creat_umol < 2.0),
                (creat_umol >= 2.0) & (creat_umol < 3.0),
                (creat_umol >= 3.0) & (creat_umol < 4.0),
                creat_umol >= 4.0
            ],
            [
                0 + (creat_umol - 0) * (1 / 0.2),
                1 + (creat_umol - 0.2) * (2 / 0.2),
                3 + (creat_umol - 0.4) * (1 / 0.2),
                4 + (creat_umol - 0.6) * (2 / 0.2),
                6 + (creat_umol - 0.8) * (1 / 0.2),
                7 + (creat_umol - 1.0) * (1 / 0.2),
                8 + (creat_umol - 1.2) * (2 / 0.2),
                10 + (creat_umol - 1.4) * (1 / 0.2),
                11 + (creat_umol - 1.6) * (2 / 0.2),
                13 + (creat_umol - 1.8) * (1 / 0.2),
                14 + (creat_umol - 2.0) * (7 / 1.0),
                21 + (creat_umol - 3.0) * (7 / 1.0),
                28.0
            ]
        )

        # Troponin points (sex-specific)
        results["troponin_points"] = np.where(
            results["sex"].str.lower() == "male",
            np.where(results["troponin_level"] >= 34, 13, 0),
            np.where(results["troponin_level"] >= 16, 13, 0)
        )

        # Cardiac arrest points
        results["arrest_points"] = np.where(results["cardiac_arrest"], 30, 0)

        # ST depression points
        results["st_points"] = np.where(results["ecg_st_depression"], 17, 0)

        # Calculate total score
        results["risk_score"] = (
            results["killip_points"] + results["sbp_points"] + results["hr_points"] +
            results["age_points"] + results["creat_points"] + results["troponin_points"] +
            results["arrest_points"] + results["st_points"]
        )

        # Determine risk categories
        results["risk_category"] = np.select(
            [
                results["risk_score"] <= 88,
                (results["risk_score"] >= 89) & (results["risk_score"] <= 118),
                results["risk_score"] > 118
            ],
            ["low", "moderate", "high"],
            default="unknown"
        )

        return results
