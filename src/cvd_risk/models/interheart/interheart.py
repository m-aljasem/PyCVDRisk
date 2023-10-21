"""
INTERHEART Risk Score.

The INTERHEART study is a global case-control study that identified 9 modifiable
risk factors for myocardial infarction. The INTERHEART risk score provides a
simple way to estimate cardiovascular risk based on these factors.

Reference:
    Yusuf S, Hawken S, Ounpuu S, et al. Effect of potentially modifiable risk
    factors associated with myocardial infarction in 52 countries (the
    INTERHEART study): case-control study. The Lancet. 2004;364(9438):937-952.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# INTERHEART risk score components and their point values
# Based on the 9 modifiable risk factors identified in the study
_INTERHEART_RISK_FACTORS = {
    "smoking": {
        "current_smoker": 2,
        "former_smoker": 1,
        "never_smoker": 0
    },
    "diabetes": {
        "yes": 2,
        "no": 0
    },
    "hypertension": {
        "yes": 1,
        "no": 0
    },
    "abdominal_obesity": {
        "yes": 1,
        "no": 0
    },
    "diet": {
        "unhealthy": 1,
        "healthy": 0
    },
    "exercise": {
        "no_regular_exercise": 1,
        "regular_exercise": 0
    },
    "alcohol": {
        "moderate_high": 1,
        "low": 0
    },
    "psychosocial_stress": {
        "high": 1,
        "low": 0
    },
    "depression": {
        "yes": 1,
        "no": 0
    }
}


class INTERHEART(RiskModel):
    """
    INTERHEART Risk Score for myocardial infarction.

    The INTERHEART risk score is based on 9 modifiable risk factors identified
    in a global case-control study. It provides a simple point-based assessment
    of cardiovascular risk.

    Risk Factors (9 points maximum):
    - Smoking (0-2 points)
    - Diabetes (0-2 points)
    - Hypertension (0-1 points)
    - Abdominal obesity (0-1 points)
    - Unhealthy diet (0-1 points)
    - No regular exercise (0-1 points)
    - Moderate/high alcohol consumption (0-1 points)
    - High psychosocial stress (0-1 points)
    - Depression (0-1 points)

    Parameters
    ----------
    age : int
        Age in years.
    sex : Literal["male", "female"]
        Biological sex.
    smoking_status : Optional[Literal["never", "former", "current"]]
        Smoking status.
    diabetes : Optional[bool]
        Diabetes mellitus.
    hypertension : Optional[bool]
        Hypertension.
    abdominal_obesity : Optional[bool]
        Abdominal obesity (waist circumference >102cm men, >88cm women).
    unhealthy_diet : Optional[bool]
        Unhealthy diet (low fruit/vegetable, high fried foods).
    no_regular_exercise : Optional[bool]
        No regular physical activity.
    moderate_high_alcohol : Optional[bool]
        Moderate to high alcohol consumption.
    high_psychosocial_stress : Optional[bool]
        High psychosocial stress.
    depression : Optional[bool]
        Depression.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: INTERHEART score (0-9)
        - risk_category: Risk classification
        - model_name: "INTERHEART"
        - model_version: "2004"
    """

    model_name = "INTERHEART"
    model_version = "2004"
    supported_regions = ["Global"]  # 52 countries worldwide

    def __init__(self) -> None:
        """Initialize INTERHEART model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for INTERHEART requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for INTERHEART calculation.
        """
        super().validate_input(patient)

        # INTERHEART doesn't require specific measurements but benefits from
        # complete risk factor information
        logger.info("INTERHEART score benefits from complete risk factor assessment")

    def _calculate_smoking_score(self, smoking_status: Optional[str]) -> int:
        """Calculate smoking component score."""
        if smoking_status == "current":
            return _INTERHEART_RISK_FACTORS["smoking"]["current_smoker"]
        elif smoking_status == "former":
            return _INTERHEART_RISK_FACTORS["smoking"]["former_smoker"]
        else:  # never or None
            return _INTERHEART_RISK_FACTORS["smoking"]["never_smoker"]

    def _calculate_diabetes_score(self, diabetes: Optional[bool]) -> int:
        """Calculate diabetes component score."""
        return _INTERHEART_RISK_FACTORS["diabetes"]["yes" if diabetes else "no"]

    def _calculate_hypertension_score(self, hypertension: Optional[bool]) -> int:
        """Calculate hypertension component score."""
        return _INTERHEART_RISK_FACTORS["hypertension"]["yes" if hypertension else "no"]

    def _calculate_abdominal_obesity_score(self, abdominal_obesity: Optional[bool]) -> int:
        """Calculate abdominal obesity component score."""
        return _INTERHEART_RISK_FACTORS["abdominal_obesity"]["yes" if abdominal_obesity else "no"]

    def _calculate_diet_score(self, unhealthy_diet: Optional[bool]) -> int:
        """Calculate diet component score."""
        return _INTERHEART_RISK_FACTORS["diet"]["unhealthy" if unhealthy_diet else "healthy"]

    def _calculate_exercise_score(self, no_regular_exercise: Optional[bool]) -> int:
        """Calculate exercise component score."""
        return _INTERHEART_RISK_FACTORS["exercise"]["no_regular_exercise" if no_regular_exercise else "regular_exercise"]

    def _calculate_alcohol_score(self, moderate_high_alcohol: Optional[bool]) -> int:
        """Calculate alcohol component score."""
        return _INTERHEART_RISK_FACTORS["alcohol"]["moderate_high" if moderate_high_alcohol else "low"]

    def _calculate_stress_score(self, high_psychosocial_stress: Optional[bool]) -> int:
        """Calculate psychosocial stress component score."""
        return _INTERHEART_RISK_FACTORS["psychosocial_stress"]["high" if high_psychosocial_stress else "low"]

    def _calculate_depression_score(self, depression: Optional[bool]) -> int:
        """Calculate depression component score."""
        return _INTERHEART_RISK_FACTORS["depression"]["yes" if depression else "no"]

    def _get_risk_category(self, score: int) -> str:
        """Get risk category based on INTERHEART score."""
        if score <= 2:
            return "low"
        elif score <= 4:
            return "moderate"
        elif score <= 6:
            return "high"
        else:  # score >= 7
            return "very_high"

    def _get_relative_risk(self, score: int) -> float:
        """Get relative risk of myocardial infarction based on score."""
        # Based on INTERHEART study findings
        # Risk increases with each point approximately
        baseline_risk = 1.0
        risk_multiplier_per_point = 1.6  # Approximate from study
        return baseline_risk * (risk_multiplier_per_point ** score)

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate INTERHEART risk score.

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

        # Calculate individual component scores
        smoking_score = self._calculate_smoking_score(getattr(patient, 'smoking_status', None))
        diabetes_score = self._calculate_diabetes_score(patient.diabetes)
        hypertension_score = self._calculate_hypertension_score(patient.hypertension)
        abdominal_obesity_score = self._calculate_abdominal_obesity_score(
            getattr(patient, 'abdominal_obesity', None))
        diet_score = self._calculate_diet_score(getattr(patient, 'unhealthy_diet', None))
        exercise_score = self._calculate_exercise_score(getattr(patient, 'no_regular_exercise', None))
        alcohol_score = self._calculate_alcohol_score(getattr(patient, 'moderate_high_alcohol', None))
        stress_score = self._calculate_stress_score(getattr(patient, 'high_psychosocial_stress', None))
        depression_score = self._calculate_depression_score(getattr(patient, 'depression', None))

        # Total INTERHEART score
        total_score = (
            smoking_score + diabetes_score + hypertension_score +
            abdominal_obesity_score + diet_score + exercise_score +
            alcohol_score + stress_score + depression_score
        )

        # Ensure score is within valid range
        total_score = max(0, min(9, total_score))

        # Get risk category and relative risk
        risk_category = self._get_risk_category(total_score)
        relative_risk = self._get_relative_risk(total_score)

        # Create metadata with component scores
        metadata = {
            "smoking_score": smoking_score,
            "diabetes_score": diabetes_score,
            "hypertension_score": hypertension_score,
            "abdominal_obesity_score": abdominal_obesity_score,
            "diet_score": diet_score,
            "exercise_score": exercise_score,
            "alcohol_score": alcohol_score,
            "stress_score": stress_score,
            "depression_score": depression_score,
            "total_score": total_score,
            "relative_risk": relative_risk,
            "note": "This is a relative risk score (not absolute 10-year risk percentage)"
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
        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Fill missing optional columns with defaults
        defaults = {
            "smoking_status": "never",
            "diabetes": False,
            "hypertension": False,
            "abdominal_obesity": False,
            "unhealthy_diet": False,
            "no_regular_exercise": False,
            "moderate_high_alcohol": False,
            "high_psychosocial_stress": False,
            "depression": False
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Vectorized calculations
        smoking_scores = results["smoking_status"].map({
            "current": _INTERHEART_RISK_FACTORS["smoking"]["current_smoker"],
            "former": _INTERHEART_RISK_FACTORS["smoking"]["former_smoker"]
        }).fillna(_INTERHEART_RISK_FACTORS["smoking"]["never_smoker"]).astype(int)

        diabetes_scores = results["diabetes"].map({
            True: _INTERHEART_RISK_FACTORS["diabetes"]["yes"],
            False: _INTERHEART_RISK_FACTORS["diabetes"]["no"]
        }).fillna(_INTERHEART_RISK_FACTORS["diabetes"]["no"]).astype(int)

        hypertension_scores = results["hypertension"].map({
            True: _INTERHEART_RISK_FACTORS["hypertension"]["yes"],
            False: _INTERHEART_RISK_FACTORS["hypertension"]["no"]
        }).fillna(_INTERHEART_RISK_FACTORS["hypertension"]["no"]).astype(int)

        abdominal_obesity_scores = results["abdominal_obesity"].map({
            True: _INTERHEART_RISK_FACTORS["abdominal_obesity"]["yes"],
            False: _INTERHEART_RISK_FACTORS["abdominal_obesity"]["no"]
        }).fillna(_INTERHEART_RISK_FACTORS["abdominal_obesity"]["no"]).astype(int)

        diet_scores = results["unhealthy_diet"].map({
            True: _INTERHEART_RISK_FACTORS["diet"]["unhealthy"],
            False: _INTERHEART_RISK_FACTORS["diet"]["healthy"]
        }).fillna(_INTERHEART_RISK_FACTORS["diet"]["healthy"]).astype(int)

        exercise_scores = results["no_regular_exercise"].map({
            True: _INTERHEART_RISK_FACTORS["exercise"]["no_regular_exercise"],
            False: _INTERHEART_RISK_FACTORS["exercise"]["regular_exercise"]
        }).fillna(_INTERHEART_RISK_FACTORS["exercise"]["regular_exercise"]).astype(int)

        alcohol_scores = results["moderate_high_alcohol"].map({
            True: _INTERHEART_RISK_FACTORS["alcohol"]["moderate_high"],
            False: _INTERHEART_RISK_FACTORS["alcohol"]["low"]
        }).fillna(_INTERHEART_RISK_FACTORS["alcohol"]["low"]).astype(int)

        stress_scores = results["high_psychosocial_stress"].map({
            True: _INTERHEART_RISK_FACTORS["psychosocial_stress"]["high"],
            False: _INTERHEART_RISK_FACTORS["psychosocial_stress"]["low"]
        }).fillna(_INTERHEART_RISK_FACTORS["psychosocial_stress"]["low"]).astype(int)

        depression_scores = results["depression"].map({
            True: _INTERHEART_RISK_FACTORS["depression"]["yes"],
            False: _INTERHEART_RISK_FACTORS["depression"]["no"]
        }).fillna(_INTERHEART_RISK_FACTORS["depression"]["no"]).astype(int)

        # Calculate total scores
        total_scores = (
            smoking_scores + diabetes_scores + hypertension_scores +
            abdominal_obesity_scores + diet_scores + exercise_scores +
            alcohol_scores + stress_scores + depression_scores
        )

        # Ensure scores are within valid range
        total_scores = np.clip(total_scores, 0, 9)

        # Vectorized categorization
        category_conditions = [
            total_scores <= 2,
            (total_scores >= 3) & (total_scores <= 4),
            (total_scores >= 5) & (total_scores <= 6),
            total_scores >= 7
        ]
        category_choices = ["low", "moderate", "high", "very_high"]

        results["risk_score"] = total_scores
        results["risk_category"] = np.select(category_conditions, category_choices, default="very_high")

        return results
