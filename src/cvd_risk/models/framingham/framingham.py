"""
Framingham Risk Score for cardiovascular disease risk prediction.

The Framingham Risk Score estimates 10-year risk of cardiovascular disease
based on traditional risk factors.

Reference:
    Wilson PW, D'Agostino RB, Levy D, Belanger AM, Silbershatz H, Kannel WB.
    Prediction of coronary heart disease using risk factor categories.
    Circulation. 1998;97(18):1837-47.

Note: This is a simplified implementation. The full Framingham model
includes multiple variants (hard CHD, all CVD, etc.).
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# Framingham point system coefficients (simplified version)
_FRAMINGHAM_POINTS = {
    "male": {
        "age": {
            "30-34": 0,
            "35-39": 2,
            "40-44": 5,
            "45-49": 6,
            "50-54": 8,
            "55-59": 10,
            "60-64": 11,
            "65-69": 12,
            "70-74": 14,
        },
        "total_cholesterol": {
            "<160": [-2, 0, 1],
            "160-199": [0, 1, 2],
            "200-239": [1, 3, 4],
            "240-279": [2, 4, 5],
            "≥280": [3, 5, 6],
        },
        "hdl": {
            "<35": 5,
            "35-44": 2,
            "45-49": 1,
            "50-59": 0,
            "≥60": -2,
        },
        "bp": {
            "optimal": [0, 0],
            "normal": [0, 1],
            "high_normal": [1, 2],
            "stage1": [2, 3],
        },
    },
    "female": {
        "age": {
            "30-34": 0,
            "35-39": 2,
            "40-44": 4,
            "45-49": 5,
            "50-54": 7,
            "55-59": 8,
            "60-64": 9,
            "65-69": 10,
            "70-74": 11,
        },
        "total_cholesterol": {
            "<160": [-2, 0, 1],
            "160-199": [0, 1, 3],
            "200-239": [1, 3, 4],
            "240-279": [3, 4, 5],
            "≥280": [4, 6, 7],
        },
        "hdl": {
            "<35": 7,
            "35-44": 4,
            "45-49": 2,
            "50-59": 1,
            "≥60": -1,
        },
        "bp": {
            "optimal": [0, 0],
            "normal": [0, 1],
            "high_normal": [1, 2],
            "stage1": [3, 4],
        },
    },
}

# 10-year risk by point totals and age
_FRAMINGHAM_RISK_TABLE = {
    "male": {
        "30-39": [1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25],
        "40-49": [2, 2, 3, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 39, 47],
        "50-59": [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 33, 38, 45, 52],
        "60-69": [5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 22, 25, 29, 33, 38, 42],
        "70-79": [8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22, 24, 27, 29, 32, 35],
    },
    "female": {
        "30-39": [1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 11, 14, 17, 22, 27, 30],
        "40-49": [2, 2, 2, 3, 4, 5, 6, 8, 10, 12, 15, 19, 23, 27, 32, 37, 40],
        "50-59": [3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 21, 25, 30, 35, 39, 43, 47],
        "60-69": [5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26, 30, 34, 38, 42, 47],
        "70-79": [8, 9, 10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 29, 32, 34, 37],
    },
}


class Framingham(RiskModel):
    """
    Framingham Risk Score for 10-year CVD risk prediction.

    The Framingham Risk Score uses a point-based system to estimate
    10-year risk of cardiovascular disease.

    Parameters
    ----------
    age : int
        Age in years (30-79 for optimal performance).
    sex : Literal["male", "female"]
        Biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mg/dL (convert from mmol/L by multiplying by 38.67).
    hdl_cholesterol : float
        HDL cholesterol in mg/dL.
    smoking : bool
        Current smoking status.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "Framingham"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk.models.framingham import Framingham
    >>> from cvd_risk.core.validation import PatientData
    >>>
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,  # mmol/L
    ...     hdl_cholesterol=1.2,  # mmol/L
    ...     smoking=True,
    ... )
    >>> model = Framingham()
    >>> result = model.calculate(patient)
    >>> print(f"10-year CVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - Model uses point-based system
    - Cholesterol values are converted from mmol/L to mg/dL internally
    - Risk estimates are for 10-year period
    - Model validated for ages 30-79 years
    """

    model_name = "Framingham"
    model_version = "1998"
    supported_regions = None  # Framingham is US-based but widely used

    def __init__(self) -> None:
        """Initialize Framingham model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for Framingham requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for Framingham calculation.
        """
        super().validate_input(patient)

        if patient.age < 30 or patient.age > 79:
            logger.warning(
                f"Age {patient.age} outside optimal range [30, 79] years. "
                "Results may have reduced accuracy."
            )

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CVD risk using Framingham Risk Score.

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

        # Calculate points
        points = 0

        # Age points
        age_group = self._get_age_group(patient.age)
        if age_group:
            points += _FRAMINGHAM_POINTS[patient.sex]["age"][age_group]

        # Cholesterol points (convert mmol/L to mg/dL)
        chol_mgdl = patient.total_cholesterol * 38.67
        hdl_mgdl = patient.hdl_cholesterol * 38.67
        chol_points = self._get_cholesterol_points(patient.sex, chol_mgdl, age_group)
        points += chol_points

        # HDL points
        hdl_points = self._get_hdl_points(patient.sex, hdl_mgdl)
        points += hdl_points

        # BP points
        bp_points = self._get_bp_points(patient.sex, patient.systolic_bp)
        points += bp_points

        # Smoking points
        if patient.smoking:
            # Smoking adds points based on age group
            if age_group in ["50-54", "55-59", "60-64", "65-69", "70-74"]:
                points += 2

        # Look up risk from table
        risk_percentage = self._lookup_risk(patient.sex, patient.age, points)

        # Categorize risk
        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update({"points": points, "age": patient.age, "sex": patient.sex})

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def _get_age_group(self, age: int) -> str:
        """Get age group for point calculation."""
        if age < 35:
            return "30-34"
        elif age < 40:
            return "35-39"
        elif age < 45:
            return "40-44"
        elif age < 50:
            return "45-49"
        elif age < 55:
            return "50-54"
        elif age < 60:
            return "55-59"
        elif age < 65:
            return "60-64"
        elif age < 70:
            return "65-69"
        elif age < 75:
            return "70-74"
        else:
            return "70-74"  # Use max for older ages

    def _get_cholesterol_points(
        self, sex: Literal["male", "female"], chol_mgdl: float, age_group: str | None
    ) -> int:
        """Get cholesterol points based on sex, cholesterol level, and age."""
        if age_group is None:
            return 0

        if chol_mgdl < 160:
            chol_cat = "<160"
        elif chol_mgdl < 200:
            chol_cat = "160-199"
        elif chol_mgdl < 240:
            chol_cat = "200-239"
        elif chol_mgdl < 280:
            chol_cat = "240-279"
        else:
            chol_cat = "≥280"

        points_list = _FRAMINGHAM_POINTS[sex]["total_cholesterol"][chol_cat]
        # Use middle value as approximation
        return points_list[1] if isinstance(points_list, list) else points_list

    def _get_hdl_points(self, sex: Literal["male", "female"], hdl_mgdl: float) -> int:
        """Get HDL points."""
        if hdl_mgdl < 35:
            hdl_cat = "<35"
        elif hdl_mgdl < 45:
            hdl_cat = "35-44"
        elif hdl_mgdl < 50:
            hdl_cat = "45-49"
        elif hdl_mgdl < 60:
            hdl_cat = "50-59"
        else:
            hdl_cat = "≥60"

        return _FRAMINGHAM_POINTS[sex]["hdl"][hdl_cat]

    def _get_bp_points(self, sex: Literal["male", "female"], sbp: float) -> int:
        """Get blood pressure points."""
        if sbp < 120:
            bp_cat = "optimal"
        elif sbp < 130:
            bp_cat = "normal"
        elif sbp < 140:
            bp_cat = "high_normal"
        else:
            bp_cat = "stage1"

        points_list = _FRAMINGHAM_POINTS[sex]["bp"][bp_cat]
        # Use treated as default (second value)
        return points_list[1] if isinstance(points_list, list) else points_list

    def _lookup_risk(self, sex: Literal["male", "female"], age: int, points: int) -> float:
        """Look up 10-year risk from points and age."""
        # Map age to age group for risk table
        if age < 40:
            age_range = "30-39"
        elif age < 50:
            age_range = "40-49"
        elif age < 60:
            age_range = "50-59"
        elif age < 70:
            age_range = "60-69"
        else:
            age_range = "70-79"

        risk_table = _FRAMINGHAM_RISK_TABLE[sex][age_range]

        # Clamp points to valid range
        points = max(0, min(points, len(risk_table) - 1))

        return float(risk_table[points])

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk."""
        if risk_percentage < 10:
            return "low"
        elif risk_percentage < 20:
            return "moderate"
        else:
            return "high"

