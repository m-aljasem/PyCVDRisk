"""
SCORE2-Diabetes Mellitus (SCORE2-DM) cardiovascular risk model.

SCORE2-DM is a specialized version of the SCORE2 cardiovascular risk prediction model
designed specifically for patients with type 2 diabetes mellitus. It provides more
accurate risk estimates for diabetic populations by incorporating diabetes-specific
risk factors including diabetes duration, glycemic control (HbA1c), and renal function (eGFR).

Reference:
    SCORE2-Diabetes Working Group and the ESC Cardiovascular Risk Collaboration.
    SCORE2-Diabetes: 10-year cardiovascular risk estimation in type 2 diabetes
    in Europe. European Heart Journal. 2023.
    DOI: 10.1093/eurheartj/ehad260
"""

import logging
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# SCORE2-DM baseline survival probabilities at 10 years
_SCORE2DM_BASELINE_SURVIVAL = {
    "male": 0.9605,
    "female": 0.9776,
}

# SCORE2-DM region and sex-specific calibration scales
_SCORE2DM_CALIBRATION_SCALES = {
    "male": {
        "low": {"scale1": -0.5699, "scale2": 0.7476},
        "moderate": {"scale1": -0.1565, "scale2": 0.8009},
        "high": {"scale1": 0.3207, "scale2": 0.9360},
        "very_high": {"scale1": 0.5836, "scale2": 0.8294},
    },
    "female": {
        "low": {"scale1": -0.7380, "scale2": 0.7019},
        "moderate": {"scale1": -0.3143, "scale2": 0.7701},
        "high": {"scale1": 0.5710, "scale2": 0.9369},
        "very_high": {"scale1": 0.9412, "scale2": 0.8329},
    },
}

# SCORE2-DM sex-specific coefficients (including diabetes-specific terms)
_SCORE2DM_COEFFICIENTS = {
    "male": {
        # SCORE2 variables
        "cage": 0.5368,
        "smoking": 0.4774,
        "csbp": 0.1322,
        "ctchol": 0.1102,
        "chdl": -0.1087,
        "smoking_cage": -0.0672,
        "csbp_cage": -0.0268,
        "ctchol_cage": -0.0181,
        "chdl_cage": 0.0095,
        # SCORE2 diabetes adjustment
        "diabetes": 0.6457,
        # SCORE2-DM additional variables
        "diabetes_cage": -0.0983,
        "diabetes_dage": -0.0998,
        "chba1c": 0.0955,
        "cegfr": -0.0591,
        "cegfr_sq": 0.0058,
        "chba1c_cage": -0.0134,
        "cegfr_cage": 0.0115,
    },
    "female": {
        # SCORE2 variables
        "cage": 0.6624,
        "smoking": 0.6139,
        "csbp": 0.1421,
        "ctchol": 0.1127,
        "chdl": -0.1568,
        "smoking_cage": -0.1122,
        "csbp_cage": -0.0167,
        "ctchol_cage": -0.0200,
        "chdl_cage": 0.0186,
        # SCORE2 diabetes adjustment
        "diabetes": 0.8096,
        # SCORE2-DM additional variables
        "diabetes_cage": -0.1272,
        "diabetes_dage": -0.118,
        "chba1c": 0.1173,
        "cegfr": -0.0640,
        "cegfr_sq": 0.0062,
        "chba1c_cage": -0.0196,
        "cegfr_cage": 0.0169,
    },
}


def _round_to_nearest_digit(number: Union[float, np.ndarray], digits: int = 0) -> Union[float, np.ndarray]:
    """
    Symmetric rounding to the nearest digit.

    This implements the same rounding logic as the R function round_to_nearest_digit.

    Parameters
    ----------
    number : float or np.ndarray
        Number(s) to round.
    digits : int, default 0
        Number of decimal places to round to.

    Returns
    -------
    float or np.ndarray
        Rounded number(s).
    """
    posneg = np.sign(number)
    number_abs = np.abs(number) * 10**digits
    number_abs = number_abs + 0.5 + np.sqrt(np.finfo(float).eps)
    number_abs = np.trunc(number_abs)
    return (number_abs / 10**digits) * posneg


class SCORE2DM(RiskModel):
    """
    SCORE2-DM cardiovascular risk prediction model (ESC 2023).

    SCORE2-DM is specifically designed for patients with type 2 diabetes mellitus.
    Valid for ages 40-69 with type 2 diabetes (without ASCVD or severe TOD).
    """

    model_name = "SCORE2-DM"
    model_version = "2023"
    supported_regions = ["low", "moderate", "high", "very_high"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient input for SCORE2-DM."""
        super().validate_input(patient)

        if patient.age < 40 or patient.age > 69:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 69]. "
                "SCORE2-DM accuracy may be reduced."
            )

        if patient.diabetes is not True:
            raise ValueError("SCORE2-DM is designed for patients with diabetes. Set diabetes=True.")

        if patient.diabetes_age is None:
            raise ValueError("diabetes_age is required for SCORE2-DM")

        if patient.hba1c is None:
            raise ValueError("HbA1c is required for SCORE2-DM")

        if patient.egfr is None:
            raise ValueError("eGFR is required for SCORE2-DM")

        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")

        # Validate diabetes age is reasonable
        if patient.diabetes_age > patient.age:
            raise ValueError("Diabetes age cannot be greater than current age")

    def _categorize_risk(self, risk_percentage: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
        """
        Categorize risk based on 2023 ESC Guidelines for diabetes.

        Risk groups based on Figure 3 from 2023 ESC Guidelines:
        - <5%: Low risk
        - 5-10%: Moderate risk
        - 10-20%: High risk
        - ≥20%: Very high risk

        Parameters
        ----------
        risk_percentage : float or np.ndarray
            Risk percentage(s) to categorize.

        Returns
        -------
        str or np.ndarray
            Risk category/categories.
        """
        if isinstance(risk_percentage, (np.ndarray, pd.Series)):
            conditions = [
                risk_percentage < 5,
                (risk_percentage >= 5) & (risk_percentage < 10),
                (risk_percentage >= 10) & (risk_percentage < 20),
                risk_percentage >= 20
            ]
            choices = ["low", "moderate", "high", "very_high"]
            return np.select(conditions, choices, default="unknown")

        # Scalar logic
        if risk_percentage < 5:
            return "low"
        elif risk_percentage < 10:
            return "moderate"
        elif risk_percentage < 20:
            return "high"
        else:
            return "very_high"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate 10-year CVD risk for a single patient."""
        self.validate_input(patient)

        # 1. Setup Constants
        sex = patient.sex.lower()
        coeffs = _SCORE2DM_COEFFICIENTS[sex]
        base_surv = _SCORE2DM_BASELINE_SURVIVAL[sex]
        scales = _SCORE2DM_CALIBRATION_SCALES[sex][patient.region]

        # 2. Transform Inputs
        cage = (patient.age - 60) / 5
        csbp = (patient.systolic_bp - 120) / 20
        ctchol = patient.total_cholesterol - 6
        chdl = (patient.hdl_cholesterol - 1.3) / 0.5
        smoking = float(patient.smoking)

        # Diabetes-specific transformations
        diabetes_age = patient.diabetes_age if patient.diabetes_age is not None else 0
        cdage = (diabetes_age - 50) / 5  # diabetes age centered at 50
        chba1c = (patient.hba1c - 31) / 9.34  # HbA1c centered at 31 mmol/mol
        cegfr = (np.log(patient.egfr) - 4.5) / 0.15  # log(eGFR) centered at log(90)

        # 3. Linear Predictor (SCORE2 + diabetes terms)
        lin_pred = (
            # SCORE2 variables
            coeffs["cage"] * cage
            + coeffs["smoking"] * smoking
            + coeffs["csbp"] * csbp
            + coeffs["ctchol"] * ctchol
            + coeffs["chdl"] * chdl
            + coeffs["smoking_cage"] * smoking * cage
            + coeffs["csbp_cage"] * csbp * cage
            + coeffs["ctchol_cage"] * ctchol * cage
            + coeffs["chdl_cage"] * chdl * cage
            # SCORE2 diabetes adjustment
            + coeffs["diabetes"] * 1.0  # diabetes indicator
            # SCORE2-DM additional variables
            + coeffs["diabetes_cage"] * cage  # diabetes * cage interaction
            + coeffs["diabetes_dage"] * cdage  # diabetes * diabetes_age
            + coeffs["chba1c"] * chba1c
            + coeffs["cegfr"] * cegfr
            + coeffs["cegfr_sq"] * (cegfr ** 2)
            + coeffs["chba1c_cage"] * chba1c * cage
            + coeffs["cegfr_cage"] * cegfr * cage
        )

        # 4. Uncalibrated Risk
        uncalibrated_risk = 1.0 - (base_surv ** np.exp(lin_pred))

        # Clip to avoid log errors
        uncalibrated_risk = np.clip(uncalibrated_risk, 1e-9, 1.0 - 1e-9)

        # 5. Calibration
        ln_neg_ln = np.log(-np.log(1 - uncalibrated_risk))
        calib_risk = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))

        # 6. Round to 1 decimal place
        risk_percent = _round_to_nearest_digit(calib_risk * 100.0, digits=1)
        risk_percent = np.clip(risk_percent, 0.0, 100.0)

        # 7. Categorize
        risk_category = self._categorize_risk(risk_percent)

        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "raw_uncalibrated": float(uncalibrated_risk),
                "linear_predictor": float(lin_pred),
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates by grouping (Sex, Region) for efficiency.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol",
                   "smoking", "region", "diabetes", "diabetes_age", "hba1c", "egfr"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        # Filter for diabetic patients only
        df = df[df["diabetes"] == True].copy()
        if df.empty:
            raise ValueError("No diabetic patients found in dataset")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = "unknown"
        results["model_name"] = self.model_name

        # Process by Sex (Major coefficient change)
        for sex in ["male", "female"]:
            sex_mask = results["sex"].str.lower() == sex
            if not sex_mask.any():
                continue

            # Get Sex Constants
            coeffs = _SCORE2DM_COEFFICIENTS[sex]
            base_surv = _SCORE2DM_BASELINE_SURVIVAL[sex]

            # Sub-loop by Region (Scale change)
            present_regions = results.loc[sex_mask, "region"].unique()

            for region in present_regions:
                if region not in _SCORE2DM_CALIBRATION_SCALES[sex]:
                    continue

                mask = sex_mask & (results["region"] == region)
                data = results[mask]
                scales = _SCORE2DM_CALIBRATION_SCALES[sex][region]

                # Vectorized Inputs
                cage = (data["age"] - 60) / 5
                csbp = (data["systolic_bp"] - 120) / 20
                ctchol = data["total_cholesterol"] - 6
                chdl = (data["hdl_cholesterol"] - 1.3) / 0.5
                smoking = data["smoking"].astype(float)

                # Diabetes-specific transformations
                cdage = (data["diabetes_age"] - 50) / 5
                chba1c = (data["hba1c"] - 31) / 9.34
                cegfr = (np.log(data["egfr"]) - 4.5) / 0.15

                # Vectorized Linear Predictor
                lin_pred = (
                    # SCORE2 variables
                    coeffs["cage"] * cage
                    + coeffs["smoking"] * smoking
                    + coeffs["csbp"] * csbp
                    + coeffs["ctchol"] * ctchol
                    + coeffs["chdl"] * chdl
                    + coeffs["smoking_cage"] * (smoking * cage)
                    + coeffs["csbp_cage"] * (csbp * cage)
                    + coeffs["ctchol_cage"] * (ctchol * cage)
                    + coeffs["chdl_cage"] * (chdl * cage)
                    # SCORE2 diabetes adjustment
                    + coeffs["diabetes"] * 1.0  # diabetes indicator
                    # SCORE2-DM additional variables
                    + coeffs["diabetes_cage"] * cage
                    + coeffs["diabetes_dage"] * cdage
                    + coeffs["chba1c"] * chba1c
                    + coeffs["cegfr"] * cegfr
                    + coeffs["cegfr_sq"] * (cegfr ** 2)
                    + coeffs["chba1c_cage"] * (chba1c * cage)
                    + coeffs["cegfr_cage"] * (cegfr * cage)
                )

                # Vectorized Uncalibrated Risk
                uncalib = 1.0 - (base_surv ** np.exp(lin_pred))
                uncalib = np.clip(uncalib, 1e-9, 1.0 - 1e-9)

                # Vectorized Calibration
                ln_neg_ln = np.log(-np.log(1 - uncalib))
                calib = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))

                # Round and assign
                results.loc[mask, "risk_score"] = _round_to_nearest_digit(calib * 100.0, digits=1)

        # Vectorized Categorization
        results["risk_category"] = self._categorize_risk(results["risk_score"].values)

        return results

