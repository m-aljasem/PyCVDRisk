"""
SCORE2-OP (Systematic COronary Risk Evaluation 2 - Older Persons) cardiovascular risk model.

SCORE2-OP extends SCORE2 to older populations (aged 70+ years) and is the updated
European cardiovascular risk prediction model published in 2021 by the ESC
Cardiovascular Risk Collaboration.

Reference:
    SCORE2 Working Group and ESC Cardiovascular Risk Collaboration. (2021).
    SCORE2 risk prediction algorithms: new models to estimate 10-year risk
    of cardiovascular disease in Europe. European Heart Journal, 42(25), 2439-2454.
    DOI: 10.1093/eurheartj/ehab309
"""

import logging
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# SCORE2-OP baseline survival probabilities at 10 years
_SCORE2OP_BASELINE_SURVIVAL = {
    "male": 0.7576,
    "female": 0.8082,
}

# SCORE2-OP region and sex-specific calibration scales (for ages >= 70)
_SCORE2OP_CALIBRATION_SCALES = {
    "male": {
        "low": {"scale1": -0.34, "scale2": 1.19},
        "moderate": {"scale1": 0.01, "scale2": 1.25},
        "high": {"scale1": 0.08, "scale2": 1.15},
        "very_high": {"scale1": 0.05, "scale2": 0.7},
    },
    "female": {
        "low": {"scale1": -0.52, "scale2": 1.01},
        "moderate": {"scale1": -0.1, "scale2": 1.1},
        "high": {"scale1": 0.38, "scale2": 1.09},
        "very_high": {"scale1": 0.38, "scale2": 0.69},
    },
}

# SCORE2-OP sex-specific coefficients (for ages >= 70)
_SCORE2OP_COEFFICIENTS = {
    "male": {
        "cage": 0.0634,
        "smoking": 0.3524,
        "csbp": 0.0094,
        "ctchol": 0.0850,
        "chdl": -0.3564,
        "diabetes": 0.4245,
        "smoking_cage": -0.0247,
        "csbp_cage": -0.0005,
        "ctchol_cage": 0.0073,
        "chdl_cage": 0.0091,
        "diabetes_cage": -0.0174,
    },
    "female": {
        "cage": 0.0789,
        "smoking": 0.4921,
        "csbp": 0.0102,
        "ctchol": 0.0605,
        "chdl": -0.3040,
        "diabetes": 0.6010,
        "smoking_cage": -0.0255,
        "csbp_cage": -0.0004,
        "ctchol_cage": -0.0009,
        "chdl_cage": 0.0154,
        "diabetes_cage": -0.0107,
    },
}

# SCORE2-OP baseline hazard shifts
_SCORE2OP_BASELINE_SHIFTS = {
    "male": -0.0929,
    "female": -0.229,
}


class SCORE2OP(RiskModel):
    """
    SCORE2-OP cardiovascular risk prediction model (ESC 2021).
    Valid for ages 70+.
    """

    model_name = "SCORE2-OP"
    model_version = "2021"
    supported_regions = ["low", "moderate", "high", "very_high"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.age < 70:
            logger.warning(
                f"Age {patient.age} is below optimal range [70, ...]. "
                "SCORE2-OP is specifically designed for older populations (>=70). "
                "Consider using SCORE2 for ages 40-69."
            )
        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")
        if patient.diabetes is None:
            raise ValueError("Diabetes status is required for SCORE2-OP")

    def _get_risk_thresholds(self, age: Union[int, float, np.ndarray]) -> Tuple:
        """
        Get clinical risk thresholds based on Age (ESC 2021 Guidelines).
        SCORE2-OP uses the same thresholds as SCORE2 for ages >=70.

        Returns
        -------
        Tuple (low_cutoff, high_cutoff)
            < low_cutoff: Low Risk
            >= high_cutoff: Very High Risk
        """
        # For SCORE2-OP (ages >=70), thresholds are fixed
        low_cut = 7.5
        high_cut = 15.0
        return low_cut, high_cut

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # 1. Setup Constants
        sex = patient.sex.lower()
        coeffs = _SCORE2OP_COEFFICIENTS[sex]
        base_surv = _SCORE2OP_BASELINE_SURVIVAL[sex]
        base_shift = _SCORE2OP_BASELINE_SHIFTS[sex]
        scales = _SCORE2OP_CALIBRATION_SCALES[sex][patient.region]

        # 2. Transform Inputs (SCORE2-OP specific transformations)
        cage = patient.age - 73  # Centered at 73 for older population
        csbp = patient.systolic_bp - 150  # Centered at 150 for older population
        ctchol = patient.total_cholesterol - 6
        chdl = patient.hdl_cholesterol - 1.4  # Centered at 1.4 for older population
        smoking = float(patient.smoking)
        diabetes = float(patient.diabetes)

        # 3. Linear Predictor
        lin_pred = (
            coeffs["cage"] * cage
            + coeffs["smoking"] * smoking
            + coeffs["csbp"] * csbp
            + coeffs["ctchol"] * ctchol
            + coeffs["chdl"] * chdl
            + coeffs["diabetes"] * diabetes
            + coeffs["smoking_cage"] * smoking * cage
            + coeffs["csbp_cage"] * csbp * cage
            + coeffs["ctchol_cage"] * ctchol * cage
            + coeffs["chdl_cage"] * chdl * cage
            + coeffs["diabetes_cage"] * diabetes * cage
            + base_shift  # Apply baseline hazard shift
        )

        # 4. Uncalibrated Risk
        uncalibrated_risk = 1.0 - (base_surv ** np.exp(lin_pred))

        # Clip to avoid log errors
        uncalibrated_risk = np.clip(uncalibrated_risk, 1e-9, 1.0 - 1e-9)

        # 5. Calibration
        ln_neg_ln = np.log(-np.log(1 - uncalibrated_risk))
        calib_risk = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))

        risk_percent = np.clip(calib_risk * 100.0, 0.0, 100.0)

        # 6. Categorize (SCORE2-OP uses fixed thresholds for ages >=70)
        low_cut, high_cut = self._get_risk_thresholds(patient.age)

        if risk_percent < low_cut:
            cat = "low"
        elif risk_percent < high_cut:
            cat = "moderate"  # Note: In SCORE2-OP terminology, the gap is "moderate"
        else:
            cat = "very_high"

        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=cat,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={"raw_uncalibrated": uncalibrated_risk}
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates by grouping (Sex, Region) for efficiency.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking", "diabetes", "region"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["model_name"] = self.model_name

        # Process by Sex (Major coefficient change)
        for sex in ["male", "female"]:
            # Create mask for this sex
            sex_mask = results["sex"].str.lower() == sex
            if not sex_mask.any():
                continue

            # Get Sex Constants
            coeffs = _SCORE2OP_COEFFICIENTS[sex]
            base_surv = _SCORE2OP_BASELINE_SURVIVAL[sex]
            base_shift = _SCORE2OP_BASELINE_SHIFTS[sex]

            # Sub-loop by Region (Scale change)
            present_regions = results.loc[sex_mask, "region"].unique()

            for region in present_regions:
                if region not in _SCORE2OP_CALIBRATION_SCALES[sex]:
                    continue  # Skip invalid regions

                # Mask for Sex AND Region
                mask = sex_mask & (results["region"] == region)
                data = results[mask]
                scales = _SCORE2OP_CALIBRATION_SCALES[sex][region]

                # Vectorized Inputs (SCORE2-OP specific transformations)
                cage = data["age"] - 73  # Centered at 73 for older population
                csbp = data["systolic_bp"] - 150  # Centered at 150 for older population
                ctchol = data["total_cholesterol"] - 6
                chdl = data["hdl_cholesterol"] - 1.4  # Centered at 1.4 for older population
                smoking = data["smoking"].astype(float)
                diabetes = data["diabetes"].astype(float)

                # Vectorized Linear Predictor
                lin_pred = (
                    coeffs["cage"] * cage
                    + coeffs["smoking"] * smoking
                    + coeffs["csbp"] * csbp
                    + coeffs["ctchol"] * ctchol
                    + coeffs["chdl"] * chdl
                    + coeffs["diabetes"] * diabetes
                    + coeffs["smoking_cage"] * (smoking * cage)
                    + coeffs["csbp_cage"] * (csbp * cage)
                    + coeffs["ctchol_cage"] * (ctchol * cage)
                    + coeffs["chdl_cage"] * (chdl * cage)
                    + coeffs["diabetes_cage"] * (diabetes * cage)
                    + base_shift  # Apply baseline hazard shift
                )

                # Vectorized Uncalibrated Risk
                uncalib = 1.0 - (base_surv ** np.exp(lin_pred))
                uncalib = np.clip(uncalib, 1e-9, 1.0 - 1e-9)

                # Vectorized Calibration
                ln_neg_ln = np.log(-np.log(1 - uncalib))
                calib = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))

                # Assign Results
                results.loc[mask, "risk_score"] = np.clip(calib * 100.0, 0.0, 100.0)

        # Vectorized Categorization (Fixed thresholds for SCORE2-OP)
        low_cuts, high_cuts = self._get_risk_thresholds(results["age"].values)

        # Numpy Select for Classification
        conditions = [
            results["risk_score"] < low_cuts,
            (results["risk_score"] >= low_cuts) & (results["risk_score"] < high_cuts),
            results["risk_score"] >= high_cuts
        ]
        choices = ["low", "moderate", "very_high"]

        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results

