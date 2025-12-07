"""
SCORE2 (Systematic COronary Risk Evaluation 2) cardiovascular risk model.

SCORE2 is the updated European cardiovascular risk prediction model
published in 2021 by the ESC Cardiovascular Risk Collaboration.

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

# SCORE2 baseline survival probabilities at 10 years
_SCORE2_BASELINE_SURVIVAL = {
    "male": 0.9605,
    "female": 0.9776,
}

# SCORE2 region and sex-specific calibration scales
_SCORE2_CALIBRATION_SCALES = {
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

# SCORE2 sex-specific coefficients
_SCORE2_COEFFICIENTS = {
    "male": {
        "cage": 0.3742,
        "smoking": 0.6012,
        "csbp": 0.2777,
        "ctchol": 0.1458,
        "chdl": -0.2698,
        "smoking_cage": -0.0755,
        "csbp_cage": -0.0255,
        "ctchol_cage": -0.0281,
        "chdl_cage": 0.0426,
    },
    "female": {
        "cage": 0.4648,
        "smoking": 0.7744,
        "csbp": 0.3131,
        "ctchol": 0.1002,
        "chdl": -0.2606,
        "smoking_cage": -0.1088,
        "csbp_cage": -0.0277,
        "ctchol_cage": -0.0226,
        "chdl_cage": 0.0613,
    },
}


class SCORE2(RiskModel):
    """
    SCORE2 cardiovascular risk prediction model (ESC 2021).
    Valid for ages 40-69.
    """

    model_name = "SCORE2"
    model_version = "2021"
    supported_regions = ["low", "moderate", "high", "very_high"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.age < 40 or patient.age > 69:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 69]. "
                "SCORE2 accuracy may be reduced."
            )
        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")

    def _get_risk_thresholds(self, age: Union[int, float, np.ndarray]) -> Tuple:
        """
        Get clinical risk thresholds based on Age (ESC 2021 Guidelines).
        
        Returns
        -------
        Tuple (low_cutoff, high_cutoff)
            < low_cutoff: Low-to-Moderate Risk
            >= high_cutoff: High Risk
            >= high_cutoff * 2 (approx): Very High Risk
        """
        # Vectorized logic for NumPy arrays
        if isinstance(age, (np.ndarray, pd.Series)):
            low_cut = np.select(
                [age < 50, (age >= 50) & (age < 70), age >= 70],
                [2.5, 5.0, 7.5],
                default=5.0
            )
            high_cut = np.select(
                [age < 50, (age >= 50) & (age < 70), age >= 70],
                [7.5, 10.0, 15.0],
                default=10.0
            )
            return low_cut, high_cut

        # Scalar logic for single patient
        if age < 50:
            return 2.5, 7.5
        elif age < 70:
            return 5.0, 10.0
        else:
            return 7.5, 15.0

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # 1. Setup Constants
        sex = patient.sex.lower()
        coeffs = _SCORE2_COEFFICIENTS[sex]
        base_surv = _SCORE2_BASELINE_SURVIVAL[sex]
        scales = _SCORE2_CALIBRATION_SCALES[sex][patient.region]

        # 2. Transform Inputs
        cage = (patient.age - 60) / 5
        csbp = (patient.systolic_bp - 120) / 20
        ctchol = patient.total_cholesterol - 6
        chdl = (patient.hdl_cholesterol - 1.3) / 0.5
        smoking = float(patient.smoking)

        # 3. Linear Predictor
        lin_pred = (
            coeffs["cage"] * cage
            + coeffs["smoking"] * smoking
            + coeffs["csbp"] * csbp
            + coeffs["ctchol"] * ctchol
            + coeffs["chdl"] * chdl
            + coeffs["smoking_cage"] * smoking * cage
            + coeffs["csbp_cage"] * csbp * cage
            + coeffs["ctchol_cage"] * ctchol * cage
            + coeffs["chdl_cage"] * chdl * cage
        )

        # 4. Uncalibrated Risk
        uncalibrated_risk = 1.0 - (base_surv ** np.exp(lin_pred))
        
        # Clip to avoid log errors
        uncalibrated_risk = np.clip(uncalibrated_risk, 1e-9, 1.0 - 1e-9)

        # 5. Calibration
        ln_neg_ln = np.log(-np.log(1 - uncalibrated_risk))
        calib_risk = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))
        
        risk_percent = np.clip(calib_risk * 100.0, 0.0, 100.0)

        # 6. Categorize (Age Adjusted)
        low_cut, high_cut = self._get_risk_thresholds(patient.age)
        
        if risk_percent < low_cut:
            cat = "low-to-moderate"
        elif risk_percent < high_cut:
            cat = "high" # Note: In SCORE2 terminology, the gap is often "high"
        else:
            cat = "very_high"
            
        # Simplified bucketing to match standard output expectations
        # You may adjust these strings to match your specific UI needs
        if risk_percent < low_cut: final_cat = "low"
        elif risk_percent < high_cut: final_cat = "moderate" 
        elif risk_percent < (high_cut * 2): final_cat = "high"
        else: final_cat = "very_high"

        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=final_cat,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={"raw_uncalibrated": uncalibrated_risk}
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates by grouping (Sex, Region) for efficiency.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking", "region"]
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
            coeffs = _SCORE2_COEFFICIENTS[sex]
            base_surv = _SCORE2_BASELINE_SURVIVAL[sex]
            
            # Sub-loop by Region (Scale change)
            # We only loop regions present in the data to save time
            present_regions = results.loc[sex_mask, "region"].unique()
            
            for region in present_regions:
                if region not in _SCORE2_CALIBRATION_SCALES[sex]:
                    continue # Skip invalid regions or handle error
                
                # Mask for Sex AND Region
                mask = sex_mask & (results["region"] == region)
                data = results[mask]
                scales = _SCORE2_CALIBRATION_SCALES[sex][region]

                # Vectorized Inputs
                cage = (data["age"] - 60) / 5
                csbp = (data["systolic_bp"] - 120) / 20
                ctchol = data["total_cholesterol"] - 6
                chdl = (data["hdl_cholesterol"] - 1.3) / 0.5
                smoking = data["smoking"].astype(float)

                # Vectorized Linear Predictor
                lin_pred = (
                    coeffs["cage"] * cage
                    + coeffs["smoking"] * smoking
                    + coeffs["csbp"] * csbp
                    + coeffs["ctchol"] * ctchol
                    + coeffs["chdl"] * chdl
                    + coeffs["smoking_cage"] * (smoking * cage)
                    + coeffs["csbp_cage"] * (csbp * cage)
                    + coeffs["ctchol_cage"] * (ctchol * cage)
                    + coeffs["chdl_cage"] * (chdl * cage)
                )

                # Vectorized Uncalibrated Risk
                uncalib = 1.0 - (base_surv ** np.exp(lin_pred))
                uncalib = np.clip(uncalib, 1e-9, 1.0 - 1e-9)

                # Vectorized Calibration
                ln_neg_ln = np.log(-np.log(1 - uncalib))
                calib = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))
                
                # Assign Results
                results.loc[mask, "risk_score"] = np.clip(calib * 100.0, 0.0, 100.0)

        # Vectorized Categorization (Age Specific)
        low_cuts, high_cuts = self._get_risk_thresholds(results["age"].values)
        
        # Numpy Select for Classification
        conditions = [
            results["risk_score"] < low_cuts,
            (results["risk_score"] >= low_cuts) & (results["risk_score"] < high_cuts),
            (results["risk_score"] >= high_cuts) & (results["risk_score"] < (high_cuts * 2)),
            results["risk_score"] >= (high_cuts * 2)
        ]
        choices = ["low", "moderate", "high", "very_high"]
        
        results["risk_category"] = np.select(conditions, choices, default="unknown")
        
        return results