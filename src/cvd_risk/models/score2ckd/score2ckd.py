"""
SCORE2-CKD (Systematic COronary Risk Evaluation 2 for Chronic Kidney Disease) cardiovascular risk model.

SCORE2-CKD is an adaptation of the SCORE2 cardiovascular risk prediction model
specifically designed for patients with chronic kidney disease. It incorporates
kidney function measures (eGFR, albuminuria) and accounts for the elevated
cardiovascular risk associated with CKD.

Reference:
    SCORE2-CKD Working Group. SCORE2-CKD: a cardiovascular risk prediction model
    for patients with chronic kidney disease. European Heart Journal. 2023.
    DOI: 10.1093/eurheartj/ehad219
"""

import logging
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# SCORE2 baseline survival probabilities at 10 years (age < 70)
_SCORE2_BASELINE_SURVIVAL = {
    "male": 0.9605,
    "female": 0.9776,
}

# SCORE2-OP baseline survival probabilities at 10 years (age >= 70)
_SCORE2_OP_BASELINE_SURVIVAL = {
    "male": 0.7576,
    "female": 0.8082,
}

# SCORE2 calibration scales (age < 70)
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

# SCORE2-OP calibration scales (age >= 70)
_SCORE2_OP_CALIBRATION_SCALES = {
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

# SCORE2 coefficients (age < 70)
_SCORE2_COEFFICIENTS = {
    "male": {
        "cage": 0.3742,
        "smoking": 0.6012,
        "csbp": 0.2777,
        "diabetes": 0.6457,
        "ctchol": 0.1458,
        "chdl": -0.2698,
        "smoking_cage": -0.0755,
        "csbp_cage": -0.0255,
        "diabetes_cage": -0.0983,
        "ctchol_cage": -0.0281,
        "chdl_cage": 0.0426,
    },
    "female": {
        "cage": 0.4648,
        "smoking": 0.7744,
        "csbp": 0.3131,
        "diabetes": 0.8096,
        "ctchol": 0.1002,
        "chdl": -0.2606,
        "smoking_cage": -0.1088,
        "csbp_cage": -0.0277,
        "diabetes_cage": -0.1272,
        "ctchol_cage": -0.0226,
        "chdl_cage": 0.0613,
    },
}

# SCORE2-OP coefficients (age >= 70)
_SCORE2_OP_COEFFICIENTS = {
    "male": {
        "cage": 0.0634,
        "diabetes": 0.4245,
        "smoking": 0.3524,
        "csbp": 0.0094,
        "ctchol": 0.0850,
        "chdl": -0.3564,
        "diabetes_cage": -0.0174,
        "smoking_cage": -0.0247,
        "csbp_cage": -0.0005,
        "ctchol_cage": 0.0073,
        "chdl_cage": 0.0091,
    },
    "female": {
        "cage": 0.0789,
        "diabetes": 0.6010,
        "smoking": 0.4921,
        "csbp": 0.0102,
        "ctchol": 0.0605,
        "chdl": -0.3040,
        "diabetes_cage": -0.0107,
        "smoking_cage": -0.0255,
        "csbp_cage": -0.0004,
        "ctchol_cage": -0.0009,
        "chdl_cage": 0.0154,
    },
}

# CKD adjustment constants
_CKD_CONSTANTS = {
    "exeGFR_offset": 87.8980,
    "exeGFR_age": -3.7891,
    "exeGFR_female": -0.7023,
    "exeGFR_tchol": -0.2941,
    "exeGFR_hdl": 1.0960,
    "exeGFR_sbp": -0.1364,
    "exeGFR_diabetes": 0.1205,
    "exeGFR_smoking": 1.3211,
    "exeGFR_age_tchol": 0.0555,
    "exeGFR_age_hdl": 0.1717,
    "exeGFR_age_sbp": 0.0059,
    "exeGFR_age_diabetes": -0.8994,
    "exeGFR_age_smoking": 0.2181,
    "egfr_adjust_age_under_70": {
        "beta1": 0.4713,
        "beta2": 0.0956,
        "beta3": -0.0802,
        "beta4": 0.0088,
    },
    "egfr_adjust_age_70_plus": {
        "beta1": 0.3072,
        "beta2": 0.0942,
        "beta3": -0.4616,
        "beta4": -0.0127,
        "beta5": -0.0098,
        "beta6": -0.0075,
    },
    "exACR_intercept": 0.9775,
    "exACR_age": 0.0159,
    "exACR_female": 0.0308,
    "exACR_tchol": 0.0185,
    "exACR_hdl": -0.0274,
    "exACR_sbp": 0.1339,
    "exACR_diabetes": 0.2171,
    "exACR_smoking": 0.0629,
    "exACR_age_tchol": -0.0062,
    "exACR_age_hdl": 0.0003,
    "exACR_age_sbp": 0.0008,
    "exACR_age_diabetes": -0.0109,
    "exACR_age_smoking": 0.0085,
    "exACR_egfr_under_60": 0.4057,
    "exACR_egfr_60_30": 0.0597,
    "exACR_egfr_over_90": -0.0916,
    "acr_adjust_age_under_70": 0.2432,
    "acr_adjust_age_70_plus": 0.2370,
    "dipstick_coefficients": {
        "negative": 0.0,
        "trace": 0.2644,
        "1+": 0.4126,
        "2+": 0.4761,
        "3+": 0.4761,
        "4+": 0.4761,
    },
}


class SCORE2CKD(RiskModel):
    """
    SCORE2-CKD cardiovascular risk prediction model for patients with chronic kidney disease.

    This model extends SCORE2 with CKD-specific adjustments for eGFR, albuminuria,
    and proteinuria measures. Valid for ages 40-69 (SCORE2) and 70+ (SCORE2-OP).
    """

    model_name = "SCORE2-CKD"
    model_version = "2023"
    supported_regions = ["low", "moderate", "high", "very_high"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.age < 40:
            logger.warning(
                f"Age {patient.age} below optimal range [40, ∞). "
                "SCORE2-CKD accuracy may be reduced."
            )
        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")

        # CKD-specific validations
        if patient.egfr is None:
            raise ValueError("eGFR is required for SCORE2-CKD calculation")
        if patient.egfr <= 0:
            raise ValueError("eGFR must be positive")

        # At least one of ACR or proteinuria trace should be provided
        if patient.acr is None and patient.proteinuria_trace is None:
            logger.warning(
                "Neither ACR nor proteinuria trace provided. "
                "Only eGFR adjustment will be applied."
            )
        elif patient.acr is not None and patient.acr <= 0:
            raise ValueError("ACR must be positive if provided")
        elif patient.proteinuria_trace is not None and patient.proteinuria_trace not in _CKD_CONSTANTS["dipstick_coefficients"]:
            raise ValueError(f"Proteinuria trace must be one of {list(_CKD_CONSTANTS['dipstick_coefficients'].keys())}")

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

    def _calculate_expected_egfr(self, patient: PatientData) -> float:
        """Calculate expected eGFR based on patient characteristics."""
        sex = patient.sex.lower()
        cage = (patient.age - 60) / 5

        exeGFR = (
            _CKD_CONSTANTS["exeGFR_offset"]
            + _CKD_CONSTANTS["exeGFR_age"] * cage
            + (0 if sex == "male" else _CKD_CONSTANTS["exeGFR_female"])
            + _CKD_CONSTANTS["exeGFR_tchol"] * patient.total_cholesterol
            + _CKD_CONSTANTS["exeGFR_hdl"] * patient.hdl_cholesterol
            + _CKD_CONSTANTS["exeGFR_sbp"] * patient.systolic_bp
            + _CKD_CONSTANTS["exeGFR_diabetes"] * float(patient.diabetes)
            + _CKD_CONSTANTS["exeGFR_smoking"] * float(patient.smoking)
            + _CKD_CONSTANTS["exeGFR_age_tchol"] * cage * patient.total_cholesterol
            + _CKD_CONSTANTS["exeGFR_age_hdl"] * cage * patient.hdl_cholesterol
            + _CKD_CONSTANTS["exeGFR_age_sbp"] * cage * patient.systolic_bp
            + _CKD_CONSTANTS["exeGFR_age_diabetes"] * cage * float(patient.diabetes)
            + _CKD_CONSTANTS["exeGFR_age_smoking"] * cage * float(patient.smoking)
        )
        return exeGFR

    def _calculate_expected_acr(self, patient: PatientData, exeGFR: float) -> float:
        """Calculate expected ACR (albumin-to-creatinine ratio)."""
        sex = patient.sex.lower()
        cage = (patient.age - 60) / 5

        exACR = 8 ** (
        _CKD_CONSTANTS["exACR_intercept"]
        + _CKD_CONSTANTS["exACR_age"] * cage
        + (0 if sex == "male" else _CKD_CONSTANTS["exACR_female"])
        + _CKD_CONSTANTS["exACR_tchol"] * (patient.total_cholesterol - 6)
        + _CKD_CONSTANTS["exACR_hdl"] * (patient.hdl_cholesterol - 1.3) / 0.5
        + _CKD_CONSTANTS["exACR_sbp"] * (patient.systolic_bp - 120) / 20
        + _CKD_CONSTANTS["exACR_diabetes"] * float(patient.diabetes)
        + _CKD_CONSTANTS["exACR_smoking"] * float(patient.smoking)
        + _CKD_CONSTANTS["exACR_age_tchol"] * cage * (patient.total_cholesterol - 6)
        + _CKD_CONSTANTS["exACR_age_hdl"] * cage * (patient.hdl_cholesterol - 1.3) / 0.5
        + _CKD_CONSTANTS["exACR_age_sbp"] * cage * (patient.systolic_bp - 120) / 20
        + _CKD_CONSTANTS["exACR_age_diabetes"] * cage * float(patient.diabetes)
        + _CKD_CONSTANTS["exACR_age_smoking"] * cage * float(patient.smoking)
        + _CKD_CONSTANTS["exACR_egfr_under_60"] * min(patient.egfr - 60, 0) / -15
        + _CKD_CONSTANTS["exACR_egfr_60_30"] * min(max(patient.egfr - 60, 0), 30) / -15
        + _CKD_CONSTANTS["exACR_egfr_over_90"] * max(patient.egfr - 90, 0) / -15
        )
        return exACR

    def _apply_egfr_adjustment(self, base_risk: float, patient: PatientData, exeGFR: float) -> float:
        """Apply eGFR-based risk adjustment."""
        if patient.age < 70:
            const = _CKD_CONSTANTS["egfr_adjust_age_under_70"]
            adjustment = (
                const["beta1"] * (min(patient.egfr, 60) / -15 - min(exeGFR, 60) / -15)
                + const["beta2"] * (min(max(patient.egfr - 60, 0), 30) / -15 - min(max(exeGFR - 60, 0), 30) / -15)
                - const["beta3"] * (patient.age - 60) / 5 * (min(patient.egfr, 60) / -15 - min(exeGFR, 60) / -15)
                + const["beta4"] * (patient.age - 60) / 5 * (min(max(patient.egfr - 60, 0), 30) / -15 - min(max(exeGFR - 60, 0), 30) / -15)
            )
        else:
            const = _CKD_CONSTANTS["egfr_adjust_age_70_plus"]
            adjustment = (
                const["beta1"] * (min(patient.egfr, 60) / -15 - min(exeGFR, 60) / -15)
                + const["beta2"] * (min(max(patient.egfr - 60, 0), 30) / -15 - min(max(exeGFR - 60, 0), 30) / -15)
                - const["beta3"] * (max(patient.egfr - 90, 0) / -15 - max(exeGFR - 90, 0) / -15)
                - const["beta4"] * (patient.age - 73) * (min(patient.egfr, 60) / -15 - min(exeGFR, 60) / -15)
                - const["beta5"] * (patient.age - 73) * (min(max(patient.egfr - 60, 0), 30) / -15 - min(max(exeGFR - 60, 0), 30) / -15)
                - const["beta6"] * (patient.age - 73) * (max(patient.egfr - 90, 0) / -15 - max(exeGFR - 90, 0) / -15)
            )

        adjusted_risk = 1 - (1 - base_risk) ** np.exp(adjustment)
        return adjusted_risk

    def _apply_acr_adjustment(self, egfr_adjusted_risk: float, patient: PatientData, exACR: float) -> float:
        """Apply ACR-based risk adjustment."""
        if patient.acr is None:
            return egfr_adjusted_risk

        beta = _CKD_CONSTANTS["acr_adjust_age_under_70"] if patient.age < 70 else _CKD_CONSTANTS["acr_adjust_age_70_plus"]
        adjustment = beta * (np.log(patient.acr) - np.log(exACR)) / np.log(8)

        acr_adjusted_risk = 1 - (1 - egfr_adjusted_risk) ** np.exp(adjustment)
        return acr_adjusted_risk

    def _apply_dipstick_adjustment(self, egfr_adjusted_risk: float, patient: PatientData) -> float:
        """Apply proteinuria dipstick-based risk adjustment."""
        if patient.proteinuria_trace is None:
            return egfr_adjusted_risk

        beta = _CKD_CONSTANTS["dipstick_coefficients"][patient.proteinuria_trace]
        dipstick_adjusted_risk = 1 - (1 - egfr_adjusted_risk) ** np.exp(beta)
        return dipstick_adjusted_risk

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # Determine which SCORE2 version to use
        is_older_population = patient.age >= 70
        sex = patient.sex.lower()

        if is_older_population:
            coeffs = _SCORE2_OP_COEFFICIENTS[sex]
            base_surv = _SCORE2_OP_BASELINE_SURVIVAL[sex]
            scales = _SCORE2_OP_CALIBRATION_SCALES[sex][patient.region]
            cage = patient.age - 73
            csbp = (patient.systolic_bp - 150) / 20
            ctchol = patient.total_cholesterol - 6
            chdl = (patient.hdl_cholesterol - 1.4) / 0.5
        else:
            coeffs = _SCORE2_COEFFICIENTS[sex]
            base_surv = _SCORE2_BASELINE_SURVIVAL[sex]
            scales = _SCORE2_CALIBRATION_SCALES[sex][patient.region]
            cage = (patient.age - 60) / 5
            csbp = (patient.systolic_bp - 120) / 20
            ctchol = patient.total_cholesterol - 6
            chdl = (patient.hdl_cholesterol - 1.3) / 0.5

        smoking = float(patient.smoking)
        diabetes = float(patient.diabetes)

        # Linear predictor
        if is_older_population:
            lin_pred = (
                coeffs["cage"] * cage
                + coeffs["diabetes"] * diabetes
                + coeffs["smoking"] * smoking
                + coeffs["csbp"] * csbp
                + coeffs["ctchol"] * ctchol
                + coeffs["chdl"] * chdl
                + coeffs["diabetes_cage"] * diabetes * cage
                + coeffs["smoking_cage"] * smoking * cage
                + coeffs["csbp_cage"] * csbp * cage
                + coeffs["ctchol_cage"] * ctchol * cage
                + coeffs["chdl_cage"] * chdl * cage
            )
            # SCORE2-OP has an additional offset
            offset = -0.0929 if sex == "male" else -0.229
            lin_pred += offset
        else:
            lin_pred = (
                coeffs["cage"] * cage
                + coeffs["smoking"] * smoking
                + coeffs["csbp"] * csbp
                + coeffs["diabetes"] * diabetes
                + coeffs["ctchol"] * ctchol
                + coeffs["chdl"] * chdl
                + coeffs["smoking_cage"] * smoking * cage
                + coeffs["csbp_cage"] * csbp * cage
                + coeffs["diabetes_cage"] * diabetes * cage
                + coeffs["ctchol_cage"] * ctchol * cage
                + coeffs["chdl_cage"] * chdl * cage
            )

        # Uncalibrated risk
        uncalibrated_risk = 1.0 - (base_surv ** np.exp(lin_pred))
        uncalibrated_risk = np.clip(uncalibrated_risk, 1e-9, 1.0 - 1e-9)

        # Calibration
        ln_neg_ln = np.log(-np.log(1 - uncalibrated_risk))
        base_risk = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))
        base_risk = np.clip(base_risk, 0.0, 1.0)

        # CKD adjustments
        exeGFR = self._calculate_expected_egfr(patient)
        egfr_adjusted_risk = self._apply_egfr_adjustment(base_risk, patient, exeGFR)

        # Apply ACR or dipstick adjustment
        if patient.acr is not None:
            exACR = self._calculate_expected_acr(patient, exeGFR)
            ckd_risk = self._apply_acr_adjustment(egfr_adjusted_risk, patient, exACR)
        elif patient.proteinuria_trace is not None:
            ckd_risk = self._apply_dipstick_adjustment(egfr_adjusted_risk, patient)
        else:
            ckd_risk = egfr_adjusted_risk

        risk_percent = np.clip(ckd_risk * 100.0, 0.0, 100.0)

        # Categorize
        low_cut, high_cut = self._get_risk_thresholds(patient.age)

        if risk_percent < low_cut:
            cat = "low-to-moderate"
        elif risk_percent < high_cut:
            cat = "high"
        else:
            cat = "very_high"

        # Simplified bucketing
        if risk_percent < low_cut:
            final_cat = "low"
        elif risk_percent < high_cut:
            final_cat = "moderate"
        elif risk_percent < (high_cut * 2):
            final_cat = "high"
        else:
            final_cat = "very_high"

        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=final_cat,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "base_score2_risk": float(base_risk * 100),
                "egfr_adjusted_risk": float(egfr_adjusted_risk * 100),
                "exeGFR": float(exeGFR),
                "exACR": float(exACR) if patient.acr is not None else None,
                "proteinuria_trace": patient.proteinuria_trace,
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates by grouping (Sex, Region, Age Group) for efficiency.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol",
                   "smoking", "diabetes", "region", "egfr"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        # Check for CKD parameters
        has_acr = "acr" in df.columns
        has_trace = "proteinuria_trace" in df.columns

        if not has_acr and not has_trace:
            logger.warning("Neither ACR nor proteinuria_trace columns found. Only eGFR adjustment will be applied.")

        results = df.copy()
        results["risk_score"] = np.nan
        results["model_name"] = self.model_name
        results["egfr_adjusted_risk"] = np.nan
        results["exeGFR"] = np.nan
        if has_acr:
            results["exACR"] = np.nan

        # Process by age group first (SCORE2 vs SCORE2-OP)
        for is_older_pop in [False, True]:
            age_mask = results["age"] >= 70 if is_older_pop else results["age"] < 70
            if not age_mask.any():
                continue

            age_data = results[age_mask]

            # Process by Sex
            for sex in ["male", "female"]:
                sex_mask = age_data["sex"].str.lower() == sex
                if not sex_mask.any():
                    continue

                sex_data = age_data[sex_mask]

                # Get sex-specific constants
                if is_older_pop:
                    coeffs = _SCORE2_OP_COEFFICIENTS[sex]
                    base_surv = _SCORE2_OP_BASELINE_SURVIVAL[sex]
                    offset = -0.0929 if sex == "male" else -0.229
                else:
                    coeffs = _SCORE2_COEFFICIENTS[sex]
                    base_surv = _SCORE2_BASELINE_SURVIVAL[sex]
                    offset = 0.0

                # Process by Region
                present_regions = sex_data["region"].unique()
                for region in present_regions:
                    if region not in self.supported_regions:
                        continue

                    region_mask = sex_data["region"] == region
                    data = sex_data[region_mask]

                    if is_older_pop:
                        scales = _SCORE2_OP_CALIBRATION_SCALES[sex][region]
                        cage = data["age"] - 73
                        csbp = (data["systolic_bp"] - 150) / 20
                        ctchol = data["total_cholesterol"] - 6
                        chdl = (data["hdl_cholesterol"] - 1.4) / 0.5
                    else:
                        scales = _SCORE2_CALIBRATION_SCALES[sex][region]
                        cage = (data["age"] - 60) / 5
                        csbp = (data["systolic_bp"] - 120) / 20
                        ctchol = data["total_cholesterol"] - 6
                        chdl = (data["hdl_cholesterol"] - 1.3) / 0.5

                    smoking = data["smoking"].astype(float)
                    diabetes = data["diabetes"].astype(float)

                    # Vectorized linear predictor
                    if is_older_pop:
                        lin_pred = (
                            coeffs["cage"] * cage
                            + coeffs["diabetes"] * diabetes
                            + coeffs["smoking"] * smoking
                            + coeffs["csbp"] * csbp
                            + coeffs["ctchol"] * ctchol
                            + coeffs["chdl"] * chdl
                            + coeffs["diabetes_cage"] * diabetes * cage
                            + coeffs["smoking_cage"] * smoking * cage
                            + coeffs["csbp_cage"] * csbp * cage
                            + coeffs["ctchol_cage"] * ctchol * cage
                            + coeffs["chdl_cage"] * chdl * cage
                            + offset
                        )
                    else:
                        lin_pred = (
                            coeffs["cage"] * cage
                            + coeffs["smoking"] * smoking
                            + coeffs["csbp"] * csbp
                            + coeffs["diabetes"] * diabetes
                            + coeffs["ctchol"] * ctchol
                            + coeffs["chdl"] * chdl
                            + coeffs["smoking_cage"] * smoking * cage
                            + coeffs["csbp_cage"] * csbp * cage
                            + coeffs["diabetes_cage"] * diabetes * cage
                            + coeffs["ctchol_cage"] * ctchol * cage
                            + coeffs["chdl_cage"] * chdl * cage
                        )

                    # Vectorized uncalibrated risk
                    uncalib = 1.0 - (base_surv ** np.exp(lin_pred))
                    uncalib = np.clip(uncalib, 1e-9, 1.0 - 1e-9)

                    # Vectorized calibration
                    ln_neg_ln = np.log(-np.log(1 - uncalib))
                    base_risk = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))
                    base_risk = np.clip(base_risk, 0.0, 1.0)

                    # CKD adjustments
                    # Calculate expected eGFR
                    exeGFR = (
                        _CKD_CONSTANTS["exeGFR_offset"]
                        + _CKD_CONSTANTS["exeGFR_age"] * cage
                        + (0 if sex == "male" else _CKD_CONSTANTS["exeGFR_female"])
                        + _CKD_CONSTANTS["exeGFR_tchol"] * data["total_cholesterol"]
                        + _CKD_CONSTANTS["exeGFR_hdl"] * data["hdl_cholesterol"]
                        + _CKD_CONSTANTS["exeGFR_sbp"] * data["systolic_bp"]
                        + _CKD_CONSTANTS["exeGFR_diabetes"] * diabetes
                        + _CKD_CONSTANTS["exeGFR_smoking"] * smoking
                        + _CKD_CONSTANTS["exeGFR_age_tchol"] * cage * data["total_cholesterol"]
                        + _CKD_CONSTANTS["exeGFR_age_hdl"] * cage * data["hdl_cholesterol"]
                        + _CKD_CONSTANTS["exeGFR_age_sbp"] * cage * data["systolic_bp"]
                        + _CKD_CONSTANTS["exeGFR_age_diabetes"] * cage * diabetes
                        + _CKD_CONSTANTS["exeGFR_age_smoking"] * cage * smoking
                    )

                    # eGFR adjustment
                    if is_older_pop:
                        const = _CKD_CONSTANTS["egfr_adjust_age_70_plus"]
                        egfr_adj = (
                            const["beta1"] * (np.minimum(data["egfr"], 60) / -15 - np.minimum(exeGFR, 60) / -15)
                            + const["beta2"] * (np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                               np.minimum(np.maximum(exeGFR - 60, 0), 30) / -15)
                            - const["beta3"] * (np.maximum(data["egfr"] - 90, 0) / -15 - np.maximum(exeGFR - 90, 0) / -15)
                            - const["beta4"] * (data["age"] - 73) * (np.minimum(data["egfr"], 60) / -15 - np.minimum(exeGFR, 60) / -15)
                            - const["beta5"] * (data["age"] - 73) * (np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                               np.minimum(np.maximum(exeGFR - 60, 0), 30) / -15)
                            - const["beta6"] * (data["age"] - 73) * (np.maximum(data["egfr"] - 90, 0) / -15 - np.maximum(exeGFR - 90, 0) / -15)
                        )
                    else:
                        const = _CKD_CONSTANTS["egfr_adjust_age_under_70"]
                        egfr_adj = (
                            const["beta1"] * (np.minimum(data["egfr"], 60) / -15 - np.minimum(exeGFR, 60) / -15)
                            + const["beta2"] * (np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                               np.minimum(np.maximum(exeGFR - 60, 0), 30) / -15)
                            - const["beta3"] * cage * (np.minimum(data["egfr"], 60) / -15 - np.minimum(exeGFR, 60) / -15)
                            + const["beta4"] * cage * (np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                               np.minimum(np.maximum(exeGFR - 60, 0), 30) / -15)
                        )

                    egfr_adjusted_risk = 1 - (1 - base_risk) ** np.exp(egfr_adj)

                    # ACR or dipstick adjustment
                    final_risk = egfr_adjusted_risk.copy()

                    if has_acr and not data["acr"].isna().all():
                        # Calculate expected ACR
                        exACR = 8 ** (
                            _CKD_CONSTANTS["exACR_intercept"]
                            + _CKD_CONSTANTS["exACR_age"] * cage
                            + (0 if sex == "male" else _CKD_CONSTANTS["exACR_female"])
                            + _CKD_CONSTANTS["exACR_tchol"] * (data["total_cholesterol"] - 6)
                            + _CKD_CONSTANTS["exACR_hdl"] * (data["hdl_cholesterol"] - 1.3) / 0.5
                            + _CKD_CONSTANTS["exACR_sbp"] * (data["systolic_bp"] - 120) / 20
                            + _CKD_CONSTANTS["exACR_diabetes"] * diabetes
                            + _CKD_CONSTANTS["exACR_smoking"] * smoking
                            + _CKD_CONSTANTS["exACR_age_tchol"] * cage * (data["total_cholesterol"] - 6)
                            + _CKD_CONSTANTS["exACR_age_hdl"] * cage * (data["hdl_cholesterol"] - 1.3) / 0.5
                            + _CKD_CONSTANTS["exACR_age_sbp"] * cage * (data["systolic_bp"] - 120) / 20
                            + _CKD_CONSTANTS["exACR_age_diabetes"] * cage * diabetes
                            + _CKD_CONSTANTS["exACR_age_smoking"] * cage * smoking
                            + _CKD_CONSTANTS["exACR_egfr_under_60"] * np.minimum(data["egfr"] - 60, 0) / -15
                            + _CKD_CONSTANTS["exACR_egfr_60_30"] * np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15
                            + _CKD_CONSTANTS["exACR_egfr_over_90"] * np.maximum(data["egfr"] - 90, 0) / -15
                        )

                        # ACR adjustment
                        beta_acr = _CKD_CONSTANTS["acr_adjust_age_under_70"] if not is_older_pop else _CKD_CONSTANTS["acr_adjust_age_70_plus"]
                        acr_adj = beta_acr * (np.log(data["acr"]) - np.log(exACR)) / np.log(8)
                        final_risk = 1 - (1 - egfr_adjusted_risk) ** np.exp(acr_adj)

                        # Store exACR for valid ACR values
                        results.loc[data.index, "exACR"] = exACR

                    elif has_trace and not data["proteinuria_trace"].isna().all():
                        # Dipstick adjustment - vectorized
                        dipstick_coeffs = data["proteinuria_trace"].map(_CKD_CONSTANTS["dipstick_coefficients"]).fillna(0)
                        final_risk = 1 - (1 - egfr_adjusted_risk) ** np.exp(dipstick_coeffs)

                    # Assign results
                    results.loc[data.index, "risk_score"] = np.clip(final_risk * 100.0, 0.0, 100.0)
                    results.loc[data.index, "egfr_adjusted_risk"] = egfr_adjusted_risk * 100.0
                    results.loc[data.index, "exeGFR"] = exeGFR

        # Vectorized categorization
        low_cuts, high_cuts = self._get_risk_thresholds(results["age"].values)

        conditions = [
            results["risk_score"] < low_cuts,
            (results["risk_score"] >= low_cuts) & (results["risk_score"] < high_cuts),
            (results["risk_score"] >= high_cuts) & (results["risk_score"] < (high_cuts * 2)),
            results["risk_score"] >= (high_cuts * 2)
        ]
        choices = ["low", "moderate", "high", "very_high"]

        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results

