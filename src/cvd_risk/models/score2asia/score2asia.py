"""
SCORE2-Asia CKD cardiovascular risk model.

SCORE2-Asia CKD is the SCORE2 model with Chronic Kidney Disease (CKD) adjustments
specifically calibrated for Asian populations. It incorporates eGFR (estimated glomerular
filtration rate) and ACR (albumin-creatinine ratio) or dipstick protein measurements
to adjust cardiovascular risk estimates.

Reference:
    Kunihiro M, et al. (2022). SCORE2-Asia risk prediction algorithms: a recalibration
    of SCORE2 risk prediction algorithms based on a large cohort study in Asia.
    European Journal of Preventive Cardiology, 29(18), 2494-2505.
    DOI: 10.1093/eurjpc/zwac176
"""

import logging
from typing import Literal, Tuple, Union, Optional

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

# SCORE2-Asia CKD sex-specific coefficients (includes diabetes)
_SCORE2_CKD_COEFFICIENTS = {
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
        "diabetes": 0.6457,
        "diabetes_cage": -0.0983,
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
        "diabetes": 0.8096,
        "diabetes_cage": -0.1272,
    },
}

# SCORE2-OP (older persons) coefficients for age >= 70
_SCORE2_OP_COEFFICIENTS = {
    "male": {
        "cage": 0.0634,
        "smoking": 0.3524,
        "csbp": 0.0094,
        "ctchol": 0.0850,
        "chdl": -0.3564,
        "smoking_cage": -0.0247,
        "csbp_cage": -0.0005,
        "ctchol_cage": 0.0073,
        "chdl_cage": 0.0091,
        "diabetes": 0.4245,
        "diabetes_cage": -0.0174,
    },
    "female": {
        "cage": 0.0789,
        "smoking": 0.4921,
        "csbp": 0.0102,
        "ctchol": 0.0605,
        "chdl": -0.3040,
        "smoking_cage": -0.0255,
        "csbp_cage": -0.0004,
        "ctchol_cage": -0.0009,
        "chdl_cage": 0.0154,
        "diabetes": 0.6010,
        "diabetes_cage": -0.0107,
    },
}


class SCORE2AsiaCKD(RiskModel):
    """
    SCORE2-Asia CKD cardiovascular risk prediction model.

    This model extends SCORE2 with Chronic Kidney Disease adjustments for Asian populations,
    incorporating eGFR and ACR measurements to refine cardiovascular risk estimates.

    Valid for ages 40-69 (SCORE2) and 70+ (SCORE2-OP).
    """

    model_name = "SCORE2-Asia CKD"
    model_version = "2022"
    supported_regions = ["low", "moderate", "high", "very_high"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")

        # Check for CKD-specific parameters
        if not hasattr(patient, 'egfr') or patient.egfr is None:
            raise ValueError("eGFR (estimated glomerular filtration rate) is required for SCORE2-Asia CKD")
        if patient.egfr <= 0:
            raise ValueError("eGFR must be positive")

        # Check if either ACR or trace is provided for CKD adjustment
        has_acr = hasattr(patient, 'acr') and patient.acr is not None
        has_trace = hasattr(patient, 'proteinuria_trace') and patient.proteinuria_trace is not None

        if not has_acr and not has_trace:
            logger.warning(
                "Neither ACR nor proteinuria trace provided. Using eGFR-only CKD adjustment."
            )

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
        """
        Calculate expected eGFR based on patient characteristics.
        Used for CKD risk adjustment.
        """
        sex = patient.sex.lower()
        age = patient.age

        # Expected eGFR calculation
        exeGFR = 87.8980
        exeGFR -= 3.7891 * (age - 60) / 5
        exeGFR -= (1 if sex == "female" else 0) * 0.7023
        exeGFR -= 0.2941 * (patient.total_cholesterol - 6)
        exeGFR += 1.0960 * (patient.hdl_cholesterol - 1.3) / 0.5
        exeGFR -= 0.1364 * (patient.systolic_bp - 120) / 20
        exeGFR += 0.1205 * float(patient.diabetes)
        exeGFR += 1.3211 * float(patient.smoking)

        # Interaction terms
        exeGFR += 0.0555 * ((age - 60) / 5) * (patient.total_cholesterol - 6)
        exeGFR += 0.1717 * ((age - 60) / 5) * (patient.hdl_cholesterol - 1.3) / 0.5
        exeGFR += 0.0059 * ((age - 60) / 5) * (patient.systolic_bp - 120) / 20
        exeGFR -= 0.8994 * ((age - 60) / 5) * float(patient.diabetes)
        exeGFR += 0.2181 * ((age - 60) / 5) * float(patient.smoking)

        return exeGFR

    def _calculate_expected_acr(self, patient: PatientData) -> float:
        """
        Calculate expected ACR (albumin-creatinine ratio) based on patient characteristics.
        """
        sex = patient.sex.lower()
        age = patient.age
        egfr = patient.egfr

        # Expected ACR calculation (base 8 for log transformation)
        exACR = 8 ** (1 - 0.0225)
        exACR *= 8 ** (0.0159 * (age - 60) / 5)
        exACR *= 8 ** ((1 if sex == "female" else 0) * 0.0308)
        exACR *= 8 ** (0.0185 * (patient.total_cholesterol - 6))
        exACR *= 8 ** (-0.0274 * (patient.hdl_cholesterol - 1.3) / 0.5)
        exACR *= 8 ** (0.1339 * (patient.systolic_bp - 120) / 20)
        exACR *= 8 ** (0.2171 * float(patient.diabetes))
        exACR *= 8 ** (0.0629 * float(patient.smoking))

        # Interaction terms
        exACR *= 8 ** (-0.0062 * (age - 60) / 5 * (patient.total_cholesterol - 6))
        exACR *= 8 ** (0.0003 * (age - 60) / 5 * (patient.hdl_cholesterol - 1.3) / 0.5)
        exACR *= 8 ** (0.0008 * (age - 60) / 5 * (patient.systolic_bp - 120) / 20)
        exACR *= 8 ** (-0.0109 * (age - 60) / 5 * float(patient.diabetes))
        exACR *= 8 ** (0.0085 * (age - 60) / 5 * float(patient.smoking))

        # eGFR terms
        exACR *= 8 ** (0.4057 * min(egfr - 60, 0) / -15)
        exACR *= 8 ** (0.0597 * min(max(egfr - 60, 0), 30) / -15)
        exACR *= 8 ** (-0.0916 * max(egfr - 90, 0) / -15)

        return exACR

    def _apply_ckd_adjustment(self, base_risk: float, patient: PatientData) -> float:
        """
        Apply CKD adjustment to the base SCORE2 risk using eGFR and ACR/trace.
        """
        age = patient.age
        egfr = patient.egfr
        exeGFR = self._calculate_expected_egfr(patient)

        # eGFR adjustment
        if age < 70:
            egfr_adjustment = np.exp(
                0.4713 * (min(egfr, 60) / -15 - min(exeGFR, 60) / -15) +
                0.0956 * (min(max(egfr - 60, 0), 30) / -15 - min(max(exeGFR - 60, 0), 30) / -15) -
                0.0802 * (age - 60) / 5 * (min(egfr, 60) / -15 - min(exeGFR, 60) / -15) +
                0.0088 * (age - 60) / 5 * (min(max(egfr - 60, 0), 30) / -15 - min(max(exeGFR - 60, 0), 30) / -15)
            )
        else:  # age >= 70
            egfr_adjustment = np.exp(
                0.3072 * (min(egfr, 60) / -15 - min(exeGFR, 60) / -15) +
                0.0942 * (min(max(egfr - 60, 0), 30) / -15 - min(max(exeGFR - 60, 0), 30) / -15) -
                0.4616 * (max(egfr - 90, 0) / -15 - max(exeGFR - 90, 0) / -15) -
                0.0127 * (age - 73) * (min(egfr, 60) / -15 - min(exeGFR, 60) / -15) -
                0.0098 * (age - 73) * (min(max(egfr - 60, 0), 30) / -15 - min(max(exeGFR - 60, 0), 30) / -15) -
                0.0075 * (age - 73) * (max(egfr - 90, 0) / -15 - max(exeGFR - 90, 0) / -15)
            )

        score_egfr = 1 - (1 - base_risk) ** egfr_adjustment

        # Check if ACR or trace is available
        has_acr = hasattr(patient, 'acr') and patient.acr is not None and patient.acr > 0
        has_trace = hasattr(patient, 'proteinuria_trace') and patient.proteinuria_trace is not None

        if not has_acr and not has_trace:
            # Use eGFR only
            return score_egfr

        # Apply ACR or trace adjustment
        if has_acr:
            # ACR adjustment
            acr = patient.acr
            exACR = self._calculate_expected_acr(patient)

            if age < 70:
                acr_adjustment = 0.2432 * (np.log(acr) / np.log(8) - np.log(exACR) / np.log(8))
            else:  # age >= 70
                acr_adjustment = 0.2370 * (np.log(acr) / np.log(8) - np.log(exACR) / np.log(8))

            score_ckd = 1 - (1 - score_egfr) ** np.exp(acr_adjustment)

        elif has_trace:
            # Dipstick trace adjustment
            trace = patient.proteinuria_trace.lower()

            # Map trace to coefficient
            trace_coeffs = {
                'negative': 0.0,
                'trace': 0.2644,
                '1+': 0.4126,
                '2+': 0.4761,
                '3+': 0.4761,
                '4+': 0.4761,
            }

            if trace not in trace_coeffs:
                logger.warning(f"Unknown trace value '{trace}', using 'negative'")
                trace_coeff = 0.0
            else:
                trace_coeff = trace_coeffs[trace]

            score_ckd = 1 - (1 - score_egfr) ** np.exp(trace_coeff)

        return score_ckd

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # 1. Setup Constants
        sex = patient.sex.lower()
        age = patient.age

        if age < 70:
            coeffs = _SCORE2_CKD_COEFFICIENTS[sex]
            base_surv = _SCORE2_BASELINE_SURVIVAL[sex]
        else:
            coeffs = _SCORE2_OP_COEFFICIENTS[sex]
            # SCORE2-OP uses different baseline survival
            base_surv = 0.7576 if sex == "male" else 0.8082

        scales = _SCORE2_CALIBRATION_SCALES[sex][patient.region]

        # 2. Transform Inputs
        if age < 70:
            cage = (age - 60) / 5
        else:
            cage = age - 73  # SCORE2-OP centering

        csbp = (patient.systolic_bp - 120) / 20
        ctchol = patient.total_cholesterol - 6
        chdl = (patient.hdl_cholesterol - 1.3) / 0.5
        smoking = float(patient.smoking)
        diabetes = float(patient.diabetes)

        # 3. Linear Predictor
        if age < 70:
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
                + coeffs["diabetes"] * diabetes
                + coeffs["diabetes_cage"] * diabetes * cage
            )
        else:
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
                + coeffs["diabetes"] * diabetes
                + coeffs["diabetes_cage"] * diabetes * cage
            )

        # 4. Uncalibrated Risk
        if age < 70:
            uncalibrated_risk = 1.0 - (base_surv ** np.exp(lin_pred))
        else:
            # SCORE2-OP uses different formula
            uncalibrated_risk = 1.0 - (base_surv ** np.exp(lin_pred - (-0.0929 if sex == "male" else -0.229)))

        # Clip to avoid log errors
        uncalibrated_risk = np.clip(uncalibrated_risk, 1e-9, 1.0 - 1e-9)

        # 5. Calibration
        ln_neg_ln = np.log(-np.log(1 - uncalibrated_risk))
        calib_risk = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))

        # 6. Apply CKD adjustment
        ckd_risk = self._apply_ckd_adjustment(calib_risk, patient)

        risk_percent = np.clip(ckd_risk * 100.0, 0.0, 100.0)

        # 7. Categorize (Age Adjusted)
        low_cut, high_cut = self._get_risk_thresholds(age)

        if risk_percent < low_cut:
            cat = "low"
        elif risk_percent < high_cut:
            cat = "moderate"
        else:
            cat = "high"

        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=cat,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "base_score2_risk": float(calib_risk * 100.0),
                "uncalibrated_risk": float(uncalibrated_risk * 100.0)
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates by grouping (Sex, Region) for efficiency.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol",
                   "smoking", "diabetes", "region", "egfr"]

        # Check for CKD parameters - at least one of acr or proteinuria_trace should be present
        has_acr = "acr" in df.columns
        has_trace = "proteinuria_trace" in df.columns

        if not has_acr and not has_trace:
            logger.warning("Neither 'acr' nor 'proteinuria_trace' columns found. Using eGFR-only CKD adjustment.")

        required_cols = required + (["acr"] if has_acr else []) + (["proteinuria_trace"] if has_trace else [])

        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["model_name"] = self.model_name

        # Process by Sex (Major coefficient change)
        for sex in ["male", "female"]:
            sex_mask = results["sex"].str.lower() == sex
            if not sex_mask.any():
                continue

            # Sub-loop by age group (< 70 vs >= 70)
            for age_group in ["under_70", "over_70"]:
                if age_group == "under_70":
                    age_mask = sex_mask & (results["age"] < 70)
                    coeffs = _SCORE2_CKD_COEFFICIENTS[sex]
                    base_surv = _SCORE2_BASELINE_SURVIVAL[sex]
                    cage_center = 60
                    cage_div = 5
                else:
                    age_mask = sex_mask & (results["age"] >= 70)
                    coeffs = _SCORE2_OP_COEFFICIENTS[sex]
                    base_surv = 0.7576 if sex == "male" else 0.8082
                    cage_center = 73
                    cage_div = 1  # No division for OP

                if not age_mask.any():
                    continue

                # Sub-sub-loop by Region
                present_regions = results.loc[age_mask, "region"].unique()

                for region in present_regions:
                    if region not in _SCORE2_CALIBRATION_SCALES[sex]:
                        continue

                    mask = age_mask & (results["region"] == region)
                    data = results[mask]
                    scales = _SCORE2_CALIBRATION_SCALES[sex][region]

                    # Vectorized inputs
                    if age_group == "under_70":
                        cage = (data["age"] - cage_center) / cage_div
                    else:
                        cage = data["age"] - cage_center

                    csbp = (data["systolic_bp"] - 120) / 20
                    ctchol = data["total_cholesterol"] - 6
                    chdl = (data["hdl_cholesterol"] - 1.3) / 0.5
                    smoking = data["smoking"].astype(float)
                    diabetes = data["diabetes"].astype(float)

                    # Vectorized linear predictor
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
                        + coeffs["diabetes"] * diabetes
                        + coeffs["diabetes_cage"] * diabetes * cage
                    )

                    # Vectorized uncalibrated risk
                    if age_group == "under_70":
                        uncalib = 1.0 - (base_surv ** np.exp(lin_pred))
                    else:
                        offset = -0.0929 if sex == "male" else -0.229
                        uncalib = 1.0 - (base_surv ** np.exp(lin_pred - offset))

                    uncalib = np.clip(uncalib, 1e-9, 1.0 - 1e-9)

                    # Vectorized calibration
                    ln_neg_ln = np.log(-np.log(1 - uncalib))
                    calib = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))

                    # Vectorized CKD adjustment
                    # First, calculate expected eGFR for each patient
                    exeGFR = 87.8980
                    exeGFR -= 3.7891 * (data["age"] - 60) / 5
                    exeGFR -= (data["sex"].str.lower() == "female").astype(float) * 0.7023
                    exeGFR -= 0.2941 * (data["total_cholesterol"] - 6)
                    exeGFR += 1.0960 * (data["hdl_cholesterol"] - 1.3) / 0.5
                    exeGFR -= 0.1364 * (data["systolic_bp"] - 120) / 20
                    exeGFR += 0.1205 * diabetes
                    exeGFR += 1.3211 * smoking

                    # Interaction terms for eGFR
                    exeGFR += 0.0555 * ((data["age"] - 60) / 5) * (data["total_cholesterol"] - 6)
                    exeGFR += 0.1717 * ((data["age"] - 60) / 5) * (data["hdl_cholesterol"] - 1.3) / 0.5
                    exeGFR += 0.0059 * ((data["age"] - 60) / 5) * (data["systolic_bp"] - 120) / 20
                    exeGFR -= 0.8994 * ((data["age"] - 60) / 5) * diabetes
                    exeGFR += 0.2181 * ((data["age"] - 60) / 5) * smoking

                    # eGFR adjustment
                    if age_group == "under_70":
                        egfr_adj = np.exp(
                            0.4713 * (np.minimum(data["egfr"], 60) / -15 - np.minimum(exeGFR, 60) / -15) +
                            0.0956 * (np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                     np.minimum(np.maximum(exeGFR - 60, 0), 30) / -15) -
                            0.0802 * (data["age"] - 60) / 5 * (np.minimum(data["egfr"], 60) / -15 -
                                                             np.minimum(exeGFR, 60) / -15) +
                            0.0088 * (data["age"] - 60) / 5 * (np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                                             np.minimum(np.maximum(exeGFR - 60, 0), 30) / -15)
                        )
                    else:
                        egfr_adj = np.exp(
                            0.3072 * (np.minimum(data["egfr"], 60) / -15 - np.minimum(exeGFR, 60) / -15) +
                            0.0942 * (np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                     np.minimum(np.maximum(exeGFR - 60, 0), 30) / -15) -
                            0.4616 * (np.maximum(data["egfr"] - 90, 0) / -15 -
                                     np.maximum(exeGFR - 90, 0) / -15) -
                            0.0127 * (data["age"] - 73) * (np.minimum(data["egfr"], 60) / -15 -
                                                         np.minimum(exeGFR, 60) / -15) -
                            0.0098 * (data["age"] - 73) * (np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                                         np.minimum(np.maximum(exeGFR - 60, 0), 30) / -15) -
                            0.0075 * (data["age"] - 73) * (np.maximum(data["egfr"] - 90, 0) / -15 -
                                                         np.maximum(exeGFR - 90, 0) / -15)
                        )

                    score_egfr = 1 - (1 - calib) ** egfr_adj

                    # Apply ACR or trace adjustment if available
                    if has_acr and data["acr"].notna().any():
                        # Calculate expected ACR
                        exACR = 8 ** (1 - 0.0225 +
                                     0.0159 * (data["age"] - 60) / 5 +
                                     (data["sex"].str.lower() == "female").astype(float) * 0.0308 +
                                     0.0185 * (data["total_cholesterol"] - 6) -
                                     0.0274 * (data["hdl_cholesterol"] - 1.3) / 0.5 +
                                     0.1339 * (data["systolic_bp"] - 120) / 20 +
                                     0.2171 * diabetes +
                                     0.0629 * smoking -
                                     0.0062 * (data["age"] - 60) / 5 * (data["total_cholesterol"] - 6) +
                                     0.0003 * (data["age"] - 60) / 5 * (data["hdl_cholesterol"] - 1.3) / 0.5 +
                                     0.0008 * (data["age"] - 60) / 5 * (data["systolic_bp"] - 120) / 20 -
                                     0.0109 * (data["age"] - 60) / 5 * diabetes +
                                     0.0085 * (data["age"] - 60) / 5 * smoking +
                                     0.4057 * np.minimum(data["egfr"] - 60, 0) / -15 +
                                     0.0597 * np.minimum(np.maximum(data["egfr"] - 60, 0), 30) / -15 -
                                     0.0916 * np.maximum(data["egfr"] - 90, 0) / -15)

                        if age_group == "under_70":
                            acr_adj = 0.2432 * (np.log(data["acr"]) / np.log(8) - np.log(exACR) / np.log(8))
                        else:
                            acr_adj = 0.2370 * (np.log(data["acr"]) / np.log(8) - np.log(exACR) / np.log(8))

                        ckd_risk = 1 - (1 - score_egfr) ** np.exp(acr_adj)
                        # Use ACR where available, otherwise use eGFR only
                        final_risk = np.where(data["acr"].notna(), ckd_risk, score_egfr)

                    elif has_trace and data["proteinuria_trace"].notna().any():
                        # Map trace to coefficients
                        trace_coeffs_map = {
                            'negative': 0.0, 'trace': 0.2644, '1+': 0.4126,
                            '2+': 0.4761, '3+': 0.4761, '4+': 0.4761
                        }

                        # Default to 0 for unknown values
                        trace_coeffs = data["proteinuria_trace"].str.lower().map(trace_coeffs_map).fillna(0.0)
                        ckd_risk = 1 - (1 - score_egfr) ** np.exp(trace_coeffs)
                        final_risk = np.where(data["proteinuria_trace"].notna(), ckd_risk, score_egfr)
                    else:
                        final_risk = score_egfr

                    # Assign results
                    results.loc[mask, "risk_score"] = np.clip(final_risk * 100.0, 0.0, 100.0)

        # Vectorized categorization
        low_cuts, high_cuts = self._get_risk_thresholds(results["age"].values)

        conditions = [
            results["risk_score"] < low_cuts,
            (results["risk_score"] >= low_cuts) & (results["risk_score"] < high_cuts),
            results["risk_score"] >= high_cuts
        ]
        choices = ["low", "moderate", "high"]

        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results
