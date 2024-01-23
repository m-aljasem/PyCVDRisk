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

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.data_classes import SCORE2PatientData
from cvd_risk.core.validation import RiskResult, validate_dataframe

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


def categorize_esc2021(risk_percent, age) -> np.ndarray:
    """
    ESC 2021 age-specific risk categories for the SCORE2 family.

    Ages <50: <2.5% low_to_moderate, 2.5-<7.5% high, >=7.5% very_high.
    Ages 50-69: <5%, 5-<10%, >=10%. Ages >=70: <7.5%, 7.5-<15%, >=15%.
    Works on scalars and arrays; NaN risk maps to "error".
    """
    risk = np.asarray(risk_percent, dtype=float)
    age = np.asarray(age, dtype=float)
    low_cut = np.select([age < 50, age < 70], [2.5, 5.0], default=7.5)
    high_cut = np.select([age < 50, age < 70], [7.5, 10.0], default=15.0)
    return np.select(
        [np.isnan(risk), risk < low_cut, risk < high_cut],
        ["error", "low_to_moderate", "high"],
        default="very_high",
    )


def _score2_risk(sex: str, region: str, age, sbp, tchol, hdl, smoking):
    """10-year calibrated SCORE2 risk (%) and uncalibrated risk (0-1); scalars or arrays."""
    c = _SCORE2_COEFFICIENTS[sex]
    scales = _SCORE2_CALIBRATION_SCALES[sex][region]
    cage = (age - 60) / 5
    csbp = (sbp - 120) / 20
    ctchol = tchol - 6
    chdl = (hdl - 1.3) / 0.5
    lin_pred = (
        c["cage"] * cage
        + c["smoking"] * smoking
        + c["csbp"] * csbp
        + c["ctchol"] * ctchol
        + c["chdl"] * chdl
        + c["smoking_cage"] * smoking * cage
        + c["csbp_cage"] * csbp * cage
        + c["ctchol_cage"] * ctchol * cage
        + c["chdl_cage"] * chdl * cage
    )
    uncalib = 1.0 - _SCORE2_BASELINE_SURVIVAL[sex] ** np.exp(lin_pred)
    # Clip to avoid log(0) in the calibration step
    uncalib = np.clip(uncalib, 1e-9, 1.0 - 1e-9)
    ln_neg_ln = np.log(-np.log(1 - uncalib))
    calib = 1.0 - np.exp(-np.exp(scales["scale1"] + scales["scale2"] * ln_neg_ln))
    return np.clip(calib * 100.0, 0.0, 100.0), uncalib


# Input ranges enforced by SCORE2PatientData, applied to batch rows too
_BATCH_RANGES = {
    "age": (18, 120),
    "systolic_bp": (50.0, 250.0),
    "total_cholesterol": (1.0, 15.0),
    "hdl_cholesterol": (0.3, 5.0),
}


class SCORE2(RiskModel[SCORE2PatientData]):
    """
    SCORE2 cardiovascular risk prediction model (ESC 2021).
    Valid for ages 40-69.

    Risk categories follow ESC 2021 (see ``categorize_esc2021``):
    "low_to_moderate", "high", "very_high".
    """

    model_name = "SCORE2"
    model_version = "2021"
    supported_regions = ["low", "moderate", "high", "very_high"]
    patient_data_class = SCORE2PatientData

    def validate_input(self, patient: SCORE2PatientData) -> None:
        super().validate_input(patient)
        if patient.age < 40 or patient.age > 69:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 69]. "
                "SCORE2 accuracy may be reduced."
            )
        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")

    def calculate(self, patient: SCORE2PatientData) -> RiskResult:
        self.validate_input(patient)
        risk_percent, uncalib = _score2_risk(
            patient.sex.lower(),
            patient.region,
            patient.age,
            patient.systolic_bp,
            patient.total_cholesterol,
            patient.hdl_cholesterol,
            float(patient.smoking),
        )
        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=str(categorize_esc2021(risk_percent, patient.age)),
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={"raw_uncalibrated": float(uncalib)},
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.

        Rows that fail validation get ``risk_score`` NaN, ``risk_category``
        "error" and a reason in the ``error`` column (same contract as
        ``RiskModel.calculate_batch``).
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking", "region"]
        validate_dataframe(df, required)

        results = df.copy()
        results["risk_score"] = np.nan
        results["model_name"] = self.model_name
        error = pd.Series(None, index=results.index, dtype=object)

        sex = results["sex"].astype(str).str.lower()
        error[~sex.isin(["male", "female"])] = "sex must be 'male' or 'female'"
        error[error.isna() & ~results["region"].isin(self.supported_regions)] = (
            f"region must be one of {self.supported_regions}"
        )
        error[error.isna() & ~results["smoking"].isin([True, False, 0, 1])] = "smoking must be boolean"
        for col, (lo, hi) in _BATCH_RANGES.items():
            values = pd.to_numeric(results[col], errors="coerce")
            error[error.isna() & ~values.between(lo, hi)] = f"{col} outside [{lo}, {hi}]"

        valid = error.isna()
        for (s, region), group in results[valid].groupby([sex[valid], results.loc[valid, "region"]]):
            risk, _ = _score2_risk(
                s,
                region,
                group["age"].astype(float),
                group["systolic_bp"].astype(float),
                group["total_cholesterol"].astype(float),
                group["hdl_cholesterol"].astype(float),
                group["smoking"].astype(float),
            )
            results.loc[group.index, "risk_score"] = risk

        results["risk_category"] = categorize_esc2021(results["risk_score"], pd.to_numeric(results["age"], errors="coerce"))
        results["error"] = error
        return results
