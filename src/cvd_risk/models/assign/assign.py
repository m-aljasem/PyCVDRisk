"""
ASSIGN (ASsessing cardiovascular risk using SIGN guidelines) cardiovascular risk model.

ASSIGN is the Scottish Intercollegiate Guidelines Network cardiovascular risk
prediction model for Scotland, with both original (v1.0) and recalibrated (v2.0) versions.

Reference:
    SIGN (Scottish Intercollegiate Guidelines Network). (2009).
    Risk estimation and the prevention of cardiovascular disease: a national clinical guideline.
    Edinburgh: SIGN. (SIGN publication no. 97)
"""

import logging
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# ASSIGN baseline survival probabilities at 10 years
_ASSIGN_BASELINE_SURVIVAL = {
    "male": {
        "v1": 0.8831,
        "v2": 0.9130,
    },
    "female": {
        "v1": 0.9365,
        "v2": 0.9666,
    },
}

# ASSIGN sex-specific coefficients for linear predictor
_ASSIGN_COEFFICIENTS = {
    "male": {
        "age": 0.05698,
        "chol": 0.22286,
        "hdl": -0.53684,
        "sbp": 0.01183,
        "dm": 0.81558,
        "fhxcvd": 0.275,
        "ncig": 0.02005,
        "simdscore10": 0.06296,
    },
    "female": {
        "age": 0.07203,
        "chol": 0.12720,
        "hdl": -0.55836,
        "sbp": 0.01064,
        "dm": 0.97727,
        "fhxcvd": 0.492,
        "ncig": 0.02724,
        "simdscore10": 0.09386,
    },
}

# ASSIGN baseline means for calibration (population averages)
_ASSIGN_BASELINE_MEANS = {
    "male": {
        "v1": {
            "age": 48.8706,
            "chol": 6.2252,
            "hdl": 1.35042,
            "sbp": 133.810,
            "dm": 0.0152905,
            "fhxcvd": 0.263762,
            "ncig": 7.95841,
            "simdscore10": 2.74038,
        },
        "v2": {
            "age": 55.3999,
            "chol": 5.5488,
            "hdl": 1.2944,
            "sbp": 141.5754,
            "dm": 0.0481,
            "fhxcvd": 0.5620,
            "ncig": 2.0560,
            "simdscore10": 1.4701,
        },
    },
    "female": {
        "v1": {
            "age": 48.7959,
            "chol": 6.40706,
            "hdl": 1.62837,
            "sbp": 130.115,
            "dm": 0.0127275,
            "fhxcvd": 0.326328,
            "ncig": 6.44058,
            "simdscore10": 2.82470,
        },
        "v2": {
            "age": 55.1321,
            "chol": 5.8199,
            "hdl": 1.6086,
            "sbp": 135.2402,
            "dm": 0.0268,
            "fhxcvd": 0.6129,
            "ncig": 1.6107,
            "simdscore10": 1.4740,
        },
    },
}


class ASSIGN(RiskModel):
    """
    ASSIGN cardiovascular risk prediction model (SIGN 2009).
    Valid for adults in Scotland. Provides both v1.0 and v2.0 versions.
    """

    model_name = "ASSIGN"
    model_version = "v1.0/v2.0"
    supported_versions = ["v1", "v2"]

    def __init__(self, version: Literal["v1", "v2"] = "v2") -> None:
        """
        Initialize ASSIGN model.

        Parameters
        ----------
        version : str
            Model version to use ('v1' for original, 'v2' for recalibrated).
        """
        super().__init__()
        if version not in self.supported_versions:
            raise ValueError(f"Version must be one of {self.supported_versions}")
        self.version = version

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        # ASSIGN is specifically for Scotland, but we don't enforce region validation
        # as it's primarily a Scottish model but can be used elsewhere

    def _calculate_linear_predictor(self, patient: PatientData) -> float:
        """Calculate the linear predictor component of ASSIGN score."""
        sex = patient.sex.lower()
        coeffs = _ASSIGN_COEFFICIENTS[sex]

        # Transform deprivation score (use townsend_deprivation if available, otherwise 0)
        # ASSIGN originally uses SIMD/10, but we can use townsend_deprivation as proxy
        deprivation_score = patient.townsend_deprivation if patient.townsend_deprivation is not None else 0.0

        # Convert boolean inputs to float
        dm = float(patient.diabetes) if patient.diabetes is not None else 0.0
        fhxcvd = float(patient.family_history) if patient.family_history is not None else 0.0

        # ASSIGN uses cigarettes per day, but PatientData only has smoking bool
        # We'll use a default value for cigarettes when smoking=True
        ncig = 10.0 if patient.smoking else 0.0  # Default 10 cigarettes/day for smokers

        lin_pred = (
            coeffs["age"] * patient.age
            + coeffs["chol"] * patient.total_cholesterol
            + coeffs["hdl"] * patient.hdl_cholesterol
            + coeffs["sbp"] * patient.systolic_bp
            + coeffs["dm"] * dm
            + coeffs["fhxcvd"] * fhxcvd
            + coeffs["ncig"] * ncig
            + coeffs["simdscore10"] * deprivation_score
        )

        return lin_pred

    def _calculate_baseline_mean(self, sex: str) -> float:
        """Calculate the baseline mean for calibration."""
        baseline_vals = _ASSIGN_BASELINE_MEANS[sex][self.version]
        coeffs = _ASSIGN_COEFFICIENTS[sex]

        baseline_mean = (
            coeffs["age"] * baseline_vals["age"]
            + coeffs["chol"] * baseline_vals["chol"]
            + coeffs["hdl"] * baseline_vals["hdl"]
            + coeffs["sbp"] * baseline_vals["sbp"]
            + coeffs["dm"] * baseline_vals["dm"]
            + coeffs["fhxcvd"] * baseline_vals["fhxcvd"]
            + coeffs["ncig"] * baseline_vals["ncig"]
            + coeffs["simdscore10"] * baseline_vals["simdscore10"]
        )

        return baseline_mean

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        sex = patient.sex.lower()
        base_survival = _ASSIGN_BASELINE_SURVIVAL[sex][self.version]

        # Calculate linear predictor and baseline mean
        lin_pred = self._calculate_linear_predictor(patient)
        baseline_mean = self._calculate_baseline_mean(sex)

        # Calculate calibrated risk score
        calibrated_diff = lin_pred - baseline_mean
        risk_score = 100 * (1 - (base_survival ** np.exp(calibrated_diff)))

        # Clip to reasonable bounds
        risk_score = np.clip(risk_score, 0.0, 100.0)

        # Categorize risk (ASSIGN uses similar thresholds to other UK models)
        if risk_score < 10:
            category = "low"
        elif risk_score < 20:
            category = "moderate"
        else:
            category = "high"

        return RiskResult(
            risk_score=float(risk_score),
            risk_category=category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "version": self.version,
                "linear_predictor": lin_pred,
                "baseline_mean": baseline_mean,
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates by grouping (Sex, Version) for efficiency.
        """
        required = [
            "age", "sex", "total_cholesterol", "hdl_cholesterol", "systolic_bp",
            "smoking"
        ]
        optional = ["diabetes", "family_history", "townsend_deprivation"]

        missing_required = set(required) - set(df.columns)
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name
        results["model_version"] = self.version

        # Process by Sex (Major coefficient change)
        for sex in ["male", "female"]:
            sex_mask = results["sex"].str.lower() == sex
            if not sex_mask.any():
                continue

            # Get sex-specific constants
            coeffs = _ASSIGN_COEFFICIENTS[sex]
            base_survival = _ASSIGN_BASELINE_SURVIVAL[sex][self.version]
            baseline_vals = _ASSIGN_BASELINE_MEANS[sex][self.version]

            # Calculate baseline mean vectorized
            baseline_mean = (
                coeffs["age"] * baseline_vals["age"]
                + coeffs["chol"] * baseline_vals["chol"]
                + coeffs["hdl"] * baseline_vals["hdl"]
                + coeffs["sbp"] * baseline_vals["sbp"]
                + coeffs["dm"] * baseline_vals["dm"]
                + coeffs["fhxcvd"] * baseline_vals["fhxcvd"]
                + coeffs["ncig"] * baseline_vals["ncig"]
                + coeffs["simdscore10"] * baseline_vals["simdscore10"]
            )

            # Vectorized linear predictor
            data = results[sex_mask]

            # Handle optional fields
            dm = data.get("diabetes", False).fillna(False).astype(float)
            fhxcvd = data.get("family_history", False).fillna(False).astype(float)
            deprivation_score = data.get("townsend_deprivation", 0.0).fillna(0.0)

            # ASSIGN uses cigarettes per day - default 10 for smokers, 0 for non-smokers
            ncig = data["smoking"].astype(float) * 10.0

            lin_pred = (
                coeffs["age"] * data["age"]
                + coeffs["chol"] * data["total_cholesterol"]
                + coeffs["hdl"] * data["hdl_cholesterol"]
                + coeffs["sbp"] * data["systolic_bp"]
                + coeffs["dm"] * dm
                + coeffs["fhxcvd"] * fhxcvd
                + coeffs["ncig"] * ncig
                + coeffs["simdscore10"] * deprivation_score
            )

            # Vectorized risk calculation
            calibrated_diff = lin_pred - baseline_mean
            risk_scores = 100 * (1 - (base_survival ** np.exp(calibrated_diff)))
            risk_scores = np.clip(risk_scores, 0.0, 100.0)

            # Assign results
            results.loc[sex_mask, "risk_score"] = risk_scores

        # Vectorized categorization
        conditions = [
            results["risk_score"] < 10,
            (results["risk_score"] >= 10) & (results["risk_score"] < 20),
            results["risk_score"] >= 20
        ]
        choices = ["low", "moderate", "high"]
        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results

# --- Example Usage ---
if __name__ == "__main__":
    from cvd_risk.core.validation import PatientData

    # Example Patient Data
    patient = PatientData(
        age=55,
        sex="male",
        total_cholesterol=6.0,    # mmol/L
        hdl_cholesterol=1.2,      # mmol/L
        systolic_bp=140,          # mmHg
        smoking=True,             # Current smoker
        diabetes=False,
        family_history=True,      # Family history of CVD
        townsend_deprivation=3.0  # Townsend deprivation score (proxy for SIMD)
    )

    try:
        # Calculate with ASSIGN v1.0 (original)
        assign_v1 = ASSIGN(version="v1")
        result_v1 = assign_v1.calculate(patient)

        # Calculate with ASSIGN v2.0 (recalibrated)
        assign_v2 = ASSIGN(version="v2")
        result_v2 = assign_v2.calculate(patient)

        print("Patient Profile:")
        print(f"  Age: {patient.age} years")
        print(f"  Sex: {patient.sex}")
        print(f"  Total Cholesterol: {patient.total_cholesterol} mmol/L")
        print(f"  HDL Cholesterol: {patient.hdl_cholesterol} mmol/L")
        print(f"  Systolic BP: {patient.systolic_bp} mmHg")
        print(f"  Diabetes: {patient.diabetes}")
        print(f"  Family History CVD: {patient.family_history}")
        print(f"  Current Smoker: {patient.smoking}")
        print(f"  Townsend Deprivation Score: {patient.townsend_deprivation}")
        print("-" * 50)
        print(f"ASSIGN Score (v1.0): {result_v1.risk_score:.1f}% - {result_v1.risk_category} risk")
        print(f"ASSIGN Score (v2.0): {result_v2.risk_score:.1f}% - {result_v2.risk_category} risk")

    except ValueError as e:
        print(f"Error: {e}")