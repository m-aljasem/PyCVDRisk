"""
SCORE (Systematic COronary Risk Evaluation) cardiovascular risk model.

SCORE is the original European cardiovascular risk prediction model
published in 2003 by the European Society of Cardiology.

Reference:
    Conroy RM, Pyörälä K, Fitzgerald AP, et al. Estimation of ten-year risk of
    fatal cardiovascular disease in Europe: the SCORE project. European Heart
    Journal. 2003;24(11):987-1003.
    DOI: 10.1016/S0195-668X(03)00114-3
"""

import logging
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# SCORE Weibull baseline survival parameters (shape and scale in years)
# From Conroy et al. (2003) European Heart Journal
_SCORE_BASELINE_PARAMETERS = {
    "high": {
        "male": {
            "CHD": {"shape": 4.705, "scale": 87.61},
            "NON_CHD": {"shape": 5.300, "scale": 107.3},
        },
        "female": {
            "CHD": {"shape": 5.681, "scale": 121.8},
            "NON_CHD": {"shape": 6.429, "scale": 152.8},
        },
    },
    "low": {
        "male": {
            "CHD": {"shape": 4.705, "scale": 104.2},
            "NON_CHD": {"shape": 5.300, "scale": 126.7},
        },
        "female": {
            "CHD": {"shape": 5.681, "scale": 144.3},
            "NON_CHD": {"shape": 6.429, "scale": 181.0},
        },
    },
}

# SCORE risk factor coefficients (applied to centered variables)
# Coefficients derived from the SCORE model for risk factor adjustments
_SCORE_COEFFICIENTS = {
    "CHD": {
        "cholesterol": 0.24,  # for (total_cholesterol - 6)
        "sbp": 0.18,          # for (systolic_bp - 120) / 20
        "smoking": 0.25,      # for smoking status
    },
    "NON_CHD": {
        "cholesterol": 0.20,  # for (total_cholesterol - 6)
        "sbp": 0.15,          # for (systolic_bp - 120) / 20
        "smoking": 0.21,      # for smoking status
    },
}


class SCORE(RiskModel):
    """
    SCORE cardiovascular risk prediction model (ESC 2003).

    Estimates 10-year risk of fatal cardiovascular disease using Weibull
    survival models calibrated for European populations.

    Valid for ages 40-65 years (optimal range).
    """

    model_name = "SCORE"
    model_version = "2003"
    supported_regions = ["low", "high"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.age < 40 or patient.age > 65:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 65]. "
                "SCORE accuracy may be reduced."
            )
        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")

    def _calculate_baseline_survival(self, age: float, shape: float, scale: float) -> float:
        """
        Calculate baseline survival probability at given age using Weibull model.

        S(t) = exp(-(t/scale)^shape)
        """
        return np.exp(-(age / scale) ** shape)

    def _get_risk_thresholds(self, age: Union[int, float, np.ndarray]) -> Tuple:
        """
        Get clinical risk thresholds based on Age (ESC 2003 Guidelines).

        Returns
        -------
        Tuple (low_cutoff, high_cutoff)
            < low_cutoff: Low Risk
            >= high_cutoff: High Risk
        """
        # Vectorized logic for NumPy arrays
        if isinstance(age, (np.ndarray, pd.Series)):
            # SCORE uses simplified thresholds compared to SCORE2
            low_cut = np.full_like(age, 1.0, dtype=float)
            high_cut = np.full_like(age, 5.0, dtype=float)
            return low_cut, high_cut

        # Scalar logic for single patient
        return 1.0, 5.0

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # 1. Setup Constants
        sex = patient.sex.lower()
        region = patient.region

        # 2. Transform Inputs (centering as in original SCORE)
        cholesterol_centered = patient.total_cholesterol - 6.0
        sbp_centered = (patient.systolic_bp - 120.0) / 20.0
        smoking = float(patient.smoking)

        # 3. Calculate risk for each endpoint (CHD and NON_CHD)
        total_risk = 0.0

        for endpoint in ["CHD", "NON_CHD"]:
            # Get Weibull parameters
            params = _SCORE_BASELINE_PARAMETERS[region][sex][endpoint]
            shape = params["shape"]
            scale = params["scale"]

            # Get coefficients
            coeffs = _SCORE_COEFFICIENTS[endpoint]

            # Calculate baseline survival at current age and age+10
            s0_age = self._calculate_baseline_survival(patient.age, shape, scale)
            s0_age_10 = self._calculate_baseline_survival(patient.age + 10, shape, scale)

            # Calculate linear predictor (risk factor adjustment)
            linear_pred = (
                coeffs["cholesterol"] * cholesterol_centered
                + coeffs["sbp"] * sbp_centered
                + coeffs["smoking"] * smoking
            )

            # Apply risk factor adjustment
            exp_adjustment = np.exp(linear_pred)
            s_age = s0_age ** exp_adjustment
            s_age_10 = s0_age_10 ** exp_adjustment

            # Calculate conditional probability (risk of event in next 10 years)
            if s_age == 0:
                risk_endpoint = 1.0
            else:
                risk_endpoint = 1.0 - (s_age_10 / s_age)

            # Clip to valid range
            risk_endpoint = np.clip(risk_endpoint, 0.0, 1.0)
            total_risk += risk_endpoint

        # Convert to percentage and clip
        risk_percent = np.clip(total_risk * 100.0, 0.0, 100.0)

        # 4. Categorize
        low_cut, high_cut = self._get_risk_thresholds(patient.age)

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
                "region": region,
                "endpoints_modeled": ["CHD", "NON_CHD"]
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates by grouping (Sex, Region) for efficiency.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "smoking", "region"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["model_name"] = self.model_name

        # Process by Sex (Major parameter difference)
        for sex in ["male", "female"]:
            sex_mask = results["sex"].str.lower() == sex
            if not sex_mask.any():
                continue

            # Sub-loop by Region
            for region in ["low", "high"]:
                region_mask = sex_mask & (results["region"] == region)
                if not region_mask.any():
                    continue

                data = results[region_mask]

                # Transform inputs
                cholesterol_centered = data["total_cholesterol"] - 6.0
                sbp_centered = (data["systolic_bp"] - 120.0) / 20.0
                smoking = data["smoking"].astype(float)

                # Calculate risk for each endpoint
                total_risk = np.zeros(len(data))

                for endpoint in ["CHD", "NON_CHD"]:
                    # Get Weibull parameters
                    params = _SCORE_BASELINE_PARAMETERS[region][sex][endpoint]
                    shape = params["shape"]
                    scale = params["scale"]

                    # Get coefficients
                    coeffs = _SCORE_COEFFICIENTS[endpoint]

                    # Vectorized baseline survival calculation
                    s0_age = np.exp(-(data["age"] / scale) ** shape)
                    s0_age_10 = np.exp(-((data["age"] + 10) / scale) ** shape)

                    # Vectorized linear predictor
                    linear_pred = (
                        coeffs["cholesterol"] * cholesterol_centered
                        + coeffs["sbp"] * sbp_centered
                        + coeffs["smoking"] * smoking
                    )

                    # Vectorized risk factor adjustment
                    exp_adjustment = np.exp(linear_pred)
                    s_age = s0_age ** exp_adjustment
                    s_age_10 = s0_age_10 ** exp_adjustment

                    # Vectorized conditional risk (avoid division by zero)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        risk_endpoint = np.where(
                            s_age == 0,
                            1.0,
                            1.0 - (s_age_10 / s_age)
                        )
                    risk_endpoint = np.clip(risk_endpoint, 0.0, 1.0)
                    total_risk += risk_endpoint

                # Convert to percentage and assign
                results.loc[region_mask, "risk_score"] = np.clip(total_risk * 100.0, 0.0, 100.0)

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