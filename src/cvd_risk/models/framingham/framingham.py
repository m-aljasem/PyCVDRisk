"""
Framingham 2008 ASCVD Risk Score (Lipid Model).

This implements the D'Agostino et al. 2008 Framingham Risk Score for
10-year risk of cardiovascular disease using traditional risk factors
including lipid measurements.

Reference:
    D'agostino, R.B., Vasan, R.S., Pencina, M.J., Wolf, P.A., Cobain, M.,
    Massaro, J.M. and Kannel, W.B., 2008. General cardiovascular risk
    profile for use in primary care: the Framingham Heart Study.
    Circulation, 117(6), pp.743-753.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# Framingham 2008 Lipid Model coefficients from D'Agostino et al.
# These match the R CVrisk package implementation
_FRAMINGHAM_COEFFICIENTS = {
    "male": {
        "ln_age": 3.06117,
        "ln_totchol": 1.12370,
        "ln_hdl": -0.93263,
        "ln_untreated_sbp": 1.93303,
        "ln_treated_sbp": 1.99881,
        "smoker": 0.65451,
        "diabetes": 0.57367,
        "group_mean": 23.9802,
        "baseline_survival": 0.88936,
    },
    "female": {
        "ln_age": 2.32888,
        "ln_totchol": 1.20904,
        "ln_hdl": -0.70833,
        "ln_untreated_sbp": 2.76157,
        "ln_treated_sbp": 2.82263,
        "smoker": 0.52873,
        "diabetes": 0.69154,
        "group_mean": 26.1931,
        "baseline_survival": 0.95012,
    },
}


class Framingham(RiskModel):
    """
    Framingham 2008 ASCVD Risk Score (Lipid Model).

    This model estimates 10-year risk of cardiovascular disease events
    (coronary death, myocardial infarction, coronary insufficiency, angina,
    ischemic stroke, hemorrhagic stroke, transient ischemic attack,
    peripheral artery disease, or heart failure).

    The model uses the D'Agostino et al. 2008 Framingham Risk Score with
    lipid measurements (total cholesterol and HDL cholesterol).

    Parameters
    ----------
    age : int
        Age in years (30-79 for optimal performance, validated up to 74).
    sex : Literal["male", "female"]
        Biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L.
    hdl_cholesterol : float
        HDL cholesterol in mmol/L.
    smoking : bool
        Current smoking status.
    diabetes : bool
        Diabetes mellitus status.
    bp_treated : bool
        Whether patient is on blood pressure medication.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage (0.0-100.0)
        - risk_category: Risk classification ("low", "moderate", "high")
        - model_name: "Framingham"
        - model_version: "2008"

    Examples
    --------
    >>> from cvd_risk.models.framingham import Framingham
    >>> from cvd_risk.core.validation import PatientData
    >>>
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=5.52,  # mmol/L (213 mg/dL)
    ...     hdl_cholesterol=1.29,    # mmol/L (50 mg/dL)
    ...     smoking=True,
    ...     diabetes=False,
    ...     bp_treated=True,
    ... )
    >>> model = Framingham()
    >>> result = model.calculate(patient)
    >>> print(f"10-year CVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - Uses log-transformed continuous variables and coefficients
    - Formula: 1 - S₀(t)^exp(ΣβᵢXᵢ - mean_risk_factors)
    - Cholesterol values expected in mmol/L (conversion factor: mg/dL ÷ 38.67)
    - Model validated for ages 30-74 years in original study
    - Risk estimates are for general CVD events, not just coronary heart disease
    """

    model_name = "Framingham"
    model_version = "2008"
    supported_regions = None  # Framingham is US-based but widely used

    def __init__(self) -> None:
        """Initialize Framingham 2008 model."""
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

        # Framingham-specific validations
        if patient.age < 30 or patient.age > 79:
            logger.warning(
                f"Age {patient.age} outside optimal range [30, 79] years. "
                "Results may have reduced accuracy."
            )

        if patient.age > 74:
            logger.warning(
                f"Age {patient.age} > 74 years. Model was validated up to age 74. "
                "Extrapolation may be unreliable."
            )

        # Check for required fields that Framingham needs
        if patient.total_cholesterol <= 0:
            raise ValueError("Total cholesterol must be > 0 mmol/L")

        if patient.hdl_cholesterol <= 0:
            raise ValueError("HDL cholesterol must be > 0 mmol/L")

        if patient.systolic_bp <= 0:
            raise ValueError("Systolic blood pressure must be > 0 mmHg")

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CVD risk using Framingham 2008 Risk Score.

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

        # Get coefficients for the patient's sex
        coeffs = _FRAMINGHAM_COEFFICIENTS[patient.sex.lower()]

        # Transform input variables to log scale
        # Framingham coefficients are calibrated for mg/dL, so convert mmol/L to mg/dL first
        ln_age = np.log(patient.age)
        ln_total_chol = np.log(patient.total_cholesterol * 38.67)  # Convert mmol/L to mg/dL
        ln_hdl = np.log(patient.hdl_cholesterol * 38.67)  # Convert mmol/L to mg/dL

        # Handle blood pressure based on treatment status
        if patient.antihypertensive:
            ln_sbp = np.log(patient.systolic_bp) * coeffs["ln_treated_sbp"]
        else:
            ln_sbp = np.log(patient.systolic_bp) * coeffs["ln_untreated_sbp"]

        # Calculate the linear predictor (sum of beta * x)
        linear_predictor = (
            coeffs["ln_age"] * ln_age
            + coeffs["ln_totchol"] * ln_total_chol
            + coeffs["ln_hdl"] * ln_hdl
            + ln_sbp  # Already multiplied by appropriate coefficient above
            + coeffs["smoker"] * int(patient.smoking)
            + coeffs["diabetes"] * int(patient.diabetes)
        )

        # Calculate risk using Framingham formula:
        # Risk = 1 - S₀(t)^exp(ΣβᵢXᵢ - mean_risk_factors)
        risk_exponent = linear_predictor - coeffs["group_mean"]
        risk_score = 1.0 - (coeffs["baseline_survival"] ** np.exp(risk_exponent))

        # Convert to percentage and clip to valid range
        risk_percentage = np.clip(risk_score * 100.0, 0.0, 100.0)

        # Categorize risk
        risk_category = self._categorize_risk(risk_percentage)

        # Create metadata
        metadata = self._get_metadata()
        metadata.update({
            "linear_predictor": float(linear_predictor),
            "risk_exponent": float(risk_exponent),
            "transformed_vars": {
                "ln_age": float(ln_age),
                "ln_total_chol_mgdl": float(ln_total_chol),  # After mg/dL conversion
                "ln_hdl_mgdl": float(ln_hdl),               # After mg/dL conversion
                "ln_sbp": float(np.log(patient.systolic_bp)),
                "antihypertensive": patient.antihypertensive,
                "cholesterol_conversion_factor": 38.67,
            },
        })

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates Framingham risk for a dataframe of patients.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with required columns. Must contain:
            - age: Age in years
            - sex: Sex ('male'/'female')
            - systolic_bp: Systolic BP in mmHg
            - total_cholesterol: Total cholesterol in mmol/L
            - hdl_cholesterol: HDL cholesterol in mmol/L
            - smoking: Smoking status (bool/int)
            - diabetes: Diabetes status (bool/int)
            - bp_treated: BP treatment status (bool/int)

        Returns
        -------
        pd.DataFrame
            Input dataframe with additional columns:
            - risk_score: 10-year CVD risk percentage
            - risk_category: Risk category
            - model_name: "Framingham"
            - model_version: "2008"
        """
        required_cols = [
            "age", "sex", "systolic_bp", "total_cholesterol",
            "hdl_cholesterol", "smoking", "diabetes", "antihypertensive"
        ]

        if not set(required_cols).issubset(df.columns):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Create copy and initialize result columns
        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name
        results["model_version"] = self.model_version

        # Process by sex for efficiency
        for sex in ["male", "female"]:
            sex_mask = results["sex"].str.lower() == sex
            if not sex_mask.any():
                continue

            coeffs = _FRAMINGHAM_COEFFICIENTS[sex]
            sex_data = results[sex_mask].copy()

            # Vectorized transformations
            # Framingham coefficients are calibrated for mg/dL, so convert mmol/L to mg/dL first
            ln_age = np.log(sex_data["age"])
            ln_total_chol = np.log(sex_data["total_cholesterol"] * 38.67)  # Convert mmol/L to mg/dL
            ln_hdl = np.log(sex_data["hdl_cholesterol"] * 38.67)  # Convert mmol/L to mg/dL
            ln_sbp = np.log(sex_data["systolic_bp"])

            # Blood pressure coefficients based on treatment
            bp_coeff = np.where(
                sex_data["antihypertensive"],
                coeffs["ln_treated_sbp"],
                coeffs["ln_untreated_sbp"]
            )
            sbp_term = ln_sbp * bp_coeff

            # Linear predictor
            linear_predictor = (
                coeffs["ln_age"] * ln_age
                + coeffs["ln_totchol"] * ln_total_chol
                + coeffs["ln_hdl"] * ln_hdl
                + sbp_term
                + coeffs["smoker"] * sex_data["smoking"].astype(float)
                + coeffs["diabetes"] * sex_data["diabetes"].astype(float)
            )

            # Risk calculation
            risk_exponent = linear_predictor - coeffs["group_mean"]
            risk_score = 1.0 - (coeffs["baseline_survival"] ** np.exp(risk_exponent))
            risk_percentage = np.clip(risk_score * 100.0, 0.0, 100.0)

            # Assign results
            results.loc[sex_mask, "risk_score"] = risk_percentage

        # Vectorized categorization
        conditions = [
            results["risk_score"] < 10,
            (results["risk_score"] >= 10) & (results["risk_score"] < 20),
            results["risk_score"] >= 20
        ]
        choices = ["low", "moderate", "high"]
        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results

    def _categorize_risk(self, risk_percentage: float) -> str:
        """
        Categorize risk based on ACC/AHA guidelines adapted for Framingham.

        Parameters
        ----------
        risk_percentage : float
            10-year risk percentage

        Returns
        -------
        str
            Risk category
        """
        if risk_percentage < 10:
            return "low"
        elif risk_percentage < 20:
            return "moderate"
        else:
            return "high"

