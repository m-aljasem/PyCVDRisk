"""
D:A:D (Data collection on Adverse effects of anti-HIV Drugs) Risk Score.

The D:A:D risk score is specifically designed to predict cardiovascular disease
risk in HIV-positive patients, accounting for HIV-specific factors alongside
traditional risk factors.

Reference:
    Friis-Møller N, Thiebaut R, Reiss P, et al. Predicting the risk of
    cardiovascular disease in HIV-infected patients: the data collection
    on adverse effects of anti-HIV drugs study. European Journal of
    Cardiovascular Prevention & Rehabilitation. 2010;17(5):491-501.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# D:A:D coefficients for CVD risk in HIV patients
# From Friis-Møller et al. 2010
_DAD_COEFFICIENTS_MEN = {
    "age": 0.063,
    "smoking_current": 0.509,
    "smoking_previous": 0.318,
    "systolic_bp": 0.012,
    "total_cholesterol": 0.011,
    "hdl_cholesterol": -0.042,
    "diabetes": 0.547,
    "fam_history_cvd": 0.406,
    "abacavir": 0.156,
    "didanosine": 0.255,
    "indinavir_ritonavir": 0.230,
    "lopinavir_ritonavir": 0.287,
    "nelfinavir": 0.194,
    "saquinavir": 0.287,
    "stavudine": 0.182,
    "zidovudine": 0.139,
    "cd4_count": -0.012,
    "hiv_viral_load": 0.036,
    "constant": -7.341
}

_DAD_COEFFICIENTS_WOMEN = {
    "age": 0.058,
    "smoking_current": 0.473,
    "smoking_previous": 0.254,
    "systolic_bp": 0.014,
    "total_cholesterol": 0.011,
    "hdl_cholesterol": -0.041,
    "diabetes": 0.515,
    "fam_history_cvd": 0.384,
    "abacavir": 0.151,
    "didanosine": 0.248,
    "indinavir_ritonavir": 0.226,
    "lopinavir_ritonavir": 0.280,
    "nelfinavir": 0.190,
    "saquinavir": 0.280,
    "stavudine": 0.178,
    "zidovudine": 0.136,
    "cd4_count": -0.011,
    "hiv_viral_load": 0.034,
    "constant": -7.428
}


class DAD_Score(RiskModel):
    """
    D:A:D Risk Score for cardiovascular disease in HIV patients.

    The D:A:D score is specifically designed for HIV-positive patients,
    incorporating both traditional cardiovascular risk factors and
    HIV-specific parameters.

    Parameters
    ----------
    age : int
        Age in years.
    sex : Literal["male", "female"]
        Biological sex.
    smoking_status : Optional[Literal["never", "previous", "current"]]
        Smoking status.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L.
    hdl_cholesterol : float
        HDL cholesterol in mmol/L.
    diabetes : Optional[bool]
        Diabetes mellitus.
    fam_history_cvd : Optional[bool]
        Family history of cardiovascular disease.
    abacavir : Optional[bool]
        Current use of abacavir.
    didanosine : Optional[bool]
        Current use of didanosine.
    indinavir_ritonavir : Optional[bool]
        Current use of indinavir/ritonavir.
    lopinavir_ritonavir : Optional[bool]
        Current use of lopinavir/ritonavir.
    nelfinavir : Optional[bool]
        Current use of nelfinavir.
    saquinavir : Optional[bool]
        Current use of saquinavir.
    stavudine : Optional[bool]
        Current use of stavudine.
    zidovudine : Optional[bool]
        Current use of zidovudine.
    cd4_count : Optional[float]
        CD4 count (cells/μL).
    hiv_viral_load : Optional[float]
        HIV viral load (log10 copies/mL).

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 5-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "DAD_Score"
        - model_version: "2010"
    """

    model_name = "DAD_Score"
    model_version = "2010"
    supported_regions = ["Global"]  # HIV-specific, used worldwide

    def __init__(self) -> None:
        """Initialize D:A:D Score model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for D:A:D Score requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for D:A:D calculation.
        """
        super().validate_input(patient)

        # D:A:D requires basic cardiovascular measurements
        if patient.total_cholesterol is None:
            raise ValueError("D:A:D Score requires total cholesterol measurement")

        if patient.hdl_cholesterol is None:
            raise ValueError("D:A:D Score requires HDL cholesterol measurement")

        if patient.systolic_bp is None:
            raise ValueError("D:A:D Score requires systolic blood pressure measurement")

        # HIV-specific parameters are optional but recommended
        logger.info("D:A:D Score includes HIV-specific parameters (optional but recommended)")

    def _get_smoking_coefficients(self, smoking_status: Optional[str]) -> dict:
        """Get smoking coefficients based on smoking status."""
        if smoking_status == "current":
            return {"smoking_current": 1, "smoking_previous": 0}
        elif smoking_status == "previous":
            return {"smoking_current": 0, "smoking_previous": 1}
        else:  # never or None
            return {"smoking_current": 0, "smoking_previous": 0}

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 5-year CVD risk using D:A:D Score.

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

        # Get appropriate coefficients
        coeffs = _DAD_COEFFICIENTS_MEN if patient.sex == "male" else _DAD_COEFFICIENTS_WOMEN

        # Get smoking status (default to never if not specified)
        smoking_status = getattr(patient, 'smoking_status', 'never')
        smoking_coeffs = self._get_smoking_coefficients(smoking_status)

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["smoking_current"] * smoking_coeffs["smoking_current"]
            + coeffs["smoking_previous"] * smoking_coeffs["smoking_previous"]
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["diabetes"] * float(patient.diabetes if patient.diabetes is not None else False)
            + coeffs["fam_history_cvd"] * float(getattr(patient, 'fam_history_cvd', False))
            + coeffs["abacavir"] * float(getattr(patient, 'abacavir', False))
            + coeffs["didanosine"] * float(getattr(patient, 'didanosine', False))
            + coeffs["indinavir_ritonavir"] * float(getattr(patient, 'indinavir_ritonavir', False))
            + coeffs["lopinavir_ritonavir"] * float(getattr(patient, 'lopinavir_ritonavir', False))
            + coeffs["nelfinavir"] * float(getattr(patient, 'nelfinavir', False))
            + coeffs["saquinavir"] * float(getattr(patient, 'saquinavir', False))
            + coeffs["stavudine"] * float(getattr(patient, 'stavudine', False))
            + coeffs["zidovudine"] * float(getattr(patient, 'zidovudine', False))
            + coeffs["cd4_count"] * (getattr(patient, 'cd4_count', 500) / 100)  # Scaled to per 100 cells
            + coeffs["hiv_viral_load"] * getattr(patient, 'hiv_viral_load', 2.0)  # Default log10 viral load
            + coeffs["constant"]
        )

        # Convert to 5-year risk probability (model predicts 5-year risk)
        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = risk_probability * 100.0

        # Ensure risk is within reasonable bounds
        risk_percentage = np.clip(risk_percentage, 0.0, 30.0)

        # Categorize risk
        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update({
            "age": patient.age,
            "sex": patient.sex,
            "smoking_status": smoking_status,
            "systolic_bp": patient.systolic_bp,
            "total_cholesterol": patient.total_cholesterol,
            "hdl_cholesterol": patient.hdl_cholesterol,
            "diabetes": patient.diabetes,
            "fam_history_cvd": getattr(patient, 'fam_history_cvd', False),
            "cd4_count": getattr(patient, 'cd4_count', 500),
            "hiv_viral_load": getattr(patient, 'hiv_viral_load', 2.0),
            "linear_predictor": linear_predictor,
            "note": "This is a 5-year risk prediction (not 10-year)"
        })

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk using D:A:D categories (for 5-year risk)."""
        if risk_percentage < 1:
            return "low"
        elif risk_percentage < 2:
            return "moderate"
        elif risk_percentage < 5:
            return "high"
        else:
            return "very_high"

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Fill missing optional columns with defaults
        defaults = {
            "smoking_status": "never",
            "diabetes": False,
            "fam_history_cvd": False,
            "abacavir": False,
            "didanosine": False,
            "indinavir_ritonavir": False,
            "lopinavir_ritonavir": False,
            "nelfinavir": False,
            "saquinavir": False,
            "stavudine": False,
            "zidovudine": False,
            "cd4_count": 500,
            "hiv_viral_load": 2.0
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Vectorized calculations
        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = _DAD_COEFFICIENTS_MEN if sex == 'male' else _DAD_COEFFICIENTS_WOMEN

            # Calculate smoking coefficients
            smoking_current = (results.loc[mask, "smoking_status"] == "current").astype(int)
            smoking_previous = (results.loc[mask, "smoking_status"] == "previous").astype(int)

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"]
                + coeffs["smoking_current"] * smoking_current
                + coeffs["smoking_previous"] * smoking_previous
                + coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"]
                + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"]
                + coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"]
                + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float)
                + coeffs["fam_history_cvd"] * results.loc[mask, "fam_history_cvd"].astype(float)
                + coeffs["abacavir"] * results.loc[mask, "abacavir"].astype(float)
                + coeffs["didanosine"] * results.loc[mask, "didanosine"].astype(float)
                + coeffs["indinavir_ritonavir"] * results.loc[mask, "indinavir_ritonavir"].astype(float)
                + coeffs["lopinavir_ritonavir"] * results.loc[mask, "lopinavir_ritonavir"].astype(float)
                + coeffs["nelfinavir"] * results.loc[mask, "nelfinavir"].astype(float)
                + coeffs["saquinavir"] * results.loc[mask, "saquinavir"].astype(float)
                + coeffs["stavudine"] * results.loc[mask, "stavudine"].astype(float)
                + coeffs["zidovudine"] * results.loc[mask, "zidovudine"].astype(float)
                + coeffs["cd4_count"] * (results.loc[mask, "cd4_count"] / 100)
                + coeffs["hiv_viral_load"] * results.loc[mask, "hiv_viral_load"]
                + coeffs["constant"]
            )

            # Convert to probability
            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        # Clip risks
        results["risk_score"] = np.clip(results["risk_score"], 0.0, 30.0)

        # Vectorized categorization
        risk_scores = results["risk_score"]
        conditions = [
            risk_scores < 1,
            (risk_scores >= 1) & (risk_scores < 2),
            (risk_scores >= 2) & (risk_scores < 5),
            risk_scores >= 5
        ]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")

        return results
