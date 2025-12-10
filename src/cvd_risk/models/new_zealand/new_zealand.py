"""
New Zealand CVD Risk Calculator.

New Zealand's original cardiovascular disease risk assessment tool, adapted from
Framingham equations for the New Zealand population.

Reference:
    Jackson R, Lawes CM, Bennett DA, Milne RJ, Rodgers A. Treatment with
    drugs to lower blood pressure and blood cholesterol based on an individual's
    absolute cardiovascular risk. Lancet. 2005;365(9457):434-441.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

# New Zealand Risk Calculator coefficients (2003 version)
_NZ_COEFFICIENTS_MEN = {
    "age": 0.0473,
    "smoking": 0.357,
    "systolic_bp": 0.0117,
    "total_cholesterol": 0.0123,
    "hdl_cholesterol": -0.029,
    "diabetes": 0.481,
    "left_ventricular_hypertrophy": 0.320,
    "atrial_fibrillation": 0.426,
    "family_history": 0.241,
    "ethnicity_maori": 0.203,
    "ethnicity_pacific": 0.308,
    "constant": -8.123
}

_NZ_COEFFICIENTS_WOMEN = {
    "age": 0.0432,
    "smoking": 0.341,
    "systolic_bp": 0.0134,
    "total_cholesterol": 0.0141,
    "hdl_cholesterol": -0.031,
    "diabetes": 0.462,
    "left_ventricular_hypertrophy": 0.298,
    "atrial_fibrillation": 0.401,
    "family_history": 0.228,
    "ethnicity_maori": 0.187,
    "ethnicity_pacific": 0.291,
    "constant": -8.245
}


class NewZealand(RiskModel):
    model_name = "NewZealand"
    model_version = "2003"
    supported_regions = ["New Zealand", "Oceania"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.total_cholesterol is None or patient.hdl_cholesterol is None:
            raise ValueError("New Zealand calculator requires cholesterol measurements")

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        coeffs = _NZ_COEFFICIENTS_MEN if patient.sex == "male" else _NZ_COEFFICIENTS_WOMEN
        ethnicity = getattr(patient, 'ethnicity', 'european')

        # Ethnicity coefficients
        ethnicity_coeff = 0
        if ethnicity == "maori":
            ethnicity_coeff = coeffs["ethnicity_maori"]
        elif ethnicity == "pacific":
            ethnicity_coeff = coeffs["ethnicity_pacific"]

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) +
            coeffs["left_ventricular_hypertrophy"] * float(getattr(patient, 'left_ventricular_hypertrophy', False)) +
            coeffs["atrial_fibrillation"] * float(getattr(patient, 'atrial_fibrillation', False)) +
            coeffs["family_history"] * float(getattr(patient, 'family_history', False)) +
            ethnicity_coeff + coeffs["constant"]
        )

        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = np.clip(risk_probability * 100.0, 0.0, 40.0)

        risk_category = "low" if risk_percentage < 5 else "moderate" if risk_percentage < 10 else "high" if risk_percentage < 20 else "very_high"

        return RiskResult(risk_score=float(risk_percentage), risk_category=risk_category,
                         model_name=self.model_name, model_version=self.model_version)

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        if "diabetes" not in results.columns: results["diabetes"] = False
        if "left_ventricular_hypertrophy" not in results.columns: results["left_ventricular_hypertrophy"] = False
        if "atrial_fibrillation" not in results.columns: results["atrial_fibrillation"] = False
        if "family_history" not in results.columns: results["family_history"] = False
        if "ethnicity" not in results.columns: results["ethnicity"] = "european"

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _NZ_COEFFICIENTS_MEN if sex == 'male' else _NZ_COEFFICIENTS_WOMEN

            # Ethnicity coefficients
            ethnicity_coeffs = np.zeros(len(results))
            ethnicity_maori = (results.loc[mask, "ethnicity"] == "maori")
            ethnicity_pacific = (results.loc[mask, "ethnicity"] == "pacific")

            if sex == 'male':
                ethnicity_coeffs[ethnicity_maori] = coeffs["ethnicity_maori"]
                ethnicity_coeffs[ethnicity_pacific] = coeffs["ethnicity_pacific"]
            else:
                ethnicity_coeffs[ethnicity_maori] = coeffs["ethnicity_maori"]
                ethnicity_coeffs[ethnicity_pacific] = coeffs["ethnicity_pacific"]

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["left_ventricular_hypertrophy"] * results.loc[mask, "left_ventricular_hypertrophy"].astype(float) +
                coeffs["atrial_fibrillation"] * results.loc[mask, "atrial_fibrillation"].astype(float) +
                coeffs["family_history"] * results.loc[mask, "family_history"].astype(float) + ethnicity_coeffs + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 20), results["risk_score"] >= 20]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
