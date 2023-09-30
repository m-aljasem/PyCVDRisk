"""
PREVENT cardiovascular risk model.

The PREVENT equations are the American Heart Association (AHA) 2023
cardiovascular risk prediction model for Total CVD, ASCVD, and Heart Failure.

Reference:
    Khan SS, Matsushita K, Yang C, et al. Development and Validation of the
    American Heart Association's PREVENT Equations. Circulation. 2024;149(5):388-404.
    DOI: 10.1161/CIRCULATIONAHA.123.067626
"""

import logging
from typing import Literal, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# PREVENT sex and outcome-specific coefficients
_PREVENT_COEFFICIENTS = {
    'female': {
        'total_cvd': {
            'intercept': -3.307728,
            'age': 0.7939329,
            'cnhdl': 0.0305239,
            'chdl': -0.1606857,
            'csbp': -0.2394003,
            'csbp2': 0.360078,
            'diabetes': 0.8667604,
            'smoker': 0.5360739,
            'cegfr': 0.6045917,
            'cegfr2': 0.0433769,
            'treated_bp': 0.3151672,
            'statin': -0.1477655,
            'treated_bp_x_csbp2': -0.0663612,
            'statin_x_cnhdl': 0.1197879,
            'age_x_cnhdl': -0.0819715,
            'age_x_chdl': 0.0306769,
            'age_x_csbp2': -0.0946348,
            'age_x_diabetes': -0.27057,
            'age_x_smoker': -0.078715,
            'age_x_cegfr': -0.1637806
        },
        'ascvd': {
            'intercept': -3.819975,
            'age': 0.719883,
            'cnhdl': 0.1176967,
            'chdl': -0.151185,
            'csbp': -0.0835358,
            'csbp2': 0.3592852,
            'diabetes': 0.8348585,
            'smoker': 0.4831078,
            'cegfr': 0.4864619,
            'cegfr2': 0.0397779,
            'treated_bp': 0.2265309,
            'statin': -0.0592374,
            'treated_bp_x_csbp2': -0.0395762,
            'statin_x_cnhdl': 0.0844423,
            'age_x_cnhdl': -0.0567839,
            'age_x_chdl': 0.0325692,
            'age_x_csbp2': -0.1035985,
            'age_x_diabetes': -0.2417542,
            'age_x_smoker': -0.0791142,
            'age_x_cegfr': 0.1671492
        },
        'hf': {
            'intercept': -4.310409,
            'age': 0.8998235,
            'cnhdl': -0.0267869,
            'chdl': -0.0057645,
            'csbp': 0.1050076,
            'csbp2': 0.4069038,
            'diabetes': 0.6161384,
            'smoker': 0.4088498,
            'cegfr': 0.3860668,
            'cegfr2': 0.0326198,
            'treated_bp': 0.329195,
            'statin': -0.0728336,
            'treated_bp_x_csbp2': -0.0795015,
            'statin_x_cnhdl': 0.1471028,
            'age_x_cnhdl': -0.0442663,
            'age_x_chdl': 0.0043961,
            'age_x_csbp2': -0.1148839,
            'age_x_diabetes': -0.2265494,
            'age_x_smoker': -0.0876956,
            'age_x_cegfr': -0.191264
        }
    },
    'male': {
        'total_cvd': {
            'intercept': -3.031168,
            'age': 0.7688528,
            'cnhdl': 0.0736174,
            'chdl': -0.0954431,
            'csbp': -0.4347345,
            'csbp2': 0.3362658,
            'diabetes': 0.7692857,
            'smoker': 0.4386871,
            'cegfr': 0.5378979,
            'cegfr2': 0.0164827,
            'treated_bp': 0.288879,
            'statin': -0.1337349,
            'treated_bp_x_csbp2': -0.0475924,
            'statin_x_cnhdl': 0.150273,
            'age_x_cnhdl': -0.0517874,
            'age_x_chdl': 0.0191169,
            'age_x_csbp2': -0.1049477,
            'age_x_diabetes': -0.2251948,
            'age_x_smoker': -0.0895067,
            'age_x_cegfr': -0.1543702
        },
        'ascvd': {
            'intercept': -3.500655,
            'age': 0.7099847,
            'cnhdl': 0.1658663,
            'chdl': -0.1144285,
            'csbp': -0.2837212,
            'csbp2': 0.3239977,
            'diabetes': 0.7189597,
            'smoker': 0.3956973,
            'cegfr': 0.3690075,
            'cegfr2': 0.0203619,
            'treated_bp': 0.2036522,
            'statin': -0.0865581,
            'treated_bp_x_csbp2': -0.0322916,
            'statin_x_cnhdl': 0.114563,
            'age_x_cnhdl': -0.0300005,
            'age_x_chdl': 0.0232747,
            'age_x_csbp2': -0.0927024,
            'age_x_diabetes': -0.2018525,
            'age_x_smoker': -0.0970527,
            'age_x_cegfr': 0.0
        },
        'hf': {
            'intercept': -3.854616,
            'age': 0.8183041,
            'cnhdl': 0.0189814,
            'chdl': -0.0129381,
            'csbp': -0.0860113,
            'csbp2': 0.3166839,
            'diabetes': 0.4807497,
            'smoker': 0.3613287,
            'cegfr': 0.3185439,
            'cegfr2': 0.0224091,
            'treated_bp': 0.2365961,
            'statin': -0.0785958,
            'treated_bp_x_csbp2': -0.0438745,
            'statin_x_cnhdl': 0.1358586,
            'age_x_cnhdl': -0.0340834,
            'age_x_chdl': 0.0062643,
            'age_x_csbp2': -0.0990374,
            'age_x_diabetes': -0.1877418,
            'age_x_smoker': -0.0928904,
            'age_x_cegfr': -0.1312154
        }
    }
}


class Prevent(RiskModel):
    """
    PREVENT cardiovascular risk prediction model (AHA 2023).

    Predicts 10-year risk for:
    - Total CVD (composite of ASCVD and heart failure)
    - ASCVD (atherosclerotic cardiovascular disease)
    - Heart Failure

    Valid for ages 30-79.
    """

    model_name = "PREVENT"
    model_version = "2023"
    supported_outcomes = ["total_cvd", "ascvd", "hf"]

    def __init__(self, outcome: Literal["total_cvd", "ascvd", "hf"] = "total_cvd") -> None:
        """
        Initialize PREVENT model.

        Parameters
        ----------
        outcome : str
            Risk outcome to calculate. One of: "total_cvd", "ascvd", "hf"
        """
        super().__init__()
        if outcome not in self.supported_outcomes:
            raise ValueError(f"Outcome must be one of {self.supported_outcomes}")
        self.outcome = outcome

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.age < 30 or patient.age > 79:
            logger.warning(
                f"Age {patient.age} outside optimal range [30, 79]. "
                "PREVENT accuracy may be reduced."
            )
        if patient.egfr is None:
            raise ValueError("eGFR is required for PREVENT calculations")
        if patient.antihypertensive is None:
            raise ValueError("Antihypertensive medication status is required for PREVENT calculations")
        if patient.statin_use is None:
            raise ValueError("Statin use status is required for PREVENT calculations")

    def _transform_variables(self, age: Union[float, np.ndarray],
                           total_chol: Union[float, np.ndarray],
                           hdl_chol: Union[float, np.ndarray],
                           sbp: Union[float, np.ndarray],
                           egfr: Union[float, np.ndarray]) -> dict:
        """
        Transform input variables according to PREVENT methodology.

        Parameters
        ----------
        age : float or array
            Age in years
        total_chol : float or array
            Total cholesterol in mmol/L (converted to mg/dL internally)
        hdl_chol : float or array
            HDL cholesterol in mmol/L (converted to mg/dL internally)
        sbp : float or array
            Systolic blood pressure in mmHg
        egfr : float or array
            eGFR in mL/min/1.73m2

        Returns
        -------
        dict
            Transformed variables
        """
        # Convert from mmol/L to mg/dL for PREVENT calculations
        tc_mgdl = total_chol * 38.67
        hdl_mgdl = hdl_chol * 38.67

        # Unit conversion (mg/dL to mmol/L) - PREVENT expects mmol/L internally
        tc_mmol = tc_mgdl / 38.67
        hdl_mmol = hdl_mgdl / 38.67
        non_hdl_mmol = tc_mmol - hdl_mmol

        # Transformations
        t_age = (age - 55.0) / 10.0
        t_cnhdl = non_hdl_mmol - 3.5
        t_chdl = (hdl_mmol - 1.3) / 0.3

        # SBP splines
        t_csbp = (np.minimum(sbp, 110.0) - 110.0) / 20.0
        t_csbp2 = (np.maximum(sbp, 110.0) - 130.0) / 20.0

        # eGFR splines
        t_cegfr = (np.minimum(egfr, 60.0) - 60.0) / -15.0
        t_cegfr2 = (np.maximum(egfr, 60.0) - 90.0) / -15.0

        return {
            't_age': t_age,
            't_cnhdl': t_cnhdl,
            't_chdl': t_chdl,
            't_csbp': t_csbp,
            't_csbp2': t_csbp2,
            't_cegfr': t_cegfr,
            't_cegfr2': t_cegfr2
        }

    def _get_risk_thresholds(self, age: Union[int, float, np.ndarray]) -> tuple:
        """
        Get clinical risk thresholds based on age (AHA/ACC Guidelines).

        Returns
        -------
        tuple (moderate_cutoff, high_cutoff)
            >= moderate_cutoff: Moderate Risk
            >= high_cutoff: High Risk
        """
        # Vectorized logic for NumPy arrays
        if isinstance(age, (np.ndarray, pd.Series)):
            moderate_cut = np.select(
                [age < 40, (age >= 40) & (age < 60), age >= 60],
                [5.0, 7.5, 10.0],
                default=7.5
            )
            high_cut = np.select(
                [age < 40, (age >= 40) & (age < 60), age >= 60],
                [10.0, 15.0, 20.0],
                default=15.0
            )
            return moderate_cut, high_cut

        # Scalar logic for single patient
        if age < 40:
            return 5.0, 10.0
        elif age < 60:
            return 7.5, 15.0
        else:
            return 10.0, 20.0

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate PREVENT risk for a single patient.

        Parameters
        ----------
        patient : PatientData
            Patient information

        Returns
        -------
        RiskResult
            Risk calculation results
        """
        self.validate_input(patient)

        # Get coefficients
        sex = patient.sex.lower()
        coeffs = _PREVENT_COEFFICIENTS[sex][self.outcome]

        # Transform variables
        tf = self._transform_variables(
            patient.age, patient.total_cholesterol, patient.hdl_cholesterol,
            patient.systolic_bp, patient.egfr
        )

        # Indicator variables
        i_db = float(patient.diabetes or False)
        i_smk = float(patient.smoking)
        i_htn = float(patient.antihypertensive or False)
        i_statin = float(patient.statin_use or False)

        # Calculate log-odds
        log_odds = coeffs['intercept']

        # Main effects
        log_odds += coeffs['age'] * tf['t_age']
        log_odds += coeffs['cnhdl'] * tf['t_cnhdl']
        log_odds += coeffs['chdl'] * tf['t_chdl']
        log_odds += coeffs['csbp'] * tf['t_csbp']
        log_odds += coeffs['csbp2'] * tf['t_csbp2']
        log_odds += coeffs['diabetes'] * i_db
        log_odds += coeffs['smoker'] * i_smk
        log_odds += coeffs['cegfr'] * tf['t_cegfr']
        log_odds += coeffs['cegfr2'] * tf['t_cegfr2']
        log_odds += coeffs['treated_bp'] * i_htn
        log_odds += coeffs['statin'] * i_statin

        # Interactions
        log_odds += coeffs['treated_bp_x_csbp2'] * (i_htn * tf['t_csbp2'])
        log_odds += coeffs['statin_x_cnhdl'] * (i_statin * tf['t_cnhdl'])

        # Age interactions
        log_odds += coeffs['age_x_cnhdl'] * (tf['t_age'] * tf['t_cnhdl'])
        log_odds += coeffs['age_x_chdl'] * (tf['t_age'] * tf['t_chdl'])
        log_odds += coeffs['age_x_csbp2'] * (tf['t_age'] * tf['t_csbp2'])
        log_odds += coeffs['age_x_diabetes'] * (tf['t_age'] * i_db)
        log_odds += coeffs['age_x_smoker'] * (tf['t_age'] * i_smk)
        log_odds += coeffs['age_x_cegfr'] * (tf['t_age'] * tf['t_cegfr'])

        # Convert to risk percentage
        risk_percent = np.exp(log_odds) / (1 + np.exp(log_odds)) * 100.0
        risk_percent = np.clip(risk_percent, 0.0, 100.0)

        # Categorize risk
        moderate_cut, high_cut = self._get_risk_thresholds(patient.age)

        if risk_percent < moderate_cut:
            category = "low"
        elif risk_percent < high_cut:
            category = "moderate"
        else:
            category = "high"

        outcome_name = {
            "total_cvd": "Total CVD",
            "ascvd": "ASCVD",
            "hf": "Heart Failure"
        }[self.outcome]

        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "outcome": self.outcome,
                "outcome_name": outcome_name,
                "log_odds": float(log_odds)
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with required columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added risk columns
        """
        required = ["age", "sex", "total_cholesterol", "hdl_cholesterol",
                   "systolic_bp", "egfr", "diabetes", "smoking",
                   "antihypertensive", "statin_use"]

        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = "unknown"
        results["model_name"] = self.model_name
        results["outcome"] = self.outcome

        # Process by sex for efficiency
        for sex in ["male", "female"]:
            sex_mask = results["sex"].str.lower() == sex
            if not sex_mask.any():
                continue

            coeffs = _PREVENT_COEFFICIENTS[sex][self.outcome]
            data = results[sex_mask]

            # Vectorized transformations (inputs are in mmol/L, converted internally)
            tf = self._transform_variables(
                data["age"].values, data["total_cholesterol"].values,
                data["hdl_cholesterol"].values, data["systolic_bp"].values,
                data["egfr"].values
            )

            # Indicator variables
            i_db = data["diabetes"].fillna(False).astype(float).values
            i_smk = data["smoking"].astype(float).values
            i_htn = data["antihypertensive"].fillna(False).astype(float).values
            i_statin = data["statin_use"].fillna(False).astype(float).values

            # Vectorized log-odds calculation
            log_odds = coeffs['intercept'] + \
                coeffs['age'] * tf['t_age'] + \
                coeffs['cnhdl'] * tf['t_cnhdl'] + \
                coeffs['chdl'] * tf['t_chdl'] + \
                coeffs['csbp'] * tf['t_csbp'] + \
                coeffs['csbp2'] * tf['t_csbp2'] + \
                coeffs['diabetes'] * i_db + \
                coeffs['smoker'] * i_smk + \
                coeffs['cegfr'] * tf['t_cegfr'] + \
                coeffs['cegfr2'] * tf['t_cegfr2'] + \
                coeffs['treated_bp'] * i_htn + \
                coeffs['statin'] * i_statin + \
                coeffs['treated_bp_x_csbp2'] * (i_htn * tf['t_csbp2']) + \
                coeffs['statin_x_cnhdl'] * (i_statin * tf['t_cnhdl']) + \
                coeffs['age_x_cnhdl'] * (tf['t_age'] * tf['t_cnhdl']) + \
                coeffs['age_x_chdl'] * (tf['t_age'] * tf['t_chdl']) + \
                coeffs['age_x_csbp2'] * (tf['t_age'] * tf['t_csbp2']) + \
                coeffs['age_x_diabetes'] * (tf['t_age'] * i_db) + \
                coeffs['age_x_smoker'] * (tf['t_age'] * i_smk) + \
                coeffs['age_x_cegfr'] * (tf['t_age'] * tf['t_cegfr'])

            # Convert to risk percentage
            risk_percent = np.exp(log_odds) / (1 + np.exp(log_odds)) * 100.0
            risk_percent = np.clip(risk_percent, 0.0, 100.0)

            results.loc[sex_mask, "risk_score"] = risk_percent

        # Vectorized categorization
        moderate_cuts, high_cuts = self._get_risk_thresholds(results["age"].values)

        conditions = [
            results["risk_score"] < moderate_cuts,
            (results["risk_score"] >= moderate_cuts) & (results["risk_score"] < high_cuts),
            results["risk_score"] >= high_cuts
        ]
        choices = ["low", "moderate", "high"]

        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results

