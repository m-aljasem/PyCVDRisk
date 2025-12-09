"""
ASCVD (Atherosclerotic Cardiovascular Disease) Pooled Cohort Equations.

The ASCVD Pooled Cohort Equations estimate 10-year risk of atherosclerotic
cardiovascular disease for primary prevention, incorporating race/ethnicity
and diabetes status.

Reference:
    Goff DC Jr, Lloyd-Jones DM, Bennett G, et al. 2013 ACC/AHA Guideline
    on the Assessment of Cardiovascular Risk: A Report of the American
    College of Cardiology/American Heart Association Task Force on
    Practice Guidelines. Circulation. 2014;129(25 Suppl 2):S49-73.

This implementation includes:
- Race-specific equations for White/Other and African American populations
- Diabetes as a risk factor
- Treatment for hypertension
- Vectorized batch calculations
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# ASCVD Pooled Cohort Equations coefficients (2013 ACC/AHA)
# Full implementation with race and diabetes support
_ASCVD_COEFFICIENTS = {
    # White/Other race, no diabetes
    "white_male_no_diabetes": {
        "ln_age": 3.06117,
        "ln_age_squared": -1.12370,
        "ln_total_chol": 1.93303,
        "ln_age_ln_total_chol": -0.52551,
        "ln_hdl": -1.47906,
        "ln_age_ln_hdl": 0.31583,
        "ln_sbp_ln_age_untreated": 0.93338,
        "ln_sbp_ln_age_treated": 1.99881,
        "smoking": 0.65451,
        "smoking_ln_age": -0.57763,
        "constant": -23.9802,
    },
    "white_female_no_diabetes": {
        "ln_age": 2.32888,
        "ln_age_squared": -1.12370,
        "ln_total_chol": 1.20904,
        "ln_age_ln_total_chol": -0.30784,
        "ln_hdl": -0.70833,
        "ln_age_ln_hdl": 0.39163,
        "ln_sbp_ln_age_untreated": 1.34065,
        "ln_sbp_ln_age_treated": 2.00168,
        "smoking": 0.52873,
        "smoking_ln_age": -0.48660,
        "constant": -26.1931,
    },
    # African American, no diabetes
    "black_male_no_diabetes": {
        "ln_age": 2.46996,
        "ln_age_squared": 0.0,
        "ln_total_chol": 0.30203,
        "ln_age_ln_total_chol": 0.0,
        "ln_hdl": -0.30712,
        "ln_age_ln_hdl": 0.0,
        "ln_sbp_ln_age_untreated": 1.91694,
        "ln_sbp_ln_age_treated": 2.06128,
        "smoking": 0.54907,
        "smoking_ln_age": 0.0,
        "constant": -29.1817,
    },
    "black_female_no_diabetes": {
        "ln_age": 17.1141,
        "ln_age_squared": 0.0,
        "ln_total_chol": 0.93968,
        "ln_age_ln_total_chol": 0.0,
        "ln_hdl": -18.9196,
        "ln_age_ln_hdl": 4.58466,
        "ln_sbp_ln_age_untreated": 29.5717,
        "ln_sbp_ln_age_treated": -6.4321,
        "smoking": 0.39525,
        "smoking_ln_age": 0.0,
        "constant": -146.593,
    },
    # White/Other race, with diabetes
    "white_male_diabetes": {
        "ln_age": 3.06117,
        "ln_age_squared": -1.12370,
        "ln_total_chol": 1.93303,
        "ln_age_ln_total_chol": -0.52551,
        "ln_hdl": -1.47906,
        "ln_age_ln_hdl": 0.31583,
        "ln_sbp_ln_age_untreated": 0.93338,
        "ln_sbp_ln_age_treated": 1.99881,
        "smoking": 0.65451,
        "smoking_ln_age": -0.57763,
        "constant": -23.9802,
        "diabetes_adjustment": 0.57367,
    },
    "white_female_diabetes": {
        "ln_age": 2.32888,
        "ln_age_squared": -1.12370,
        "ln_total_chol": 1.20904,
        "ln_age_ln_total_chol": -0.30784,
        "ln_hdl": -0.70833,
        "ln_age_ln_hdl": 0.39163,
        "ln_sbp_ln_age_untreated": 1.34065,
        "ln_sbp_ln_age_treated": 2.00168,
        "smoking": 0.52873,
        "smoking_ln_age": -0.48660,
        "constant": -26.1931,
        "diabetes_adjustment": 0.33791,
    },
    # African American, with diabetes
    "black_male_diabetes": {
        "ln_age": 2.46996,
        "ln_age_squared": 0.0,
        "ln_total_chol": 0.30203,
        "ln_age_ln_total_chol": 0.0,
        "ln_hdl": -0.30712,
        "ln_age_ln_hdl": 0.0,
        "ln_sbp_ln_age_untreated": 1.91694,
        "ln_sbp_ln_age_treated": 2.06128,
        "smoking": 0.54907,
        "smoking_ln_age": 0.0,
        "constant": -29.1817,
        "diabetes_adjustment": 0.64562,
    },
    "black_female_diabetes": {
        "ln_age": 17.1141,
        "ln_age_squared": 0.0,
        "ln_total_chol": 0.93968,
        "ln_age_ln_total_chol": 0.0,
        "ln_hdl": -18.9196,
        "ln_age_ln_hdl": 4.58466,
        "ln_sbp_ln_age_untreated": 29.5717,
        "ln_sbp_ln_age_treated": -6.4321,
        "smoking": 0.39525,
        "smoking_ln_age": 0.0,
        "constant": -146.593,
        "diabetes_adjustment": 0.40081,
    },
}

# Baseline survival probabilities at 10 years (same for all groups)
_ASCVD_BASELINE_SURVIVAL = {"male": 0.91436, "female": 0.96652}


class ASCVD(RiskModel):
    """
    ASCVD Pooled Cohort Equations for 10-year CVD risk prediction.

    The ASCVD model estimates 10-year risk of atherosclerotic cardiovascular
    disease using the Pooled Cohort Equations from the 2013 ACC/AHA guidelines.

    Parameters
    ----------
    age : int
        Age in years (40-79 for optimal performance).
    sex : Literal["male", "female"]
        Biological sex.
    race : Optional[Literal["white", "black", "other"]]
        Race/ethnicity. "other" is grouped with "white". If None, defaults to "white".
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L.
    hdl_cholesterol : float
        HDL cholesterol in mmol/L.
    smoking : bool
        Current smoking status.
    diabetes : Optional[bool]
        Diabetes status. If None, assumed to be False.
    antihypertensive : Optional[bool]
        Use of antihypertensive medication. If None, assumed to be False.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year ASCVD risk as percentage
        - risk_category: Risk classification
        - model_name: "ASCVD"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk.models.ascvd import ASCVD
    >>> from cvd_risk.core.validation import PatientData
    >>>
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     race="white",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,
    ...     hdl_cholesterol=1.2,
    ...     smoking=True,
    ...     diabetes=False,
    ...     antihypertensive=False,
    ... )
    >>> model = ASCVD()
    >>> result = model.calculate(patient)
    >>> print(f"10-year ASCVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - Model validated for ages 40-79 years
    - This implementation uses white/non-African American coefficients
    - Full model includes race-specific equations
    - Risk estimates are for 10-year period
    """

    model_name = "ASCVD"
    model_version = "2013"
    supported_regions = None  # US-based model

    def __init__(self) -> None:
        """Initialize ASCVD model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for ASCVD requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for ASCVD calculation.
        """
        super().validate_input(patient)

        if patient.age < 40 or patient.age > 79:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 79] years. "
                "Results may have reduced accuracy."
            )

        if patient.race is None:
            logger.warning("Race not specified, defaulting to 'white'")
        elif patient.race not in ["white", "black", "other"]:
            raise ValueError(f"Race must be one of ['white', 'black', 'other'], got {patient.race}")

        if patient.diabetes is None:
            logger.warning("Diabetes status not specified, assuming no diabetes")

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year ASCVD risk using Pooled Cohort Equations.

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

        # Convert cholesterol from mmol/L to mg/dL (as expected by the equations)
        total_chol_mgdl = patient.total_cholesterol * 38.67
        hdl_mgdl = patient.hdl_cholesterol * 38.67

        # Determine coefficient set based on race and diabetes
        race = patient.race if patient.race is not None else "white"
        race = "white" if race == "other" else race  # Group "other" with "white"
        diabetes = patient.diabetes if patient.diabetes is not None else False

        coeff_key = f"{race}_{patient.sex}_{'diabetes' if diabetes else 'no_diabetes'}"
        coeffs = _ASCVD_COEFFICIENTS[coeff_key]
        baseline_surv = _ASCVD_BASELINE_SURVIVAL[patient.sex]

        # Calculate linear predictor
        ln_age = np.log(patient.age)
        ln_total_chol = np.log(total_chol_mgdl)
        ln_hdl = np.log(hdl_mgdl)
        ln_sbp = np.log(patient.systolic_bp)

        # Check if on antihypertensive treatment
        treated = patient.antihypertensive if patient.antihypertensive is not None else False

        # Build linear predictor
        linear_predictor = (
            coeffs["ln_age"] * ln_age
            + coeffs["ln_age_squared"] * ln_age * ln_age
            + coeffs["ln_total_chol"] * ln_total_chol
            + coeffs["ln_age_ln_total_chol"] * ln_age * ln_total_chol
            + coeffs["ln_hdl"] * ln_hdl
            + coeffs["ln_age_ln_hdl"] * ln_age * ln_hdl
            + (coeffs["ln_sbp_ln_age_treated"] if treated else coeffs["ln_sbp_ln_age_untreated"])
            * ln_sbp * ln_age
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["smoking_ln_age"] * float(patient.smoking) * ln_age
            + coeffs["constant"]
        )

        # Add diabetes adjustment if applicable
        if diabetes and "diabetes_adjustment" in coeffs:
            linear_predictor += coeffs["diabetes_adjustment"]

        # Calculate risk
        risk_percentage = (1.0 - (baseline_surv ** np.exp(linear_predictor))) * 100.0
        risk_percentage = np.clip(risk_percentage, 0.0, 100.0)

        # Categorize risk (ASCVD categories)
        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update(
            {
                "age": patient.age,
                "sex": patient.sex,
                "race": race,
                "diabetes": diabetes,
                "linear_predictor": linear_predictor,
                "treated": treated,
                "total_chol_mgdl": total_chol_mgdl,
                "hdl_mgdl": hdl_mgdl,
            }
        )

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk using ASCVD categories."""
        if risk_percentage < 5:
            return "low"
        elif risk_percentage < 7.5:
            return "borderline"
        elif risk_percentage < 20:
            return "intermediate"
        else:
            return "high"

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates ASCVD risk for multiple patients efficiently.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Fill missing optional columns with defaults
        if "race" not in results.columns:
            results["race"] = "white"
        if "diabetes" not in results.columns:
            results["diabetes"] = False
        if "antihypertensive" not in results.columns:
            results["antihypertensive"] = False

        # Convert cholesterol from mmol/L to mg/dL
        total_chol_mgdl = results["total_cholesterol"] * 38.67
        hdl_mgdl = results["hdl_cholesterol"] * 38.67

        # Group "other" race with "white"
        race_clean = results["race"].fillna("white").replace({"other": "white"})

        # Create coefficient keys
        coeff_keys = (
            race_clean.astype(str) + "_" +
            results["sex"].astype(str) + "_" +
            results["diabetes"].astype(bool).astype(str).replace({"True": "diabetes", "False": "no_diabetes"})
        )

        # Vectorized calculations
        ln_age = np.log(results["age"])
        ln_total_chol = np.log(total_chol_mgdl)
        ln_hdl = np.log(hdl_mgdl)
        ln_sbp = np.log(results["systolic_bp"])
        smoking = results["smoking"].astype(float)
        treated = results["antihypertensive"].astype(bool)

        # Initialize linear predictor
        linear_predictor = np.zeros(len(results))

        # Process each coefficient set
        for coeff_key in coeff_keys.unique():
            mask = coeff_keys == coeff_key
            if not mask.any():
                continue

            coeffs = _ASCVD_COEFFICIENTS[coeff_key]
            baseline_surv = _ASCVD_BASELINE_SURVIVAL[coeff_key.split("_")[1]]  # Extract sex from key

            # Calculate linear predictor for this group
            group_lp = (
                coeffs["ln_age"] * ln_age[mask]
                + coeffs["ln_age_squared"] * ln_age[mask] * ln_age[mask]
                + coeffs["ln_total_chol"] * ln_total_chol[mask]
                + coeffs["ln_age_ln_total_chol"] * ln_age[mask] * ln_total_chol[mask]
                + coeffs["ln_hdl"] * ln_hdl[mask]
                + coeffs["ln_age_ln_hdl"] * ln_age[mask] * ln_hdl[mask]
                + np.where(treated[mask],
                          coeffs["ln_sbp_ln_age_treated"],
                          coeffs["ln_sbp_ln_age_untreated"]) * ln_sbp[mask] * ln_age[mask]
                + coeffs["smoking"] * smoking[mask]
                + coeffs["smoking_ln_age"] * smoking[mask] * ln_age[mask]
                + coeffs["constant"]
            )

            # Add diabetes adjustment if applicable
            if "diabetes_adjustment" in coeffs:
                group_lp += coeffs["diabetes_adjustment"]

            # Calculate risk
            risk = 1.0 - (baseline_surv ** np.exp(group_lp))
            risk = np.clip(risk, 0.0, 1.0)
            results.loc[mask, "risk_score"] = risk * 100.0

        # Vectorized risk categorization
        risk_scores = results["risk_score"]
        conditions = [
            risk_scores < 5,
            (risk_scores >= 5) & (risk_scores < 7.5),
            (risk_scores >= 7.5) & (risk_scores < 20),
            risk_scores >= 20
        ]
        choices = ["low", "borderline", "intermediate", "high"]
        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results

