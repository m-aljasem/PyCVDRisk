"""
LifeCVD2 cardiovascular risk model.

LifeCVD2 is a comprehensive lifetime cardiovascular disease risk prediction model
that estimates both 10-year and lifetime CVD risk using competing risk methodology
with age-specific baseline hazards and region-specific recalibration.

Reference:
    Hageman SHJ, et al. Prediction of individual lifetime cardiovascular risk and
    potential treatment benefit: development and recalibration of the LIFE-CVD2
    model to four European risk regions. European Journal of Preventive Cardiology.
    2024. DOI: 10.1093/eurjpc/zwae174
"""

import logging
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# LifeCVD2 centering values (from model derivation)
_LIFECVD2_CENTER_AGE = 60
_LIFECVD2_CENTER_SBP = 120
_LIFECVD2_CENTER_TC = 6.0  # mmol/L
_LIFECVD2_CENTER_HDL = 1.3  # mmol/L

# LifeCVD2 coefficients for CVD mortality (male and female)
_LIFECVD2_BETA_CVD_MEN = {
    'age': 0.0977,  # Main effect (per 5 years, handled in logic)
    'smoking': [0.6425, -0.0630],  # [main effect, age interaction per 5 years]
    'sbp': [0.2894, -0.0373],      # per 20 mmHg
    'diabetes': [0.6627, -0.0911], # binary
    'tc': [0.1521, -0.0291],       # per 1 mmol/L
    'hdl': [-0.2744, 0.0424]       # per 0.5 mmol/L
}

_LIFECVD2_BETA_CVD_WOMEN = {
    'age': 0.1190,
    'smoking': [0.8202, -0.1001],
    'sbp': [0.3358, -0.0509],
    'diabetes': [0.8541, -0.1217],
    'tc': [0.1019, -0.0299],
    'hdl': [-0.2457, 0.0410]
}

# LifeCVD2 coefficients for non-CVD mortality (male and female)
_LIFECVD2_BETA_NONCVD_MEN = {
    'age': -0.0081,
    'smoking': [0.7987, -0.0682],
    'sbp': [0.0745, -0.0371],
    'diabetes': [0.4738, -0.0880],
    'tc': [-0.0878, -0.0007],
    'hdl': [0.0979, -0.0245]
}

_LIFECVD2_BETA_NONCVD_WOMEN = {
    'age': -0.1035,
    'smoking': [0.7806, -0.0154],
    'sbp': [0.0580, -0.0187],
    'diabetes': [0.5418, -0.0652],
    'tc': [-0.0461, -0.0174],
    'hdl': [-0.0524, 0.0143]
}

# LifeCVD2 recalibration scales by region and sex
_LIFECVD2_RECALIBRATION = {
    'low': {
        'male': {'cvd': [-1.1835, 0.8139], 'noncvd': [0.4309, 1.1526]},
        'female': {'cvd': [-1.4978, 0.7438], 'noncvd': [1.2916, 1.2578]}
    },
    'moderate': {
        'male': {'cvd': [-0.8411, 0.8303], 'noncvd': [0.2261, 1.1082]},
        'female': {'cvd': [-1.1887, 0.7590], 'noncvd': [1.2425, 1.2582]}
    },
    'high': {
        'male': {'cvd': [-0.2823, 0.9427], 'noncvd': [0.2180, 1.0521]},
        'female': {'cvd': [-0.2276, 0.9132], 'noncvd': [0.5902, 1.0991]}
    },
    'very_high': {
        'male': {'cvd': [-0.0096, 0.8783], 'noncvd': [-0.4217, 0.8938]},
        'female': {'cvd': [0.2217, 0.8675], 'noncvd': [-0.3845, 0.8872]}
    }
}

# LifeCVD2 baseline survival probabilities (1-year, age-specific)
# Full resolution data from Supplementary Table 4 (Page 19)
# Format: Age -> [male_cvd, female_cvd, male_noncvd, female_noncvd]
_LIFECVD2_BASELINE_SURVIVAL = {
    35: [0.99905, 0.99975, 0.99921, 0.99964], 36: [0.99912, 0.99970, 0.99923, 0.99960],
    37: [0.99915, 0.99964, 0.99922, 0.99955], 38: [0.99912, 0.99957, 0.99920, 0.99950],
    39: [0.99906, 0.99951, 0.99916, 0.99945], 40: [0.99897, 0.99944, 0.99911, 0.99939],
    41: [0.99886, 0.99936, 0.99906, 0.99934], 42: [0.99873, 0.99929, 0.99899, 0.99928],
    43: [0.99859, 0.99921, 0.99892, 0.99921], 44: [0.99837, 0.99912, 0.99884, 0.99915],
    45: [0.99810, 0.99902, 0.99874, 0.99909], 46: [0.99782, 0.99893, 0.99862, 0.99902],
    47: [0.99757, 0.99885, 0.99850, 0.99894], 48: [0.99739, 0.99879, 0.99835, 0.99885],
    49: [0.99724, 0.99875, 0.99818, 0.99873], 50: [0.99710, 0.99871, 0.99799, 0.99859],
    51: [0.99697, 0.99867, 0.99777, 0.99843], 52: [0.99683, 0.99862, 0.99754, 0.99827],
    53: [0.99666, 0.99856, 0.99731, 0.99810], 54: [0.99647, 0.99847, 0.99705, 0.99792],
    55: [0.99626, 0.99836, 0.99678, 0.99772], 56: [0.99604, 0.99824, 0.99649, 0.99752],
    57: [0.99585, 0.99812, 0.99617, 0.99730], 58: [0.99568, 0.99801, 0.99581, 0.99708],
    59: [0.99554, 0.99790, 0.99539, 0.99686], 60: [0.99541, 0.99778, 0.99492, 0.99662],
    61: [0.99527, 0.99766, 0.99442, 0.99637], 62: [0.99511, 0.99752, 0.99390, 0.99609],
    63: [0.99488, 0.99737, 0.99337, 0.99576], 64: [0.99455, 0.99720, 0.99284, 0.99537],
    65: [0.99415, 0.99701, 0.99227, 0.99492], 66: [0.99370, 0.99680, 0.99166, 0.99443],
    67: [0.99323, 0.99656, 0.99097, 0.99391], 68: [0.99278, 0.99631, 0.99019, 0.99335],
    69: [0.99232, 0.99602, 0.98929, 0.99277], 70: [0.99184, 0.99570, 0.98826, 0.99213],
    71: [0.99131, 0.99533, 0.98712, 0.99142], 72: [0.99072, 0.99490, 0.98586, 0.99062],
    73: [0.99005, 0.99438, 0.98450, 0.98970], 74: [0.98926, 0.99373, 0.98299, 0.98865],
    75: [0.98836, 0.99294, 0.98130, 0.98742], 76: [0.98734, 0.99202, 0.97939, 0.98603],
    77: [0.98621, 0.99100, 0.97722, 0.98445], 78: [0.98497, 0.98988, 0.97473, 0.98264],
    79: [0.98359, 0.98857, 0.97170, 0.98044], 80: [0.98201, 0.98705, 0.96816, 0.97785],
    81: [0.98017, 0.98531, 0.96414, 0.97492], 82: [0.97801, 0.98335, 0.95971, 0.97170],
    83: [0.97534, 0.98112, 0.95484, 0.96822], 84: [0.97152, 0.97839, 0.94918, 0.96432],
    85: [0.96682, 0.97520, 0.94274, 0.95992], 86: [0.96166, 0.97161, 0.93559, 0.95494],
    87: [0.95646, 0.96768, 0.92781, 0.94933], 88: [0.95143, 0.96338, 0.91932, 0.94292],
    89: [0.94587, 0.95830, 0.90943, 0.93512], 90: [0.94009, 0.95259, 0.89842, 0.92606],
    91: [0.93461, 0.94648, 0.88671, 0.91596], 92: [0.92996, 0.94021, 0.87473, 0.90507],
    93: [0.92647, 0.93397, 0.86279, 0.89349], 94: [0.92337, 0.92755, 0.85053, 0.88056],
    95: [0.92060, 0.92089, 0.83786, 0.86630], 96: [0.91827, 0.91401, 0.82481, 0.85085],
    97: [0.91651, 0.90691, 0.81139, 0.83435], 98: [0.91543, 0.89962, 0.79762, 0.81694],
    99: [0.91515, 0.89215, 0.78351, 0.79878], 100: [0.91579, 0.88449, 0.76909, 0.77999]
}


class LifeCVD2(RiskModel):
    """
    LifeCVD2 lifetime cardiovascular risk prediction model.

    LifeCVD2 estimates both 10-year and lifetime CVD risk using competing risk
    methodology with age-specific baseline hazards and region-specific recalibration.

    Valid for ages 35-99 years.
    Requires: age, sex, region, systolic_bp, total_cholesterol, hdl_cholesterol,
    smoking status, and diabetes status.
    """

    model_name = "LifeCVD2"
    model_version = "1.0"
    supported_regions = ["low", "moderate", "high", "very_high"]

    def __init__(self, clinical_practice_mode: bool = True) -> None:
        """
        Initialize LifeCVD2 model.

        Parameters
        ----------
        clinical_practice_mode : bool
            If True (default), ignores diabetes coefficients as per Supplementary Table 3
            footnote ("For use in clinical practice, this coefficient should be ignored").
            If False, includes diabetes coefficients (for research/reproduction purposes).
        """
        super().__init__()
        self.clinical_practice_mode = clinical_practice_mode
        # Load baseline survival for all ages (now using full resolution data)
        self._baseline_survival = self._load_baseline_table()

    def _load_baseline_table(self) -> dict[int, list[float]]:
        """
        Load baseline survival table.

        Returns
        -------
        dict[int, list[float]]
            Age-indexed baseline survival probabilities.
        """
        # The constant now contains the full resolution 1-year data from Supplement Table 4.
        # No interpolation is required, so we simply return the dictionary.
        return _LIFECVD2_BASELINE_SURVIVAL

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient data for LifeCVD2 calculation.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid or outside model's applicable range.
        """
        super().validate_input(patient)

        if patient.age < 35 or patient.age > 99:
            logger.warning(
                f"Age {patient.age} outside optimal range [35, 99]. "
                "LifeCVD2 accuracy may be reduced."
            )

        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")

        if patient.diabetes is None:
            raise ValueError("Diabetes status is required for LifeCVD2")

    def _get_linear_predictor(self, age: int, sex: str, sbp: float, tc: float,
                            hdl: float, smoker: bool, diabetes: bool, endpoint: str) -> float:
        """
        Calculate the linear predictor for CVD or non-CVD mortality.

        Parameters
        ----------
        age : int
            Patient age in years
        sex : str
            'male' or 'female'
        sbp : float
            Systolic blood pressure in mmHg
        tc : float
            Total cholesterol in mmol/L
        hdl : float
            HDL cholesterol in mmol/L
        smoker : bool
            Smoking status
        diabetes : bool
            Diabetes status
        endpoint : str
            'cvd' or 'noncvd'

        Returns
        -------
        float
            Linear predictor value
        """
        # Select coefficients based on sex and endpoint
        if sex == 'male':
            coeffs = _LIFECVD2_BETA_CVD_MEN if endpoint == 'cvd' else _LIFECVD2_BETA_NONCVD_MEN
        else:
            coeffs = _LIFECVD2_BETA_CVD_WOMEN if endpoint == 'cvd' else _LIFECVD2_BETA_NONCVD_WOMEN

        # Age term (centered at 60, per 5 years)
        lp = coeffs['age'] * ((age - _LIFECVD2_CENTER_AGE) / 5.0)

        # Smoking (binary with age interaction)
        smoking_main, smoking_int = coeffs['smoking']
        lp += smoking_main * float(smoker)
        lp += smoking_int * float(smoker) * ((age - _LIFECVD2_CENTER_AGE) / 5.0)

        # Systolic BP (centered at 120, per 20 mmHg)
        sbp_unit = (sbp - _LIFECVD2_CENTER_SBP) / 20.0
        sbp_main, sbp_int = coeffs['sbp']
        lp += sbp_main * sbp_unit
        lp += sbp_int * sbp_unit * ((age - _LIFECVD2_CENTER_AGE) / 5.0)

        # Diabetes (binary with age interaction)
        # UPDATED: Conditional logic based on clinical_practice_mode
        if not (self.clinical_practice_mode and diabetes):
            diabetes_main, diabetes_int = coeffs['diabetes']
            lp += diabetes_main * float(diabetes)
            lp += diabetes_int * float(diabetes) * ((age - _LIFECVD2_CENTER_AGE) / 5.0)

        # Total cholesterol (centered at 6, per 1 mmol/L)
        tc_unit = tc - _LIFECVD2_CENTER_TC
        tc_main, tc_int = coeffs['tc']
        lp += tc_main * tc_unit
        lp += tc_int * tc_unit * ((age - _LIFECVD2_CENTER_AGE) / 5.0)

        # HDL cholesterol (centered at 1.3, per 0.5 mmol/L)
        hdl_unit = (hdl - _LIFECVD2_CENTER_HDL) / 0.5
        hdl_main, hdl_int = coeffs['hdl']
        lp += hdl_main * hdl_unit
        lp += hdl_int * hdl_unit * ((age - _LIFECVD2_CENTER_AGE) / 5.0)

        return lp

    def _recalibrate_risk(self, raw_survival: float, region: str, sex: str, endpoint: str) -> float:
        """
        Apply region-specific recalibration to raw survival probability.

        Parameters
        ----------
        raw_survival : float
            Raw 1-year survival probability
        region : str
            Risk region ('low', 'moderate', 'high', 'very_high')
        sex : str
            'male' or 'female'
        endpoint : str
            'cvd' or 'noncvd'

        Returns
        -------
        float
            Recalibrated 1-year survival probability
        """
        # Clip survival to avoid log errors
        raw_survival = np.clip(raw_survival, 1e-9, 1.0 - 1e-9)

        # Calculate log(-log(survival))
        log_neg_log_S = np.log(-np.log(raw_survival))

        # Get recalibration scales
        scales = _LIFECVD2_RECALIBRATION[region][sex][endpoint]
        scale1, scale2 = scales

        # Calculate recalibrated hazard
        new_hazard = np.exp(scale1 + scale2 * log_neg_log_S)

        # Convert back to survival
        new_survival = np.exp(-new_hazard)

        return new_survival

    def _compute_lifetime_risk(self, age: int, sex: str, region: str, sbp: float,
                             tc: float, hdl: float, smoker: bool, diabetes: bool) -> dict:
        """
        Compute lifetime CVD risk metrics.

        Parameters
        ----------
        age : int
            Starting age
        sex : str
            'male' or 'female'
        region : str
            Risk region
        sbp : float
            Systolic BP
        tc : float
            Total cholesterol
        hdl : float
            HDL cholesterol
        smoker : bool
            Smoking status
        diabetes : bool
            Diabetes status

        Returns
        -------
        dict
            Dictionary with risk metrics
        """
        current_age = max(35, min(99, age))  # Clamp to valid range

        # Initialize tracking variables
        cumulative_surv_cvd_free = 1.0  # Survival without CVD
        cumulative_risk_cvd = 0.0       # Cumulative CVD risk
        cum_risk_10y = 0.0              # 10-year risk

        surv_curve = []
        ages = []

        # Year-by-year calculation
        for t_age in range(current_age, 101):
            # Get baseline survival for this age interval
            if t_age not in self._baseline_survival:
                break  # Beyond model range

            baselines = self._baseline_survival[t_age]
            s0_cvd = baselines[0] if sex == 'male' else baselines[1]
            s0_noncvd = baselines[2] if sex == 'male' else baselines[3]

            # Calculate linear predictors
            lp_cvd = self._get_linear_predictor(t_age, sex, sbp, tc, hdl, smoker, diabetes, 'cvd')
            lp_noncvd = self._get_linear_predictor(t_age, sex, sbp, tc, hdl, smoker, diabetes, 'noncvd')

            # Calculate uncalibrated 1-year survival
            raw_surv_cvd = s0_cvd ** np.exp(lp_cvd)
            raw_surv_noncvd = s0_noncvd ** np.exp(lp_noncvd)

            # Recalibrate to region
            cal_surv_cvd = self._recalibrate_risk(raw_surv_cvd, region, sex, 'cvd')
            cal_surv_noncvd = self._recalibrate_risk(raw_surv_noncvd, region, sex, 'noncvd')

            # Convert to 1-year probabilities
            prob_cvd = 1.0 - cal_surv_cvd
            prob_noncvd = 1.0 - cal_surv_noncvd

            # Apply to cumulative survival (life table method)
            incident_cvd = cumulative_surv_cvd_free * prob_cvd

            # Update cumulative risk
            cumulative_risk_cvd += incident_cvd

            # Track 10-year risk
            if t_age < current_age + 10:
                cum_risk_10y += incident_cvd

            # Update survival (surviving both CVD and non-CVD death)
            cumulative_surv_cvd_free *= (1.0 - prob_cvd - prob_noncvd)

            surv_curve.append(cumulative_surv_cvd_free)
            ages.append(t_age)

        # Calculate CVD-free life expectancy (median survival age)
        median_age = 100.0
        for i, s in enumerate(surv_curve):
            if s < 0.5:
                # Linear interpolation for precision
                if i > 0:
                    prev_s = surv_curve[i-1]
                    prev_age = ages[i-1]
                    curr_age = ages[i]
                    frac = (prev_s - 0.5) / (prev_s - s)
                    median_age = prev_age + frac
                else:
                    median_age = ages[i]
                break

        cvd_free_years = median_age - current_age

        return {
            '10_year_risk': cum_risk_10y,
            'lifetime_risk': cumulative_risk_cvd,
            'cvd_free_life_expectancy': cvd_free_years,
            'median_survival_age': median_age
        }

    def _get_risk_categories(self, ten_year_risk: float, lifetime_risk: float) -> Tuple[str, str]:
        """
        Categorize risk based on 10-year and lifetime risk thresholds.

        Parameters
        ----------
        ten_year_risk : float
            10-year CVD risk (0-1)
        lifetime_risk : float
            Lifetime CVD risk (0-1)

        Returns
        -------
        Tuple[str, str]
            (ten_year_category, lifetime_category)
        """
        # 10-year risk categories (as percentages)
        ten_year_pct = ten_year_risk * 100
        if ten_year_pct < 5:
            ten_year_cat = "low"
        elif ten_year_pct < 10:
            ten_year_cat = "moderate"
        elif ten_year_pct < 20:
            ten_year_cat = "high"
        else:
            ten_year_cat = "very_high"

        # Lifetime risk categories (as percentages)
        lifetime_pct = lifetime_risk * 100
        if lifetime_pct < 20:
            lifetime_cat = "low"
        elif lifetime_pct < 30:
            lifetime_cat = "moderate"
        elif lifetime_pct < 40:
            lifetime_cat = "high"
        else:
            lifetime_cat = "very_high"

        return ten_year_cat, lifetime_cat

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate lifetime CVD risk for a single patient.

        Returns the 10-year CVD risk as the primary risk score, with lifetime
        risk and CVD-free life expectancy in metadata.

        Parameters
        ----------
        patient : PatientData
            Validated patient data

        Returns
        -------
        RiskResult
            Risk calculation result with 10-year risk as primary score
        """
        self.validate_input(patient)

        # Extract patient data
        age = patient.age
        sex = patient.sex
        region = patient.region.lower()
        sbp = patient.systolic_bp
        tc = patient.total_cholesterol
        hdl = patient.hdl_cholesterol
        smoker = patient.smoking
        diabetes = patient.diabetes

        # Compute lifetime risk metrics
        risk_metrics = self._compute_lifetime_risk(age, sex, region, sbp, tc, hdl, smoker, diabetes)

        # Get risk categories
        ten_year_cat, lifetime_cat = self._get_risk_categories(
            risk_metrics['10_year_risk'], risk_metrics['lifetime_risk']
        )

        # Use 10-year risk as primary score (following clinical convention)
        primary_risk_score = risk_metrics['10_year_risk'] * 100.0  # Convert to percentage

        return RiskResult(
            risk_score=float(primary_risk_score),
            risk_category=ten_year_cat,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "lifetime_risk_percent": risk_metrics['lifetime_risk'] * 100.0,
                "lifetime_risk_category": lifetime_cat,
                "cvd_free_life_expectancy_years": risk_metrics['cvd_free_life_expectancy'],
                "median_survival_age": risk_metrics['median_survival_age'],
                "ten_year_risk_percent": risk_metrics['10_year_risk'] * 100.0,
                "region": region
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate lifetime CVD risk for a batch of patients.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with required columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added risk calculation columns
        """
        required = ["age", "sex", "region", "systolic_bp", "total_cholesterol",
                   "hdl_cholesterol", "smoking", "diabetes"]

        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["lifetime_risk_percent"] = np.nan
        results["lifetime_risk_category"] = ""
        results["cvd_free_life_expectancy_years"] = np.nan
        results["median_survival_age"] = np.nan
        results["model_name"] = self.model_name

        # Process each row
        for idx, row in results.iterrows():
            try:
                # Convert row to PatientData for validation
                patient_dict = row[required].to_dict()
                patient = PatientData(**patient_dict)

                # Calculate risk
                result = self.calculate(patient)

                # Update results
                results.loc[idx, "risk_score"] = result.risk_score
                results.loc[idx, "risk_category"] = result.risk_category
                results.loc[idx, "lifetime_risk_percent"] = result.calculation_metadata["lifetime_risk_percent"]
                results.loc[idx, "lifetime_risk_category"] = result.calculation_metadata["lifetime_risk_category"]
                results.loc[idx, "cvd_free_life_expectancy_years"] = result.calculation_metadata["cvd_free_life_expectancy_years"]
                results.loc[idx, "median_survival_age"] = result.calculation_metadata["median_survival_age"]

            except Exception as e:
                logger.warning(f"Error calculating risk for row {idx}: {e}")
                results.loc[idx, "risk_score"] = None
                results.loc[idx, "risk_category"] = "error"
                results.loc[idx, "error"] = str(e)

        return results