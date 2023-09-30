"""
SMART-REACH (Secondary Manifestations of Arterial Disease - Reduction of Atherothrombosis for Continued Health)
Cardiovascular Life Expectancy and Risk Model.

This model estimates life expectancy without recurrent cardiovascular events and 
10-year risk of recurrent cardiovascular events (Myocardial Infarction, Stroke, CV Death)
in patients with established vascular disease (CAD, CVD, or PAD).

Reference:
    Kaasenbrood L, et al. (2018).
    Estimated Life Expectancy Without Recurrent Cardiovascular Events in Patients 
    With Vascular Disease: The SMART-REACH Model.
    Journal of the American Heart Association.
    DOI: 10.1161/JAHA.118.009217
"""

import logging
from typing import Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# CONSTANTS & COEFFICIENTS (From Table S2 and Table S3)
# -------------------------------------------------------------------------

# Age-specific 1-year baseline survivals (Table S2)
# Structure: Age: (S0_CV_event, S0_NonCV_death)
_SMART_REACH_BASELINE = {
    45: (1.0000, 1.0000), 46: (0.8539, 0.9855), 47: (0.8420, 1.0000), 48: (0.9088, 0.9950),
    49: (0.9172, 1.0000), 50: (0.8464, 1.0000), 51: (0.7297, 0.9949), 52: (0.8081, 0.9958),
    53: (0.8980, 1.0000), 54: (0.8155, 0.9896), 55: (0.7609, 0.9966), 56: (0.8113, 0.9935),
    57: (0.8173, 0.9842), 58: (0.7939, 0.9869), 59: (0.8382, 0.9935), 60: (0.8333, 0.9938),
    61: (0.8257, 0.9934), 62: (0.8000, 0.9734), 63: (0.7930, 0.9683), 64: (0.7962, 0.9768),
    65: (0.7807, 0.9725), 66: (0.7731, 0.9724), 67: (0.8118, 0.9586), 68: (0.7325, 0.9683),
    69: (0.7671, 0.9720), 70: (0.7236, 0.9539), 71: (0.6690, 0.9439), 72: (0.7173, 0.9469),
    73: (0.6978, 0.9299), 74: (0.6074, 0.9369), 75: (0.6880, 0.9537), 76: (0.6473, 0.9172),
    77: (0.7034, 0.9018), 78: (0.6904, 0.9280), 79: (0.6507, 0.8622), 80: (0.5946, 0.8688),
    81: (0.5328, 0.8381), 82: (0.4954, 0.8647), 83: (0.5376, 0.8478), 84: (0.4403, 0.8125),
    85: (0.5043, 0.7855), 86: (0.5509, 0.7284), 87: (0.5480, 0.7685), 88: (0.3889, 0.7197),
    89: (0.3048, 0.6469)
}

# Coefficients for Model A (Cardiovascular Events) - Table S3
# Note: SBP and Creatinine are scaled (per 10 units) in the calculation logic
_COEFFS_CV = {
    "male": 0.0720,
    "current_smoker": 0.4309,
    "diabetes": 0.4357,
    "sbp_per_10": -0.2814,
    "sbp_per_10_sq": 0.0010,
    "chol": -0.3671,
    "chol_sq": 0.0356,
    "creat_per_10": 0.0612,
    "loc_2": 0.3176,
    "loc_3": 0.2896,
    "afib": 0.2143,
    "chf": 0.4447
}

# Coefficients for Model B (Non-Cardiovascular Death) - Table S3
_COEFFS_NON_CV = {
    "male": 0.5986,
    "current_smoker": 4.2538,
    "age_interaction_smoker": -0.0486,  # Multiplied by Age if Smoker
    "diabetes": 0.4065,
    "sbp_per_10": -0.0074,
    "chol": -0.0030,
    "creat_per_10": -0.0189,
    "creat_per_10_sq": 0.0001,
    "loc_2": 0.1442,
    "loc_3": 0.5694,
    "afib": 0.3213,
    "chf": 0.2061
}

# Geographic Recalibration Offsets (Table S3 Footnotes)
# Added to the Linear Predictor (A and B)
_REGIONAL_OFFSETS = {
    "western_europe": {"A": -0.4246, "B": 0.1232},  # SMART-like
    "north_america":  {"A": 0.1552,  "B": 0.4134},  # REACH-like
}


class SMART_REACH(RiskModel):
    """
    SMART-REACH Model for recurrent cardiovascular events.
    
    Applicable population: Patients with established CAD, Cerebrovascular disease, or PAD.
    Age Range: 45 - 90 (due to baseline survival table limitations).
    """

    model_name = "SMART-REACH"
    model_version = "2018"
    supported_regions = ["western_europe", "north_america"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validates patient data against model constraints.
        """
        super().validate_input(patient)
        
        # Table S2 ranges from 45 to 89
        if patient.age < 45:
            logger.warning(f"Age {patient.age} is below the validation range (45). Model may be inaccurate.")
        if patient.age >= 90:
            raise ValueError("SMART-REACH model does not support age >= 90 due to lack of baseline survival data.")

        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")
            
        # Ensure necessary medical history is present
        required_attrs = [
            "systolic_bp", "total_cholesterol", "creatinine", 
            "has_atrial_fibrillation", "has_heart_failure", 
            "disease_locations_count"
        ]
        
        # Check for missing custom attributes often not in standard PatientData
        # Assuming PatientData is extensible or these are passed in a dict
        if not hasattr(patient, "creatinine"):
             raise ValueError("SMART-REACH requires 'creatinine' (µmol/L).")
        if not hasattr(patient, "disease_locations_count"):
             raise ValueError("SMART-REACH requires 'disease_locations_count' (1, 2, or 3).")

    def _get_linear_predictors(
        self, 
        patient: PatientData, 
        current_sim_age: float
    ) -> Tuple[float, float]:
        """
        Calculates Linear Predictors for CV Events (A) and Non-CV Death (B).
        
        Note: Model B (Non-CV) contains an Age x Smoking interaction term 
        that changes as the simulation advances in years.
        """
        
        # 1. Prepare Variables
        is_male = 1.0 if patient.sex.lower() == "male" else 0.0
        is_smoker = float(patient.smoking)
        has_diabetes = float(patient.diabetes)
        has_afib = float(getattr(patient, "has_atrial_fibrillation", False))
        has_chf = float(getattr(patient, "has_heart_failure", False))
        
        # Disease Locations (Reference is 1 location)
        # S3: "if two locations... add X, if three locations... add Y"
        loc_count = getattr(patient, "disease_locations_count", 1)
        is_loc2 = 1.0 if loc_count == 2 else 0.0
        is_loc3 = 1.0 if loc_count >= 3 else 0.0

        # Continuous variables with scaling per Table S3
        sbp_10 = patient.systolic_bp / 10.0
        chol = patient.total_cholesterol # mmol/L
        creat_10 = patient.creatinine / 10.0 # µmol/L per 10

        # 2. Calculate Linear Predictor A (Cardiovascular Events)
        lp_a = (
            _COEFFS_CV["male"] * is_male +
            _COEFFS_CV["current_smoker"] * is_smoker +
            _COEFFS_CV["diabetes"] * has_diabetes +
            _COEFFS_CV["sbp_per_10"] * sbp_10 +
            _COEFFS_CV["sbp_per_10_sq"] * (sbp_10 ** 2) +
            _COEFFS_CV["chol"] * chol +
            _COEFFS_CV["chol_sq"] * (chol ** 2) +
            _COEFFS_CV["creat_per_10"] * creat_10 +
            _COEFFS_CV["loc_2"] * is_loc2 +
            _COEFFS_CV["loc_3"] * is_loc3 +
            _COEFFS_CV["afib"] * has_afib +
            _COEFFS_CV["chf"] * has_chf
        )

        # 3. Calculate Linear Predictor B (Non-Cardiovascular Death)
        # Note the age interaction term uses the CURRENT simulation age
        lp_b = (
            _COEFFS_NON_CV["male"] * is_male +
            _COEFFS_NON_CV["current_smoker"] * is_smoker +
            _COEFFS_NON_CV["age_interaction_smoker"] * (current_sim_age * is_smoker) +
            _COEFFS_NON_CV["diabetes"] * has_diabetes +
            _COEFFS_NON_CV["sbp_per_10"] * sbp_10 +
            _COEFFS_NON_CV["chol"] * chol +
            _COEFFS_NON_CV["creat_per_10"] * creat_10 +
            _COEFFS_NON_CV["creat_per_10_sq"] * (creat_10 ** 2) +
            _COEFFS_NON_CV["loc_2"] * is_loc2 +
            _COEFFS_NON_CV["loc_3"] * is_loc3 +
            _COEFFS_NON_CV["afib"] * has_afib +
            _COEFFS_NON_CV["chf"] * has_chf
        )

        # 4. Apply Geographic Recalibration
        offsets = _REGIONAL_OFFSETS[patient.region]
        lp_a += offsets["A"]
        lp_b += offsets["B"]

        return lp_a, lp_b

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculates 10-year risk of recurrent cardiovascular events.
        """
        self.validate_input(patient)

        cumulative_survival_prob = 1.0
        cumulative_cv_risk = 0.0
        
        current_age = int(patient.age)
        max_simulation_years = 10

        # Iterate year by year to accumulate risk (Competing Risk integration)
        for year in range(max_simulation_years):
            
            # Stop if age exceeds table limits (89)
            if current_age not in _SMART_REACH_BASELINE:
                break
                
            # Get Baseline Survivals for this specific age step
            s0_cv, s0_ncv = _SMART_REACH_BASELINE[current_age]
            
            # Get Linear Predictors (LP_B depends on current_age)
            lp_a, lp_b = self._get_linear_predictors(patient, float(current_age))
            
            # Calculate 1-year Cause-Specific Hazards
            # Formula: 1 - S0^exp(LP)
            prob_cv_event_this_year = 1.0 - (s0_cv ** np.exp(lp_a))
            prob_ncv_death_this_year = 1.0 - (s0_ncv ** np.exp(lp_b))
            
            # Competing Risk Logic:
            # The absolute risk added this year is the probability of having survived 
            # up to the start of this year, multiplied by the hazard of CV event this year.
            absolute_risk_increment = cumulative_survival_prob * prob_cv_event_this_year
            cumulative_cv_risk += absolute_risk_increment
            
            # Update cumulative survival (surviving both CV event and Non-CV death)
            # Prob of surviving this year = 1 - (Hazard_CV + Hazard_NonCV)
            # Note: This is an approximation for discrete time steps.
            prob_survive_year = 1.0 - prob_cv_event_this_year - prob_ncv_death_this_year
            
            # Clamp to 0 to avoid negative probability issues in extreme high risk
            prob_survive_year = max(0.0, prob_survive_year)
            
            cumulative_survival_prob *= prob_survive_year
            
            current_age += 1

        risk_percent = cumulative_cv_risk * 100.0

        # Categorization (Arbitrary based on clinical usage for secondary prevention)
        # SMART-REACH doesn't define strict categories in the paper like SCORE2,
        # but >20% or >30% is often considered very high in secondary prevention.
        if risk_percent < 10:
            cat = "low" # Relative to this high-risk population
        elif risk_percent < 20:
            cat = "moderate"
        elif risk_percent < 30:
            cat = "high"
        else:
            cat = "very_high"

        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=cat,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "10_year_survival_free_of_events": cumulative_survival_prob
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for SMART-REACH.
        Requires iteration over 10 years of prediction time due to age-varying baseline.
        """
        required = [
            "age", "sex", "smoking", "diabetes", "systolic_bp", 
            "total_cholesterol", "creatinine", "disease_locations_count",
            "has_atrial_fibrillation", "has_heart_failure", "region"
        ]
        
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        n = len(results)
        
        # Initialize Accumulators
        results["cumulative_survival"] = 1.0
        results["cumulative_cv_risk"] = 0.0
        results["current_sim_age"] = results["age"].astype(int)
        
        # Pre-calculate static parts of Linear Predictors (optimization)
        # Convert columns to numpy for speed
        sex_male = (results["sex"].str.lower() == "male").astype(float).values
        smoking = results["smoking"].astype(float).values
        diabetes = results["diabetes"].astype(float).values
        sbp_10 = (results["systolic_bp"] / 10.0).values
        chol = results["total_cholesterol"].values
        creat_10 = (results["creatinine"] / 10.0).values
        afib = results["has_atrial_fibrillation"].astype(float).values
        chf = results["has_heart_failure"].astype(float).values
        
        # Location Dummies
        loc2 = (results["disease_locations_count"] == 2).astype(float).values
        loc3 = (results["disease_locations_count"] >= 3).astype(float).values

        # Regional Offsets
        offset_a = np.zeros(n)
        offset_b = np.zeros(n)
        for region, vals in _REGIONAL_OFFSETS.items():
            mask = results["region"] == region
            offset_a[mask] = vals["A"]
            offset_b[mask] = vals["B"]

        # Calculate Static components of LP A (CV) and LP B (Non-CV)
        # Note: LP B has an age interaction added inside the loop
        
        # LP A Static
        lp_a_base = (
            _COEFFS_CV["male"] * sex_male +
            _COEFFS_CV["current_smoker"] * smoking +
            _COEFFS_CV["diabetes"] * diabetes +
            _COEFFS_CV["sbp_per_10"] * sbp_10 +
            _COEFFS_CV["sbp_per_10_sq"] * (sbp_10**2) +
            _COEFFS_CV["chol"] * chol +
            _COEFFS_CV["chol_sq"] * (chol**2) +
            _COEFFS_CV["creat_per_10"] * creat_10 +
            _COEFFS_CV["loc_2"] * loc2 +
            _COEFFS_CV["loc_3"] * loc3 +
            _COEFFS_CV["afib"] * afib +
            _COEFFS_CV["chf"] * chf +
            offset_a
        )

        # LP B Static (excluding age*smoking)
        lp_b_static = (
            _COEFFS_NON_CV["male"] * sex_male +
            _COEFFS_NON_CV["current_smoker"] * smoking +
            _COEFFS_NON_CV["diabetes"] * diabetes +
            _COEFFS_NON_CV["sbp_per_10"] * sbp_10 +
            _COEFFS_NON_CV["chol"] * chol +
            _COEFFS_NON_CV["creat_per_10"] * creat_10 +
            _COEFFS_NON_CV["creat_per_10_sq"] * (creat_10**2) +
            _COEFFS_NON_CV["loc_2"] * loc2 +
            _COEFFS_NON_CV["loc_3"] * loc3 +
            _COEFFS_NON_CV["afib"] * afib +
            _COEFFS_NON_CV["chf"] * chf +
            offset_b
        )

        # ---------------------------------------------------------------------
        # 10-YEAR SIMULATION LOOP
        # ---------------------------------------------------------------------
        for year in range(10):
            # Create a vector of baseline survivals for the current age of each patient
            # Map current_sim_age to S0 values. 
            # If age > 89, fill with nan or handle. Here we assume constant high risk or stop.
            
            # Using map/reindex for lookup
            current_ages = results["current_sim_age"].values
            
            # Fast Lookup using numpy array indexing if Age is index, 
            # but dict mapping is safer for robustness
            s0_cv = np.array([_SMART_REACH_BASELINE.get(a, (1.0, 1.0))[0] for a in current_ages])
            s0_ncv = np.array([_SMART_REACH_BASELINE.get(a, (1.0, 1.0))[1] for a in current_ages])
            
            # Handle out of bounds (age >= 90) by making s0 = 1 (no risk calculated) 
            # or handling gracefully. 
            # In this implementation, if age missing, risk added is 0.
            
            # Update LP B with current age interaction
            lp_b_dynamic = lp_b_static + (
                _COEFFS_NON_CV["age_interaction_smoker"] * (current_ages * smoking)
            )
            
            # Calculate Hazards
            risk_cv_year = 1.0 - np.power(s0_cv, np.exp(lp_a_base))
            risk_ncv_year = 1.0 - np.power(s0_ncv, np.exp(lp_b_dynamic))
            
            # Accumulate
            # Increment = Prob_Alive_Start_Of_Year * Prob_CV_Event_This_Year
            increment = results["cumulative_survival"].values * risk_cv_year
            results["cumulative_cv_risk"] += increment
            
            # Update Survival
            surv_year = 1.0 - risk_cv_year - risk_ncv_year
            surv_year = np.maximum(0.0, surv_year) # Clamp
            
            results["cumulative_survival"] *= surv_year
            results["current_sim_age"] += 1

        # Final Formatting
        results["risk_score"] = np.clip(results["cumulative_cv_risk"] * 100.0, 0.0, 100.0)
        results["model_name"] = self.model_name
        
        # Categorization
        conditions = [
            results["risk_score"] < 10,
            (results["risk_score"] >= 10) & (results["risk_score"] < 20),
            (results["risk_score"] >= 20) & (results["risk_score"] < 30),
            results["risk_score"] >= 30
        ]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        
        # Cleanup temp columns
        results = results.drop(columns=["cumulative_survival", "cumulative_cv_risk", "current_sim_age"])
        
        return results