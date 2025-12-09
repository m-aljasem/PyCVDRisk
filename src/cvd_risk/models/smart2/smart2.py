"""
SMART2 (Secondary Manifestations of ARTerial disease 2) risk model.

SMART2 is the updated risk prediction model for estimating 10-year risk of 
recurrent atherosclerotic cardiovascular disease (ASCVD) events (myocardial 
infarction, stroke, or vascular death) in patients with established ASCVD.

Reference:
    Hageman, S. H., et al. (2022). Estimation of recurrent atherosclerotic 
    cardiovascular event risk in patients with established cardiovascular disease: 
    the updated SMART2 algorithm. European Heart Journal, 43(18), 1715-1727.
    DOI: 10.1093/eurheartj/ehac056
"""

import logging
from typing import Literal, Union, Dict, Any

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# MODEL CONSTANTS (Source: Supplementary Table 2 & 3)
# -------------------------------------------------------------------------

# The 10-year baseline risk (S0(10)) and Mean Linear Predictor
_SMART2_BASELINE_RISK_10YR = 0.165822797
_SMART2_MEAN_LINEAR_PREDICTOR = -0.0463729

# Beta Coefficients (Supplementary Table 2)
_SMART2_COEFFICIENTS = {
    "age": -0.03496022,
    "age_sq": 0.000551072,
    "male_sex": 0.287658743,
    "current_smoking": 0.345583271,
    "diabetes": 0.318170659,
    "systolic_bp": 0.018913154,
    "log_non_hdl": 0.540364249,
    "log_hscrp": 0.151760173,
    "egfr": -0.03967521,
    "egfr_sq": 0.000218613,
    "years_since_dx": 0.047699585,
    "years_since_dx_sq": -0.00164973,
    # Disease History Indicators
    "hist_cad": 0.294701954,        # Coronary Artery Disease
    "hist_cevd": 0.34831786,        # Cerebrovascular Disease
    "hist_pad": 0.22446658,         # Peripheral Artery Disease
    "hist_aaa": 0.330356631,        # Abdominal Aortic Aneurysm
    # Treatment
    "antithrombotics": -0.21072103, # Aspirin or equivalent
}

# Regional Calibration Ratios (Expected/Observed) - Supplementary Table 2 & 3
_SMART2_REGIONAL_RATIOS = {
    "europe_low": 0.81590,         # e.g., UK, Netherlands (Low Risk)
    "europe_moderate": 0.6973285,  # e.g., Sweden (Moderate Risk)
    "europe_high": 0.5085825,      # e.g., Estonia, Poland (High Risk)
    "europe_very_high": 0.2285371, # e.g., Russia, Ukraine (Very High Risk)
    "north_america": 0.4714961,
    "latin_america": 0.5075670,
    "asia": 0.4043255,             # Excluding Japan
    "japan": 0.8825678,
    "australia": 1.0040808,
}


class SMART2(RiskModel):
    """
    SMART2 Risk Prediction Model (2022 Update).
    
    Target Population: Patients aged 40-80 with established ASCVD.
    Outcome: 10-year risk of fatal or non-fatal MI, stroke, or vascular death.
    """

    model_name = "SMART2"
    model_version = "2022"
    supported_regions = list(_SMART2_REGIONAL_RATIOS.keys())

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validates input data against SMART2 derivation ranges.
        Note: PatientData needs to be extended or dynamically accessed for 
        specific SMART2 history fields (hsCRP, years_since_dx, etc).
        """
        super().validate_input(patient)
        
        if patient.age < 40 or patient.age > 80:
            logger.warning(
                f"Age {patient.age} is outside the derivation range [40, 80]. "
                "SMART2 accuracy may be reduced."
            )
        
        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(
                f"Region '{patient.region}' not supported. "
                f"Must be one of: {self.supported_regions}"
            )
        
        # Validation for required extra attributes
        required_attrs = [
            "hs_crp", "egfr", "years_since_diagnosis", 
            "has_cad", "has_cevd", "has_pad", "has_aaa"
        ]
        missing = [attr for attr in required_attrs if not hasattr(patient, attr)]
        if missing:
            raise AttributeError(f"PatientData missing required SMART2 attributes: {missing}")

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        # 1. Setup Variables & Transformations
        # Note: Non-HDL is Total Chol - HDL
        non_hdl = patient.total_cholesterol - patient.hdl_cholesterol
        
        # Log transformations (using max(value, epsilon) to prevent log(0))
        log_nhdl = np.log(max(non_hdl, 0.1))
        log_hscrp = np.log(max(patient.hs_crp, 0.1))
        
        # Squared transformations
        age_sq = patient.age ** 2
        egfr_sq = patient.egfr ** 2
        years_sq = patient.years_since_diagnosis ** 2

        # Boolean to Float
        is_male = float(patient.sex.lower() == "male")
        is_smoker = float(patient.smoking)
        has_diabetes = float(patient.diabetes) # Assuming attribute exists
        
        # Disease History (1 if present, 0 if not)
        # Note: Polyvascular patients add multiple coefficients
        h_cad = float(patient.has_cad)
        h_cevd = float(patient.has_cevd)
        h_pad = float(patient.has_pad)
        h_aaa = float(patient.has_aaa)
        
        # Treatment
        # Default to 0 if not specified, though usually ASCVD pts are on aspirin
        on_antithromb = float(getattr(patient, "on_antithrombotics", 0))

        # 2. Calculate Linear Predictor (LP)
        # Sum of (Beta * Value)
        lp = (
            _SMART2_COEFFICIENTS["age"] * patient.age +
            _SMART2_COEFFICIENTS["age_sq"] * age_sq +
            _SMART2_COEFFICIENTS["male_sex"] * is_male +
            _SMART2_COEFFICIENTS["current_smoking"] * is_smoker +
            _SMART2_COEFFICIENTS["diabetes"] * has_diabetes +
            _SMART2_COEFFICIENTS["systolic_bp"] * patient.systolic_bp +
            _SMART2_COEFFICIENTS["log_non_hdl"] * log_nhdl +
            _SMART2_COEFFICIENTS["log_hscrp"] * log_hscrp +
            _SMART2_COEFFICIENTS["egfr"] * patient.egfr +
            _SMART2_COEFFICIENTS["egfr_sq"] * egfr_sq +
            _SMART2_COEFFICIENTS["years_since_dx"] * patient.years_since_diagnosis +
            _SMART2_COEFFICIENTS["years_since_dx_sq"] * years_sq +
            _SMART2_COEFFICIENTS["hist_cad"] * h_cad +
            _SMART2_COEFFICIENTS["hist_cevd"] * h_cevd +
            _SMART2_COEFFICIENTS["hist_pad"] * h_pad +
            _SMART2_COEFFICIENTS["hist_aaa"] * h_aaa +
            _SMART2_COEFFICIENTS["antithrombotics"] * on_antithromb
        )

        # 3. Apply Recalibration
        # Formula: Risk = 1 - (1 - Baseline)^exp(LP - LP_mean - ln(EO_ratio))
        
        eo_ratio = _SMART2_REGIONAL_RATIOS[patient.region]
        
        # The exponent represents the hazard ratio relative to the derivation population mean
        exponent_term = np.exp(
            lp - _SMART2_MEAN_LINEAR_PREDICTOR - np.log(eo_ratio)
        )
        
        # 4. Final Risk Calculation
        # (1 - Baseline) is the baseline survival
        risk_prob = 1.0 - (1.0 - _SMART2_BASELINE_RISK_10YR) ** exponent_term
        
        risk_percent = np.clip(risk_prob * 100.0, 0.0, 100.0)

        # 5. Categorization
        # Since all patients have established ASCVD, they are clinically "Very High Risk".
        # However, for residual risk stratification (Clinical Utility section of paper):
        # <10% is often considered "Lower residual risk"
        # 10-20% "Moderate residual risk"
        # >20% "High residual risk"
        # >30% "Very high residual risk"
        if risk_percent < 10.0:
            cat = "lower_residual"
        elif risk_percent < 20.0:
            cat = "moderate_residual"
        elif risk_percent < 30.0:
            cat = "high_residual"
        else:
            cat = "very_high_residual"

        return RiskResult(
            risk_score=float(risk_percent),
            risk_category=cat,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "linear_predictor": lp,
                "region_ratio": eo_ratio
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        """
        required = [
            "age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", 
            "hs_crp", "egfr", "smoking", "diabetes", "region",
            "years_since_diagnosis", "has_cad", "has_cevd", "has_pad", "has_aaa"
        ]
        
        # Check columns (ignoring case for robustness if needed, but strict here)
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        results = df.copy()
        
        # Handle optional antithrombotics column (assume 0 if missing)
        if "on_antithrombotics" not in results.columns:
            results["on_antithrombotics"] = 0.0

        # --- Vectorized Transformations ---
        
        # Non-HDL
        non_hdl = results["total_cholesterol"] - results["hdl_cholesterol"]
        # Log Transforms (clip to epsilon to avoid log(0))
        log_nhdl = np.log(np.maximum(non_hdl, 0.1))
        log_hscrp = np.log(np.maximum(results["hs_crp"], 0.1))
        
        # Squares
        age_sq = np.square(results["age"])
        egfr_sq = np.square(results["egfr"])
        years_sq = np.square(results["years_since_diagnosis"])
        
        # Booleans/Indicators
        is_male = (results["sex"].str.lower() == "male").astype(float)
        is_smoker = results["smoking"].astype(float)
        has_db = results["diabetes"].astype(float)
        
        # Map Regions to Ratios
        # Use map, fill NaN with 1.0 (or handle error) to prevent crash on unknown region
        eo_ratios = results["region"].map(_SMART2_REGIONAL_RATIOS)
        if eo_ratios.isna().any():
            invalid_regions = results.loc[eo_ratios.isna(), "region"].unique()
            raise ValueError(f"Invalid regions found in batch data: {invalid_regions}")

        # --- Linear Predictor Calculation ---
        
        lp = (
            _SMART2_COEFFICIENTS["age"] * results["age"] +
            _SMART2_COEFFICIENTS["age_sq"] * age_sq +
            _SMART2_COEFFICIENTS["male_sex"] * is_male +
            _SMART2_COEFFICIENTS["current_smoking"] * is_smoker +
            _SMART2_COEFFICIENTS["diabetes"] * has_db +
            _SMART2_COEFFICIENTS["systolic_bp"] * results["systolic_bp"] +
            _SMART2_COEFFICIENTS["log_non_hdl"] * log_nhdl +
            _SMART2_COEFFICIENTS["log_hscrp"] * log_hscrp +
            _SMART2_COEFFICIENTS["egfr"] * results["egfr"] +
            _SMART2_COEFFICIENTS["egfr_sq"] * egfr_sq +
            _SMART2_COEFFICIENTS["years_since_dx"] * results["years_since_diagnosis"] +
            _SMART2_COEFFICIENTS["years_since_dx_sq"] * years_sq +
            _SMART2_COEFFICIENTS["hist_cad"] * results["has_cad"].astype(float) +
            _SMART2_COEFFICIENTS["hist_cevd"] * results["has_cevd"].astype(float) +
            _SMART2_COEFFICIENTS["hist_pad"] * results["has_pad"].astype(float) +
            _SMART2_COEFFICIENTS["hist_aaa"] * results["has_aaa"].astype(float) +
            _SMART2_COEFFICIENTS["antithrombotics"] * results["on_antithrombotics"].astype(float)
        )

        # --- Apply Formula ---
        
        # Exponent = exp(LP_i - LP_mean - ln(EO_ratio))
        # Note: np.log of the ratio
        exponent_term = np.exp(
            lp - _SMART2_MEAN_LINEAR_PREDICTOR - np.log(eo_ratios)
        )
        
        # Risk = 1 - (1 - Baseline)^Exponent
        risk_prob = 1.0 - np.power((1.0 - _SMART2_BASELINE_RISK_10YR), exponent_term)
        
        # Assign Score
        results["risk_score"] = np.clip(risk_prob * 100.0, 0.0, 100.0)
        results["model_name"] = self.model_name
        
        # --- Vectorized Categorization ---
        conditions = [
            results["risk_score"] < 10.0,
            (results["risk_score"] >= 10.0) & (results["risk_score"] < 20.0),
            (results["risk_score"] >= 20.0) & (results["risk_score"] < 30.0),
            results["risk_score"] >= 30.0
        ]
        choices = ["lower_residual", "moderate_residual", "high_residual", "very_high_residual"]
        
        results["risk_category"] = np.select(conditions, choices, default="unknown")
        
        return results