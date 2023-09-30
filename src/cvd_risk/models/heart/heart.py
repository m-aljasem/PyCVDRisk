"""
HEART (History, ECG, Age, Risk factors and Troponin) risk score.

The HEART score is a bedside clinical prediction tool used to stratify risk
in patients presenting to the emergency department with chest pain suggestive
of acute coronary syndrome (ACS).

Reference:
    Backus BE, Six AJ, Kelder JC, et al. A prospective validation of the HEART
    score for chest pain patients at the emergency department. Int J Cardiol.
    2013;168(3):2153-2158.
    DOI: 10.1016/j.ijcard.2013.01.255

    Six AJ, Cullen L, Backus BE, et al. The HEART score for the assessment of
    patients with chest pain in the emergency department: a multinational
    validation study. Crit Pathw Cardiol. 2013;12(3):121-126.
    DOI: 10.1097/HPC.0b013e31828b327e
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# HEART risk score categories
_HEART_CATEGORIES = {
    "low": "Low risk",
    "moderate": "Moderate risk",
    "high": "High risk"
}

# HEART score thresholds
_HEART_THRESHOLDS = {
    "low": (0, 3),      # 0-3 points: Low risk
    "moderate": (4, 6), # 4-6 points: Moderate risk
    "high": (7, 10)     # 7-10 points: High risk
}


class HEART(RiskModel):
    """
    HEART (History, ECG, Age, Risk factors, Troponin) risk score.

    The HEART score ranges from 0-10 points and stratifies patients with chest
    pain into three risk categories for major adverse cardiac events (MACE).

    Components:
    - History: 0-2 points (typical symptoms)
    - ECG: 0-2 points (abnormalities)
    - Age: 0-2 points (age groups)
    - Risk factors: 0-2 points (traditional CVD risk factors)
    - Troponin: 0-2 points (elevated levels)

    Valid for patients presenting with chest pain suggestive of ACS.
    """

    model_name = "HEART"
    model_version = "2013"
    supported_regions = None  # HEART is not region-specific

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient data for HEART score calculation."""
        super().validate_input(patient)

        # Check required HEART-specific fields
        required_fields = [
            "typical_symptoms_num", "ecg_normal", "abn_repolarisation",
            "ecg_st_depression", "presentation_hstni"
        ]

        missing_fields = []
        for field in required_fields:
            if getattr(patient, field) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required HEART fields: {missing_fields}")

        # Additional validation for HEART-specific logic
        if patient.age < 18:
            raise ValueError("HEART score is validated for adults ≥18 years")

    def _calculate_history_score(self, typical_symptoms_num: int) -> int:
        """
        Calculate History component score.

        History -
        Absence of history for coronary ischemia: nonspecific = 0
        Nonspecific + suspicious elements: moderately suspicious = 1
        Mainly suspicious elements (middle- or left-sided, heavy chest pain,
        radiation, and/or relief of symptoms by sublingual nitrates): = 2

        Parameters
        ----------
        typical_symptoms_num : int
            Number of typical symptoms (0-6)

        Returns
        -------
        int
            History score (0-2)
        """
        if typical_symptoms_num <= 1:
            return 0  # 0-1: nonspecific
        elif 2 <= typical_symptoms_num <= 4:
            return 1  # 2-4: moderately suspicious
        else:  # 5-6
            return 2  # mainly suspicious

    def _calculate_ecg_score(self, ecg_normal: bool, abn_repolarisation: bool,
                           ecg_st_depression: bool) -> int:
        """
        Calculate ECG component score.

        ECG -
        Normal ECG according to Minnesota criteria = 0
        Repolarization abnormalities without significant ST-segment
        depression or elevation = 1
        Significant ST-segment depressions or elevations in absence of
        bundle branch block, left ventricular hypertrophy, or digoxin use = 2

        Parameters
        ----------
        ecg_normal : bool
            Whether ECG is normal
        abn_repolarisation : bool
            Abnormal repolarization present
        ecg_st_depression : bool
            ST segment depression present

        Returns
        -------
        int
            ECG score (0-2)
        """
        if ecg_normal:
            return 0  # Normal ECG
        elif abn_repolarisation and not ecg_st_depression:
            return 1  # Abnormal repolarization only
        elif ecg_st_depression:
            return 2  # Significant ST depression/elevation
        else:
            return 1  # Any other abnormality counts as abnormal repolarization

    def _calculate_age_score(self, age: int) -> int:
        """
        Calculate Age component score.

        Age -
        Younger than 45 = 0
        45 to 64 years old = 1
        65 years or older = 2

        Parameters
        ----------
        age : int
            Patient age in years

        Returns
        -------
        int
            Age score (0-2)
        """
        if age < 45:
            return 0
        elif 45 <= age <= 64:
            return 1
        else:  # 65+
            return 2

    def _calculate_risk_factors_score(self, patient: PatientData) -> int:
        """
        Calculate Risk factors component score.

        Risk factors (count ≥1 for each):
        - Currently treated diabetes mellitus
        - Current or recent (<90 days) smoker
        - Diagnosed and/or treated hypertension
        - Diagnosed hypercholesterolemia
        - Family history of coronary artery disease
        - Obesity (BMI >30) - not directly available, using atherosclerotic_disease as proxy
        - History of significant atherosclerosis (CAD, MI, stroke, PAD, etc.)

        No risk factors = 0
        1-2 risk factors = 1
        ≥3 risk factors OR known atherosclerotic disease = 2

        Parameters
        ----------
        patient : PatientData
            Patient data

        Returns
        -------
        int
            Risk factors score (0-2)
        """
        risk_count = 0

        # Count individual risk factors
        if patient.diabetes:
            risk_count += 1
        if patient.smoking:
            risk_count += 1
        if patient.hypertension:
            risk_count += 1
        if patient.hyperlipidaemia:
            risk_count += 1
        if patient.family_history:
            risk_count += 1
        # Note: BMI >30 not directly available in current data structure
        # Atherosclerotic disease counts separately

        # Apply scoring logic
        if patient.atherosclerotic_disease:
            return 2  # Known atherosclerotic disease = 2 points
        elif risk_count == 0:
            return 0  # No risk factors
        elif risk_count <= 2:
            return 1  # 1-2 risk factors
        else:  # risk_count >= 3
            return 2  # 3+ risk factors

    def _calculate_troponin_score(self, presentation_hstni: float, sex: str) -> int:
        """
        Calculate Troponin component score.

        Troponin T or I -
        Below the threshold for positivity = 0
        Between 1 and 3 times the threshold for positivity = 1
        Higher than 3 times the threshold for positivity = 2

        Thresholds (99th percentile URL):
        - Male: 34 ng/L
        - Female: 16 ng/L

        Parameters
        ----------
        presentation_hstni : float
            High-sensitivity troponin I level (ng/L)
        sex : str
            Patient sex ("male" or "female")

        Returns
        -------
        int
            Troponin score (0-2)
        """
        if sex.lower() == "male":
            url = 34.0  # ng/L
        else:  # female
            url = 16.0  # ng/L

        if presentation_hstni < url:
            return 0  # Below threshold
        elif url <= presentation_hstni < (3 * url):
            return 1  # 1-3x threshold
        else:  # 3x threshold or higher
            return 2

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate HEART score for a single patient.

        Parameters
        ----------
        patient : PatientData
            Patient data with required HEART fields

        Returns
        -------
        RiskResult
            HEART score and risk category
        """
        self.validate_input(patient)

        # Calculate component scores
        history_score = self._calculate_history_score(patient.typical_symptoms_num)
        ecg_score = self._calculate_ecg_score(
            patient.ecg_normal, patient.abn_repolarisation, patient.ecg_st_depression
        )
        age_score = self._calculate_age_score(patient.age)
        risk_score = self._calculate_risk_factors_score(patient)
        troponin_score = self._calculate_troponin_score(
            patient.presentation_hstni, patient.sex
        )

        # Total HEART score
        total_score = history_score + ecg_score + age_score + risk_score + troponin_score

        # Determine risk category
        if total_score <= 3:
            category = "low"
        elif total_score <= 6:
            category = "moderate"
        else:
            category = "high"

        return RiskResult(
            risk_score=float(total_score),
            risk_category=category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "history_score": history_score,
                "ecg_score": ecg_score,
                "age_score": age_score,
                "risk_factors_score": risk_score,
                "troponin_score": troponin_score,
                "score_range": (0, 10)
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate HEART scores for a batch of patients (vectorized).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with patient data. Must include HEART-specific columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with original columns plus HEART score results.
        """
        required_columns = [
            "age", "sex", "typical_symptoms_num", "ecg_normal",
            "abn_repolarisation", "ecg_st_depression", "diabetes",
            "smoking", "hypertension", "hyperlipidaemia",
            "family_history", "atherosclerotic_disease", "presentation_hstni"
        ]

        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        results = df.copy()
        results["heart_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Vectorized component calculations
        # History score
        conditions_hist = [
            results["typical_symptoms_num"] <= 1,
            (results["typical_symptoms_num"] >= 2) & (results["typical_symptoms_num"] <= 4),
            results["typical_symptoms_num"] >= 5
        ]
        choices_hist = [0, 1, 2]
        results["history_score"] = np.select(conditions_hist, choices_hist, default=0)

        # ECG score - this is more complex, need to handle logic
        results["ecg_score"] = 0
        # Normal ECG = 0
        normal_mask = results["ecg_normal"] == True
        results.loc[normal_mask, "ecg_score"] = 0
        # ST depression = 2
        st_dep_mask = results["ecg_st_depression"] == True
        results.loc[st_dep_mask, "ecg_score"] = 2
        # Abnormal repolarization without ST depression = 1
        abn_rep_mask = (results["abn_repolarisation"] == True) & (results["ecg_st_depression"] == False)
        results.loc[abn_rep_mask, "ecg_score"] = 1

        # Age score
        conditions_age = [
            results["age"] < 45,
            (results["age"] >= 45) & (results["age"] <= 64),
            results["age"] >= 65
        ]
        choices_age = [0, 1, 2]
        results["age_score"] = np.select(conditions_age, choices_age, default=0)

        # Risk factors score
        # Count risk factors
        risk_factors = ["diabetes", "smoking", "hypertension", "hyperlipidaemia", "family_history"]
        risk_count = results[risk_factors].sum(axis=1)

        # Apply scoring logic
        results["risk_factors_score"] = 0
        # Known atherosclerotic disease = 2
        atherosclerotic_mask = results["atherosclerotic_disease"] == True
        results.loc[atherosclerotic_mask, "risk_factors_score"] = 2
        # No risk factors = 0 (already set)
        # 1-2 risk factors = 1
        risk_1_2_mask = (risk_count >= 1) & (risk_count <= 2) & (~atherosclerotic_mask)
        results.loc[risk_1_2_mask, "risk_factors_score"] = 1
        # 3+ risk factors = 2
        risk_3_plus_mask = (risk_count >= 3) & (~atherosclerotic_mask)
        results.loc[risk_3_plus_mask, "risk_factors_score"] = 2

        # Troponin score
        results["troponin_score"] = 0
        # Get URL based on sex
        url = np.where(results["sex"].str.lower() == "male", 34.0, 16.0)

        # Below threshold = 0 (already set)
        # 1-3x threshold = 1
        troponin_1_3x = (results["presentation_hstni"] >= url) & (results["presentation_hstni"] < 3 * url)
        results.loc[troponin_1_3x, "troponin_score"] = 1
        # 3x+ threshold = 2
        troponin_3x_plus = results["presentation_hstni"] >= 3 * url
        results.loc[troponin_3x_plus, "troponin_score"] = 2

        # Total HEART score
        component_cols = ["history_score", "ecg_score", "age_score", "risk_factors_score", "troponin_score"]
        results["heart_score"] = results[component_cols].sum(axis=1)

        # Risk categories
        conditions_cat = [
            results["heart_score"] <= 3,
            (results["heart_score"] >= 4) & (results["heart_score"] <= 6),
            results["heart_score"] >= 7
        ]
        choices_cat = ["low", "moderate", "high"]
        results["risk_category"] = np.select(conditions_cat, choices_cat, default="unknown")

        return results
