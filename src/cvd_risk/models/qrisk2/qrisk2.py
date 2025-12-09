"""
QRISK2 (QRISK2-2014) cardiovascular risk model.

QRISK2 is a cardiovascular disease risk prediction algorithm developed
by ClinRisk Ltd. for the UK population. It predicts the 10-year risk of
cardiovascular disease (CVD) including coronary heart disease and stroke.

The algorithm was developed using data from the QResearch database and
includes additional risk factors compared to earlier models.

Reference:
    Hippisley-Cox J, Coupland C, Vinogradova Y, et al. Predicting cardiovascular
    risk in England and Wales: prospective derivation and validation of QRISK2.
    BMJ. 2008;336(7659):1475-1482. doi:10.1136/bmj.39609.449676.25

Additional terms from ClinRisk Ltd.:
    The initial version of this file, to be found at
    http://svn.clinrisk.co.uk/opensource/qrisk2, faithfully implements QRISK2-2014.
    ClinRisk Ltd. have released this code under the GNU Lesser General Public
    License to enable others to implement the algorithm faithfully.
    However, the nature of the GNU Lesser General Public License is such that
    we cannot prevent, for example, someone accidentally altering the coefficients,
    getting the inputs wrong, or just poor programming.
    ClinRisk Ltd. stress, therefore, that it is the responsibility of the end user
    to check that the source that they receive produces the same results as the
    original code posted at http://svn.clinrisk.co.uk/opensource/qrisk2.
    Inaccurate implementations of risk scores can lead to wrong patients being
    given the wrong treatment.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# QRISK2 baseline survival probabilities at 10 years
_QRISK2_BASELINE_SURVIVAL = {
    "male": 0.977699398994446,
    "female": 0.988948762416840,
}

# QRISK2 ethnicity risk scores (ethrisk mapping)
_QRISK2_ETHRISK_MAPPING = {
    "white": 1,
    "south_asian": 3,
    "black": 4,
    "chinese": 5,
    "mixed": 2,
    "other": 2,
}

# QRISK2 smoking categories (smoke_cat mapping)
# 0 = non-smoker, 1 = ex-smoker, 2 = light smoker (<10/day), 3 = moderate (10-19/day), 4 = heavy (20+/day)
_QRISK2_SMOKING_MAPPING = {
    0: 0,  # non-smoker -> 0
    1: 1,  # ex-smoker -> 1
    2: 2,  # light smoker -> 2
    3: 3,  # moderate smoker -> 3
    4: 4,  # heavy smoker -> 4
}

# QRISK2 coefficients for females
_QRISK2_FEMALE_COEFFICIENTS = {
    "Iethrisk": [
        0.0,
        0.0,
        0.2671958047902151500000000,
        0.7147534261793343500000000,
        0.3702894474455115700000000,
        0.2073797362620235500000000,
        -0.1744149722741736900000000,
        -0.3271878654368842200000000,
        -0.2200617876129250500000000,
        -0.2090388032466696800000000
    ],
    "Ismoke": [
        0.0,
        0.1947480856528854800000000,
        0.6229400520450627500000000,
        0.7405819891143352600000000,
        0.9134392684576959600000000
    ],
    "age_1": 3.8734583855051343000000000,
    "age_2": 0.1346634304478384600000000,
    "bmi_1": -0.1557872403333062600000000,
    "bmi_2": -3.7727795566691125000000000,
    "rati": 0.1525695208919679600000000,
    "sbp": 0.0132165300119653560000000,
    "town": 0.0643647529864017080000000,
    "b_AF": 1.4235421148946676000000000,
    "b_ra": 0.3021462511553648100000000,
    "b_renal": 0.8614743039721416400000000,
    "b_treatedhyp": 0.5889355458733703800000000,
    "b_type1": 1.6684783657502795000000000,
    "b_type2": 1.1350165062510138000000000,
    "fh_cvd": 0.5133972775738673300000000,
    # Interaction terms
    "age_1_smoke_1": 0.6891139747579299000000000,
    "age_1_smoke_2": 0.6942632802121626600000000,
    "age_1_smoke_3": -1.6952388644218186000000000,
    "age_1_smoke_4": -1.2150150940219255000000000,
    "age_1_b_AF": -3.5855215448190969000000000,
    "age_1_b_renal": -3.0766647922469192000000000,
    "age_1_b_treatedhyp": -4.0295302811880314000000000,
    "age_1_b_type1": -0.3344110567405778600000000,
    "age_1_b_type2": -3.3144806806620530000000000,
    "age_1_bmi_1": -5.5933905797230006000000000,
    "age_1_bmi_2": 64.3635572837688980000000000,
    "age_1_fh_cvd": 0.8605433761217157200000000,
    "age_1_sbp": -0.0509321154551188590000000,
    "age_1_town": 0.1518664540724453700000000,
    "age_2_smoke_1": -0.1765395485882681500000000,
    "age_2_smoke_2": -0.2323836483278573000000000,
    "age_2_smoke_3": 0.2734395770551826300000000,
    "age_2_smoke_4": 0.1432552287454152700000000,
    "age_2_b_AF": 0.4986871390807032200000000,
    "age_2_b_renal": 0.4393033615664938600000000,
    "age_2_b_treatedhyp": 0.6904385790303250200000000,
    "age_2_b_type1": -0.1734316566060327700000000,
    "age_2_b_type2": 0.4864930655867949500000000,
    "age_2_bmi_1": 1.5223341309207974000000000,
    "age_2_bmi_2": -12.7413436207964070000000000,
    "age_2_fh_cvd": -0.2756708481415109900000000,
    "age_2_sbp": 0.0073790750039744186000000,
    "age_2_town": -0.0487465462679640900000000,
}

# QRISK2 coefficients for males
_QRISK2_MALE_COEFFICIENTS = {
    "Iethrisk": [
        0.0,
        0.0,
        0.3567133647493443400000000,
        0.5369559608176189800000000,
        0.5190878419529624300000000,
        0.2182992106490147000000000,
        -0.3474174705898491800000000,
        -0.3674730037922803700000000,
        -0.3749664891426142700000000,
        -0.1926947742531604500000000
    ],
    "Ismoke": [
        0.0,
        0.2784649664157046200000000,
        0.6067834395168959500000000,
        0.7103835060989258700000000,
        0.8626172339181202900000000
    ],
    "age_1": -17.6225543381945610000000000,
    "age_2": 0.0241873189298273640000000,
    "bmi_1": 1.7320282704272665000000000,
    "bmi_2": -7.2311754066699754000000000,
    "rati": 0.1751387974012235100000000,
    "sbp": 0.0101676305179196900000000,
    "town": 0.0298177271496720960000000,
    "b_AF": 0.9890997526189402300000000,
    "b_ra": 0.2541886209118611200000000,
    "b_renal": 0.7949789230438320000000000,
    "b_treatedhyp": 0.6229359479868044100000000,
    "b_type1": 1.3330353321463930000000000,
    "b_type2": 0.9372956828151940400000000,
    "fh_cvd": 0.5923353736582422900000000,
    # Interaction terms
    "age_1_smoke_1": 0.9243747443632776000000000,
    "age_1_smoke_2": 1.9597527500081284000000000,
    "age_1_smoke_3": 2.9993544847631153000000000,
    "age_1_smoke_4": 5.0370735254768100000000000,
    "age_1_b_AF": 8.2354205455482727000000000,
    "age_1_b_renal": -3.9747389951976779000000000,
    "age_1_b_treatedhyp": 7.8737743159167728000000000,
    "age_1_b_type1": 5.4238504414460937000000000,
    "age_1_b_type2": 5.062416180653014100000000000,
    "age_1_bmi_1": 33.5437525167394240000000000,
    "age_1_bmi_2": -129.9766738257203800000000000,
    "age_1_fh_cvd": 1.9279963874659789000000000,
    "age_1_sbp": 0.0523440892175620200000000,
    "age_1_town": -0.1730588074963540200000000,
    "age_2_smoke_1": -0.0034466074038854394000000,
    "age_2_smoke_2": -0.0050703431499952954000000,
    "age_2_smoke_3": 0.0003216059799916440800000,
    "age_2_smoke_4": 0.0031312537144240087000000,
    "age_2_b_AF": 0.0073291937255039966000000,
    "age_2_b_renal": -0.0261557073286531780000000,
    "age_2_b_treatedhyp": 0.0085556382622618121000000,
    "age_2_b_type1": 0.0020586479482670723000000,
    "age_2_b_type2": -0.0002328590770854172900000,
    "age_2_bmi_1": 0.0811847212080794990000000,
    "age_2_bmi_2": -0.2558919068850948300000000,
    "age_2_fh_cvd": -0.0056729073729663406000000,
    "age_2_sbp": -0.0000536584257307299330000,
    "age_2_town": -0.0010763305052605857000000,
}

# Centring values for continuous variables
_QRISK2_CENTRES = {
    "female": {
        "age_1": 2.099778413772583,
        "age_2": 4.409069538116455,
        "bmi_1": 0.154046609997749,
        "bmi_2": 0.144072100520134,
        "rati": 3.554229259490967,
        "sbp": 125.773628234863280,
        "town": 0.032508373260498,
    },
    "male": {
        "age_1": 0.232008963823318,
        "age_2": 18.577636718750000,
        "bmi_1": 0.146408438682556,
        "bmi_2": 0.140651300549507,
        "rati": 4.377167701721191,
        "sbp": 131.038314819335940,
        "town": 0.151332527399063,
    }
}


class QRISK2(RiskModel):
    """
    QRISK2 cardiovascular disease risk prediction model (ClinRisk 2014).

    QRISK2 predicts 10-year risk of CVD (coronary heart disease and stroke)
    for individuals aged 25-84 years. The model includes traditional risk
    factors plus additional variables like ethnicity, family history,
    BMI, rheumatoid arthritis, renal disease, and deprivation score.

    Valid for ages 25-84 years.
    """

    model_name = "QRISK2"
    model_version = "2014"
    supported_regions = None  # UK-specific model

    def __init__(self) -> None:
        super().__init__()

    def _get_ethrisk(self, ethnicity: Optional[str]) -> int:
        """Map ethnicity to QRISK2 ethrisk score."""
        if ethnicity is None:
            return 1  # Default to white
        return _QRISK2_ETHRISK_MAPPING.get(ethnicity, 1)

    def _get_smoke_cat(self, smoking_category: Optional[int], smoking: bool) -> int:
        """Map smoking category to QRISK2 smoke_cat."""
        if smoking_category is not None:
            return _QRISK2_SMOKING_MAPPING.get(smoking_category, 0)
        elif smoking:
            # If smoking=True but no category specified, assume moderate smoker
            return 3
        else:
            return 0

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient data for QRISK2 applicability."""
        super().validate_input(patient)

        # QRISK2 specific validation
        if patient.age < 25 or patient.age > 84:
            raise ValueError(f"Age {patient.age} outside QRISK2 range [25, 84]")

        if patient.bmi is None:
            raise ValueError("BMI is required for QRISK2")
        if patient.bmi < 20 or patient.bmi > 40:
            raise ValueError(f"BMI {patient.bmi} outside QRISK2 range [20, 40]")

        if patient.systolic_bp < 70 or patient.systolic_bp > 210:
            raise ValueError(f"Systolic BP {patient.systolic_bp} outside QRISK2 range [70, 210]")

        rati = patient.total_cholesterol / patient.hdl_cholesterol
        if rati < 1 or rati > 12:
            raise ValueError(f"Cholesterol ratio {rati:.2f} outside QRISK2 range [1, 12]")

        if patient.townsend_deprivation is not None:
            if patient.townsend_deprivation < -7 or patient.townsend_deprivation > 11:
                raise ValueError(f"Townsend score {patient.townsend_deprivation} outside QRISK2 range [-7, 11]")

    def _transform_variables(self, patient: PatientData, sex: str) -> Dict[str, float]:
        """Transform input variables according to QRISK2 algorithm."""
        centres = _QRISK2_CENTRES[sex]

        # Age transformations
        dage = patient.age / 10.0
        if sex == "female":
            age_1 = np.sqrt(dage) - centres["age_1"]
            age_2 = dage - centres["age_2"]
        else:  # male
            age_1 = 1.0 / dage - centres["age_1"]
            age_2 = dage ** 2 - centres["age_2"]

        # BMI transformations
        dbmi = patient.bmi / 10.0
        bmi_1 = (dbmi ** -2) - centres["bmi_1"]
        bmi_2 = ((dbmi ** -2) * np.log(dbmi)) - centres["bmi_2"]

        # Other continuous variables
        rati = (patient.total_cholesterol / patient.hdl_cholesterol) - centres["rati"]
        sbp = patient.systolic_bp - centres["sbp"]
        town = patient.townsend_deprivation if patient.townsend_deprivation is not None else 0.0
        town = town - centres["town"]

        return {
            "age_1": age_1,
            "age_2": age_2,
            "bmi_1": bmi_1,
            "bmi_2": bmi_2,
            "rati": rati,
            "sbp": sbp,
            "town": town,
        }

    def _calculate_linear_predictor(self, patient: PatientData, sex: str) -> float:
        """Calculate the linear predictor for QRISK2."""
        coeffs = _QRISK2_FEMALE_COEFFICIENTS if sex == "female" else _QRISK2_MALE_COEFFICIENTS

        # Get transformed variables
        trans_vars = self._transform_variables(patient, sex)

        # Get categorical mappings
        ethrisk = self._get_ethrisk(patient.ethnicity)
        smoke_cat = self._get_smoke_cat(patient.smoking_category, patient.smoking)

        # Start with sum
        a = 0.0

        # Add ethnicity and smoking categorical terms
        a += coeffs["Iethrisk"][ethrisk]
        a += coeffs["Ismoke"][smoke_cat]

        # Add continuous variable terms
        a += trans_vars["age_1"] * coeffs["age_1"]
        a += trans_vars["age_2"] * coeffs["age_2"]
        a += trans_vars["bmi_1"] * coeffs["bmi_1"]
        a += trans_vars["bmi_2"] * coeffs["bmi_2"]
        a += trans_vars["rati"] * coeffs["rati"]
        a += trans_vars["sbp"] * coeffs["sbp"]
        a += trans_vars["town"] * coeffs["town"]

        # Add boolean terms
        a += (patient.atrial_fibrillation or False) * coeffs["b_AF"]
        a += (patient.rheumatoid_arthritis or False) * coeffs["b_ra"]
        a += (patient.renal_disease or False) * coeffs["b_renal"]
        a += (patient.treated_hypertension or False) * coeffs["b_treatedhyp"]
        a += (patient.type1_diabetes or False) * coeffs["b_type1"]
        a += (patient.type2_diabetes or False) * coeffs["b_type2"]
        a += (patient.family_history or False) * coeffs["fh_cvd"]

        # Add interaction terms
        smoke_terms = {
            1: "age_1_smoke_1",
            2: "age_1_smoke_2",
            3: "age_1_smoke_3",
            4: "age_1_smoke_4",
        }

        if smoke_cat in smoke_terms:
            a += trans_vars["age_1"] * coeffs[smoke_terms[smoke_cat]]

        smoke_terms_age2 = {
            1: "age_2_smoke_1",
            2: "age_2_smoke_2",
            3: "age_2_smoke_3",
            4: "age_2_smoke_4",
        }

        if smoke_cat in smoke_terms_age2:
            a += trans_vars["age_2"] * coeffs[smoke_terms_age2[smoke_cat]]

        # Boolean interactions with age_1
        if patient.atrial_fibrillation:
            a += trans_vars["age_1"] * coeffs["age_1_b_AF"]
        if patient.renal_disease:
            a += trans_vars["age_1"] * coeffs["age_1_b_renal"]
        if patient.treated_hypertension:
            a += trans_vars["age_1"] * coeffs["age_1_b_treatedhyp"]
        if patient.type1_diabetes:
            a += trans_vars["age_1"] * coeffs["age_1_b_type1"]
        if patient.type2_diabetes:
            a += trans_vars["age_1"] * coeffs["age_1_b_type2"]
        if patient.family_history:
            a += trans_vars["age_1"] * coeffs["age_1_fh_cvd"]

        a += trans_vars["age_1"] * trans_vars["bmi_1"] * coeffs["age_1_bmi_1"]
        a += trans_vars["age_1"] * trans_vars["bmi_2"] * coeffs["age_1_bmi_2"]
        a += trans_vars["age_1"] * trans_vars["sbp"] * coeffs["age_1_sbp"]
        a += trans_vars["age_1"] * trans_vars["town"] * coeffs["age_1_town"]

        # Boolean interactions with age_2
        if patient.atrial_fibrillation:
            a += trans_vars["age_2"] * coeffs["age_2_b_AF"]
        if patient.renal_disease:
            a += trans_vars["age_2"] * coeffs["age_2_b_renal"]
        if patient.treated_hypertension:
            a += trans_vars["age_2"] * coeffs["age_2_b_treatedhyp"]
        if patient.type1_diabetes:
            a += trans_vars["age_2"] * coeffs["age_2_b_type1"]
        if patient.type2_diabetes:
            a += trans_vars["age_2"] * coeffs["age_2_b_type2"]
        if patient.family_history:
            a += trans_vars["age_2"] * coeffs["age_2_fh_cvd"]

        a += trans_vars["age_2"] * trans_vars["bmi_1"] * coeffs["age_2_bmi_1"]
        a += trans_vars["age_2"] * trans_vars["bmi_2"] * coeffs["age_2_bmi_2"]
        a += trans_vars["age_2"] * trans_vars["sbp"] * coeffs["age_2_sbp"]
        a += trans_vars["age_2"] * trans_vars["town"] * coeffs["age_2_town"]

        return a

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into QRISK2 categories."""
        # QRISK2 uses threshold of 20% for high risk
        # Additional thresholds based on clinical guidelines
        if risk_score < 10:
            return "low"
        elif risk_score < 20:
            return "moderate"
        else:
            return "high"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate QRISK2 risk for a single patient."""
        self.validate_input(patient)

        sex = patient.sex.lower()
        base_surv = _QRISK2_BASELINE_SURVIVAL[sex]

        # Calculate linear predictor
        lin_pred = self._calculate_linear_predictor(patient, sex)

        # Calculate risk score
        risk_score = 100.0 * (1.0 - (base_surv ** np.exp(lin_pred)))

        # Ensure risk is within bounds
        risk_score = np.clip(risk_score, 0.0, 100.0)

        # Categorize risk
        risk_category = self._categorize_risk(risk_score)

        return RiskResult(
            risk_score=float(risk_score),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "linear_predictor": lin_pred,
                "baseline_survival": base_surv,
            }
        )

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized calculation for high-throughput analysis."""
        required = [
            "age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "bmi"
        ]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = "unknown"
        results["model_name"] = self.model_name

        # Process by sex for efficiency
        for sex in ["male", "female"]:
            sex_mask = results["sex"].str.lower() == sex
            if not sex_mask.any():
                continue

            data = results[sex_mask].copy()
            base_surv = _QRISK2_BASELINE_SURVIVAL[sex]
            coeffs = _QRISK2_FEMALE_COEFFICIENTS if sex == "female" else _QRISK2_MALE_COEFFICIENTS
            centres = _QRISK2_CENTRES[sex]

            # Vectorized transformations
            dage = data["age"] / 10.0

            if sex == "female":
                age_1 = np.sqrt(dage) - centres["age_1"]
                age_2 = dage - centres["age_2"]
            else:
                age_1 = 1.0 / dage - centres["age_1"]
                age_2 = (dage ** 2) - centres["age_2"]

            dbmi = data["bmi"] / 10.0
            bmi_1 = (dbmi ** -2) - centres["bmi_1"]
            bmi_2 = ((dbmi ** -2) * np.log(dbmi)) - centres["bmi_2"]

            rati = (data["total_cholesterol"] / data["hdl_cholesterol"]) - centres["rati"]
            sbp = data["systolic_bp"] - centres["sbp"]

            # Townsend score (default to 0 if missing)
            town = data.get("townsend_deprivation", 0.0).fillna(0.0) - centres["town"]

            # Vectorized linear predictor calculation
            a = np.zeros(len(data))

            # Ethnicity and smoking mappings
            ethrisk_vals = []
            smoke_cat_vals = []

            for idx, row in data.iterrows():
                # Ethnicity mapping
                ethnicity = row.get("ethnicity")
                ethrisk = self._get_ethrisk(ethnicity)
                ethrisk_vals.append(ethrisk)

                # Smoking mapping
                smoking_cat = row.get("smoking_category")
                smoking = row.get("smoking", False)
                smoke_cat = self._get_smoke_cat(smoking_cat, smoking)
                smoke_cat_vals.append(smoke_cat)

            # Add categorical terms
            for i, (eth, smoke) in enumerate(zip(ethrisk_vals, smoke_cat_vals)):
                a[i] += coeffs["Iethrisk"][eth]
                a[i] += coeffs["Ismoke"][smoke]

            # Add continuous terms
            a += age_1 * coeffs["age_1"]
            a += age_2 * coeffs["age_2"]
            a += bmi_1 * coeffs["bmi_1"]
            a += bmi_2 * coeffs["bmi_2"]
            a += rati * coeffs["rati"]
            a += sbp * coeffs["sbp"]
            a += town * coeffs["town"]

            # Boolean terms
            bool_cols = {
                "atrial_fibrillation": "b_AF",
                "rheumatoid_arthritis": "b_ra",
                "renal_disease": "b_renal",
                "treated_hypertension": "b_treatedhyp",
                "type1_diabetes": "b_type1",
                "type2_diabetes": "b_type2",
                "family_history": "fh_cvd",
            }

            for col, coeff_key in bool_cols.items():
                if col in data.columns:
                    bool_vals = data[col].fillna(False).astype(bool)
                    a += bool_vals * coeffs[coeff_key]

            # Interaction terms - vectorized where possible
            smoke_cat_series = pd.Series(smoke_cat_vals, index=data.index)

            # Age_1 smoking interactions
            a += np.where(smoke_cat_series == 1, age_1 * coeffs["age_1_smoke_1"], 0)
            a += np.where(smoke_cat_series == 2, age_1 * coeffs["age_1_smoke_2"], 0)
            a += np.where(smoke_cat_series == 3, age_1 * coeffs["age_1_smoke_3"], 0)
            a += np.where(smoke_cat_series == 4, age_1 * coeffs["age_1_smoke_4"], 0)

            # Boolean interactions with age_1
            for col, coeff_key in {
                "atrial_fibrillation": "age_1_b_AF",
                "renal_disease": "age_1_b_renal",
                "treated_hypertension": "age_1_b_treatedhyp",
                "type1_diabetes": "age_1_b_type1",
                "type2_diabetes": "age_1_b_type2",
                "family_history": "age_1_fh_cvd",
            }.items():
                if col in data.columns:
                    bool_vals = data[col].fillna(False).astype(bool)
                    a += bool_vals * age_1 * coeffs[coeff_key]

            # Continuous interactions with age_1
            a += age_1 * bmi_1 * coeffs["age_1_bmi_1"]
            a += age_1 * bmi_2 * coeffs["age_1_bmi_2"]
            a += age_1 * sbp * coeffs["age_1_sbp"]
            a += age_1 * town * coeffs["age_1_town"]

            # Age_2 smoking interactions
            a += np.where(smoke_cat_series == 1, age_2 * coeffs["age_2_smoke_1"], 0)
            a += np.where(smoke_cat_series == 2, age_2 * coeffs["age_2_smoke_2"], 0)
            a += np.where(smoke_cat_series == 3, age_2 * coeffs["age_2_smoke_3"], 0)
            a += np.where(smoke_cat_series == 4, age_2 * coeffs["age_2_smoke_4"], 0)

            # Boolean interactions with age_2
            for col, coeff_key in {
                "atrial_fibrillation": "age_2_b_AF",
                "renal_disease": "age_2_b_renal",
                "treated_hypertension": "age_2_b_treatedhyp",
                "type1_diabetes": "age_2_b_type1",
                "type2_diabetes": "age_2_b_type2",
                "family_history": "age_2_fh_cvd",
            }.items():
                if col in data.columns:
                    bool_vals = data[col].fillna(False).astype(bool)
                    a += bool_vals * age_2 * coeffs[coeff_key]

            # Continuous interactions with age_2
            a += age_2 * bmi_1 * coeffs["age_2_bmi_1"]
            a += age_2 * bmi_2 * coeffs["age_2_bmi_2"]
            a += age_2 * sbp * coeffs["age_2_sbp"]
            a += age_2 * town * coeffs["age_2_town"]

            # Calculate final risk scores
            risk_scores = 100.0 * (1.0 - (base_surv ** np.exp(a)))
            risk_scores = np.clip(risk_scores, 0.0, 100.0)

            # Categorize risks
            risk_categories = [self._categorize_risk(score) for score in risk_scores]

            # Update results
            results.loc[sex_mask, "risk_score"] = risk_scores
            results.loc[sex_mask, "risk_category"] = risk_categories

        return results

