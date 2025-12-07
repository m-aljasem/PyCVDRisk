"""
WHO CVD Risk Chart for cardiovascular disease risk prediction.

The WHO CVD Risk Charts provide region-specific risk estimates based
on simplified risk factor profiles using lookup tables.

Reference:
    WHO CVD Risk Chart Working Group. World Health Organization
    cardiovascular disease risk charts: revised models to estimate
    risk in 21 global regions. Lancet Glob Health. 2019;7(10):e1332-e1345.
"""

import logging
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# WHO epidemiological subregions
_WHO_SUPPORTED_REGIONS = [
    "AFR_D", "AFR_E", "AMR_A", "AMR_B", "AMR_D",
    "EMR_B", "EMR_D", "EUR_A", "EUR_B", "EUR_C",
    "SEAR_B", "SEAR_D", "WPR_A", "WPR_B"
]

# Risk score mappings to numeric values for categorization
_WHO_RISK_SCORE_MAPPING = {
    "<10%": 5.0,
    "10% to <20%": 15.0,
    "20% to <30%": 25.0,
    "30% to <40%": 35.0,
    ">=40%": 45.0
}


class WHO(RiskModel):
    """
    WHO CVD Risk Chart for cardiovascular disease risk prediction.

    The WHO model provides region-specific 10-year CVD risk estimates
    using lookup tables based on discretized risk factors.

    Parameters
    ----------
    age : int
        Age in years (18+ for optimal performance).
    sex : Literal["male", "female"]
        Biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L.
    smoking : bool
        Current smoking status.
    diabetes : Optional[bool]
        Diabetes status.
    region : str
        WHO epidemiological subregion (required).

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage (numeric value)
        - risk_category: Risk classification
        - model_name: "WHO"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk.models.who import WHO
    >>> from cvd_risk.core.validation import PatientData
    >>>
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,
    ...     smoking=True,
    ...     diabetes=False,
    ...     region="EUR_A"
    ... )
    >>> model = WHO()
    >>> result = model.calculate(patient)
    >>> print(f"10-year CVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - Uses discretized risk factor categories
    - Region-specific estimates for 14 WHO epidemiological subregions
    - Based on WHO/ISH risk prediction charts
    """

    model_name = "WHO"
    model_version = "2019"
    supported_regions = _WHO_SUPPORTED_REGIONS

    def __init__(self) -> None:
        """Initialize WHO model."""
        super().__init__()
        self._load_risk_data()

    def _load_risk_data(self) -> None:
        """Load WHO risk score lookup table."""
        data_path = os.path.join(os.path.dirname(__file__), "WHO_ISH_Scores.csv")
        try:
            self._risk_data = pd.read_csv(data_path, dtype={'refv': str})
            # Create lookup dictionary for faster access
            self._lookup_dict = {}
            for _, row in self._risk_data.iterrows():
                self._lookup_dict[str(row['refv'])] = {
                    region: row[region] for region in _WHO_SUPPORTED_REGIONS
                }
        except FileNotFoundError:
            raise FileNotFoundError(f"WHO risk data file not found at {data_path}")

    def _discretize_age(self, age: int) -> int:
        """Discretize age according to WHO categories."""
        if age > 17 and age < 50:
            return 40
        elif age >= 50 and age < 60:
            return 50
        elif age >= 60 and age < 70:
            return 60
        else:  # age >= 70
            return 70

    def _discretize_cholesterol(self, cholesterol: float) -> int:
        """Discretize total cholesterol according to WHO categories."""
        if cholesterol > 0 and cholesterol < 4.5:
            return 4
        elif cholesterol >= 4.5 and cholesterol < 5.5:
            return 5
        elif cholesterol >= 5.5 and cholesterol < 6.5:
            return 6
        elif cholesterol >= 6.5 and cholesterol < 7.5:
            return 7
        else:  # cholesterol >= 7.5
            return 8

    def _discretize_sbp(self, sbp: float) -> int:
        """Discretize systolic blood pressure according to WHO categories."""
        if sbp > 0 and sbp < 140:
            return 120
        elif sbp >= 140 and sbp < 160:
            return 140
        elif sbp >= 160 and sbp < 180:
            return 160
        else:  # sbp >= 180
            return 180

    def _create_lookup_value(self, patient: PatientData) -> str:
        """Create lookup value by concatenating discretized parameters."""
        age_disc = self._discretize_age(patient.age)
        gdr = 1 if patient.sex == "male" else 0  # 1=male, 0=female
        smk = 1 if patient.smoking else 0
        sbp_disc = self._discretize_sbp(patient.systolic_bp)
        dm = 1 if patient.diabetes else 0
        chl_disc = self._discretize_cholesterol(patient.total_cholesterol)

        # Format: age + gdr + dm + smk + sbp + chl (all as single digits)
        return f"{age_disc}{gdr}{dm}{smk}{sbp_disc}{chl_disc}"

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient input for WHO model."""
        super().validate_input(patient)

        if patient.region is None or patient.region not in self.supported_regions:
            raise ValueError(f"Region must be one of {self.supported_regions}")

        if patient.age < 18:
            logger.warning(f"Age {patient.age} is below 18 years. Results may be unreliable.")

        if patient.age > 100:
            logger.warning(f"Age {patient.age} is greater than 100 years. Results may be unreliable.")

        if patient.systolic_bp < 90:
            logger.warning(f"Systolic BP {patient.systolic_bp} is below 90 mmHg. Results may be unreliable.")

        if patient.systolic_bp > 250:
            logger.warning(f"Systolic BP {patient.systolic_bp} is over 250 mmHg. Results may be unreliable.")

        if patient.total_cholesterol > 10:
            logger.warning(
                f"Total cholesterol {patient.total_cholesterol} is greater than 10 mmol/L. "
                "Ensure all values are in units of mmol/L."
            )

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate 10-year CVD risk using WHO lookup table."""
        self.validate_input(patient)

        # Create lookup value from discretized parameters
        lookup_value = self._create_lookup_value(patient)

        # Get risk score from lookup table
        try:
            region_scores = self._lookup_dict[lookup_value]
            risk_score_text = region_scores[patient.region]
        except KeyError:
            raise ValueError(f"Lookup value {lookup_value} not found in risk data")

        # Convert categorical risk score to numeric value
        risk_score_numeric = _WHO_RISK_SCORE_MAPPING.get(risk_score_text, 0.0)

        risk_category = self._categorize_risk(risk_score_numeric)

        metadata = self._get_metadata()
        metadata.update(
            {
                "age": patient.age,
                "sex": patient.sex,
                "region": patient.region,
                "lookup_value": lookup_value,
                "risk_score_text": risk_score_text,
                "discretized_age": self._discretize_age(patient.age),
                "discretized_sbp": self._discretize_sbp(patient.systolic_bp),
                "discretized_cholesterol": self._discretize_cholesterol(patient.total_cholesterol),
            }
        )

        return RiskResult(
            risk_score=float(risk_score_numeric),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk using WHO categories."""
        if risk_percentage < 10:
            return "low"
        elif risk_percentage < 20:
            return "moderate"
        elif risk_percentage < 30:
            return "high"
        elif risk_percentage < 40:
            return "very_high"
        else:  # >= 40%
            return "very_high"

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        Calculates WHO risk scores for multiple patients.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "smoking", "diabetes", "region"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_score_text"] = ""
        results["model_name"] = self.model_name

        # Vectorized discretization
        results["age_disc"] = results["age"].apply(self._discretize_age)
        results["sbp_disc"] = results["systolic_bp"].apply(self._discretize_sbp)
        results["chl_disc"] = results["total_cholesterol"].apply(self._discretize_cholesterol)

        # Convert sex to numeric (1=male, 0=female)
        results["gdr"] = (results["sex"].str.lower() == "male").astype(int)
        results["smk"] = results["smoking"].astype(int)
        results["dm"] = results["diabetes"].astype(int)

        # Create lookup values
        results["lookup_value"] = (
            results["age_disc"].astype(str)
            + results["gdr"].astype(str)
            + results["dm"].astype(str)
            + results["smk"].astype(str)
            + results["sbp_disc"].astype(str)
            + results["chl_disc"].astype(str)
        )

        # Vectorized lookup and risk score assignment
        for idx, row in results.iterrows():
            try:
                region_scores = self._lookup_dict[row["lookup_value"]]
                risk_score_text = region_scores[row["region"]]
                risk_score_numeric = _WHO_RISK_SCORE_MAPPING.get(risk_score_text, 0.0)
                results.at[idx, "risk_score"] = risk_score_numeric
                results.at[idx, "risk_score_text"] = risk_score_text
            except KeyError:
                logger.warning(f"Lookup value {row['lookup_value']} not found for region {row['region']}")
                results.at[idx, "risk_score"] = np.nan
                results.at[idx, "risk_score_text"] = "unknown"

        # Vectorized categorization
        conditions = [
            results["risk_score"] < 10,
            (results["risk_score"] >= 10) & (results["risk_score"] < 20),
            (results["risk_score"] >= 20) & (results["risk_score"] < 30),
            results["risk_score"] >= 30
        ]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results

