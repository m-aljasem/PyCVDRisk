"""
Globorisk cardiovascular disease risk prediction model.

Globorisk provides country-specific cardiovascular disease risk
estimates that account for local mortality rates.

Reference:
    Ueda P, Woodward M, Lu Y, et al. Laboratory-based and
    office-based risk scores and charts to predict 10-year risk
    of cardiovascular disease in 182 countries: a pooled analysis
    of prospective cohorts and health surveys. Lancet Diabetes
    Endocrinol. 2017;5(3):196-213.
"""

import logging
import lzma
import pickle
from typing import Literal, Optional
import os

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# LAC (Latin America and Caribbean) countries
LAC_COUNTRIES = {
    "ARG", "ATG", "BHS", "BLZ", "BOL", "BRA", "BRB", "CHL", "COL",
    "CRI", "CUB", "DOM", "ECU", "GRD", "GTM", "GUY", "HND", "HTI",
    "JAM", "LCA", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "SUR",
    "TTO", "URY", "VCT", "VEN"
}

# Load Globorisk data using relative path
_GLOBORISK_DATA = None
try:
    # Get the directory of this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(module_dir, "globorisk_data.pkl.xz")

    with lzma.open(data_file, 'rb') as f:
        _GLOBORISK_DATA = pickle.load(f)
    logger.info("Loaded Globorisk data successfully")
except Exception as e:
    logger.error(f"Failed to load Globorisk data: {e}")
    _GLOBORISK_DATA = None


class Globorisk(RiskModel):
    """
    Globorisk cardiovascular disease risk prediction model.

    Globorisk provides country-specific 10-year CVD risk estimates
    that account for local cardiovascular mortality rates.

    Valid for ages 40-74 years. Supports 182 countries with
    country-specific coefficients and baseline rates.

    Parameters
    ----------
    country : str, optional
        ISO 3-letter country code (e.g., 'USA', 'GBR', 'DEU').
        If None, uses global coefficients.
    year : int, optional
        Baseline year for risk calculation (2000-2020).
        If None, uses the latest available year (2020).
    version : Literal["lab", "office", "fatal"]
        Model version to use:
        - "lab": Laboratory-based (requires total_cholesterol)
        - "office": Office-based (requires bmi)
        - "fatal": Fatal CVD only (requires total_cholesterol)

    Attributes
    ----------
    model_name : str
        Name of the model ("Globorisk")
    model_version : str
        Version identifier ("2017")
    supported_versions : list
        Available model versions
    supported_regions : str
        Description of supported regions
    """

    model_name = "Globorisk"
    model_version = "2017"
    supported_versions = ["lab", "office", "fatal"]
    supported_regions = "182 countries (ISO 3-letter codes)"

    def __init__(
        self,
        country: Optional[str] = None,
        year: Optional[int] = None,
        version: Literal["lab", "office", "fatal"] = "lab"
    ) -> None:
        """Initialize Globorisk model.

        Parameters
        ----------
        country : str, optional
            ISO 3-letter country code. If None, uses global coefficients.
        year : int, optional
            Baseline year (2000-2020). If None, uses latest year.
        version : str
            Model version: 'lab', 'office', or 'fatal'.
        """
        super().__init__()
        self.country = country.upper() if country else None
        self.year = year
        self.version = version

        # Validate version
        if version not in self.supported_versions:
            raise ValueError(f"Version must be one of {self.supported_versions}")

        # Validate inputs
        if _GLOBORISK_DATA is None:
            raise RuntimeError("Globorisk data not loaded. Check data file installation.")

        if self.country and self.country not in _GLOBORISK_DATA['cvd_rates']:
            logger.warning(f"Country {self.country} not found in database. Using global coefficients.")
            self.country = None

        if self.year and (self.year < 2000 or self.year > 2020):
            raise ValueError(f"Year {self.year} must be between 2000 and 2020")

        # Determine if country is LAC
        self.is_lac = self.country in LAC_COUNTRIES if self.country else False

    def __init__(
        self,
        country: Optional[str] = None,
        year: Optional[int] = None,
        version: Literal["lab", "office", "fatal"] = "lab"
    ) -> None:
        """Initialize Globorisk model.

        Parameters
        ----------
        country : str, optional
            ISO 3-letter country code. If None, uses global coefficients.
        year : int, optional
            Baseline year (2000-2020). If None, uses latest available year.
        version : str
            Model version: 'lab', 'office', or 'fatal'.
        """
        super().__init__()
        self.country = country.upper() if country else None
        self.year = year
        self.version = version

        # Validate inputs
        if _GLOBORISK_DATA is None:
            raise RuntimeError("Globorisk data not loaded. Check data file.")

        if self.country and self.country not in _GLOBORISK_DATA['cvd_rates']:
            logger.warning(f"Country {self.country} not found in database. Using global coefficients.")
            self.country = None

        if self.year and (self.year < 2000 or self.year > 2020):
            raise ValueError(f"Year {self.year} outside valid range [2000, 2020]")

        # Determine if country is LAC
        self.is_lac = self.country in LAC_COUNTRIES if self.country else False

    @classmethod
    def get_available_countries(cls) -> list:
        """Get list of available country codes."""
        if _GLOBORISK_DATA is None:
            return []
        return sorted(_GLOBORISK_DATA['cvd_rates'].keys())

    @classmethod
    def get_lac_countries(cls) -> list:
        """Get list of Latin America and Caribbean countries."""
        return sorted(LAC_COUNTRIES)

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient input for Globorisk."""
        super().validate_input(patient)

        if _GLOBORISK_DATA is None:
            raise RuntimeError("Globorisk data not loaded. Check data file.")

        if patient.age < 40 or patient.age > 74:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 74] years. "
                "Results may have reduced accuracy."
            )

        # Check required fields based on version
        if self.version in ["lab", "fatal"]:
            # These versions require total cholesterol - already validated in PatientData
            pass
        elif self.version == "office":
            if patient.bmi is None:
                raise ValueError("BMI required for 'office' version")

        # Validate country if specified
        if self.country and self.country.upper() not in _GLOBORISK_DATA['cvd_rates']:
            logger.warning(f"Country {self.country} not found in database. Using global coefficients.")

        # Validate year range
        if self.year and (self.year < 2000 or self.year > 2020):
            logger.warning(f"Year {self.year} outside valid range [2000, 2020]. Using default year.")

        # Diabetes is optional - handled in calculation
        if patient.diabetes is None:
            logger.debug("Diabetes status not provided, assuming False")

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CVD risk using Globorisk.

        Parameters
        ----------
        patient : PatientData
            Patient information including demographics and risk factors

        Returns
        -------
        RiskResult
            Risk calculation results with score, category, and metadata

        Raises
        ------
        ValueError
            If required patient data is missing or invalid
        RuntimeError
            If model data is not loaded
        """
        self.validate_input(patient)

        # Get coefficients for this version and region
        coeffs = self._get_coefficients()

        # Get CVD rates for this country/year
        baseline_rates = self._get_baseline_rates(patient)

        # Get risk factor means for centering
        rf_means = self._get_risk_factor_means(patient)

        # Prepare patient data
        sex_numeric = 0 if patient.sex == "female" else 1
        age_centered = patient.age
        agec = int(np.clip((age_centered - 40) // 5, 0, 6))  # age category 0-6

        # Center risk factors using population means
        sbp_c = (patient.systolic_bp / 10.0) - rf_means['mean_sbp']
        dm_c = float(patient.diabetes or False) - rf_means['mean_dm']
        smk_c = float(patient.smoking) - rf_means['mean_smk']

        if self.version in ["lab", "fatal"]:
            tc_c = patient.total_cholesterol - rf_means['mean_tc']
        elif self.version == "office":
            bmi_c = (patient.bmi / 5.0) - rf_means['mean_bmi']

        # Calculate hazard ratios for each year (0-9)
        hr_list = []
        for t in range(10):
            if self.version in ["lab", "fatal"]:
                # Laboratory-based or fatal version
                hr = np.exp(
                    sbp_c * coeffs.get("main_sbpc", 0) +
                    tc_c * coeffs.get("main_tcc", 0) +
                    dm_c * coeffs.get("main__Idm_1", 0) +
                    smk_c * coeffs.get("main_smok", 0) +
                    sex_numeric * dm_c * coeffs.get("main_sexdm", 0) +
                    sex_numeric * smk_c * coeffs.get("main_sexsmok", 0) +
                    (age_centered + t) * sbp_c * coeffs.get("tvc_sbpc", 0) +
                    (age_centered + t) * tc_c * coeffs.get("tvc_tcc", 0) +
                    (age_centered + t) * dm_c * coeffs.get("tvc_dm", 0) +
                    (age_centered + t) * smk_c * coeffs.get("tvc_smok", 0)
                )
            elif self.version == "office":
                # Office-based version
                hr = np.exp(
                    sbp_c * coeffs.get("main_sbpc", 0) +
                    bmi_c * coeffs.get("main_bmi5c", 0) +
                    smk_c * coeffs.get("main_smokc", 0) +
                    sex_numeric * smk_c * coeffs.get("main_sexsmokc", 0) +
                    sex_numeric * sbp_c * coeffs.get("main_sbpsexc", 0) +
                    (age_centered + t) * sbp_c * coeffs.get("tvc_sbpc", 0) +
                    (age_centered + t) * smk_c * coeffs.get("tvc_smokc", 0) +
                    (age_centered + t) * bmi_c * coeffs.get("tvc_bmi5c", 0)
                )
            hr_list.append(hr)

        # Calculate hazard rates by multiplying by baseline rates
        hazard_rates = []
        for t in range(10):
            if t in baseline_rates:
                hz = hr_list[t] * baseline_rates[t]
            else:
                hz = 0.0
            hazard_rates.append(hz)

        # Calculate survival probabilities
        survival_probs = [np.exp(-hz) for hz in hazard_rates]

        # Calculate cumulative survival
        cumulative_survival = 1.0
        for surv in survival_probs:
            cumulative_survival *= surv

        # Calculate 10-year risk
        risk_score = (1.0 - cumulative_survival) * 100.0
        risk_score = np.clip(risk_score, 0.0, 100.0)

        risk_category = self._categorize_risk(risk_score)

        metadata = self._get_metadata()
        metadata.update({
            "country": self.country,
            "year": self.year,
            "version": self.version,
            "is_lac": self.is_lac,
            "age": patient.age,
            "sex": patient.sex,
            "hazard_rates": hazard_rates,
            "survival_probs": survival_probs,
            "cumulative_survival": cumulative_survival
        })

        return RiskResult(
            risk_score=float(risk_score),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=f"{self.model_version}_{self.version}",
            calculation_metadata=metadata,
        )

    def _get_coefficients(self) -> dict:
        """Get coefficients for the current version and region."""
        lac_key = 1 if self.is_lac else 0

        if self.version not in _GLOBORISK_DATA['coefficients']:
            raise ValueError(f"Version {self.version} not found in coefficients data")

        if lac_key not in _GLOBORISK_DATA['coefficients'][self.version]:
            # Fall back to non-LAC coefficients if LAC not available
            lac_key = 0
            if lac_key not in _GLOBORISK_DATA['coefficients'][self.version]:
                raise ValueError(f"No coefficients found for version {self.version}")

        return _GLOBORISK_DATA['coefficients'][self.version][lac_key]

    def _get_baseline_rates(self, patient: PatientData) -> dict:
        """Get baseline CVD rates for the current country/year."""
        sex_numeric = 0 if patient.sex == "female" else 1
        age = patient.age

        # Determine CVD type based on version
        if self.version == "fatal":
            cvd_type = "F"
        else:
            cvd_type = "FNF"

        # Get country data or use default
        country = self.country or "USA"  # Default fallback

        if country not in _GLOBORISK_DATA['cvd_rates']:
            logger.warning(f"Country {country} not found, using USA as fallback")
            country = "USA"

        if sex_numeric not in _GLOBORISK_DATA['cvd_rates'][country]:
            logger.warning(f"Sex {sex_numeric} not found for {country}, using sex 0")
            sex_numeric = 0

        if cvd_type not in _GLOBORISK_DATA['cvd_rates'][country][sex_numeric]:
            logger.warning(f"CVD type {cvd_type} not found, using FNF")
            cvd_type = "FNF"

        # Get year data
        year = self.year or 2020  # Default to latest year
        if year not in _GLOBORISK_DATA['cvd_rates'][country][sex_numeric][cvd_type]:
            # Find closest available year
            available_years = list(_GLOBORISK_DATA['cvd_rates'][country][sex_numeric][cvd_type].keys())
            year = max(available_years) if available_years else 2020

        # Get age-specific rates
        if age not in _GLOBORISK_DATA['cvd_rates'][country][sex_numeric][cvd_type][year]:
            # Find closest available age
            available_ages = list(_GLOBORISK_DATA['cvd_rates'][country][sex_numeric][cvd_type][year].keys())
            age = min(available_ages, key=lambda x: abs(x - age))  # Closest age
            logger.warning(f"Age {patient.age} not found, using age {age}")

        return _GLOBORISK_DATA['cvd_rates'][country][sex_numeric][cvd_type][year][age]

    def _get_risk_factor_means(self, patient: PatientData) -> dict:
        """Get risk factor means for centering."""
        sex_numeric = 0 if patient.sex == "female" else 1
        agec = int(np.clip((patient.age - 40) // 5, 0, 6)) + 1  # age category 1-7 in data

        country = self.country or "USA"  # Default fallback

        if country not in _GLOBORISK_DATA['risk_factors']:
            logger.warning(f"Country {country} not found for risk factors, using USA")
            country = "USA"

        if sex_numeric not in _GLOBORISK_DATA['risk_factors'][country]:
            logger.warning(f"Sex {sex_numeric} not found for {country} risk factors, using sex 0")
            sex_numeric = 0

        if agec not in _GLOBORISK_DATA['risk_factors'][country][sex_numeric]:
            logger.warning(f"Age category {agec} not found for {country}, using agec 1")
            agec = 1

        return _GLOBORISK_DATA['risk_factors'][country][sex_numeric][agec]

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk."""
        if risk_percentage < 10:
            return "low"
        elif risk_percentage < 20:
            return "moderate"
        else:
            return "high"

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch calculation for multiple patients.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with patient data. Must contain:
            - Required: age, sex, systolic_bp, smoking
            - For 'lab'/'fatal' versions: total_cholesterol
            - For 'office' version: bmi
            - Optional: country, year, version (defaults will be used if missing)

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'risk_score', 'risk_category', and 'model_name' columns

        Examples
        --------
        >>> import pandas as pd
        >>> from cvd_risk.models.globorisk import Globorisk
        >>>
        >>> df = pd.DataFrame({
        ...     'age': [55, 60],
        ...     'sex': ['male', 'female'],
        ...     'systolic_bp': [140.0, 130.0],
        ...     'total_cholesterol': [6.0, 5.5],
        ...     'smoking': [True, False],
        ...     'country': ['USA', 'GBR']
        ... })
        >>> model = Globorisk()
        >>> results = model.calculate_batch(df)
        """
        if _GLOBORISK_DATA is None:
            raise RuntimeError("Globorisk data not loaded")

        # Check for required columns
        base_required = ["age", "sex", "systolic_bp", "smoking"]
        df_copy = df.copy()

        # Add optional columns with defaults if missing
        if 'country' not in df_copy.columns:
            df_copy['country'] = None
        if 'year' not in df_copy.columns:
            df_copy['year'] = None
        if 'version' not in df_copy.columns:
            df_copy['version'] = 'lab'

        # Convert NaN values to None for proper processing
        df_copy['country'] = df_copy['country'].where(pd.notna(df_copy['country']), None)
        df_copy['year'] = df_copy['year'].where(pd.notna(df_copy['year']), None)
        df_copy['version'] = df_copy['version'].where(pd.notna(df_copy['version']), 'lab')

        # Validate required columns based on versions present
        versions_present = df_copy['version'].unique()
        for version in versions_present:
            if version in ['lab', 'fatal']:
                required_cols = base_required + ['total_cholesterol']
            elif version == 'office':
                required_cols = base_required + ['bmi']
            else:
                required_cols = base_required

            missing = set(required_cols) - set(df_copy.columns)
            if missing:
                raise ValueError(f"Missing required columns for version '{version}': {missing}")

        # Initialize result columns
        df_copy["risk_score"] = np.nan
        df_copy["risk_category"] = "unknown"
        df_copy["model_name"] = self.model_name

        # Process each row individually for now (simpler approach)
        # TODO: Optimize with grouping when None handling is fixed
        for idx, row in df_copy.iterrows():
            country = row['country']
            year = row['year']
            version = row['version']

            # Create model instance for this configuration
            temp_model = Globorisk(country=country, year=year, version=version)

            # Convert row to PatientData
            patient_data = temp_model._row_to_patient_data(row, version)

            # Calculate risk
            result = temp_model.calculate(patient_data)

            # Store results
            df_copy.loc[idx, "risk_score"] = result.risk_score
            df_copy.loc[idx, "risk_category"] = result.risk_category

        return df_copy

    def _row_to_patient_data(self, row: pd.Series, version: str) -> PatientData:
        """Convert DataFrame row to PatientData object."""
        # Build the data dict
        data = {
            'age': int(row['age']),
            'sex': row['sex'],
            'systolic_bp': float(row['systolic_bp']),
            'smoking': bool(row['smoking']),
        }

        # Add version-specific required fields
        if version in ['lab', 'fatal']:
            data['total_cholesterol'] = float(row['total_cholesterol'])
            data['hdl_cholesterol'] = float(row.get('hdl_cholesterol', 1.2))  # Default if missing
        elif version == 'office':
            data['bmi'] = float(row['bmi'])
            data['total_cholesterol'] = float(row.get('total_cholesterol', 6.0))  # Default if missing
            data['hdl_cholesterol'] = float(row.get('hdl_cholesterol', 1.2))  # Default if missing

        # Add optional fields if present
        if 'diabetes' in row.index and pd.notna(row['diabetes']):
            data['diabetes'] = bool(row['diabetes'])
        if 'bmi' in row.index and pd.notna(row['bmi']) and version != 'office':
            data['bmi'] = float(row['bmi'])

        return PatientData(**data)

