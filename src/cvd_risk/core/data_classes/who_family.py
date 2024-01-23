"""
Patient data classes for WHO family models.

Includes data classes for WHO and Globorisk models.
"""

from typing import Optional

from pydantic import Field

from cvd_risk.core.data_classes.base import BasePatientData


# WHO epidemiological subregions
WHO_REGIONS = [
    "AFR_D", "AFR_E",  # Africa
    "AMR_A", "AMR_B", "AMR_D",  # Americas
    "EMR_B", "EMR_D",  # Eastern Mediterranean
    "EUR_A", "EUR_B", "EUR_C",  # Europe
    "SEAR_B", "SEAR_D",  # South-East Asia
    "WPR_A", "WPR_B",  # Western Pacific
]


class WHOPatientData(BasePatientData):
    """
    Patient data for WHO CVD Risk Charts.

    WHO risk charts estimate 10-year CVD risk for different world regions.

    Additional Fields
    -----------------
    region : str
        WHO epidemiological subregion (e.g., AFR_D, EUR_A, SEAR_B).
    diabetes : Optional[bool]
        Diabetes status.
    """

    region: str = Field(
        ..., description="WHO epidemiological subregion (e.g., AFR_D, EUR_A)"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")


class GloboriskPatientData(BasePatientData):
    """
    Patient data for Globorisk model.

    Globorisk provides country-specific CVD risk estimates.

    Additional Fields
    -----------------
    region : str
        Country or region code.
    diabetes : Optional[bool]
        Diabetes status.
    """

    region: str = Field(..., description="Country or region code")
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    bmi: Optional[float] = Field(
        default=None, ge=10.0, le=60.0, description="Body mass index in kg/m²"
    )
