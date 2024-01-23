"""
Patient data classes for Framingham family models.

Includes data classes for Framingham, ASCVD, and Reynolds models.
"""

from typing import Literal, Optional

from pydantic import Field

from cvd_risk.core.data_classes.base import BasePatientData


class FraminghamPatientData(BasePatientData):
    """
    Patient data for Framingham Risk Score (2008).

    Framingham estimates 10-year CVD risk using traditional risk factors.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    antihypertensive : Optional[bool]
        Currently on antihypertensive medication.
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    antihypertensive: Optional[bool] = Field(
        default=None, description="On antihypertensive medication"
    )


class ASCVDPatientData(BasePatientData):
    """
    Patient data for ACC/AHA ASCVD Pooled Cohort Equations.

    ASCVD estimates 10-year risk of atherosclerotic CVD events.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    antihypertensive : Optional[bool]
        Currently on antihypertensive medication.
    race : Optional[Literal["white", "black", "other"]]
        Race category for risk adjustment.
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    antihypertensive: Optional[bool] = Field(
        default=None, description="On antihypertensive medication"
    )
    race: Optional[Literal["white", "black", "other"]] = Field(
        default=None, description="Race: white, black, or other"
    )


class ReynoldsPatientData(BasePatientData):
    """
    Patient data for Reynolds Risk Score.

    Reynolds incorporates hs-CRP and family history for improved risk prediction.

    Additional Fields
    -----------------
    family_history : Optional[bool]
        Parental history of MI before age 60.
    hs_crp : Optional[float]
        High-sensitivity C-reactive protein in mg/L.
    """

    family_history: Optional[bool] = Field(
        default=None, description="Parental MI before age 60"
    )
    hs_crp: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="hs-CRP in mg/L"
    )
