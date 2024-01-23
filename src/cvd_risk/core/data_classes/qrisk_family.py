"""
Patient data classes for QRISK family models.

Includes data classes for QRISK2 and QRISK3 models.
"""

from typing import Literal, Optional

from pydantic import Field

from cvd_risk.core.data_classes.base import BasePatientData


class QRISK2PatientData(BasePatientData):
    """
    Patient data for QRISK2 model.

    QRISK2 estimates 10-year CVD risk with ethnicity and medical conditions.

    Additional Fields
    -----------------
    bmi : Optional[float]
        Body mass index in kg/m².
    ethnicity : Optional[Literal[...]]
        Ethnicity category for risk adjustment.
    family_history : Optional[bool]
        Family history of CVD in first-degree relative before age 60.
    diabetes : Optional[bool]
        Diabetes status.
    treated_hypertension : Optional[bool]
        On blood pressure treatment.
    rheumatoid_arthritis : Optional[bool]
        Rheumatoid arthritis diagnosis.
    atrial_fibrillation : Optional[bool]
        Atrial fibrillation diagnosis.
    renal_disease : Optional[bool]
        Chronic kidney disease (stage 4-5).
    townsend_deprivation : Optional[float]
        Townsend deprivation score.
    """

    bmi: Optional[float] = Field(
        default=None, ge=10.0, le=60.0, description="Body mass index in kg/m²"
    )
    ethnicity: Optional[
        Literal["white", "south_asian", "black", "chinese", "mixed", "other"]
    ] = Field(default=None, description="Ethnicity category")
    family_history: Optional[bool] = Field(
        default=None, description="Family history of CVD"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    treated_hypertension: Optional[bool] = Field(
        default=None, description="On blood pressure treatment"
    )
    rheumatoid_arthritis: Optional[bool] = Field(
        default=None, description="Rheumatoid arthritis"
    )
    atrial_fibrillation: Optional[bool] = Field(
        default=None, description="Atrial fibrillation"
    )
    renal_disease: Optional[bool] = Field(
        default=None, description="Chronic kidney disease stage 4-5"
    )
    townsend_deprivation: Optional[float] = Field(
        default=None, ge=-10.0, le=10.0, description="Townsend deprivation score"
    )
    smoking_category: Optional[Literal[1, 2, 3, 4]] = Field(
        default=None, description="Smoking category: 1=non-smoker, 2=ex-smoker, 3=light, 4=heavy"
    )
    type1_diabetes: Optional[bool] = Field(
        default=None, description="Type 1 diabetes"
    )
    type2_diabetes: Optional[bool] = Field(
        default=None, description="Type 2 diabetes"
    )


class QRISK3PatientData(QRISK2PatientData):
    """
    Patient data for QRISK3 model.

    QRISK3 extends QRISK2 with additional medical conditions and measurements.
    Inherits all fields from QRISK2PatientData.

    Additional Fields
    -----------------
    atypical_antipsychotics : Optional[bool]
        Use of atypical antipsychotic medication.
    impotence2 : Optional[bool]
        Erectile dysfunction (males only).
    corticosteroids : Optional[bool]
        Systemic corticosteroid use.
    migraine : Optional[bool]
        Migraine diagnosis.
    systemic_lupus : Optional[bool]
        Systemic lupus erythematosus.
    severe_mental_illness : Optional[bool]
        Severe mental illness diagnosis.
    type1_diabetes : Optional[bool]
        Type 1 diabetes specifically.
    type2_diabetes : Optional[bool]
        Type 2 diabetes specifically.
    sbps5 : Optional[float]
        Systolic BP variability (SD of 5 readings).
    smoking_category : Optional[Literal[1, 2, 3, 4]]
        Smoking category: 1=non, 2=ex, 3=light, 4=heavy.
    """

    atypical_antipsychotics: Optional[bool] = Field(
        default=None, description="Atypical antipsychotic use"
    )
    impotence2: Optional[bool] = Field(
        default=None, description="Erectile dysfunction (males only)"
    )
    corticosteroids: Optional[bool] = Field(
        default=None, description="Systemic corticosteroid use"
    )
    migraine: Optional[bool] = Field(default=None, description="Migraine diagnosis")
    systemic_lupus: Optional[bool] = Field(
        default=None, description="Systemic lupus erythematosus"
    )
    severe_mental_illness: Optional[bool] = Field(
        default=None, description="Severe mental illness"
    )
    type1_diabetes: Optional[bool] = Field(
        default=None, description="Type 1 diabetes"
    )
    type2_diabetes: Optional[bool] = Field(
        default=None, description="Type 2 diabetes"
    )
    sbps5: Optional[float] = Field(
        default=None, ge=0.0, le=50.0, description="Systolic BP variability (SD)"
    )
    smoking_category: Optional[Literal[1, 2, 3, 4]] = Field(
        default=None, description="Smoking category: 1=non, 2=ex, 3=light, 4=heavy"
    )
