"""
Patient data classes for other CVD risk models.

Includes data classes for regional models, secondary prevention models,
and models requiring only simple fields.
"""

from typing import Literal, Optional

from pydantic import Field

from cvd_risk.core.data_classes.base import BasePatientData


class SimplePatientData(BasePatientData):
    """
    Simple patient data for models using only core fields.

    Used by models that don't require additional fields beyond the base class.
    Adds only optional region and diabetes fields for flexibility.

    Additional Fields
    -----------------
    region : Optional[str]
        Geographic region if applicable.
    diabetes : Optional[bool]
        Diabetes status.
    """

    region: Optional[str] = Field(default=None, description="Geographic region")
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")


class ASSIGNPatientData(BasePatientData):
    """
    Patient data for ASSIGN Score (Scotland).

    ASSIGN estimates 10-year CVD risk with socioeconomic factors.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    family_history : Optional[bool]
        Family history of CVD.
    rheumatoid_arthritis : Optional[bool]
        Rheumatoid arthritis diagnosis.
    simd_score : Optional[float]
        Scottish Index of Multiple Deprivation.
    cigarettes_per_day : Optional[int]
        Number of cigarettes smoked per day.
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    family_history: Optional[bool] = Field(
        default=None, description="Family history of CVD"
    )
    rheumatoid_arthritis: Optional[bool] = Field(
        default=None, description="Rheumatoid arthritis"
    )
    simd_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="SIMD deprivation score"
    )
    cigarettes_per_day: Optional[int] = Field(
        default=None, ge=0, le=100, description="Cigarettes per day"
    )
    townsend_deprivation: Optional[float] = Field(
        default=None, ge=-10.0, le=10.0, description="Townsend deprivation score"
    )


class PROCAMPatientData(BasePatientData):
    """
    Patient data for PROCAM Score (Germany).

    PROCAM estimates 10-year coronary heart disease risk.

    Additional Fields
    -----------------
    ldl_cholesterol : Optional[float]
        LDL cholesterol in mmol/L.
    triglycerides : Optional[float]
        Triglycerides in mmol/L.
    diabetes : Optional[bool]
        Diabetes status.
    family_history : Optional[bool]
        Family history of MI before age 60.
    """

    ldl_cholesterol: Optional[float] = Field(
        default=None, ge=1.0, le=10.0, description="LDL cholesterol in mmol/L"
    )
    triglycerides: Optional[float] = Field(
        default=None, ge=0.5, le=20.0, description="Triglycerides in mmol/L"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    family_history: Optional[bool] = Field(
        default=None, description="Family history of MI"
    )


class REGICORPatientData(BasePatientData):
    """
    Patient data for REGICOR Score (Spain).

    REGICOR is the Framingham-based Spanish cardiovascular risk function.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    antihypertensive : Optional[bool]
        On antihypertensive medication.
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    antihypertensive: Optional[bool] = Field(
        default=None, description="On antihypertensive medication"
    )


class SMART2PatientData(BasePatientData):
    """
    Patient data for SMART2 (Secondary Manifestations of ARTerial disease).

    SMART2 estimates CVD risk for secondary prevention patients.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    egfr : Optional[float]
        Estimated GFR in ml/min/1.73m².
    hs_crp : Optional[float]
        High-sensitivity CRP in mg/L.
    years_since_diagnosis : Optional[int]
        Years since vascular disease diagnosis.
    vascular_disease_location : Optional[str]
        Location of vascular disease.
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    egfr: Optional[float] = Field(
        default=None, ge=5.0, le=200.0, description="eGFR in ml/min/1.73m²"
    )
    hs_crp: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="hs-CRP in mg/L"
    )
    years_since_diagnosis: Optional[int] = Field(
        default=None, ge=0, le=50, description="Years since vascular disease"
    )
    vascular_disease_location: Optional[str] = Field(
        default=None, description="Location of vascular disease"
    )
    region: Optional[str] = Field(
        default=None, description="CVD risk region (SCORE2: low, moderate, high, very_high; WHO: epidemiological subregions)"
    )
    has_cad: Optional[bool] = Field(default=None, description="History of coronary artery disease")
    has_cevd: Optional[bool] = Field(default=None, description="History of cerebrovascular disease")
    has_pad: Optional[bool] = Field(default=None, description="History of peripheral artery disease")
    has_aaa: Optional[bool] = Field(default=None, description="History of abdominal aortic aneurysm")


class SMARTREACHPatientData(SMART2PatientData):
    """
    Patient data for SMART-REACH model.

    SMART-REACH extends SMART2 with additional vascular event history.
    Inherits all fields from SMART2PatientData.

    Additional Fields
    -----------------
    polyvascular_disease : Optional[bool]
        Disease in multiple vascular beds.
    prior_mi : Optional[bool]
        Prior myocardial infarction.
    prior_stroke : Optional[bool]
        Prior ischemic stroke.
    """

    polyvascular_disease: Optional[bool] = Field(
        default=None, description="Disease in multiple vascular beds"
    )
    prior_mi: Optional[bool] = Field(
        default=None, description="Prior myocardial infarction"
    )
    prior_stroke: Optional[bool] = Field(
        default=None, description="Prior ischemic stroke"
    )
    creatinine: Optional[float] = Field(
        default=None, ge=0.1, le=20.0, description="Creatinine level in mg/dL"
    )


class LifeCVD2PatientData(BasePatientData):
    """
    Patient data for LIFE-CVD2 lifetime risk model.

    LIFE-CVD2 estimates lifetime CVD risk and life-years gained from treatment.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    family_history : Optional[bool]
        Family history of CVD.
    bmi : Optional[float]
        Body mass index in kg/m².
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    family_history: Optional[bool] = Field(
        default=None, description="Family history of CVD"
    )
    bmi: Optional[float] = Field(
        default=None, ge=10.0, le=60.0, description="BMI in kg/m²"
    )
    region: Optional[str] = Field(
        default=None, description="CVD risk region (SCORE2: low, moderate, high, very_high; WHO: epidemiological subregions)"
    )


class PreventPatientData(BasePatientData):
    """
    Patient data for PREVENT equations (AHA 2023).

    PREVENT estimates 10 and 30-year ASCVD, heart failure, and CVD risk.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    antihypertensive : Optional[bool]
        On antihypertensive medication.
    statin_use : Optional[bool]
        Current statin use.
    egfr : Optional[float]
        Estimated GFR in ml/min/1.73m².
    bmi : Optional[float]
        Body mass index in kg/m².
    hba1c : Optional[float]
        HbA1c in % (for diabetes-enhanced model).
    uacr : Optional[float]
        Urine albumin-to-creatinine ratio in mg/g.
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    antihypertensive: Optional[bool] = Field(
        default=None, description="On antihypertensive medication"
    )
    statin_use: Optional[bool] = Field(default=None, description="Current statin use")
    egfr: Optional[float] = Field(
        default=None, ge=5.0, le=200.0, description="eGFR in ml/min/1.73m²"
    )
    bmi: Optional[float] = Field(
        default=None, ge=10.0, le=60.0, description="BMI in kg/m²"
    )
    hba1c: Optional[float] = Field(
        default=None, ge=4.0, le=20.0, description="HbA1c in %"
    )
    uacr: Optional[float] = Field(
        default=None, ge=0.1, le=5000.0, description="UACR in mg/g"
    )


class DIAL2PatientData(BasePatientData):
    """
    Patient data for DIAL2 (Diabetes Lifetime-perspective prediction).

    DIAL2 estimates CVD risk specifically for type 2 diabetes patients.

    Additional Fields
    -----------------
    diabetes_duration : Optional[int]
        Duration of diabetes in years.
    hba1c : Optional[float]
        HbA1c in mmol/mol.
    egfr : Optional[float]
        Estimated GFR in ml/min/1.73m².
    acr : Optional[float]
        Albumin-to-creatinine ratio in mg/mmol.
    bmi : Optional[float]
        Body mass index in kg/m².
    antihypertensive : Optional[bool]
        On antihypertensive medication.
    lipid_lowering : Optional[bool]
        On lipid-lowering medication.
    """

    diabetes_duration: Optional[int] = Field(
        default=None, ge=0, le=80, description="Diabetes duration in years"
    )
    hba1c: Optional[float] = Field(
        default=None, ge=20.0, le=200.0, description="HbA1c in mmol/mol"
    )
    egfr: Optional[float] = Field(
        default=None, ge=5.0, le=200.0, description="eGFR in ml/min/1.73m²"
    )
    acr: Optional[float] = Field(
        default=None, ge=0.1, le=500.0, description="ACR in mg/mmol"
    )
    bmi: Optional[float] = Field(
        default=None, ge=10.0, le=60.0, description="BMI in kg/m²"
    )
    antihypertensive: Optional[bool] = Field(
        default=None, description="On antihypertensive medication"
    )
    lipid_lowering: Optional[bool] = Field(
        default=None, description="On lipid-lowering medication"
    )
    region: Optional[str] = Field(
        default=None, description="CVD risk region (SCORE2: low, moderate, high, very_high; WHO: epidemiological subregions)"
    )


class SingaporePatientData(BasePatientData):
    """
    Patient data for the Singapore-modified Framingham Risk Score 2023 (SG-FRS-2023).

    Additional Fields
    -----------------
    ethnicity : Literal["chinese", "malay", "indian"]
        Ethnic group; selects the SG-FRS-2023 risk table.
    antihypertensive : bool
        On blood pressure treatment (selects the treated SBP column).
    diabetes : Optional[bool]
        SG-FRS-2023 does not apply to people with diabetes.
    """

    ethnicity: Literal["chinese", "malay", "indian"] = Field(
        ..., description="Ethnic group: chinese, malay or indian"
    )
    antihypertensive: bool = Field(..., description="On antihypertensive treatment")
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
