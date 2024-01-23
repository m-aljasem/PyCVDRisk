"""Input validation models for patient data using Pydantic.

DEPRECATION NOTICE: The monolithic PatientData class is maintained for backward
compatibility. New code should use the model-specific data classes from
cvd_risk.core.data_classes (e.g., SCORE2PatientData, QRISK3PatientData).
"""

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cvd_risk.core.data_classes.base import BasePatientData


class PatientData(BasePatientData):
    """
    DEPRECATED: Use model-specific data classes instead.
    
    This class is maintained for backward compatibility. For new code,
    import from cvd_risk.core.data_classes:
    
    - SCORE2PatientData for SCORE2 models
    - QRISK3PatientData for QRISK models
    - FraminghamPatientData for Framingham
    - etc.
    
    Validated patient data structure for CVD risk calculation.

    This model ensures type safety and value validation for patient
    inputs across all risk models. All values are validated against
    clinical ranges.

    Attributes
    ----------
    age : int
        Patient age in years. Typical range: 40-100 years.
    sex : Literal["male", "female"]
        Patient biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg. Typical range: 90-250 mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L. Typical range: 2.5-10.0 mmol/L.
    hdl_cholesterol : float
        HDL cholesterol in mmol/L. Typical range: 0.5-3.5 mmol/L.
    smoking : bool
        Current smoking status.
    region : Optional[str]
        CVD risk region (for SCORE2: low, moderate, high, very_high;
        for WHO: AFR_D, AFR_E, AMR_A, AMR_B, AMR_D, EMR_B, EMR_D,
        EUR_A, EUR_B, EUR_C, SEAR_B, SEAR_D, WPR_A, WPR_B).
        None if not applicable to the model.
    diabetes : Optional[bool]
        Diabetes status. None if not applicable to the model.
    ethnicity : Optional[Literal["white", "south_asian", "black", "chinese", "mixed", "other"]]
        Ethnicity (for QRISK3). None if not applicable.
    bmi : Optional[float]
        Body mass index in kg/m². None if not applicable.
    family_history : Optional[bool]
        Family history of CVD. None if not applicable.
    antihypertensive : Optional[bool]
        Use of antihypertensive medication. None if not applicable.
    statin_use : Optional[bool]
        Current statin use. None if not applicable.

    Examples
    --------
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,
    ...     hdl_cholesterol=1.2,
    ...     smoking=True,
    ...     region="moderate"
    ... )
    """

    # Extended fields beyond BasePatientData for backward compatibility
    region: Optional[str] = Field(
        default=None, description="CVD risk region (SCORE2: low, moderate, high, very_high; WHO: epidemiological subregions)"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    ethnicity: Optional[
        Literal["white", "south_asian", "black", "chinese", "mixed", "other"]
    ] = Field(default=None, description="Ethnicity for QRISK3")
    bmi: Optional[float] = Field(
        default=None, ge=10.0, le=60.0, description="Body mass index in kg/m²"
    )
    family_history: Optional[bool] = Field(
        default=None, description="Family history of CVD"
    )
    antihypertensive: Optional[bool] = Field(
        default=None, description="Use of antihypertensive medication"
    )
    statin_use: Optional[bool] = Field(
        default=None, description="Current statin use"
    )
    race: Optional[Literal["white", "black", "other"]] = Field(
        default=None, description="Race for ASCVD model (white, black, other)"
    )
    # TIMI-specific fields
    hypertension: Optional[bool] = Field(
        default=None, description="Hypertension diagnosis"
    )
    hyperlipidaemia: Optional[bool] = Field(
        default=None, description="Hyperlipidaemia diagnosis"
    )
    previous_pci: Optional[bool] = Field(
        default=None, description="Previous percutaneous coronary intervention"
    )
    previous_cabg: Optional[bool] = Field(
        default=None, description="Previous coronary artery bypass grafting"
    )
    aspirin_use: Optional[bool] = Field(
        default=None, description="Current aspirin use"
    )
    angina_episodes_24h: Optional[int] = Field(
        default=None, ge=0, le=50, description="Number of angina episodes in 24 hours"
    )
    ecg_st_depression: Optional[bool] = Field(
        default=None, description="ECG ST segment depression"
    )
    troponin_level: Optional[float] = Field(
        default=None, ge=0.0, le=1000.0, description="Troponin level (ng/mL)"
    )
    # QRISK3-specific fields
    atrial_fibrillation: Optional[bool] = Field(
        default=None, description="Atrial fibrillation (AF)"
    )
    atypical_antipsychotics: Optional[bool] = Field(
        default=None, description="Use of atypical antipsychotics"
    )
    impotence2: Optional[bool] = Field(
        default=None, description="Erectile dysfunction (males only)"
    )
    corticosteroids: Optional[bool] = Field(
        default=None, description="Systemic corticosteroids"
    )
    migraine: Optional[bool] = Field(
        default=None, description="Migraine"
    )
    rheumatoid_arthritis: Optional[bool] = Field(
        default=None, description="Rheumatoid arthritis"
    )
    renal_disease: Optional[bool] = Field(
        default=None, description="Severe renal disease"
    )
    systemic_lupus: Optional[bool] = Field(
        default=None, description="Systemic lupus erythematosus"
    )
    severe_mental_illness: Optional[bool] = Field(
        default=None, description="Severe mental illness"
    )
    treated_hypertension: Optional[bool] = Field(
        default=None, description="Treated hypertension"
    )
    type1_diabetes: Optional[bool] = Field(
        default=None, description="Type 1 diabetes"
    )
    type2_diabetes: Optional[bool] = Field(
        default=None, description="Type 2 diabetes"
    )
    diabetes_age: Optional[int] = Field(
        default=None, ge=1, le=120, description="Age when diabetes was diagnosed (years)"
    )
    hba1c: Optional[float] = Field(
        default=None, ge=10.0, le=200.0, description="Glycated hemoglobin (HbA1c) in mmol/mol"
    )
    egfr: Optional[float] = Field(
        default=None, ge=5.0, le=200.0, description="Estimated glomerular filtration rate (eGFR) in ml/min/1.73m²"
    )
    acr: Optional[float] = Field(
        default=None, ge=0.1, le=2000.0, description="Albumin-to-creatinine ratio (ACR) in mg/g"
    )
    proteinuria_trace: Optional[Literal["negative", "trace", "1+", "2+", "3+", "4+"]] = Field(
        default=None, description="Proteinuria dipstick test result"
    )
    sbps5: Optional[float] = Field(
        default=None, ge=0.0, le=50.0, description="Systolic BP variability (SD of 5 readings)"
    )
    smoking_category: Optional[Literal[1, 2, 3, 4]] = Field(
        default=None, description="Smoking category: 1=non-smoker, 2=ex-smoker, 3=light, 4=heavy"
    )
    townsend_deprivation: Optional[float] = Field(
        default=None, ge=-10.0, le=10.0, description="Townsend deprivation score"
    )
    # EDACS-specific fields
    sweating: Optional[bool] = Field(
        default=None, description="Sweating (diaphoresis) symptom"
    )
    pain_radiation: Optional[bool] = Field(
        default=None, description="Pain radiates to arm, shoulder, neck, or jaw"
    )
    pleuritic: Optional[bool] = Field(
        default=None, description="Pain occurred or worsened with inspiration"
    )
    palpation: Optional[bool] = Field(
        default=None, description="Pain is reproduced by palpation"
    )
    ecg_twi: Optional[bool] = Field(
        default=None, description="ECG shows T-wave inversion"
    )
    presentation_hstni: Optional[float] = Field(
        default=None, ge=0.0, le=50000.0, description="Presentation high-sensitivity troponin I level"
    )
    second_hstni: Optional[float] = Field(
        default=None, ge=0.0, le=50000.0, description="Second high-sensitivity troponin I level"
    )
    # HEART-specific fields
    typical_symptoms_num: Optional[int] = Field(
        default=None, ge=0, le=6, description="Number of typical cardiac symptoms (0-6)"
    )
    ecg_normal: Optional[bool] = Field(
        default=None, description="ECG normal (no significant abnormalities)"
    )
    abn_repolarisation: Optional[bool] = Field(
        default=None, description="Abnormal repolarization on ECG"
    )
    atherosclerotic_disease: Optional[bool] = Field(
        default=None, description="Known atherosclerotic disease (CAD, MI, stroke, PAD, etc.)"
    )
    # GRACE2-specific fields
    heart_rate: Optional[float] = Field(
        default=None, ge=30.0, le=300.0, description="Heart rate in beats per minute"
    )
    creatinine: Optional[float] = Field(
        default=None, ge=0.1, le=20.0, description="Creatinine level in mg/dL"
    )
    killip_class: Optional[Literal[1, 2, 3, 4]] = Field(
        default=None, description="Killip classification: 1=No CHF, 2=Rales/JVD, 3=Pulmonary edema, 4=Cardiogenic shock"
    )
    cardiac_arrest: Optional[bool] = Field(
        default=None, description="Cardiac arrest at presentation"
    )

    # CHADS2 / CHA2DS2-VASc / HAS-BLED fields
    congestive_heart_failure: Optional[bool] = Field(
        default=None, description="Congestive heart failure / LV dysfunction"
    )
    prior_stroke_tia: Optional[bool] = Field(
        default=None, description="Prior stroke, TIA or thromboembolism"
    )
    vascular_disease: Optional[bool] = Field(
        default=None, description="Vascular disease (prior MI, PAD or aortic plaque)"
    )
    abnormal_renal_function: Optional[bool] = Field(
        default=None, description="Abnormal renal function (HAS-BLED)"
    )
    abnormal_liver_function: Optional[bool] = Field(
        default=None, description="Abnormal liver function (HAS-BLED)"
    )
    bleeding_history: Optional[bool] = Field(
        default=None, description="Prior major bleeding or predisposition"
    )
    labile_inr: Optional[bool] = Field(
        default=None, description="Labile INR (time in therapeutic range < 60%)"
    )
    antiplatelet_drugs: Optional[bool] = Field(
        default=None, description="Antiplatelet or NSAID use"
    )
    alcohol_abuse: Optional[bool] = Field(
        default=None, description="Alcohol use >= 8 drinks/week"
    )
    # Note: validate_cholesterol and validate_hdl are inherited from BasePatientData

    model_config = ConfigDict(frozen=True, validate_assignment=True)


class RiskResult(BaseModel):
    """
    Risk calculation result with metadata.

    Attributes
    ----------
    risk_score : float
        Calculated risk score (typically percentage).
    risk_category : str
        Categorical risk classification (e.g., "low", "moderate", "high").
    confidence_interval_lower : Optional[float]
        Lower bound of 95% confidence interval if available.
    confidence_interval_upper : Optional[float]
        Upper bound of 95% confidence interval if available.
    model_name : str
        Name of the risk model used.
    model_version : str
        Version identifier of the model algorithm.
    calculation_metadata : dict
        Additional metadata about the calculation.
    """

    risk_score: float = Field(..., description="Calculated risk score (percentage)")
    risk_category: str = Field(..., description="Risk category classification")
    confidence_interval_lower: Optional[float] = Field(
        default=None, description="Lower 95% CI bound"
    )
    confidence_interval_upper: Optional[float] = Field(
        default=None, description="Upper 95% CI bound"
    )
    model_name: str = Field(..., description="Name of risk model")
    model_version: str = Field(..., description="Model version/algorithm identifier")
    calculation_metadata: dict = Field(
        default_factory=dict, description="Additional calculation metadata"
    )

    model_config = ConfigDict(frozen=True)


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """
    Validate pandas DataFrame has required columns for risk calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with patient data.
    required_columns : list[str]
        List of required column names.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df

