"""
Patient data classes for acute care and emergency models.

Includes data classes for TIMI, GRACE2, EDACS, and HEART models.
"""

from typing import Literal, Optional

from pydantic import Field

from cvd_risk.core.data_classes.base import BasePatientData


class TIMIPatientData(BasePatientData):
    """
    Patient data for TIMI Risk Score (UA/NSTEMI).

    TIMI stratifies risk in patients with unstable angina and NSTEMI.

    Additional Fields
    -----------------
    hypertension : Optional[bool]
        Hypertension diagnosis.
    hyperlipidaemia : Optional[bool]
        Hyperlipidaemia diagnosis.
    diabetes : Optional[bool]
        Diabetes status.
    family_history : Optional[bool]
        Family history of CAD.
    previous_pci : Optional[bool]
        Previous percutaneous coronary intervention.
    previous_cabg : Optional[bool]
        Previous coronary artery bypass grafting.
    aspirin_use : Optional[bool]
        Aspirin use in last 7 days.
    angina_episodes_24h : Optional[int]
        Number of angina episodes in 24 hours.
    ecg_st_depression : Optional[bool]
        ECG ST segment depression >= 0.5mm.
    troponin_level : Optional[float]
        Troponin level (ng/mL).
    """

    hypertension: Optional[bool] = Field(
        default=None, description="Hypertension diagnosis"
    )
    hyperlipidaemia: Optional[bool] = Field(
        default=None, description="Hyperlipidaemia diagnosis"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    family_history: Optional[bool] = Field(
        default=None, description="Family history of CAD"
    )
    previous_pci: Optional[bool] = Field(default=None, description="Previous PCI")
    previous_cabg: Optional[bool] = Field(default=None, description="Previous CABG")
    aspirin_use: Optional[bool] = Field(
        default=None, description="Aspirin use in last 7 days"
    )
    angina_episodes_24h: Optional[int] = Field(
        default=None, ge=0, le=50, description="Angina episodes in 24h"
    )
    ecg_st_depression: Optional[bool] = Field(
        default=None, description="ST depression >= 0.5mm"
    )
    troponin_level: Optional[float] = Field(
        default=None, ge=0.0, le=1000.0, description="Troponin in ng/mL"
    )


class GRACE2PatientData(BasePatientData):
    """
    Patient data for GRACE2 model (Acute Coronary Syndromes).

    GRACE2 estimates mortality risk in ACS patients.

    Additional Fields
    -----------------
    heart_rate : Optional[float]
        Heart rate in beats per minute.
    creatinine : Optional[float]
        Serum creatinine in mg/dL.
    killip_class : Optional[Literal[1, 2, 3, 4]]
        Killip classification for heart failure.
    cardiac_arrest : Optional[bool]
        Cardiac arrest at presentation.
    ecg_st_depression : Optional[bool]
        ST segment depression on ECG.
    troponin_level : Optional[float]
        Troponin level (ng/mL).
    """

    heart_rate: Optional[float] = Field(
        default=None, ge=30.0, le=300.0, description="Heart rate in bpm"
    )
    creatinine: Optional[float] = Field(
        default=None, ge=0.1, le=20.0, description="Creatinine in mg/dL"
    )
    killip_class: Optional[Literal[1, 2, 3, 4]] = Field(
        default=None, description="Killip class: 1=No CHF, 2=Rales, 3=Edema, 4=Shock"
    )
    cardiac_arrest: Optional[bool] = Field(
        default=None, description="Cardiac arrest at presentation"
    )
    ecg_st_depression: Optional[bool] = Field(
        default=None, description="ST depression on ECG"
    )
    troponin_level: Optional[float] = Field(
        default=None, ge=0.0, le=1000.0, description="Troponin in ng/mL"
    )


class EDACSPatientData(BasePatientData):
    """
    Patient data for EDACS (Emergency Department Assessment of Chest pain Score).

    EDACS helps identify low-risk chest pain patients in the ED.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    hypertension : Optional[bool]
        Hypertension diagnosis.
    hyperlipidaemia : Optional[bool]
        Hyperlipidaemia diagnosis.
    family_history : Optional[bool]
        Family history of CAD.
    sweating : Optional[bool]
        Diaphoresis symptom.
    pain_radiation : Optional[bool]
        Pain radiating to arm/shoulder/neck/jaw.
    pleuritic : Optional[bool]
        Pain with inspiration.
    palpation : Optional[bool]
        Pain reproduced by palpation.
    ecg_st_depression : Optional[bool]
        ST depression on ECG.
    ecg_twi : Optional[bool]
        T-wave inversion on ECG.
    presentation_hstni : Optional[float]
        Initial high-sensitivity troponin I.
    second_hstni : Optional[float]
        Second high-sensitivity troponin I.
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    hypertension: Optional[bool] = Field(default=None, description="Hypertension")
    hyperlipidaemia: Optional[bool] = Field(
        default=None, description="Hyperlipidaemia"
    )
    family_history: Optional[bool] = Field(
        default=None, description="Family history of CAD"
    )
    sweating: Optional[bool] = Field(default=None, description="Diaphoresis")
    pain_radiation: Optional[bool] = Field(
        default=None, description="Pain radiation to arm/shoulder/neck/jaw"
    )
    pleuritic: Optional[bool] = Field(
        default=None, description="Pain with inspiration"
    )
    palpation: Optional[bool] = Field(
        default=None, description="Pain reproduced by palpation"
    )
    ecg_st_depression: Optional[bool] = Field(
        default=None, description="ST depression"
    )
    ecg_twi: Optional[bool] = Field(default=None, description="T-wave inversion")
    presentation_hstni: Optional[float] = Field(
        default=None, ge=0.0, le=50000.0, description="Initial hs-TnI"
    )
    second_hstni: Optional[float] = Field(
        default=None, ge=0.0, le=50000.0, description="Second hs-TnI"
    )


class HEARTPatientData(BasePatientData):
    """
    Patient data for HEART Score.

    HEART Score stratifies chest pain patients for major adverse cardiac events.

    Additional Fields
    -----------------
    diabetes : Optional[bool]
        Diabetes status.
    hypertension : Optional[bool]
        Hypertension diagnosis.
    hyperlipidaemia : Optional[bool]
        Hyperlipidaemia diagnosis.
    family_history : Optional[bool]
        Family history of CAD.
    atherosclerotic_disease : Optional[bool]
        Known atherosclerotic disease.
    typical_symptoms_num : Optional[int]
        Number of typical cardiac symptoms (0-6).
    ecg_normal : Optional[bool]
        ECG normal (no significant abnormalities).
    abn_repolarisation : Optional[bool]
        Abnormal repolarization on ECG.
    ecg_st_depression : Optional[bool]
        ST depression on ECG.
    presentation_hstni : Optional[float]
        High-sensitivity troponin I at presentation.
    """

    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    hypertension: Optional[bool] = Field(default=None, description="Hypertension")
    hyperlipidaemia: Optional[bool] = Field(
        default=None, description="Hyperlipidaemia"
    )
    family_history: Optional[bool] = Field(
        default=None, description="Family history of CAD"
    )
    atherosclerotic_disease: Optional[bool] = Field(
        default=None, description="Known atherosclerotic disease"
    )
    typical_symptoms_num: Optional[int] = Field(
        default=None, ge=0, le=6, description="Number of typical symptoms"
    )
    ecg_normal: Optional[bool] = Field(default=None, description="ECG normal")
    abn_repolarisation: Optional[bool] = Field(
        default=None, description="Abnormal ECG repolarization"
    )
    ecg_st_depression: Optional[bool] = Field(
        default=None, description="ST depression on ECG"
    )
    presentation_hstni: Optional[float] = Field(
        default=None, ge=0.0, le=50000.0, description="hs-TnI at presentation"
    )
