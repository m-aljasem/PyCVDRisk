"""Core module for CVD risk calculator base classes and validation."""

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

# Import base data class
from cvd_risk.core.data_classes.base import BasePatientData

# Import model-specific data classes
from cvd_risk.core.data_classes.score_family import (
    SCORE2PatientData,
    SCOREPatientData,
    SCORE2OPPatientData,
    SCORE2DMPatientData,
    SCORE2CKDPatientData,
    SCORE2AsiaPatientData,
)
from cvd_risk.core.data_classes.qrisk_family import (
    QRISK2PatientData,
    QRISK3PatientData,
)
from cvd_risk.core.data_classes.framingham_family import (
    FraminghamPatientData,
    ASCVDPatientData,
    ReynoldsPatientData,
)
from cvd_risk.core.data_classes.acute_care import (
    TIMIPatientData,
    GRACE2PatientData,
    EDACSPatientData,
    HEARTPatientData,
)
from cvd_risk.core.data_classes.who_family import (
    WHOPatientData,
    GloboriskPatientData,
)
from cvd_risk.core.data_classes.other_models import (
    ASSIGNPatientData,
    PROCAMPatientData,
    REGICORPatientData,
    SMART2PatientData,
    SMARTREACHPatientData,
    LifeCVD2PatientData,
    PreventPatientData,
    DIAL2PatientData,
    SimplePatientData,
    SingaporePatientData,
)

__all__ = [
    # Core classes
    "RiskModel",
    "RiskResult",
    # Backward compatible PatientData
    "PatientData",
    # Base data class
    "BasePatientData",
    # SCORE family
    "SCORE2PatientData",
    "SCOREPatientData",
    "SCORE2OPPatientData",
    "SCORE2DMPatientData",
    "SCORE2CKDPatientData",
    "SCORE2AsiaPatientData",
    # QRISK family
    "QRISK2PatientData",
    "QRISK3PatientData",
    # Framingham family
    "FraminghamPatientData",
    "ASCVDPatientData",
    "ReynoldsPatientData",
    # Acute care
    "TIMIPatientData",
    "GRACE2PatientData",
    "EDACSPatientData",
    "HEARTPatientData",
    # WHO family
    "WHOPatientData",
    "GloboriskPatientData",
    # Other models
    "ASSIGNPatientData",
    "PROCAMPatientData",
    "REGICORPatientData",
    "SMART2PatientData",
    "SMARTREACHPatientData",
    "LifeCVD2PatientData",
    "PreventPatientData",
    "DIAL2PatientData",
    "SimplePatientData",
    "SingaporePatientData",
]
