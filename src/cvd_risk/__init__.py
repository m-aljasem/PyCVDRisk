"""
CVD Risk Calculator: Production-grade cardiovascular risk assessment package.

This package provides implementations of major cardiovascular disease risk
prediction models for epidemiological research and clinical decision support.
"""

__version__ = "0.1.0"
__author__ = "CVD Risk Calculator Contributors"
__license__ = "MIT"

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

# Import all models for easy access
from cvd_risk.models import (
    # Primary prevention models
    ASCVD,
    Framingham,
    Globorisk,
    Prevent,
    QRISK2,
    QRISK3,
    SCORE,
    SCORE2,
    WHO,
    # Secondary prevention models
    SMART2,
    SMART_REACH,
    # Diabetes-specific models
    DIAL2,
    SCORE2DM,
    # CKD-specific models
    SCORE2CKD,
    SCORE2OP,
    # Region-specific models
    ASSIGN,
    SCORE2AsiaCKD,
    # Lifetime risk models
    LifeCVD2,
    # Acute coronary syndrome models
    GRACE2,
    TIMI,
    # Emergency department models
    EDACS,
    HEART,
)

__all__ = [
    "RiskModel",
    "PatientData",
    "RiskResult",
    # Primary prevention
    "ASCVD",
    "Framingham",
    "Globorisk",
    "Prevent",
    "QRISK2",
    "QRISK3",
    "SCORE",
    "SCORE2",
    "WHO",
    # Secondary prevention
    "SMART2",
    "SMART_REACH",
    # Diabetes-specific
    "DIAL2",
    "SCORE2DM",
    # CKD-specific
    "SCORE2CKD",
    "SCORE2OP",
    # Region-specific
    "ASSIGN",
    "SCORE2AsiaCKD",
    # Lifetime risk
    "LifeCVD2",
    # Acute coronary syndrome
    "GRACE2",
    "TIMI",
    # Emergency department
    "EDACS",
    "HEART",
]

