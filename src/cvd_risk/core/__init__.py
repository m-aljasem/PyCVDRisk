"""Core module for CVD risk calculator base classes and validation."""

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

__all__ = [
    "RiskModel",
    "PatientData",
    "RiskResult",
]

