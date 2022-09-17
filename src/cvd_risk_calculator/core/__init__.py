"""Core module for CVD risk calculator base classes and validation."""

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult

__all__ = [
    "RiskModel",
    "PatientData",
    "RiskResult",
]

