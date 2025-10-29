"""
CVD Risk Calculator: Production-grade cardiovascular risk assessment package.

This package provides implementations of major cardiovascular disease risk
prediction models for epidemiological research and clinical decision support.
"""

__version__ = "0.1.0"
__author__ = "CVD Risk Calculator Contributors"
__license__ = "MIT"

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult

__all__ = [
    "RiskModel",
    "PatientData",
    "RiskResult",
]

