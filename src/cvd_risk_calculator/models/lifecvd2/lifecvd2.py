"""
LifeCVD2 cardiovascular risk model (placeholder).

This module provides the scaffolding for the LifeCVD2 model. The
implementation will be added in a future iteration.
"""

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult


class LifeCVD2(RiskModel):
    """Placeholder for the LifeCVD2 cardiovascular risk model."""

    model_name = "LifeCVD2"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("LifeCVD2 model is not implemented yet.")

