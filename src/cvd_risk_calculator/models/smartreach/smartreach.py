"""
SMART-REACH cardiovascular risk model (placeholder).

This module defines the structure for the SMART-REACH model. Detailed
calculation logic will be provided in a future release.
"""

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult


class SmartReach(RiskModel):
    """Placeholder for the SMART-REACH cardiovascular risk model."""

    model_name = "SMART-REACH"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("SMART-REACH model is not implemented yet.")

