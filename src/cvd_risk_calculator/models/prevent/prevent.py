"""
PREVENT cardiovascular risk model (placeholder).

This module defines the PREVENT model scaffolding. Full algorithmic
implementation will be added later.
"""

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult


class Prevent(RiskModel):
    """Placeholder for the PREVENT cardiovascular risk model."""

    model_name = "PREVENT"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("PREVENT model is not implemented yet.")

