"""
QRISK cardiovascular risk model (placeholder).

This module defines the structure for the QRISK model. Implementation
details will be added in a future release.
"""

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult


class QRISK(RiskModel):
    """Placeholder for the QRISK cardiovascular risk model."""

    model_name = "QRISK"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("QRISK model is not implemented yet.")

