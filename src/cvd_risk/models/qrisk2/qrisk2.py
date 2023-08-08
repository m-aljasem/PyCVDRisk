"""
QRISK2 cardiovascular risk model (placeholder).

This module provides the scaffolding for the QRISK2 model. The full
algorithmic implementation will be added later.
"""

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult


class QRISK2(RiskModel):
    """Placeholder for the QRISK2 cardiovascular risk model."""

    model_name = "QRISK2"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("QRISK2 model is not implemented yet.")

