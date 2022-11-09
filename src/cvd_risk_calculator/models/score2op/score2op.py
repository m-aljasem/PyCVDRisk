"""
SCORE2-OP (older persons) cardiovascular risk model (placeholder).

This module provides the scaffolding for the SCORE2-OP model that
extends SCORE2 to older populations. Full implementation will follow.
"""

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult


class SCORE2OP(RiskModel):
    """Placeholder for the SCORE2-OP cardiovascular risk model."""

    model_name = "SCORE2-OP"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("SCORE2-OP model is not implemented yet.")

