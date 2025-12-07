"""
SCORE2-Diabetes Mellitus (SCORE2-DM) cardiovascular risk model (placeholder).

This module sets up the structure for the SCORE2-DM model. The detailed
algorithm will be added in a future update.
"""

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult


class SCORE2DM(RiskModel):
    """Placeholder for the SCORE2-DM cardiovascular risk model."""

    model_name = "SCORE2-DM"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("SCORE2-DM model is not implemented yet.")

