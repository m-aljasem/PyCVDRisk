"""
SCORE (Systematic COronary Risk Evaluation) cardiovascular risk model (placeholder).

The original SCORE model estimates 10-year risk of fatal cardiovascular
events. This file provides the scaffolding for a future implementation.
"""

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult


class SCORE(RiskModel):
    """Placeholder for the SCORE cardiovascular risk model."""

    model_name = "SCORE"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("SCORE model is not implemented yet.")

