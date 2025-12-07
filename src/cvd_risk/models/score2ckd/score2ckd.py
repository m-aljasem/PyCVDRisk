"""
SCORE2-CKD (chronic kidney disease) cardiovascular risk model (placeholder).

This module sets up the structure for the SCORE2-CKD model tailored
to patients with chronic kidney disease. Implementation will be added later.
"""

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult


class SCORE2CKD(RiskModel):
    """Placeholder for the SCORE2-CKD cardiovascular risk model."""

    model_name = "SCORE2-CKD"
    model_version = "placeholder"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate risk for a single patient (not yet implemented)."""
        raise NotImplementedError("SCORE2-CKD model is not implemented yet.")

