"""uniscore: route each person to their country's guideline CVD risk score."""

from cvd_risk.uniscore.uniscore import SCORES, guideline_score, uniscore

__all__ = ["SCORES", "guideline_score", "uniscore"]
