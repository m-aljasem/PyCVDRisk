"""SCORE (2003), ESC: 10-year risk of fatal CVD."""

from cvd_risk.score.coefficients import REGIONS
from cvd_risk.score.score import score

__all__ = ["REGIONS", "score"]
