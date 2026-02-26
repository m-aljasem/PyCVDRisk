"""Suita score (Japan, 2014: 10-year coronary heart disease) and Suita CVD risk model (2020: CHD or stroke)."""

from cvd_risk.suita.suita import bp_category, suita, suita_risk
from cvd_risk.suita.suita_cvd import suita_cvd, suita_cvd_risk

__all__ = ["bp_category", "suita", "suita_cvd", "suita_cvd_risk", "suita_risk"]
