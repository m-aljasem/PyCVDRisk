"""Verified, vectorised implementations of cardiovascular risk scores."""

from cvd_risk.globorisk import globorisk
from cvd_risk.score2 import score2, score2_diabetes, score2_op

__all__ = ["globorisk", "score2", "score2_op", "score2_diabetes"]
__version__ = "2.0.0.dev0"
