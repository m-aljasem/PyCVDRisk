"""CVD risk prediction models."""

from cvd_risk_calculator.models.ascvd import ASCVD
from cvd_risk_calculator.models.framingham import Framingham
from cvd_risk_calculator.models.globorisk import Globorisk
from cvd_risk_calculator.models.qrisk3 import QRISK3
from cvd_risk_calculator.models.score2 import SCORE2
from cvd_risk_calculator.models.smart2 import SMART2
from cvd_risk_calculator.models.who import WHO

__all__ = [
    "SCORE2",
    "Framingham",
    "ASCVD",
    "QRISK3",
    "SMART2",
    "WHO",
    "Globorisk",
]

