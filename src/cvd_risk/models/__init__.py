"""CVD risk prediction models."""

from cvd_risk.models.ascvd import ASCVD
from cvd_risk.models.framingham import Framingham
from cvd_risk.models.globorisk import Globorisk
from cvd_risk.models.qrisk3 import QRISK3
from cvd_risk.models.score2 import SCORE2
from cvd_risk.models.smart2 import SMART2
from cvd_risk.models.who import WHO

__all__ = [
    "SCORE2",
    "Framingham",
    "ASCVD",
    "QRISK3",
    "SMART2",
    "WHO",
    "Globorisk",
]

