"""2019 WHO CVD risk models (laboratory and non-laboratory)."""

from cvd_risk.who2019.coefficients import REGIONS
from cvd_risk.who2019.who2019 import who2019

__all__ = ["REGIONS", "who2019"]
