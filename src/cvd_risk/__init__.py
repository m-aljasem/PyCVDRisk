"""
CVD Risk Calculator: Compatibility import layer.

Import from cvd_risk instead of cvd_risk_calculator for convenience.
"""

# Import everything from cvd_risk_calculator for compatibility
import cvd_risk_calculator
from cvd_risk_calculator import *

__all__ = cvd_risk_calculator.__all__
