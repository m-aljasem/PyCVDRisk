"""CVD risk prediction models."""

# Primary prevention models
from cvd_risk.models.ascvd import ASCVD
from cvd_risk.models.framingham import Framingham
from cvd_risk.models.globorisk import Globorisk
from cvd_risk.models.prevent import Prevent
from cvd_risk.models.qrisk2 import QRISK2
from cvd_risk.models.qrisk3 import QRISK3
from cvd_risk.models.score import SCORE
from cvd_risk.models.score2 import SCORE2
from cvd_risk.models.who import WHO

# Secondary prevention models
from cvd_risk.models.smart2 import SMART2
from cvd_risk.models.smartreach import SMART_REACH

# Diabetes-specific models
from cvd_risk.models.dial2 import DIAL2
from cvd_risk.models.score2dm import SCORE2DM

# CKD-specific models
from cvd_risk.models.score2ckd import SCORE2CKD
from cvd_risk.models.score2op import SCORE2OP

# Region-specific models
from cvd_risk.models.assign import ASSIGN  # Scotland-specific
from cvd_risk.models.score2asia import SCORE2AsiaCKD  # Asia-specific

# Lifetime risk models
from cvd_risk.models.lifecvd2 import LifeCVD2

# Acute coronary syndrome models
from cvd_risk.models.grace2 import GRACE2
from cvd_risk.models.timi import TIMI

# Emergency department models
from cvd_risk.models.edacs import EDACS
from cvd_risk.models.heart import HEART

__all__ = [
    # Primary prevention
    "ASCVD",
    "Framingham",
    "Globorisk",
    "Prevent",
    "QRISK2",
    "QRISK3",
    "SCORE",
    "SCORE2",
    "WHO",
    # Secondary prevention
    "SMART2",
    "SMART_REACH",
    # Diabetes-specific
    "DIAL2",
    "SCORE2DM",
    # CKD-specific
    "SCORE2CKD",
    "SCORE2OP",
    # Region-specific
    "ASSIGN",
    "SCORE2AsiaCKD",
    # Lifetime risk
    "LifeCVD2",
    # Acute coronary syndrome
    "GRACE2",
    "TIMI",
    # Emergency department
    "EDACS",
    "HEART",
]

