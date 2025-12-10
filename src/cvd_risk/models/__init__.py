"""CVD risk prediction models."""

# Primary prevention models
from cvd_risk.models.ascvd import ASCVD
from cvd_risk.models.framingham import Framingham
from cvd_risk.models.globorisk import Globorisk
from cvd_risk.models.prevent import Prevent
from cvd_risk.models.procam import PROCAM
from cvd_risk.models.reynolds import Reynolds
from cvd_risk.models.finrisk import FINRISK
from cvd_risk.models.regicor import REGICOR
from cvd_risk.models.progetto_cuore import ProgettoCUORE
from cvd_risk.models.risc_score import RISC_Score
from cvd_risk.models.aric_update import ARIC_Update
from cvd_risk.models.jackson_heart import JacksonHeart
from cvd_risk.models.cardia import CARDIA
from cvd_risk.models.rotterdam_study import RotterdamStudy
from cvd_risk.models.heinz_nixdorf import HeinzNixdorf
from cvd_risk.models.epic_norfolk import EPIC_Norfolk
from cvd_risk.models.singapore import Singapore
from cvd_risk.models.predict import PREDICT
from cvd_risk.models.new_zealand import NewZealand
from cvd_risk.models.dundee import Dundee
from cvd_risk.models.malaysian_cvd import Malaysian_CVD
from cvd_risk.models.gulf_race import Gulf_RACE
from cvd_risk.models.cambridge import Cambridge
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

# HIV-specific models
from cvd_risk.models.dad_score import DAD_Score

# CKD-specific models
from cvd_risk.models.score2ckd import SCORE2CKD
from cvd_risk.models.score2op import SCORE2OP

# Region-specific models
from cvd_risk.models.assign import ASSIGN  # Scotland-specific
from cvd_risk.models.score2asia import SCORE2AsiaCKD  # Asia-specific

# Lifetime risk models
from cvd_risk.models.lifecvd2 import LifeCVD2

# Global risk scores
from cvd_risk.models.interheart import INTERHEART

# Acute coronary syndrome models
from cvd_risk.models.grace2 import GRACE2
from cvd_risk.models.timi import TIMI

# Emergency department models
from cvd_risk.models.edacs import EDACS
from cvd_risk.models.heart import HEART

# Atrial fibrillation models
from cvd_risk.models.chads2 import CHADS2
from cvd_risk.models.cha2ds2_vasc import CHA2DS2_VASc

# Bleeding risk models
from cvd_risk.models.has_bled import HAS_BLED

__all__ = [
    # Primary prevention
    "ASCVD",
    "Framingham",
    "Globorisk",
    "Prevent",
    "PROCAM",
    "Reynolds",
    "FINRISK",
    "REGICOR",
    "ProgettoCUORE",
    "RISC_Score",
    "ARIC_Update",
    "JacksonHeart",
    "CARDIA",
    "RotterdamStudy",
    "HeinzNixdorf",
    "EPIC_Norfolk",
    "Singapore",
    "PREDICT",
    "NewZealand",
    "Dundee",
    "Malaysian_CVD",
    "Gulf_RACE",
    "Cambridge",
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
    # HIV-specific
    "DAD_Score",
    # CKD-specific
    "SCORE2CKD",
    "SCORE2OP",
    # Region-specific
    "ASSIGN",
    "SCORE2AsiaCKD",
    # Lifetime risk
    "LifeCVD2",
    # Global risk scores
    "INTERHEART",
    # Acute coronary syndrome
    "GRACE2",
    "TIMI",
    # Emergency department
    "EDACS",
    "HEART",
    # Atrial fibrillation
    "CHADS2",
    "CHA2DS2_VASc",
    # Bleeding risk
    "HAS_BLED",
]

