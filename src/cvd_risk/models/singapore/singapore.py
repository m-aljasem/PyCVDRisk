"""
Singapore-modified Framingham Risk Score 2023 (SG-FRS-2023).

Points-based 10-year coronary risk (non-fatal MI or coronary death) for
Chinese, Malay and Indian adults aged 20-79 in Singapore, recalibrated from
the NCEP ATP III Framingham tables.

Not applicable to people with cardiovascular disease, familial
hypercholesterolaemia, diabetes mellitus, or CKD (eGFR <60 and/or ACR >=3 mg/mmol).
Does not predict stroke or total ASCVD.

Reference:
    Agency for Care Effectiveness (ACE), Singapore. Additional resource for
    CV risk assessment using SG-FRS-2023 (accompanies ACE Clinical Guideline
    "Lipid management: focus on cardiovascular risk", 15 Dec 2023),
    Tables A1-A4.
    Lim CGY et al. Recalibrated Singapore-Modified Framingham Risk Score 2023.
Risk categories: ACE Clinical Guideline "Lipid management: focus on
    cardiovascular risk" (Dec 2023): low <5%, borderline 5-<10%,
    intermediate 10-20%, high >20%.
"""

import bisect

from cvd_risk.core.base import RiskModel
from cvd_risk.core.data_classes import SingaporePatientData
from cvd_risk.core.validation import RiskResult

# Tables A1 (men) and A3 (women). Age bands use lower bounds.
_AGE_BANDS = [20, 35, 40, 45, 50, 55, 60, 65, 70, 75]
_AGE_POINTS = {
    "male": [-9, -4, 0, 3, 6, 8, 10, 11, 12, 13],
    "female": [-7, -3, 0, 3, 6, 8, 10, 12, 14, 16],
}

# Columns: age 20-39, 40-49, 50-59, 60-69, 70-79
_DECADE_BANDS = [20, 40, 50, 60, 70]
# Rows: total cholesterol <4.1, 4.1-5.1, 5.2-6.1, 6.2-7.2, >=7.3 mmol/L
_TC_BANDS = [4.1, 5.2, 6.2, 7.3]
_TC_POINTS = {
    "male": [
        [0, 0, 0, 0, 0],
        [4, 3, 2, 1, 0],
        [7, 5, 3, 1, 0],
        [9, 6, 4, 2, 1],
        [11, 8, 5, 3, 1],
    ],
    "female": [
        [0, 0, 0, 0, 0],
        [4, 3, 2, 1, 1],
        [8, 6, 4, 2, 1],
        [11, 8, 5, 3, 2],
        [13, 10, 7, 4, 2],
    ],
}
_SMOKER_POINTS = {
    "male": [8, 5, 3, 1, 1],  # 70-79 changed to 1 in SG-FRS-2023 (footnote a)
    "female": [9, 7, 4, 2, 1],
}

# HDL rows: <1.0, 1.0-1.2, 1.3-1.5, >=1.6 mmol/L (same for both sexes)
_HDL_BANDS = [1.0, 1.3, 1.6]
_HDL_POINTS = [2, 1, 0, -1]

# SBP rows: <120, 120-129, 130-139, 140-159, >=160 mmHg
_SBP_BANDS = [120, 130, 140, 160]
_SBP_POINTS = {
    "male": {"untreated": [0, 0, 1, 1, 2], "treated": [0, 1, 2, 2, 3]},
    "female": {"untreated": [0, 1, 2, 3, 4], "treated": [0, 3, 4, 5, 6]},
}

# Tables A2 (men, points -5..20) and A4 (women, points 0..27): 10-year risk %.
# None means "<1%".
_RISK_TABLE = {
    "male": {
        "first_points": -5,
        "chinese": [None] * 9 + [1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 7, 9, 11, 14, 18, 23, 28],
        "malay": [None] * 7 + [1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 7, 9, 11, 14, 18, 22, 28, 34, 42],
        "indian": [None] * 6 + [1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 10, 12, 15, 19, 24, 30, 37, 45, 54],
    },
    "female": {
        "first_points": 0,
        "chinese": [None] * 13 + [1, 1, 1, 1, 2, 2, 3, 4, 5, 7, 8, 11, 14, 18, 22],
        "malay": [None] * 10 + [1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 7, 9, 11, 14, 18, 23, 29, 36],
        "indian": [None] * 9 + [1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 10, 12, 16, 20, 25, 31, 39, 47],
    },
}

# Value reported for the table's "<1%" cells.
_BELOW_ONE_PERCENT = 0.5


def _band(value: float, bounds: list) -> int:
    """Index of the half-open band [bounds[i-1], bounds[i]) that holds value."""
    return bisect.bisect_right(bounds, value)


class Singapore(RiskModel[SingaporePatientData]):
    """Singapore-modified Framingham Risk Score 2023 (10-year coronary risk)."""

    model_name = "Singapore"
    model_version = "SG-FRS-2023"
    supported_regions = ["Singapore"]
    patient_data_class = SingaporePatientData

    def validate_input(self, patient: SingaporePatientData) -> None:
        super().validate_input(patient)
        if not 20 <= patient.age <= 79:
            raise ValueError("SG-FRS-2023 applies to ages 20-79")
        if patient.diabetes:
            raise ValueError("SG-FRS-2023 does not apply to people with diabetes mellitus")
        if patient.ethnicity not in ("chinese", "malay", "indian"):
            raise ValueError("SG-FRS-2023 requires ethnicity 'chinese', 'malay' or 'indian'")
        if patient.antihypertensive is None:
            raise ValueError("SG-FRS-2023 requires antihypertensive (treated BP) status")

    def points(self, patient: SingaporePatientData) -> dict:
        """Points per risk factor from Tables A1/A3."""
        sex = patient.sex
        decade = _band(patient.age, _DECADE_BANDS) - 1
        # shortcut: mmol/L bands as printed (4.1-5.1, 5.2-6.1, ...) treated as half-open,
        # so 5.15 counts as 5.2-6.1; upgrade if ACE publishes exact cut-points.
        tc_row = _band(patient.total_cholesterol, _TC_BANDS)
        sbp_col = "treated" if patient.antihypertensive else "untreated"
        return {
            "age": _AGE_POINTS[sex][_band(patient.age, _AGE_BANDS) - 1],
            "total_cholesterol": _TC_POINTS[sex][tc_row][decade],
            "smoking": _SMOKER_POINTS[sex][decade] if patient.smoking else 0,
            "hdl_cholesterol": _HDL_POINTS[_band(patient.hdl_cholesterol, _HDL_BANDS)],
            "systolic_bp": _SBP_POINTS[sex][sbp_col][_band(patient.systolic_bp, _SBP_BANDS)],
        }

    def calculate(self, patient: SingaporePatientData) -> RiskResult:
        self.validate_input(patient)
        pts = self.points(patient)
        total = sum(pts.values())

        table = _RISK_TABLE[patient.sex]
        column = table[patient.ethnicity]
        idx = total - table["first_points"]
        capped = idx >= len(column)  # above the table: report its top value
        risk = column[min(max(idx, 0), len(column) - 1)]

        if risk is None:
            risk_score, display = _BELOW_ONE_PERCENT, "<1%"
        else:
            risk_score, display = float(risk), f"{'>=' if capped else ''}{risk}%"

        if risk_score < 5:
            category = "low"
        elif risk_score < 10:
            category = "borderline"
        elif risk_score <= 20:
            category = "intermediate"
        else:
            category = "high"

        return RiskResult(
            risk_score=risk_score,
            risk_category=category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata={
                "total_points": total,
                "points": pts,
                "risk_display": display,
                "outcome": "10-year non-fatal MI or coronary death",
            },
        )
