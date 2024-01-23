"""Known-answer tests for CHADS2, CHA2DS2-VASc and HAS-BLED (published maximum scores)."""

from cvd_risk.core.validation import PatientData
from cvd_risk.models import CHA2DS2_VASc, CHADS2, HAS_BLED

BASE = dict(age=76, sex="female", systolic_bp=150.0, total_cholesterol=5.0, hdl_cholesterol=1.3, smoking=False)


def test_stroke_scores_all_factors_present() -> None:
    patient = PatientData(**BASE, congestive_heart_failure=True, hypertension=True, diabetes=True,
                          prior_stroke_tia=True, vascular_disease=True)
    assert CHADS2().calculate(patient).risk_score == 6
    assert CHA2DS2_VASc().calculate(patient).risk_score == 9


def test_has_bled_all_factors_present() -> None:
    patient = PatientData(**BASE, hypertension=True, abnormal_renal_function=True, abnormal_liver_function=True,
                          prior_stroke_tia=True, bleeding_history=True, labile_inr=True,
                          antiplatelet_drugs=True, alcohol_abuse=True)
    assert HAS_BLED().calculate(patient).risk_score == 9
