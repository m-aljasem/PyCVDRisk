"""PREDICT (New Zealand) risk equations."""

from cvd_risk.predict.predict import (
    ACS_TYPE,
    ETHNICITY,
    POLICY_ETHNICITY,
    SMOKING,
    predict_bleed,
    predict_cvd,
    predict_cvd_bmi,
    predict_cvd_policy,
    predict_post_acs,
    predict_post_cvd,
    predict_t2d,
)

__all__ = ["ACS_TYPE", "ETHNICITY", "POLICY_ETHNICITY", "SMOKING", "predict_bleed", "predict_cvd", "predict_cvd_bmi",
           "predict_cvd_policy", "predict_post_acs", "predict_post_cvd", "predict_t2d"]
