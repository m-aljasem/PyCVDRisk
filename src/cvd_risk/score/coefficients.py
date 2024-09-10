"""SCORE (2003) coefficients as printed in Conroy et al., Eur Heart J 2003;24:987-1003,
Appendix A, Tables A and B. reference/score/check_coefficients.py compares them with the PDF.
"""

SOURCE = ("Conroy RM et al. Estimation of ten-year risk of fatal cardiovascular disease in Europe: "
          "the SCORE project. Eur Heart J 2003;24:987-1003, Appendix A, Tables A and B")

REGIONS = ("low", "high")

# Table A: Weibull baseline, (alpha, p) for CHD and non-CHD CVD, by region and sex.
BASELINE = {
    ("low", "male"): {"chd": (-22.1, 4.71), "non_chd": (-26.7, 5.64)},
    ("low", "female"): {"chd": (-29.8, 6.36), "non_chd": (-31.0, 6.62)},
    ("high", "male"): {"chd": (-21.0, 4.62), "non_chd": (-25.7, 5.47)},
    ("high", "female"): {"chd": (-28.7, 6.23), "non_chd": (-30.0, 6.42)},
}

# Table B: w = b_chol (chol - 6) + b_sbp (sbp - 120) + b_smoker current
BETA = {
    "chd": {"smoker": 0.71, "chol": 0.24, "sbp": 0.018},
    "non_chd": {"smoker": 0.63, "chol": 0.02, "sbp": 0.022},
}

AGE_RANGE = (40, 66)  # the charts cover ages 40-65 (shortcut: not stated as a validity range)
