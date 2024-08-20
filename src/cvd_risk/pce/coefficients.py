"""Pooled Cohort Equations (2013 ACC/AHA), coefficients as printed.

Source: Goff DC Jr et al. 2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk.
J Am Coll Cardiol 2014;63:2935-59 (Circulation 2014;129:S49-73), Appendix 7, Table A.
Author manuscript PMC4700825, saved as reference/pce/source/goff2013_PMC4700825.xml;
reference/pce/check_coefficients.py compares every value below with that table.
"""

SOURCE = ("Goff DC Jr et al. 2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk. "
          "J Am Coll Cardiol 2014;63:2935-59, Appendix 7, Table A")

# Terms: ln_age, ln_age_sq, ln_tc, ln_age_tc, ln_hdl, ln_age_hdl, ln_sbp_treated,
# ln_age_sbp_treated, ln_sbp_untreated, ln_age_sbp_untreated, smoker, ln_age_smoker, diabetes.
# Risk = 1 - s0 ** exp(sum - mean). Missing terms are 0 ("N/A" in the table).
MODELS = {
    ("female", "white"): {
        "ln_age": -29.799, "ln_age_sq": 4.884, "ln_tc": 13.540, "ln_age_tc": -3.114, "ln_hdl": -13.578,
        "ln_age_hdl": 3.149, "ln_sbp_treated": 2.019, "ln_sbp_untreated": 1.957, "smoker": 7.574,
        "ln_age_smoker": -1.665, "diabetes": 0.661, "mean": -29.18, "s0": 0.9665,
    },
    ("female", "african_american"): {
        "ln_age": 17.114, "ln_tc": 0.940, "ln_hdl": -18.920, "ln_age_hdl": 4.475, "ln_sbp_treated": 29.291,
        "ln_age_sbp_treated": -6.432, "ln_sbp_untreated": 27.820, "ln_age_sbp_untreated": -6.087,
        "smoker": 0.691, "diabetes": 0.874, "mean": 86.61, "s0": 0.9533,
    },
    ("male", "white"): {
        "ln_age": 12.344, "ln_tc": 11.853, "ln_age_tc": -2.664, "ln_hdl": -7.990, "ln_age_hdl": 1.769,
        "ln_sbp_treated": 1.797, "ln_sbp_untreated": 1.764, "smoker": 7.837, "ln_age_smoker": -1.795,
        "diabetes": 0.658, "mean": 61.18, "s0": 0.9144,
    },
    ("male", "african_american"): {
        "ln_age": 2.469, "ln_tc": 0.302, "ln_hdl": -0.307, "ln_sbp_treated": 1.916, "ln_sbp_untreated": 1.809,
        "smoker": 0.549, "diabetes": 0.645, "mean": 19.54, "s0": 0.8954,
    },
}

AGE_RANGE = (40, 80)  # "40 to 79 years of age" (Recommendation 1)
