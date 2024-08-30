"""Framingham general CVD risk profile (D'Agostino et al. 2008), coefficients as printed.

Lab model: Circulation 2008;117:743-753, Table 2 (coefficients, baseline survival); the mean
sum of beta*x is printed in the worked examples ("Score Sheet") and in the Framingham Heart
Study calculator spreadsheet (gencardio_lipids.xls, sheet 3, "Const").
BMI (office) model: online supplement, Table I; the mean sum is printed only in the
calculator spreadsheet (gencardio_bmi.xls, sheet 3, "Const").
reference/framingham/check_coefficients.py compares every value with the PDF text.
"""

SOURCE = ("D'Agostino RB Sr et al. General cardiovascular risk profile for use in primary care: "
          "the Framingham Heart Study. Circulation 2008;117:743-753, Table 2 and supplementary Table I; "
          "Framingham Heart Study calculator spreadsheets gencardio_lipids.xls / gencardio_bmi.xls")

# Terms: ln_age, ln_tc, ln_hdl, ln_bmi, ln_sbp_untreated, ln_sbp_treated, smoker, diabetes.
# Risk = 1 - s0 ** exp(sum - mean).
MODELS = {
    "lab": {
        "female": {"ln_age": 2.32888, "ln_tc": 1.20904, "ln_hdl": -0.70833, "ln_sbp_untreated": 2.76157,
                   "ln_sbp_treated": 2.82263, "smoker": 0.52873, "diabetes": 0.69154,
                   "s0": 0.95012, "mean": 26.1931},
        "male": {"ln_age": 3.06117, "ln_tc": 1.12370, "ln_hdl": -0.93263, "ln_sbp_untreated": 1.93303,
                 "ln_sbp_treated": 1.99881, "smoker": 0.65451, "diabetes": 0.57367,
                 "s0": 0.88936, "mean": 23.9802},
    },
    "bmi": {
        "female": {"ln_age": 2.72107, "ln_bmi": 0.51125, "ln_sbp_untreated": 2.81291, "ln_sbp_treated": 2.88267,
                   "smoker": 0.61868, "diabetes": 0.77763, "s0": 0.94833, "mean": 26.0145},
        "male": {"ln_age": 3.11296, "ln_bmi": 0.79277, "ln_sbp_untreated": 1.85508, "ln_sbp_treated": 1.92672,
                 "smoker": 0.70953, "diabetes": 0.53160, "s0": 0.88431, "mean": 23.9388},
    },
}

AGE_RANGE = (30, 75)  # derivation sample aged 30-74 (Methods)
