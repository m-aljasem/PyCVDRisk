"""Reynolds Risk Score (model B) coefficients as printed.
reference/reynolds/check_coefficients.py compares every value with the PDF text.
"""

SOURCE = ("Women: Ridker PM et al. JAMA 2007;297:611-9 (reprinted with corrections), Box, model B. "
          "Men: Ridker PM et al. Circulation 2008;118:2243-51, Appendix C")

# 10-year risk = 1 - S0 ** exp(B - MEAN); B = sum of coefficient x term.
# Terms: ln(age) for men, age for women; ln of SBP (mm Hg), total and HDL cholesterol (mg/dL),
# hsCRP (mg/L); 0/1 for current smoking and parental history of MI before 60;
# HbA1c (%) if diabetic (women only; the men's cohort was non-diabetic).
WOMEN = {"age": 0.0799, "ln_sbp": 3.137, "ln_hscrp": 0.180, "ln_tc": 1.382, "ln_hdl": -1.172,
         "hba1c_if_diabetic": 0.134, "smoker": 0.818, "parental_history": 0.438,
         "s0": 0.98634, "mean": 22.325}
MEN = {"ln_age": 4.385, "ln_sbp": 2.607, "ln_tc": 0.963, "ln_hdl": -0.772, "smoker": 0.405,
       "ln_hscrp": 0.102, "parental_history": 0.541, "s0": 0.8990, "mean": 33.097}
