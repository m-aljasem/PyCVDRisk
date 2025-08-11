"""Coefficients: Wilson et al. 1998, Table 6 (b-coefficients of the categorical total-cholesterol model;
Marrugat et al., Rev Esp Cardiol 2003, Table 1 prints them to 4 decimals). Calibration: Girona
prevalences and S0 from Rev Esp Cardiol 2003 (Table 1, Appendix); Framingham G and S0 from Wilson.

Category order: total cholesterol <160, 160-199, 200-239, 240-279, >=280 mg/dL;
HDL <35, 35-44, 45-49, 50-59, >=60 mg/dL; blood pressure optimal, normal, high normal, stage I,
stage II+. Reference categories have coefficient 0.
Men's optimal blood pressure: Wilson prints -0.00226, Rev Esp Cardiol +0.0023; Wilson (the source of
the function) is used. The difference is 0.0045 in the linear predictor.
"""

BETA = {
    "male": {"age": 0.04826, "age2": 0.0,
             "tc": (-0.65945, 0.0, 0.17692, 0.50539, 0.65713), "hdl": (0.49744, 0.24310, 0.0, -0.05107, -0.48660),
             "bp": (-0.00226, 0.0, 0.28320, 0.52168, 0.61859), "diabetes": 0.42839, "smoker": 0.52337},
    "female": {"age": 0.33766, "age2": -0.00268,
               "tc": (-0.26138, 0.0, 0.20771, 0.24385, 0.53513), "hdl": (0.84312, 0.37796, 0.19785, 0.0, -0.42951),
               "bp": (-0.53363, 0.0, -0.06773, 0.26288, 0.46573), "diabetes": 0.59626, "smoker": 0.29246},
}

# Girona: mean age, mean squared age and the % of men/women in each category (Rev Esp Cardiol Table 1)
GIRONA = {
    "male": {"age": 54.6, "age2": 0.0, "tc": (5.9, 20.7, 34.8, 28.1, 10.5), "hdl": (17.0, 31.9, 12.7, 20.9, 17.5),
             "bp": (18.1, 19.3, 20.1, 28.4, 14.1), "diabetes": 9.3, "smoker": 42.3},
    "female": {"age": 54.2, "age2": 3054.9, "tc": (4.9, 11.8, 35.3, 24.6, 13.4), "hdl": (3.8, 17.0, 12.3, 26.0, 40.9),
               "bp": (27.5, 16.5, 16.2, 28.6, 11.5), "diabetes": 6.7, "smoker": 12.0},
}
# Framingham: sum(b*Xmean) and 10-year S0 as printed by Wilson (Appendix, Equations 1 and 4)
G_FRAMINGHAM = {"male": 3.0975, "female": 9.92545}
S0 = {("male", "gerona"): 0.951, ("female", "gerona"): 0.978,  # Rev Esp Cardiol: 1 - incidence 4.9% / 2.2%
      ("male", "framingham"): 0.90015, ("female", "framingham"): 0.96246}
