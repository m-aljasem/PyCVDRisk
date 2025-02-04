"""Marrugat et al. 2003, Table 1 (coefficients) and Table 2 (means, % of population per category).

Category order: total cholesterol <160, 160-199, 200-239, 240-279, >=280 mg/dL;
HDL <35, 35-44, 45-49, 50-59, >59 mg/dL; blood pressure optimal, normal, high normal, stage I,
stage II+. Table 2 leaves the reference categories blank (coefficient 0); they are 0 here.
"""

BETA = {
    "male": {"age": 0.05, "age2": 0.0,
             "tc": (-0.38, 0.00, 0.57, 0.74, 0.83), "hdl": (0.61, 0.37, 0.00, 0.00, -0.46),
             "bp": (0.09, 0.00, 0.42, 0.66, 0.90), "diabetes": 0.53, "smoker": 0.73},
    "female": {"age": 0.17, "age2": -0.001,
               "tc": (-0.21, 0.00, 0.44, 0.56, 0.89), "hdl": (0.73, 0.60, 0.60, 0.00, -0.54),
               "bp": (-0.74, 0.00, -0.37, 0.22, 0.61), "diabetes": 0.87, "smoker": 0.98},
}

# age and age2 are means; the rest are percentages; event10 is the 10-year event rate (1 - S)
MEANS = {
    ("male", "framingham"): {"age": 48.3, "age2": 0.0, "tc": (7.5, 0, 39.0, 16.5, 5.7),
                             "hdl": (19.2, 35.7, 0, 19.0, 10.6), "bp": (20.2, 0, 20.2, 22.5, 12.8),
                             "diabetes": 5.0, "smoker": 40.3, "event10": 0.057},
    ("male", "gerona"): {"age": 54.6, "age2": 0.0, "tc": (5.9, 0, 34.8, 28.1, 10.5),
                         "hdl": (17.0, 31.9, 0, 20.9, 17.5), "bp": (18.1, 0, 20.1, 28.4, 14.7),
                         "diabetes": 9.3, "smoker": 42.3, "event10": 0.035},
    ("female", "framingham"): {"age": 49.6, "age2": 2604.5, "tc": (7.9, 0, 32.7, 20.0, 9.1),
                               "hdl": (4.3, 14.9, 12.4, 0, 40.7), "bp": (34.8, 0, 15.0, 18.6, 10.0),
                               "diabetes": 3.8, "smoker": 37.8, "event10": 0.012},
    ("female", "gerona"): {"age": 54.2, "age2": 2931.7, "tc": (4.9, 0, 35.3, 24.6, 13.4),
                           "hdl": (3.8, 17.0, 12.3, 0, 40.9), "bp": (25.5, 0, 16.2, 28.6, 11.5),
                           "diabetes": 6.8, "smoker": 12.0, "event10": 0.011},
}
