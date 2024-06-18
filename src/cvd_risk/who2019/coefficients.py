"""Published coefficients for the 2019 WHO CVD risk models.

Log hazard ratios: supplementary appendix 1, Table 1.6 (7 decimals as printed;
checked against the PDF text by reference/who2019/check_coefficients.py).
Baseline 10-year survival: main paper Table 2 prints 3 decimals; the full values
below are those of the authors' Stata program `whocvdrisk`, as distributed in
WHORiskCalculator 1.1.0, and round to the printed ones.
Recalibration parameters (recalibration.csv): authors' Stata data file, via
WHORiskCalculator 1.1.0 (MIT), exported by reference/who2019/extract_data.R.
"""

SOURCE = (
    "WHO CVD Risk Chart Working Group. World Health Organization cardiovascular disease "
    "risk charts: revised models to estimate risk in 21 global regions. Lancet Glob Health "
    "2019;7:e1332-e1345. doi:10.1016/S2214-109X(19)30318-3"
)

# Terms (centred): age - 60, sbp - 120, total_chol - 6, bmi - 25; every term
# also enters as an interaction with (age - 60).
# Model per endpoint: p = 1 - s0 ** exp(lp); recalibrated on the cloglog scale.
LAB = {
    "mi": {
        "male": {"age": 0.0719227, "chol": 0.2284944, "sbp": 0.0132183, "diabetes": 0.6410114,
                 "smoker": 0.5638109, "chol_age": -0.0045806, "sbp_age": -0.0001576,
                 "diabetes_age": -0.0124966, "smoker_age": -0.0182545, "s0": 0.9540108},
        "female": {"age": 0.1020713, "chol": 0.2050377, "sbp": 0.015823, "diabetes": 1.070358,
                   "smoker": 1.053223, "chol_age": -0.0051932, "sbp_age": -0.0001378,
                   "diabetes_age": -0.0234174, "smoker_age": -0.0332666, "s0": 0.9894159},
    },
    "stroke": {
        "male": {"age": 0.0986578, "chol": 0.029526, "sbp": 0.0222629, "diabetes": 0.6268712,
                 "smoker": 0.4981217, "chol_age": 0.00142, "sbp_age": -0.0004147,
                 "diabetes_age": -0.026302, "smoker_age": -0.0150561, "s0": 0.9849257},
        "female": {"age": 0.1056632, "chol": 0.0257782, "sbp": 0.0206278, "diabetes": 0.8581998,
                   "smoker": 0.7443627, "chol_age": -0.0021387, "sbp_age": -0.0004897,
                   "diabetes_age": -0.0209826, "smoker_age": -0.0200822, "s0": 0.9890569},
    },
}

NONLAB = {
    "mi": {
        "male": {"age": 0.073593, "bmi": 0.0337219, "sbp": 0.0133937, "smoker": 0.5954767,
                 "bmi_age": -0.0010432, "sbp_age": -0.0001837, "smoker_age": -0.0200831,
                 "s0": 0.9544258},
        "female": {"age": 0.1049418, "bmi": 0.0257616, "sbp": 0.016726, "smoker": 1.093132,
                   "bmi_age": -0.0006537, "sbp_age": -0.0001966, "smoker_age": -0.0343739,
                   "s0": 0.9887124},
    },
    "stroke": {
        "male": {"age": 0.097674, "bmi": 0.0159518, "sbp": 0.0227294, "smoker": 0.4999862,
                 "bmi_age": -0.0003516, "sbp_age": -0.0004374, "smoker_age": -0.0153895,
                 "s0": 0.984826},
        "female": {"age": 0.1046105, "bmi": 0.0036406, "sbp": 0.0216741, "smoker": 0.7399405,
                   "bmi_age": -0.0000129, "sbp_age": -0.0005311, "smoker_age": -0.0203997,
                   "s0": 0.9885706},
    },
}

REGIONS = (
    "north_africa_middle_east", "central_subsaharan_africa", "eastern_subsaharan_africa",
    "southern_subsaharan_africa", "western_subsaharan_africa", "southern_latin_america",
    "high_income_north_america", "caribbean", "andean_latin_america", "central_latin_america",
    "tropical_latin_america", "east_asia", "south_asia", "southeast_asia", "central_asia",
    "high_income_asia_pacific", "western_europe", "central_europe", "eastern_europe",
    "oceania", "australasia",
)

# Ages 40-80 years (half-open [40, 81) for fractional ages); calendar year of the
# recalibration target population: 1990-2017 (GBD 2017).
AGE_RANGE = (40, 81)
YEAR_RANGE = (1990, 2017)
