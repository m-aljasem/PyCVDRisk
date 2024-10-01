"""SMART-REACH model (Kaasenbrood et al., J Am Heart Assoc 2018;7:e009217).

Coefficients are taken from the authors' calculator spreadsheet (Data S2,
reference/smart_reach/source/kaasenbrood2018_calculator.xls), which carries more digits than
the supplement's Table S3; reference/smart_reach/check_coefficients.py checks both.
Units: SBP mmHg, total cholesterol mmol/L, creatinine umol/L.
"""

SOURCE = ("Kaasenbrood L et al. J Am Heart Assoc 2018;7:e009217, Data S1 Tables S2-S3 and the "
          "SMART-REACH calculator spreadsheet (Data S2)")

# Two Fine-Gray-type models on an age time axis: "cvd" = MI, stroke or CVD death; "non_cvd" = non-CVD death.
CVD = {
    "male": 0.07202612, "smoker": 0.4309255, "diabetes": 0.4356513,
    "sbp": -0.02813967, "sbp_sq": 9.665775e-05, "tc": -0.3670944, "tc_sq": 0.03555509,
    "creatinine": 0.006115205, "creatinine_sq": 0.0,
    "two_locations": 0.3176464, "three_locations": 0.2895387, "af": 0.2143259, "chf": 0.4446873,
    "smoker_age": 0.0,
}
NON_CVD = {
    "male": 0.598602, "smoker": 4.25382, "diabetes": 0.406509,
    "sbp": -0.007410907, "sbp_sq": 0.0, "tc": -0.002985867, "tc_sq": 0.0,
    "creatinine": -0.01886007, "creatinine_sq": 8.048823e-05,
    "two_locations": 0.1441609, "three_locations": 0.5693571, "af": 0.3213421, "chf": 0.2060817,
    "smoker_age": -0.04861788,  # times attained age, for current smokers
}
# Region shifts (cvd, non_cvd): REACH Western Europe is the reference.
REGIONS = {"western_europe": (0.0, 0.0), "north_america": (0.1552, 0.4134), "netherlands": (-0.4246, 0.1232)}

# Table S2: age -> (1-year baseline survival free of MI/stroke/CVD death, 1-year non-CVD survival).
BASELINE = {
    45: (1.0000, 1.0000),
    46: (0.8539, 0.9855),
    47: (0.8420, 1.0000),
    48: (0.9088, 0.9950),
    49: (0.9172, 1.0000),
    50: (0.8464, 1.0000),
    51: (0.7297, 0.9949),
    52: (0.8081, 0.9958),
    53: (0.8980, 1.0000),
    54: (0.8155, 0.9896),
    55: (0.7609, 0.9966),
    56: (0.8113, 0.9935),
    57: (0.8173, 0.9842),
    58: (0.7939, 0.9869),
    59: (0.8382, 0.9935),
    60: (0.8333, 0.9938),
    61: (0.8257, 0.9934),
    62: (0.8000, 0.9734),
    63: (0.7930, 0.9683),
    64: (0.7962, 0.9768),
    65: (0.7807, 0.9725),
    66: (0.7731, 0.9724),
    67: (0.8118, 0.9586),
    68: (0.7325, 0.9683),
    69: (0.7671, 0.9720),
    70: (0.7236, 0.9539),
    71: (0.6690, 0.9439),
    72: (0.7173, 0.9469),
    73: (0.6978, 0.9299),
    74: (0.6074, 0.9369),
    75: (0.6880, 0.9537),
    76: (0.6473, 0.9172),
    77: (0.7034, 0.9018),
    78: (0.6904, 0.9280),
    79: (0.6507, 0.8622),
    80: (0.5946, 0.8688),
    81: (0.5328, 0.8381),
    82: (0.4954, 0.8647),
    83: (0.5376, 0.8478),
    84: (0.4403, 0.8125),
    85: (0.5043, 0.7855),
    86: (0.5509, 0.7284),
    87: (0.5480, 0.7685),
    88: (0.3889, 0.7197),
    89: (0.3048, 0.6469),
}
