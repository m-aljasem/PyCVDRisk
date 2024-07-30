"""Published coefficients for the SCORE2 family.

Every value is cited to its publication. Transcription status: each value was
transcribed twice independently (here, and in the third-party R package
RiskScorescvd 0.3.1) and the two agree exactly. reference/score2/check_coefficients.py
compares every SCORE2, SCORE2-OP and SCORE2-Diabetes value with the supplement text.
"""

SCORE2_SOURCE = (
    "SCORE2 working group and ESC Cardiovascular risk collaboration. "
    "Eur Heart J 2021;42:2439-2454. doi:10.1093/eurheartj/ehab309 (supplementary methods)"
)
SCORE2_OP_SOURCE = (
    "SCORE2-OP working group and ESC Cardiovascular risk collaboration. "
    "Eur Heart J 2021;42:2455-2467. doi:10.1093/eurheartj/ehab312 (supplementary methods)"
)
SCORE2_DIABETES_SOURCE = (
    "SCORE2-Diabetes Working Group and the ESC Cardiovascular Risk Collaboration. "
    "Eur Heart J 2023;44:2544-2556. doi:10.1093/eurheartj/ehad260 (supplementary methods)"
)

SCORE2_ASIA_PACIFIC_SOURCE = (
    "SCORE2 Asia-Pacific writing group. Risk prediction of cardiovascular disease in the "
    "Asia-Pacific region: the SCORE2 Asia-Pacific model. Eur Heart J 2025. "
    "doi:10.1093/eurheartj/ehae609, Supplementary Table 5 (rescaling factors); "
    "core model = SCORE2 (Supplementary Methods, Box 1). Source file saved in "
    "reference/score2/source/score2_asia_pacific_ehae609_supplement.docx"
)

REGIONS = ("low", "moderate", "high", "very_high")

# Terms (centred/scaled as in the source):
#   age  = (age - 60) / 5        sbp  = (sbp - 120) / 20
#   tc   = total_chol - 6        hdl  = (hdl - 1.3) / 0.5
# Model: uncal = 1 - S0 ** exp(lp);  risk = 1 - exp(-exp(s1 + s2 * ln(-ln(1 - uncal))))
SCORE2 = {
    "male": {
        "age": 0.3742, "smoker": 0.6012, "sbp": 0.2777, "diabetes": 0.6457,
        "tc": 0.1458, "hdl": -0.2698,
        "age_smoker": -0.0755, "age_sbp": -0.0255, "age_tc": -0.0281,
        "age_hdl": 0.0426, "age_diabetes": -0.0983,
        "s0": 0.9605,
    },
    "female": {
        "age": 0.4648, "smoker": 0.7744, "sbp": 0.3131, "diabetes": 0.8096,
        "tc": 0.1002, "hdl": -0.2606,
        "age_smoker": -0.1088, "age_sbp": -0.0277, "age_tc": -0.0226,
        "age_hdl": 0.0613, "age_diabetes": -0.1272,
        "s0": 0.9776,
    },
}

# Region recalibration (scale1, scale2), indexed like REGIONS. Source: SCORE2.
SCORE2_SCALES = {
    "male": ((-0.5699, 0.7476), (-0.1565, 0.8009), (0.3207, 0.9360), (0.5836, 0.8294)),
    "female": ((-0.7380, 0.7019), (-0.3143, 0.7701), (0.5710, 0.9369), (0.9412, 0.8329)),
}

# SCORE2 Asia-Pacific: SCORE2 core model (SCORE2 above) recalibrated to four
# Asia-Pacific risk regions. Values as printed (9 decimals) in Supplementary Table 5.
SCORE2_ASIA_PACIFIC_SCALES = {
    "male": ((-0.375229965, 0.62020875), (0.284885676, 0.778128607),
             (0.778231091, 0.844985356), (0.608975204, 0.679014197)),
    "female": ((-0.986572446, 0.536743779), (0.08278687, 0.718980326),
               (0.611474287, 0.703624072), (0.502751798, 0.555577072)),
}

# Terms: age = age - 73;  sbp = sbp - 150;  tc = total_chol - 6;  hdl = hdl - 1.4
# Model: uncal = 1 - S0 ** exp(lp - mean_lp); same recalibration form as SCORE2.
SCORE2_OP = {
    "male": {
        "age": 0.0634, "diabetes": 0.4245, "smoker": 0.3524, "sbp": 0.0094,
        "tc": 0.0850, "hdl": -0.3564,
        "age_diabetes": -0.0174, "age_smoker": -0.0247, "age_sbp": -0.0005,
        "age_tc": 0.0073, "age_hdl": 0.0091,
        "s0": 0.7576, "mean_lp": 0.0929,
    },
    "female": {
        "age": 0.0789, "diabetes": 0.6010, "smoker": 0.4921, "sbp": 0.0102,
        "tc": 0.0605, "hdl": -0.3040,
        "age_diabetes": -0.0107, "age_smoker": -0.0255, "age_sbp": -0.0004,
        "age_tc": -0.0009, "age_hdl": 0.0154,
        "s0": 0.8082, "mean_lp": 0.2290,
    },
}

SCORE2_OP_SCALES = {
    "male": ((-0.34, 1.19), (0.01, 1.25), (0.08, 1.15), (0.05, 0.70)),
    "female": ((-0.52, 1.01), (-0.10, 1.10), (0.38, 1.09), (0.38, 0.69)),
}

# Terms as SCORE2, plus:
#   agediag = (age_at_diabetes_diagnosis - 50) / 5   (multiplied by diabetes)
#   hba1c   = (hba1c_mmol_mol - 31) / 9.34
#   egfr    = (ln(egfr) - 4.5) / 0.15
# Baseline survival and region scales are SCORE2's.
SCORE2_DIABETES = {
    "male": {
        "age": 0.5368, "smoker": 0.4774, "sbp": 0.1322, "diabetes": 0.6457,
        "tc": 0.1102, "hdl": -0.1087,
        "age_smoker": -0.0672, "age_sbp": -0.0268, "age_diabetes": -0.0983,
        "age_tc": -0.0181, "age_hdl": 0.0095,
        "agediag": -0.0998, "hba1c": 0.0955, "egfr": -0.0591, "egfr2": 0.0058,
        "age_hba1c": -0.0134, "age_egfr": 0.0115,
        "s0": 0.9605,
    },
    "female": {
        "age": 0.6624, "smoker": 0.6139, "sbp": 0.1421, "diabetes": 0.8096,
        "tc": 0.1127, "hdl": -0.1568,
        "age_smoker": -0.1122, "age_sbp": -0.0167, "age_diabetes": -0.1272,
        "age_tc": -0.0200, "age_hdl": 0.0186,
        "agediag": -0.1180, "hba1c": 0.1173, "egfr": -0.0640, "egfr2": 0.0062,
        "age_hba1c": -0.0196, "age_egfr": 0.0169,
        "s0": 0.9776,
    },
}

# Valid ages in years, half-open [lo, hi) so fractional ages (e.g. 69.9) fall in
# exactly one variant. SCORE2: 40-69; SCORE2-OP: 70-89 (ESC 2021 guideline).
AGE_RANGE = {
    "score2": (40, 70), "score2_op": (70, 90), "score2_diabetes": (40, 70),
    "score2_asia_pacific": (40, 70),
}

SCORE2_CKD_SOURCE = (
    "Matsushita K, Kaptoge S, Hageman SHJ, et al. Including measures of chronic kidney disease "
    "to improve cardiovascular risk prediction by SCORE2 and SCORE2-OP. Eur J Prev Cardiol "
    "2023;30:8-16. doi:10.1093/eurjpc/zwac176, Supplementary Tables A-C"
)

# CKD add-ons (Supplementary Tables A-C). Applied to SCORE2 (age < 70) or
# SCORE2-OP (age 70-89):  risk_new = 1 - (1 - risk) ** exp(delta).
# Expected eGFR and expected log8(ACR) from traditional predictors, SCORE2 centring
# (a = (age-60)/5, sbp = (sbp-120)/20, tc = total_chol-6, hdl = (hdl-1.3)/0.5).
CKD_EXPECTED_EGFR = {
    "constant": 87.8980, "age": -3.7891, "female": -0.7023, "tc": -0.2941, "hdl": 1.0960,
    "sbp": -0.1364, "diabetes": 0.1205, "smoker": 1.3211, "age_tc": 0.0555, "age_hdl": 0.1717,
    "age_sbp": 0.0059, "age_diabetes": -0.8994, "age_smoker": 0.2181,
}
# log8(expected ACR / 8); also uses the observed-eGFR spline (lt60, 60-90, ge90; per -15).
CKD_EXPECTED_LOG8_ACR = {
    "constant": -0.0225, "age": 0.0159, "female": 0.0308, "tc": 0.0185, "hdl": -0.0274,
    "sbp": 0.1339, "diabetes": 0.2171, "smoker": 0.0629, "age_tc": -0.0062, "age_hdl": 0.0003,
    "age_sbp": 0.0008, "age_diabetes": -0.0109, "age_smoker": 0.0085,
    "egfr_lt60": 0.4057, "egfr_60_90": 0.0597, "egfr_ge90": -0.0916,
}
# eGFR add-on: log HRs for (observed - expected) eGFR spline segments, and their
# interactions with age ((age-60)/5 for SCORE2; age-73 for SCORE2-OP). SCORE2 has
# no >= 90 segment ("not taken into account", Table A footnote).
CKD_EGFR_ADDON = {
    "score2": {"lt60": 0.4713, "60_90": 0.0956, "ge90": 0.0,
               "age_lt60": -0.0802, "age_60_90": 0.0088, "age_ge90": 0.0},
    "score2_op": {"lt60": 0.3072, "60_90": 0.0942, "ge90": -0.4616,
                  "age_lt60": -0.0127, "age_60_90": -0.0098, "age_ge90": -0.0075},
}
CKD_ACR_ADDON = {"score2": 0.2432, "score2_op": 0.2370}  # per log8(ACR) - log8(expected ACR)
DIPSTICK = ("negative", "trace", "1+", "2+", "3+")
CKD_DIPSTICK_ADDON = {  # indexed like DIPSTICK
    "score2": (0.0, 0.2644, 0.4126, 0.4761, 0.4761),      # "++ or more"
    "score2_op": (0.0, 0.2517, 0.3872, 0.5458, 0.4207),   # "+++ or more"
}
