"""Globorisk model coefficients.

Source: the authors' official R package globorisk 1.0.2 (Boyer C, Danaei G,
Hajifathalian K, et al.; MIT licence), internal data `coefs`, at full stored
precision. The papers print these rounded:
  lab, fatal: Hajifathalian K, et al. Lancet Diabetes Endocrinol 2015;3:339-55.
              doi:10.1016/S2213-8587(15)00081-9
  office:     Ueda P, et al. Lancet Diabetes Endocrinol 2017;5:196-213.
              doi:10.1016/S2213-8587(17)30015-3
Country baseline hazards and population mean risk factors are in data.npz,
built by reference/globorisk/build_package_data.py from the same package.

Model (time-varying Cox): for follow-up year t = 0..T-1 and attained age
a + t (a = integer age at baseline), hazard ratio
  HR_t = exp(sum(main_k * x_k) + sum(tvc_k * (a + t) * x_k) + sex interactions)
with every x centred on the country/sex/age-group population mean; risk =
1 - prod_t exp(-HR_t * baseline_hazard(country, sex, year + t, a + t)).
Sex enters as 1 = female. SBP is per 10 mmHg, BMI per 5 kg/m^2.
"""

LAB = {
    "main_sbp": 0.30701285600662231, "main_tc": 0.61490607261657715,
    "main_dm": 1.4753053188323975, "main_smk": 1.8466835021972656,
    "main_female_dm": 0.40504580736160278, "main_female_smk": 0.32538318634033203,
    "tvc_sbp": -0.002247117692604661, "tvc_tc": -0.0068656518124043941,
    "tvc_dm": -0.013209526427090168, "tvc_smk": -0.02205285057425499,
}

FATAL = {
    "main_sbp": 0.52381980419158936, "main_tc": 0.57925760746002197,
    "main_dm": 1.743310809135437, "main_smk": 1.4712588787078857,
    "main_female_dm": 0.47825440764427185, "main_female_smk": 0.25174456834793091,
    "tvc_sbp": -0.0046451152302324772, "tvc_tc": -0.0064377072267234325,
    "tvc_dm": -0.013938142918050289, "tvc_smk": -0.013711881823837757,
}

OFFICE = {
    "main_sbp": 0.30374115705490112, "main_bmi": 0.32446500658988953,
    "main_smk": 1.7951201200485229, "main_female_smk": 0.35278141498565674,
    "tvc_sbp": -0.0020986464805901051, "tvc_smk": -0.021474074572324753,
    "tvc_bmi": -0.0030154697597026825,
}

# Valid inputs per the authors' package: age 40-80 at baseline, baseline year
# 2000-2020, follow-up 1-10 years.
AGE_RANGE = (40, 81)  # half-open; ages are truncated to whole years as in the source
YEAR_RANGE = (2000, 2020)
