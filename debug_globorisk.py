#!/usr/bin/env python3
"""
Debug script to compare intermediate calculations between Python and R implementations.
"""

import sys
import os
import subprocess
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cvd_risk.models.globorisk import Globorisk
from cvd_risk.core.validation import PatientData

def debug_python_calculation():
    """Debug Python calculation step by step."""

    # Test case: sex=0, age=52, sbp=140, tc=4.5, dm=1, smk=0, iso=AFG, year=2000, version=lab
    patient = PatientData(
        age=52,
        sex="male",  # sex=0 -> male
        systolic_bp=140.0,
        total_cholesterol=4.5,
        hdl_cholesterol=1.2,
        smoking=False,  # smk=0 -> False
        diabetes=True   # dm=1 -> True
    )

    print("=== Python Implementation Debug ===")
    print(f"Patient: age={patient.age}, sex={patient.sex}, sbp={patient.systolic_bp}, tc={patient.total_cholesterol}, smoking={patient.smoking}, diabetes={patient.diabetes}")

    model = Globorisk(country="AFG", year=2000, version="lab")
    print(f"Model: country={model.country}, year={model.year}, version={model.version}, is_lac={model.is_lac}")

    # Get coefficients
    coeffs = model._get_coefficients()
    print(f"Coefficients keys: {list(coeffs.keys())}")
    print(f"Sample coeffs: main_sbpc={coeffs.get('main_sbpc', 'N/A')}, main_tcc={coeffs.get('main_tcc', 'N/A')}")

    # Get baseline rates
    baseline_rates = model._get_baseline_rates(patient)
    print(f"Baseline rates keys: {list(baseline_rates.keys()) if isinstance(baseline_rates, dict) else 'Not dict'}")
    if isinstance(baseline_rates, dict) and baseline_rates:
        first_key = list(baseline_rates.keys())[0]
        print(f"Sample baseline rate: cvd_{first_key} = {baseline_rates[first_key]}")

    # Get risk factor means
    rf_means = model._get_risk_factor_means(patient)
    print(f"Risk factor means: sbp={rf_means.get('mean_sbp', 'N/A')}, tc={rf_means.get('mean_tc', 'N/A')}, dm={rf_means.get('mean_dm', 'N/A')}")

    # Calculate centered values
    sex_numeric = 0 if patient.sex == "male" else 1
    age_centered = patient.age
    agec = int((age_centered - 40) // 5) + 1

    sbp_c = (patient.systolic_bp / 10.0) - rf_means['mean_sbp']
    tc_c = patient.total_cholesterol - rf_means['mean_tc']
    dm_c = float(patient.diabetes) - rf_means['mean_dm']
    smk_c = float(patient.smoking) - rf_means['mean_smk']

    print(f"Centered values: sbp_c={sbp_c:.3f}, tc_c={tc_c:.3f}, dm_c={dm_c:.3f}, smk_c={smk_c:.3f}")
    print(f"Sex numeric: {sex_numeric}, age_centered: {age_centered}, agec: {agec}")

    # Calculate first hazard ratio (t=0)
    hr_0 = (
        sbp_c * coeffs.get("main_sbpc", 0) +
        tc_c * coeffs.get("main_tcc", 0) +
        dm_c * coeffs.get("main__Idm_1", 0) +
        smk_c * coeffs.get("main_smok", 0) +
        sex_numeric * dm_c * coeffs.get("main_sexdm", 0) +
        sex_numeric * smk_c * coeffs.get("main_sexsmok", 0) +
        (age_centered + 0) * sbp_c * coeffs.get("tvc_sbpc", 0) +
        (age_centered + 0) * tc_c * coeffs.get("tvc_tcc", 0) +
        (age_centered + 0) * dm_c * coeffs.get("tvc_dm", 0) +
        (age_centered + 0) * smk_c * coeffs.get("tvc_smok", 0)
    )
    hr_0 = np.exp(hr_0)
    print(f"HR at t=0: {hr_0:.6f}")

    # Calculate hazard rate and survival for t=0
    if 0 in baseline_rates:
        hz_0 = hr_0 * baseline_rates[0]
        surv_0 = np.exp(-hz_0)
        print(f"Hazard rate t=0: {hz_0:.6f}, Survival t=0: {surv_0:.6f}")

    # Run full calculation
    result = model.calculate(patient)
    print(f"Final result: {result.risk_score:.3f}%")
    print(f"Metadata: cumulative_survival={result.calculation_metadata.get('cumulative_survival', 'N/A')}")

def debug_r_calculation():
    """Debug R calculation by running R script with detailed output."""

    rlang_path = os.path.join(os.path.dirname(__file__), 'src', 'cvd_risk', 'models', 'globorisk', 'rlang', 'R')
    sysdata_path = os.path.join(rlang_path, 'sysdata.rda')

    # R script for detailed debugging
    r_script = '''
# Load the sysdata first
load("{sysdata_path}")

# Load the globorisk source directly
source(file.path("{rlang_path}", "globorisk.R"))

# Test case
sex <- 0  # 0=man, 1=woman
age <- 52
sbp <- 140
tc <- 4.5
dm <- 1
smk <- 0
iso <- "AFG"
year <- 2000
version <- "lab"
time <- 10

cat("=== R Implementation Debug ===\\n")
cat(sprintf("Patient: sex=%d, age=%d, sbp=%.1f, tc=%.1f, dm=%d, smk=%d, iso=%s, year=%d\\n",
           sex, age, sbp, tc, dm, smk, iso, year))

# Create data frame like the function does
d <- data.frame(
    iso = toupper(iso),
    sex = as.integer(sex),
    year = as.integer(year),
    age = as.integer(trunc(age)),
    agec = as.integer(ifelse(age < 85, trunc(age / 5) - 7, 10)),
    sbp = sbp / 10,
    tc = tc,
    dm = as.integer(dm),
    smk = as.integer(smk),
    stringsAsFactors = FALSE
)

cat(sprintf("Data frame agec: %d\\n", d$agec))

# Get coefficients
coefs <- subset(coefs, type == "lab" & lac == 0)  # Non-LAC for AFG
cat(sprintf("Coefficients: main_sbpc=%.6f, main_tcc=%.6f, main__Idm_1=%.6f\\n",
           coefs$main_sbpc, coefs$main_tcc, coefs$main__Idm_1))

# Merge risk factor levels
d <- merge(d, rf, by = c("iso", "sex", "agec"), all.x = TRUE, sort = FALSE)
cat(sprintf("Risk factors: mean_sbp=%.3f, mean_tc=%.3f, mean_dm=%.3f, mean_smk=%.3f\\n",
           d$mean_sbp, d$mean_tc, d$mean_dm, d$mean_smk))

# Center values
d$sbp_c <- d$sbp - d$mean_sbp
d$tc_c <- d$tc - d$mean_tc
d$dm_c <- d$dm - d$mean_dm
d$smk_c <- d$smk - d$mean_smk

cat(sprintf("Centered: sbp_c=%.3f, tc_c=%.3f, dm_c=%.3f, smk_c=%.3f\\n",
           d$sbp_c, d$tc_c, d$dm_c, d$smk_c))

# Merge baseline cvd rates
cvdr_f <- subset(cvdr, type == "FNF")  # lab version uses FNF
d <- merge(d, cvdr_f, by = c("iso", "year", "sex", "agec", "age"), all.x = TRUE, sort = FALSE)

# Calculate hazard ratio for t=0
hr_0 <- exp(
    d$sbp_c * coefs$main_sbpc +
    d$tc_c * coefs$main_tcc +
    d$dm_c * coefs$main__Idm_1 +
    d$smk_c * coefs$main_smok +
    d$sex * d$dm_c * coefs$main_sexdm +
    d$sex * d$smk_c * coefs$main_sexsmok +
    (d$age + 0) * d$sbp_c * coefs$tvc_sbpc +
    (d$age + 0) * d$tc_c * coefs$tvc_tcc +
    (d$age + 0) * d$dm_c * coefs$tvc_dm +
    (d$age + 0) * d$smk_c * coefs$tvc_smok
)

cat(sprintf("HR at t=0: %.6f\\n", hr_0))

# Calculate hazard rate and survival for t=0
hz_0 <- hr_0 * d$cvd_0
surv_0 <- exp(-hz_0)
cat(sprintf("Hazard rate t=0: %.6f, Survival t=0: %.6f\\n", hz_0, surv_0))

# Run full calculation
result <- globorisk(
    sex = sex, age = age, sbp = sbp, tc = tc, dm = dm, smk = smk,
    iso = iso, year = year, version = version, time = time, type = "all"
)

cat(sprintf("Final result: %.3f%%\\n", result$globorisk * 100))
cat(sprintf("Cumulative survival: %.6f\\n", result$totsurv))
'''.format(sysdata_path=sysdata_path, rlang_path=rlang_path)

    # Write R script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_script)
        temp_script = f.name

    try:
        # Run R script
        result = subprocess.run(
            ['Rscript', temp_script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )

        print("=== R Implementation Debug ===")
        print(result.stdout)
        if result.stderr:
            print("R stderr:", result.stderr)

    finally:
        os.unlink(temp_script)

if __name__ == "__main__":
    debug_python_calculation()
    print("\n" + "="*50 + "\n")
    debug_r_calculation()
