#!/usr/bin/env python3
"""
Validation script to compare Python Globorisk implementation with R package.

This script runs test cases using both Python and R implementations and compares results.
"""

import subprocess
import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cvd_risk.models.globorisk import Globorisk
from cvd_risk.core.validation import PatientData

def run_r_globorisk_test(sex, age, sbp, tc, dm, smk, iso, year, version, time=10):
    """Run a single test case using the R globorisk source code."""

    rlang_path = os.path.join(os.path.dirname(__file__), 'src', 'cvd_risk', 'models', 'globorisk', 'rlang', 'R')
    sysdata_path = os.path.join(rlang_path, 'sysdata.rda')

    # Create R script that loads the data and source
    r_script = f"""
# Load the sysdata first
load("{sysdata_path}")

# Load the globorisk source directly
source(file.path("{rlang_path}", "globorisk.R"))

# Test case
sex <- {sex}  # 0=man, 1=woman
age <- {age}
sbp <- {sbp}
tc <- {tc}
dm <- {dm}
smk <- {smk}
iso <- "{iso}"
year <- {year}
version <- "{version}"
time <- {time}

try({{
    result <- globorisk(
        sex = sex,
        age = age,
        sbp = sbp,
        tc = tc,
        dm = dm,
        smk = smk,
        iso = iso,
        year = year,
        version = version,
        time = time,
        type = "risk"
    )

    # Return the result
    cat(result)
}}, silent = FALSE)
"""

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

        if result.returncode == 0:
            # Extract the result from output
            output = result.stdout.strip()
            try:
                return float(output)
            except ValueError:
                print(f"R script output: {output}")
                print(f"R script stderr: {result.stderr}")
                return None
        else:
            print(f"R script failed: {result.stderr}")
            return None

    finally:
        # Clean up temp file
        os.unlink(temp_script)

def test_case_python(sex, age, sbp, tc, dm, smk, iso, year, version):
    """Run a test case using the Python implementation."""

    # Convert R sex coding to Python (0=man -> male, 1=woman -> female)
    # R uses: 0=man/male, 1=woman/female
    # Python uses: "male", "female"
    py_sex = "male" if sex == 0 else "female"

    # Create patient data
    patient_data = {
        'age': age,
        'sex': py_sex,
        'systolic_bp': sbp,
        'total_cholesterol': tc,
        'smoking': bool(smk),
        'diabetes': bool(dm)
    }

    # Add HDL cholesterol (default value)
    patient_data['hdl_cholesterol'] = 1.2

    # Add BMI for office version (required)
    if version == "office":
        patient_data['bmi'] = 25.0  # Default BMI

    patient = PatientData(**patient_data)

    try:
        model = Globorisk(country=iso, year=year, version=version)
        result = model.calculate(patient)
        return result.risk_score
    except Exception as e:
        print(f"Python error: {e}")
        return None

def run_validation_tests():
    """Run comprehensive validation tests comparing Python vs R."""

    print("=== Globorisk Python vs R Validation ===\n")

    # Test cases from the original R package documentation
    test_cases = [
        # (sex, age, sbp, tc, dm, smk, iso, year, version)
        (0, 52, 140, 4.5, 1, 0, "AFG", 2000, "lab"),    # Male
        (1, 60, 160, 5.0, 1, 1, "AFG", 2000, "lab"),    # Female
        (0, 65, 170, 5.0, 1, 1, "USA", 2020, "lab"),    # Male, USA
        (1, 55, 130, 6.2, 0, 0, "GBR", 2010, "office"), # Female, office version
        (0, 58, 145, 5.8, 1, 1, "DEU", 2015, "fatal"),   # Male, fatal version
    ]

    results = []

    for i, (sex, age, sbp, tc, dm, smk, iso, year, version) in enumerate(test_cases, 1):
        print(f"Test Case {i}: sex={sex}, age={age}, sbp={sbp}, tc={tc}, dm={dm}, smk={smk}, iso={iso}, year={year}, version={version}")

        # Run Python implementation
        py_result = test_case_python(sex, age, sbp, tc, dm, smk, iso, year, version)
        print(".3f")

        # Run R implementation
        r_result = run_r_globorisk_test(sex, age, sbp, tc, dm, smk, iso, year, version)
        print(".3f")

        # Compare results
        if py_result is not None and r_result is not None:
            diff = abs(py_result - r_result)
            pct_diff = (diff / r_result) * 100 if r_result != 0 else 0
            print(".4f")
            status = "✓ PASS" if pct_diff < 1.0 else "✗ FAIL"  # Allow 1% tolerance
            print(f"  Status: {status}")
        else:
            print("  Status: ✗ ERROR - Missing result")

        print()
        results.append((py_result, r_result, py_result - r_result if py_result and r_result else None))

    # Summary
    print("=== Summary ===")
    valid_results = [(py, r) for py, r, _ in results if py is not None and r is not None]
    if valid_results:
        py_results, r_results = zip(*valid_results)
        mean_py = np.mean(py_results)
        mean_r = np.mean(r_results)
        mean_diff = np.mean([abs(py - r) for py, r in valid_results])

        print(f"Valid test cases: {len(valid_results)}")
        print(".3f")
        print(".3f")
        print(".4f")

        # Check if all differences are within tolerance
        all_pass = all(abs(py - r) / r < 0.01 for py, r in valid_results if r != 0)
        print(f"Overall status: {'✓ PASS' if all_pass else '✗ FAIL'}")
    else:
        print("No valid results to compare")

if __name__ == "__main__":
    run_validation_tests()
