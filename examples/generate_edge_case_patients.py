"""
Edge Case Patient Generator for CVD Risk Model Stress Testing

This script generates edge case patient data designed to stress test CVD risk
models. It creates patients with boundary values, extreme combinations, and
potentially problematic inputs to ensure models handle edge cases gracefully.

WARNING: These are artificial edge cases for testing purposes only.
They may not represent realistic clinical scenarios.

Usage:
    python generate_edge_case_patients.py

Output:
    - data/edge_case_patients.csv: Edge case patients in CSV format
    - data/edge_case_patients.json: Same data in JSON format

Categories of Edge Cases:
    - Boundary values: Exact min/max allowed values
    - Extreme combinations: All risk factors present/absent
    - Biologically implausible: HDL > total cholesterol, etc.
    - Precision issues: Very small/large decimal values
    - Age extremes: Very young/old patients
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def create_boundary_patients() -> List[Dict]:
    """Create patients with exact boundary values."""
    patients = []

    # Age boundaries
    for age in [18, 120]:
        for sex in ["male", "female"]:
            patients.append({
                "patient_id": f"boundary_age_{age}_{sex}",
                "age": age,
                "sex": sex,
                "systolic_bp": 120.0,
                "total_cholesterol": 5.0,
                "hdl_cholesterol": 1.3,
                "smoking": False,
                "region": "moderate",
                "diabetes": False,
                "ethnicity": "white",
                "bmi": 25.0,
                "family_history": False,
                "antihypertensive": False,
                "edge_case_type": "boundary_age",
                "description": f"Age boundary: {age} years"
            })

    # Blood pressure boundaries
    for bp in [50.0, 250.0]:
        patients.append({
            "patient_id": f"boundary_bp_{bp}",
            "age": 55,
            "sex": "male",
            "systolic_bp": bp,
            "total_cholesterol": 5.0,
            "hdl_cholesterol": 1.3,
            "smoking": False,
            "region": "moderate",
            "diabetes": False,
            "ethnicity": "white",
            "bmi": 25.0,
            "family_history": False,
            "antihypertensive": False,
            "edge_case_type": "boundary_bp",
            "description": f"Systolic BP boundary: {bp} mmHg"
        })

    # Cholesterol boundaries
    for tc in [1.0, 15.0]:
        for hdl in [0.3, 5.0]:
            patients.append({
                "patient_id": f"boundary_chol_{tc}_{hdl}",
                "age": 55,
                "sex": "male",
                "systolic_bp": 120.0,
                "total_cholesterol": tc,
                "hdl_cholesterol": hdl,
                "smoking": False,
                "region": "moderate",
                "diabetes": False,
                "ethnicity": "white",
                "bmi": 25.0,
                "family_history": False,
                "antihypertensive": False,
                "edge_case_type": "boundary_cholesterol",
                "description": f"Cholesterol boundary: TC={tc}, HDL={hdl}"
            })

    # BMI boundaries
    for bmi in [10.0, 60.0]:
        patients.append({
            "patient_id": f"boundary_bmi_{bmi}",
            "age": 55,
            "sex": "male",
            "systolic_bp": 120.0,
            "total_cholesterol": 5.0,
            "hdl_cholesterol": 1.3,
            "smoking": False,
            "region": "moderate",
            "diabetes": False,
            "ethnicity": "white",
            "bmi": bmi,
            "family_history": False,
            "antihypertensive": False,
            "edge_case_type": "boundary_bmi",
            "description": f"BMI boundary: {bmi}"
        })

    return patients


def create_extreme_combinations() -> List[Dict]:
    """Create patients with extreme combinations of risk factors."""
    patients = []

    # All risk factors present (maximum risk)
    patients.append({
        "patient_id": "extreme_all_risks",
        "age": 70,
        "sex": "male",
        "systolic_bp": 200.0,
        "total_cholesterol": 12.0,
        "hdl_cholesterol": 0.5,
        "smoking": True,
        "region": "very_high",
        "diabetes": True,
        "ethnicity": "south_asian",
        "bmi": 40.0,
        "family_history": True,
        "antihypertensive": True,
        "edge_case_type": "extreme_risk_max",
        "description": "All risk factors maximized"
    })

    # No risk factors (minimum risk)
    patients.append({
        "patient_id": "extreme_no_risks",
        "age": 30,
        "sex": "female",
        "systolic_bp": 100.0,
        "total_cholesterol": 3.0,
        "hdl_cholesterol": 2.5,
        "smoking": False,
        "region": "low",
        "diabetes": False,
        "ethnicity": "white",
        "bmi": 20.0,
        "family_history": False,
        "antihypertensive": False,
        "edge_case_type": "extreme_risk_min",
        "description": "All risk factors minimized"
    })

    # Contradictory risk factors (young with high risks)
    patients.append({
        "patient_id": "extreme_young_high_risk",
        "age": 25,
        "sex": "male",
        "systolic_bp": 180.0,
        "total_cholesterol": 10.0,
        "hdl_cholesterol": 0.8,
        "smoking": True,
        "region": "very_high",
        "diabetes": True,
        "ethnicity": "south_asian",
        "bmi": 35.0,
        "family_history": True,
        "antihypertensive": True,
        "edge_case_type": "extreme_young_risk",
        "description": "Young age with maximum risk factors"
    })

    return patients


def create_biological_anomalies() -> List[Dict]:
    """Create patients with biologically implausible values."""
    patients = []

    # HDL > Total cholesterol (impossible)
    patients.append({
        "patient_id": "anomaly_hdl_higher_tc",
        "age": 55,
        "sex": "male",
        "systolic_bp": 120.0,
        "total_cholesterol": 4.0,
        "hdl_cholesterol": 6.0,  # HDL > TC (biologically impossible)
        "smoking": False,
        "region": "moderate",
        "diabetes": False,
        "ethnicity": "white",
        "bmi": 25.0,
        "family_history": False,
        "antihypertensive": False,
        "edge_case_type": "biological_anomaly",
        "description": "HDL > Total cholesterol (biologically impossible)"
    })

    # Extremely low HDL with high TC
    patients.append({
        "patient_id": "anomaly_extreme_ratios",
        "age": 55,
        "sex": "male",
        "systolic_bp": 120.0,
        "total_cholesterol": 10.0,
        "hdl_cholesterol": 0.4,  # Very low HDL
        "smoking": True,
        "region": "high",
        "diabetes": True,
        "ethnicity": "white",
        "bmi": 30.0,
        "family_history": True,
        "antihypertensive": False,
        "edge_case_type": "biological_anomaly",
        "description": "Extreme TC/HDL ratio"
    })

    return patients


def create_precision_edge_cases() -> List[Dict]:
    """Create patients with floating point precision edge cases."""
    patients = []

    # Very small decimals
    patients.append({
        "patient_id": "precision_tiny_values",
        "age": 55,
        "sex": "male",
        "systolic_bp": 120.000001,  # Tiny decimal
        "total_cholesterol": 5.0000001,
        "hdl_cholesterol": 1.30000001,
        "smoking": False,
        "region": "moderate",
        "diabetes": False,
        "ethnicity": "white",
        "bmi": 25.000001,
        "family_history": False,
        "antihypertensive": False,
        "edge_case_type": "precision_tiny",
        "description": "Very small decimal values"
    })

    # Many decimal places
    patients.append({
        "patient_id": "precision_many_decimals",
        "age": 55,
        "sex": "male",
        "systolic_bp": 120.123456789,
        "total_cholesterol": 5.123456789,
        "hdl_cholesterol": 1.123456789,
        "smoking": False,
        "region": "moderate",
        "diabetes": False,
        "ethnicity": "white",
        "bmi": 25.123456789,
        "family_history": False,
        "antihypertensive": False,
        "edge_case_type": "precision_many_decimals",
        "description": "Many decimal places"
    })

    # Values that might cause division by zero or log issues
    patients.append({
        "patient_id": "precision_near_zero",
        "age": 55,
        "sex": "male",
        "systolic_bp": 50.0001,  # Very close to minimum
        "total_cholesterol": 1.0001,
        "hdl_cholesterol": 0.3001,
        "smoking": False,
        "region": "moderate",
        "diabetes": False,
        "ethnicity": "white",
        "bmi": 10.0001,
        "family_history": False,
        "antihypertensive": False,
        "edge_case_type": "precision_near_zero",
        "description": "Values very close to minimum bounds"
    })

    return patients


def create_model_specific_edge_cases() -> List[Dict]:
    """Create edge cases specific to different CVD risk models."""
    patients = []

    # SCORE2 age range edges (optimal range is 40-69)
    for age in [39, 40, 69, 70]:  # Just outside/boundary of optimal range
        patients.append({
            "patient_id": f"score2_age_edge_{age}",
            "age": age,
            "sex": "male",
            "systolic_bp": 140.0,
            "total_cholesterol": 6.0,
            "hdl_cholesterol": 1.2,
            "smoking": True,
            "region": "moderate",
            "diabetes": False,
            "ethnicity": "white",
            "bmi": 25.0,
            "family_history": False,
            "antihypertensive": False,
            "edge_case_type": "model_specific_score2",
            "description": f"SCORE2 age boundary: {age} (optimal range 40-69)"
        })

    # QRISK3 ethnicity combinations with extreme values
    ethnicities = ["white", "south_asian", "black", "chinese", "mixed", "other"]
    for ethnicity in ethnicities:
        patients.append({
            "patient_id": f"qrisk3_ethnicity_{ethnicity}",
            "age": 55,
            "sex": "male",
            "systolic_bp": 140.0,
            "total_cholesterol": 6.0,
            "hdl_cholesterol": 1.0,
            "smoking": True,
            "region": "moderate",
            "diabetes": True,
            "ethnicity": ethnicity,
            "bmi": 35.0,
            "family_history": True,
            "antihypertensive": True,
            "edge_case_type": "model_specific_qrisk3",
            "description": f"QRISK3 ethnicity: {ethnicity} with high risk factors"
        })

    return patients


def create_random_extremes() -> List[Dict]:
    """Create random patients with extreme but valid values."""
    patients = []
    random.seed(42)  # For reproducibility

    for i in range(20):
        patients.append({
            "patient_id": f"random_extreme_{i:02d}",
            "age": random.choice([25, 30, 80, 85, 90]),  # Age extremes
            "sex": random.choice(["male", "female"]),
            "systolic_bp": random.choice([60.0, 70.0, 200.0, 220.0]),  # BP extremes
            "total_cholesterol": random.choice([2.0, 2.5, 12.0, 14.0]),  # Chol extremes
            "hdl_cholesterol": random.choice([0.5, 0.8, 4.0, 4.5]),  # HDL extremes
            "smoking": random.choice([True, False]),
            "region": random.choice(["low", "moderate", "high", "very_high"]),
            "diabetes": random.choice([True, False]),
            "ethnicity": random.choice(["white", "south_asian", "black", "chinese", "mixed", "other"]),
            "bmi": random.choice([15.0, 18.0, 45.0, 55.0]),  # BMI extremes
            "family_history": random.choice([True, False]),
            "antihypertensive": random.choice([True, False]),
            "edge_case_type": "random_extreme",
            "description": f"Random combination of extreme values #{i+1}"
        })

    return patients


def generate_edge_case_patients() -> pd.DataFrame:
    """Generate comprehensive edge case patient dataset."""
    print("Generating edge case patients for CVD risk model stress testing...")

    all_patients = []

    # Generate different categories of edge cases
    all_patients.extend(create_boundary_patients())
    all_patients.extend(create_extreme_combinations())
    all_patients.extend(create_biological_anomalies())
    all_patients.extend(create_precision_edge_cases())
    all_patients.extend(create_model_specific_edge_cases())
    all_patients.extend(create_random_extremes())

    # Convert to DataFrame
    df = pd.DataFrame(all_patients)

    # Ensure we don't have any validation issues by clamping values
    df['age'] = df['age'].clip(18, 120)
    df['systolic_bp'] = df['systolic_bp'].clip(50.0, 250.0)
    df['total_cholesterol'] = df['total_cholesterol'].clip(1.0, 15.0)
    df['hdl_cholesterol'] = df['hdl_cholesterol'].clip(0.3, 5.0)
    df['bmi'] = df['bmi'].clip(10.0, 60.0)

    return df


def save_edge_case_data(df: pd.DataFrame, output_dir: str = "data") -> None:
    """Save edge case patient data to CSV and JSON formats."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    csv_path = output_path / "edge_case_patients.csv"
    json_path = output_path / "edge_case_patients.json"

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} edge case patients to {csv_path}")

    # Save JSON
    json_data = df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved {len(df)} edge case patients to {json_path}")


def display_edge_case_summary(df: pd.DataFrame) -> None:
    """Display summary of edge case categories."""
    print("\n" + "=" * 60)
    print("EDGE CASE PATIENT DATA SUMMARY")
    print("=" * 60)

    print(f"Total edge case patients: {len(df)}")
    print(f"Edge case types: {df['edge_case_type'].nunique()}")

    print("\nEdge case categories:")
    type_counts = df['edge_case_type'].value_counts()
    for case_type, count in type_counts.items():
        print(f"  {case_type}: {count} patients")

    print("\nData ranges:")
    print(f"  Age: {df['age'].min()}-{df['age'].max()} years")
    print(f"  Systolic BP: {df['systolic_bp'].min():.1f}-{df['systolic_bp'].max():.1f} mmHg")
    print(f"  Total Cholesterol: {df['total_cholesterol'].min():.1f}-{df['total_cholesterol'].max():.1f} mmol/L")
    print(f"  HDL Cholesterol: {df['hdl_cholesterol'].min():.1f}-{df['hdl_cholesterol'].max():.1f} mmol/L")
    print(f"  BMI: {df['bmi'].min():.1f}-{df['bmi'].max():.1f}")

    # Check for potential issues
    issues = []
    hdl_higher = (df['hdl_cholesterol'] > df['total_cholesterol']).sum()
    if hdl_higher > 0:
        issues.append(f"{hdl_higher} patients have HDL > Total cholesterol")

    if issues:
        print("\n⚠️  Potential issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No obvious data validation issues detected")


def main():
    """Main function to generate and save edge case patient data."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate edge case patients
    df = generate_edge_case_patients()

    # Display summary
    display_edge_case_summary(df)

    # Save data
    save_edge_case_data(df)

    print("\nEdge case generation complete!")
    print("⚠️  WARNING: These edge cases may cause models to behave unexpectedly.")
    print("   Use for testing error handling and boundary conditions only.")
    print("\nExample usage with SCORE2 model:")
    print("  from cvd_risk_calculator.models.score2 import SCORE2")
    print("  model = SCORE2()")
    print("  try:")
    print("      results = model.calculate_batch(df)")
    print("  except Exception as e:")
    print("      print(f'Edge case detected: {e}')")


if __name__ == "__main__":
    main()
