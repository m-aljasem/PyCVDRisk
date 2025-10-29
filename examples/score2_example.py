"""
SCORE2 Risk Model Example

This script demonstrates how to use the SCORE2 cardiovascular risk
prediction model for single patients and batch processing.

Example Output:
    Patient 1 (55-year-old male smoker):
      10-year CVD risk: 12.5%
      Risk category: high

    Patient 2 (50-year-old female non-smoker):
      10-year CVD risk: 2.3%
      Risk category: low
"""

import pandas as pd

from cvd_risk_calculator.core.validation import PatientData
from cvd_risk_calculator.models.score2 import SCORE2


def example_single_patient() -> None:
    """Example: Calculate risk for a single patient."""
    print("=" * 60)
    print("Example 1: Single Patient Calculation")
    print("=" * 60)

    # Create patient data
    patient = PatientData(
        age=55,
        sex="male",
        systolic_bp=140.0,
        total_cholesterol=6.0,
        hdl_cholesterol=1.2,
        smoking=True,
        region="moderate",
    )

    # Initialize model and calculate
    model = SCORE2()
    result = model.calculate(patient)

    # Display results
    print(f"\nPatient Profile:")
    print(f"  Age: {patient.age} years")
    print(f"  Sex: {patient.sex}")
    print(f"  Systolic BP: {patient.systolic_bp} mmHg")
    print(f"  Total Cholesterol: {patient.total_cholesterol} mmol/L")
    print(f"  HDL Cholesterol: {patient.hdl_cholesterol} mmol/L")
    print(f"  Smoking: {patient.smoking}")
    print(f"  Region: {patient.region}")

    print(f"\nSCORE2 Results:")
    print(f"  10-year CVD risk: {result.risk_score:.1f}%")
    print(f"  Risk category: {result.risk_category}")
    print(f"  Model: {result.model_name} v{result.model_version}")


def example_batch_processing() -> None:
    """Example: Process multiple patients at once."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    # Create DataFrame with multiple patients
    df = pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003", "P004"],
            "age": [55, 50, 65, 45],
            "sex": ["male", "female", "male", "female"],
            "systolic_bp": [140.0, 130.0, 160.0, 110.0],
            "total_cholesterol": [6.0, 5.5, 7.5, 4.5],
            "hdl_cholesterol": [1.2, 1.5, 0.9, 1.8],
            "smoking": [True, False, True, False],
            "region": ["moderate", "low", "very_high", "low"],
        }
    )

    print("\nInput Data:")
    print(df.to_string(index=False))

    # Calculate risks
    model = SCORE2()
    results = model.calculate_batch(df)

    print("\nResults:")
    display_cols = ["patient_id", "age", "sex", "risk_score", "risk_category"]
    print(results[display_cols].to_string(index=False))

    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Mean risk: {results['risk_score'].mean():.1f}%")
    print(f"  Median risk: {results['risk_score'].median():.1f}%")
    print(f"  Risk category distribution:")
    print(results["risk_category"].value_counts().to_string())


def example_risk_comparison() -> None:
    """Example: Compare risks across different patient profiles."""
    print("\n" + "=" * 60)
    print("Example 3: Risk Comparison by Profile")
    print("=" * 60)

    model = SCORE2()

    # Define base patient
    base_patient = PatientData(
        age=55,
        sex="male",
        systolic_bp=140.0,
        total_cholesterol=6.0,
        hdl_cholesterol=1.2,
        smoking=False,
        region="moderate",
    )

    # Calculate baseline risk
    base_result = model.calculate(base_patient)
    print(f"\nBaseline Patient (non-smoker):")
    print(f"  Risk: {base_result.risk_score:.1f}%")

    # Compare with smoker
    smoker_patient = PatientData(
        age=55,
        sex="male",
        systolic_bp=140.0,
        total_cholesterol=6.0,
        hdl_cholesterol=1.2,
        smoking=True,  # Changed
        region="moderate",
    )
    smoker_result = model.calculate(smoker_patient)
    print(f"\nSame patient but smoker:")
    print(f"  Risk: {smoker_result.risk_score:.1f}%")
    print(f"  Risk increase: {smoker_result.risk_score - base_result.risk_score:.1f}%")

    # Compare by region
    print("\n" + "-" * 60)
    print("Risk by Region (same patient profile):")
    for region in ["low", "moderate", "high", "very_high"]:
        region_patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region=region,
        )
        region_result = model.calculate(region_patient)
        print(f"  {region:12s}: {region_result.risk_score:.1f}%")


if __name__ == "__main__":
    example_single_patient()
    example_batch_processing()
    example_risk_comparison()

