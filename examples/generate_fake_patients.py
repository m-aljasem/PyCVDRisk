"""
Fake Patient Data Generator for CVD Risk Model Validation

This script generates realistic fake patient data for testing and validating
cardiovascular disease risk prediction models. The generated data includes
all fields required by different CVD risk models (SCORE2, QRISK3, Framingham, etc.).

Usage:
    python generate_fake_patients.py

Output:
    - fake_patients.csv: 1000 fake patients in CSV format
    - fake_patients.json: Same data in JSON format for alternative use

Data Characteristics:
    - 1000 patients total
    - Balanced gender distribution (approximately 50/50)
    - Age distribution reflecting adult population (30-80 years)
    - Realistic clinical ranges for all parameters
    - Region distribution based on European CVD risk patterns
    - Ethnicity distribution for QRISK3 compatibility
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def generate_age_distribution(n_patients: int) -> List[int]:
    """Generate realistic age distribution (30-80 years, skewed toward middle age)."""
    # Use beta distribution to create realistic age distribution
    # Beta(2, 3) gives a distribution that peaks around middle age
    ages_raw = np.random.beta(2, 3, n_patients)
    ages = np.round(30 + ages_raw * 50).astype(int)  # Scale to 30-80 range
    return ages.tolist()


def generate_blood_pressure_distribution(n_patients: int) -> List[float]:
    """Generate realistic systolic blood pressure distribution."""
    # Normal distribution with slight hypertension bias
    bp = np.random.normal(135, 20, n_patients)
    # Clip to clinical range
    bp = np.clip(bp, 90, 220)
    return np.round(bp, 1).tolist()


def generate_cholesterol_distribution(n_patients: int, hdl: bool = False) -> List[float]:
    """Generate realistic cholesterol distribution."""
    if hdl:
        # HDL cholesterol (typically 0.8-2.0 mmol/L)
        chol = np.random.normal(1.4, 0.4, n_patients)
        chol = np.clip(chol, 0.5, 3.5)
    else:
        # Total cholesterol (typically 4.0-7.0 mmol/L)
        chol = np.random.normal(5.5, 1.2, n_patients)
        chol = np.clip(chol, 3.0, 10.0)

    return np.round(chol, 2).tolist()


def generate_bmi_distribution(n_patients: int) -> List[float]:
    """Generate realistic BMI distribution."""
    # Normal distribution with overweight bias
    bmi = np.random.normal(27, 5, n_patients)
    bmi = np.clip(bmi, 18, 50)
    return np.round(bmi, 1).tolist()


def generate_fake_patients(n_patients: int = 1000) -> pd.DataFrame:
    """
    Generate fake patient data for CVD risk model validation.

    Parameters
    ----------
    n_patients : int, default 1000
        Number of fake patients to generate

    Returns
    -------
    pd.DataFrame
        DataFrame with fake patient data
    """
    print(f"Generating {n_patients} fake patients...")

    # Generate base demographics
    ages = generate_age_distribution(n_patients)
    sexes = random.choices(["male", "female"], k=n_patients)

    # Generate clinical measurements
    systolic_bp = generate_blood_pressure_distribution(n_patients)
    total_cholesterol = generate_cholesterol_distribution(n_patients, hdl=False)
    hdl_cholesterol = generate_cholesterol_distribution(n_patients, hdl=True)
    bmi = generate_bmi_distribution(n_patients)

    # Generate binary risk factors
    smoking = random.choices([True, False], weights=[0.25, 0.75], k=n_patients)
    diabetes = random.choices([True, False], weights=[0.15, 0.85], k=n_patients)
    family_history = random.choices([True, False], weights=[0.2, 0.8], k=n_patients)
    antihypertensive = random.choices([True, False], weights=[0.3, 0.7], k=n_patients)

    # Generate categorical variables
    regions = random.choices(
        ["low", "moderate", "high", "very_high"],
        weights=[0.3, 0.4, 0.2, 0.1],  # European distribution
        k=n_patients
    )

    ethnicities = random.choices(
        ["white", "south_asian", "black", "chinese", "mixed", "other"],
        weights=[0.75, 0.1, 0.05, 0.03, 0.05, 0.02],
        k=n_patients
    )

    # Create DataFrame
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(1, n_patients + 1)],
        "age": ages,
        "sex": sexes,
        "systolic_bp": systolic_bp,
        "total_cholesterol": total_cholesterol,
        "hdl_cholesterol": hdl_cholesterol,
        "smoking": smoking,
        "region": regions,
        "diabetes": diabetes,
        "ethnicity": ethnicities,
        "bmi": bmi,
        "family_history": family_history,
        "antihypertensive": antihypertensive,
    })

    # Ensure HDL is always less than total cholesterol (biological constraint)
    hdl_too_high = df["hdl_cholesterol"] >= df["total_cholesterol"]
    if hdl_too_high.any():
        # Adjust HDL to be reasonable compared to total cholesterol
        df.loc[hdl_too_high, "hdl_cholesterol"] = (
            df.loc[hdl_too_high, "total_cholesterol"] * np.random.uniform(0.2, 0.5, hdl_too_high.sum())
        ).round(2)

    return df


def save_fake_data(df: pd.DataFrame, output_dir: str = "data") -> None:
    """Save fake patient data to CSV and JSON formats."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    csv_path = output_path / "fake_patients.csv"
    json_path = output_path / "fake_patients.json"

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} patients to {csv_path}")

    # Save JSON
    json_data = df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved {len(df)} patients to {json_path}")


def display_summary(df: pd.DataFrame) -> None:
    """Display summary statistics of the generated data."""
    print("\n" + "=" * 60)
    print("FAKE PATIENT DATA SUMMARY")
    print("=" * 60)

    print(f"Total patients: {len(df)}")
    print(f"Gender distribution: {df['sex'].value_counts().to_dict()}")
    print(f"Age range: {df['age'].min()}-{df['age'].max()} years")
    print(f"Mean age: {df['age'].mean():.1f} years")

    print("\nClinical Parameters (mean ± std):")
    clinical_cols = ["systolic_bp", "total_cholesterol", "hdl_cholesterol", "bmi"]
    for col in clinical_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  {col}: {mean_val:.1f} ± {std_val:.1f}")

    print("\nRisk Factors (%):")
    binary_cols = ["smoking", "diabetes", "family_history", "antihypertensive"]
    for col in binary_cols:
        pct = (df[col].sum() / len(df)) * 100
        print(f"  {col}: {pct:.1f}%")

    print("\nCategorical Distributions:")
    print(f"Regions: {df['region'].value_counts().to_dict()}")
    print(f"Ethnicities: {df['ethnicity'].value_counts().to_dict()}")


def main():
    """Main function to generate and save fake patient data."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate fake patients
    df = generate_fake_patients(1000)

    # Display summary
    display_summary(df)

    # Save data
    save_fake_data(df)

    print("\nGeneration complete! Use this data to validate your CVD risk models.")
    print("Example usage with SCORE2 model:")
    print("  from cvd_risk_calculator.models.score2 import SCORE2")
    print("  model = SCORE2()")
    print("  results = model.calculate_batch(df)")


if __name__ == "__main__":
    main()
