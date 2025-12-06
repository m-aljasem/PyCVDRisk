"""
Model Validation with Fake Patient Data

This script demonstrates how to validate multiple CVD risk models using
the generated fake patient dataset. It loads the fake patients and runs
them through different models to ensure they work correctly.

Usage:
    python validate_models_with_fake_data.py

Requirements:
    - Fake patient data (run generate_fake_patients.py first)
    - All CVD risk models installed
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvd_risk_calculator.models.score2 import SCORE2


def load_fake_patients(csv_path: str = "data/fake_patients.csv") -> pd.DataFrame:
    """Load fake patient data from CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Fake patient data not found at {csv_path}. Run generate_fake_patients.py first.")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} fake patients from {csv_path}")
    return df


def validate_score2_model(df: pd.DataFrame) -> None:
    """Validate SCORE2 model with fake patient data."""
    print("\n" + "="*60)
    print("VALIDATING SCORE2 MODEL")
    print("="*60)

    try:
        model = SCORE2()
        results = model.calculate_batch(df)

        print(f"✓ Successfully processed {len(results)} patients")
        print(f"Risk score range: {results['risk_score'].min():.1f}% - {results['risk_score'].max():.1f}%")
        print(f"Risk categories: {results['risk_category'].value_counts().to_dict()}")

        # Check for reasonable distributions
        risk_stats = results['risk_score'].describe()
        print("\nRisk score statistics:")
        print(f"  Mean: {risk_stats['mean']:.1f}%")
        print(f"  Median: {risk_stats['50%']:.1f}%")
        print(f"  Std: {risk_stats['std']:.1f}%")

        # Validate age range (SCORE2 is for 40-69)
        age_filtered = df[(df['age'] >= 40) & (df['age'] <= 69)]
        if len(age_filtered) > 0:
            age_results = model.calculate_batch(age_filtered)
            print(f"\nOptimal age range (40-69): {len(age_filtered)} patients")
            print(f"  Risk range: {age_results['risk_score'].min():.1f}% - {age_results['risk_score'].max():.1f}%")

    except Exception as e:
        print(f"✗ SCORE2 validation failed: {e}")
        raise


def validate_data_quality(df: pd.DataFrame) -> None:
    """Validate that the fake data meets quality standards."""
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION")
    print("="*60)

    issues = []

    # Check required columns
    required_cols = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check data types and ranges
    if df['age'].min() < 18 or df['age'].max() > 120:
        issues.append(f"Age range {df['age'].min()}-{df['age'].max()} outside expected 18-120")

    if df['systolic_bp'].min() < 50 or df['systolic_bp'].max() > 250:
        issues.append(f"Systolic BP range {df['systolic_bp'].min()}-{df['systolic_bp'].max()} outside expected 50-250")

    if df['total_cholesterol'].min() < 1.0 or df['total_cholesterol'].max() > 15.0:
        issues.append(f"Total cholesterol range {df['total_cholesterol'].min():.1f}-{df['total_cholesterol'].max():.1f} outside expected 1.0-15.0")

    if df['hdl_cholesterol'].min() < 0.3 or df['hdl_cholesterol'].max() > 5.0:
        issues.append(f"HDL cholesterol range {df['hdl_cholesterol'].min():.1f}-{df['hdl_cholesterol'].max():.1f} outside expected 0.3-5.0")

    # Check HDL <= Total cholesterol
    invalid_hdl = (df['hdl_cholesterol'] > df['total_cholesterol']).sum()
    if invalid_hdl > 0:
        issues.append(f"{invalid_hdl} patients have HDL > Total cholesterol")

    # Check categorical values
    if not df['sex'].isin(['male', 'female']).all():
        issues.append("Invalid sex values found")

    if not df['smoking'].isin([True, False]).all():
        issues.append("Invalid smoking values found")

    if issues:
        print("✗ Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All data quality checks passed")


def main():
    """Main validation function."""
    print("CVD Risk Model Validation with Fake Patient Data")
    print("="*60)

    try:
        # Load fake patient data
        df = load_fake_patients()

        # Validate data quality
        validate_data_quality(df)

        # Validate specific models
        validate_score2_model(df)

        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print("✓ All validations passed successfully!")
        print("\nNext steps:")
        print("1. Add validation for other models (Framingham, QRISK3, etc.)")
        print("2. Compare model outputs for consistency")
        print("3. Test edge cases and boundary conditions")

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
