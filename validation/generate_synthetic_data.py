"""
Synthetic Validation Dataset Generator for PyCVDRisk.

Generates a synthetic dataset of N hypothetical patients using Monte Carlo
sampling with uniform distributions across the full physiological range.
This ensures robust testing including edge cases.

Reference:
    Methods Section - Synthetic Validation Dataset
    Monte Carlo sampling with uniform distributions for rigorous edge-case testing.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_validation_data(
    n_patients: int = 10000,
    seed: int = 42,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Generate synthetic patient dataset using Monte Carlo sampling.
    
    Covariates are sampled uniformly across their full physiological range
    to ensure edge cases (e.g., extreme hypertension, boundary ages) are
    rigorously tested.
    
    Parameters
    ----------
    n_patients : int
        Number of synthetic patients to generate (default: 10,000)
    seed : int
        Random seed for reproducibility (default: 42)
    output_format : str
        Output format: 'csv', 'parquet', or 'both' (default: 'csv')
        
    Returns
    -------
    pd.DataFrame
        Synthetic patient dataset with all covariates
        
    Notes
    -----
    All continuous variables are sampled from uniform distributions.
    Categorical/binary variables are sampled with equal probability.
    This differs from real-world distributions but ensures comprehensive
    coverage of the input space for validation purposes.
    """
    rng = np.random.default_rng(seed)
    
    # =========================================================================
    # CORE DEMOGRAPHICS
    # =========================================================================
    age = rng.integers(low=20, high=91, size=n_patients)  # 20-90 years
    sex = rng.choice(["male", "female"], size=n_patients)
    
    # =========================================================================
    # CARDIOVASCULAR PARAMETERS
    # =========================================================================
    # Blood Pressure (mmHg) - Uniform across physiological range
    systolic_bp = rng.uniform(low=90.0, high=200.0, size=n_patients)
    diastolic_bp = rng.uniform(low=50.0, high=120.0, size=n_patients)
    
    # Ensure SBP > DBP (physiologically required)
    # If violated, swap values
    swap_mask = systolic_bp < diastolic_bp
    systolic_bp[swap_mask], diastolic_bp[swap_mask] = (
        diastolic_bp[swap_mask], systolic_bp[swap_mask]
    )
    
    # Heart rate (bpm)
    heart_rate = rng.uniform(low=40.0, high=180.0, size=n_patients)
    
    # =========================================================================
    # LIPID PROFILE (mmol/L)
    # =========================================================================
    # Note: Paper mentions mg/dL for cholesterol (100-400 mg/dL)
    # Converting: 1 mmol/L ≈ 38.67 mg/dL
    # 100 mg/dL ≈ 2.59 mmol/L, 400 mg/dL ≈ 10.34 mmol/L
    total_cholesterol = rng.uniform(low=2.5, high=10.5, size=n_patients)  # ~100-400 mg/dL
    
    # HDL cholesterol (0.5-3.5 mmol/L typical clinical range)
    hdl_cholesterol = rng.uniform(low=0.5, high=3.5, size=n_patients)
    
    # LDL cholesterol - derived or direct
    # Ensure LDL doesn't exceed TC - HDL physiologically
    ldl_cholesterol = rng.uniform(low=1.0, high=7.0, size=n_patients)
    ldl_cholesterol = np.minimum(ldl_cholesterol, total_cholesterol - hdl_cholesterol - 0.5)
    ldl_cholesterol = np.maximum(ldl_cholesterol, 0.5)  # Floor at 0.5
    
    # Non-HDL cholesterol (TC - HDL)
    non_hdl_cholesterol = total_cholesterol - hdl_cholesterol
    
    # Triglycerides (0.5-5.0 mmol/L)
    triglycerides = rng.uniform(low=0.5, high=5.0, size=n_patients)
    
    # =========================================================================
    # DIABETES PARAMETERS
    # =========================================================================
    diabetes = rng.choice([True, False], size=n_patients)
    type1_diabetes = np.zeros(n_patients, dtype=bool)
    type2_diabetes = np.zeros(n_patients, dtype=bool)
    
    # Among diabetics: 10% Type 1, 90% Type 2
    diabetic_mask = diabetes
    n_diabetics = diabetic_mask.sum()
    type1_mask = rng.choice([True, False], size=n_diabetics, p=[0.1, 0.9])
    type1_diabetes[diabetic_mask] = type1_mask
    type2_diabetes[diabetic_mask] = ~type1_mask
    
    # HbA1c (mmol/mol) - higher in diabetics
    hba1c = np.where(
        diabetes,
        rng.uniform(low=42.0, high=120.0, size=n_patients),  # Diabetic range
        rng.uniform(low=20.0, high=42.0, size=n_patients)    # Non-diabetic range
    )
    
    # Diabetes age at diagnosis (only for diabetics)
    diabetes_age = np.where(
        diabetes,
        rng.integers(low=1, high=age + 1),  # Diagnosed before current age
        np.nan
    )
    
    # =========================================================================
    # RENAL FUNCTION
    # =========================================================================
    # eGFR (ml/min/1.73m²) - 15-120 range covers CKD stages
    egfr = rng.uniform(low=15.0, high=120.0, size=n_patients)
    
    # Creatinine (mg/dL) - inversely related to eGFR
    creatinine = rng.uniform(low=0.5, high=5.0, size=n_patients)
    
    # ACR - Albumin-to-creatinine ratio (mg/g)
    acr = rng.uniform(low=1.0, high=500.0, size=n_patients)
    
    # =========================================================================
    # ANTHROPOMETRICS
    # =========================================================================
    # BMI (kg/m²)
    bmi = rng.uniform(low=15.0, high=50.0, size=n_patients)
    
    # Height (cm) and Weight (kg) for derivation if needed
    height = rng.uniform(low=140.0, high=200.0, size=n_patients)
    weight = bmi * ((height / 100) ** 2)
    
    # Waist circumference (cm)
    waist_circumference = rng.uniform(low=60.0, high=150.0, size=n_patients)
    
    # =========================================================================
    # BEHAVIORAL / LIFESTYLE
    # =========================================================================
    smoking = rng.choice([True, False], size=n_patients)
    
    # Smoking category: 1=non-smoker, 2=ex-smoker, 3=light, 4=heavy
    smoking_category = np.where(
        smoking,
        rng.choice([3, 4], size=n_patients),  # Current smokers: light or heavy
        rng.choice([1, 2], size=n_patients)   # Non-current: never or ex
    )
    
    alcohol_use = rng.choice([True, False], size=n_patients)
    
    # =========================================================================
    # MEDICATIONS
    # =========================================================================
    antihypertensive = rng.choice([True, False], size=n_patients)
    statin_use = rng.choice([True, False], size=n_patients)
    aspirin_use = rng.choice([True, False], size=n_patients)
    
    # =========================================================================
    # COMORBIDITIES & HISTORY
    # =========================================================================
    family_history = rng.choice([True, False], size=n_patients)
    hypertension = rng.choice([True, False], size=n_patients)
    treated_hypertension = hypertension & antihypertensive  # Logical constraint
    
    atrial_fibrillation = rng.choice([True, False], size=n_patients, p=[0.1, 0.9])
    previous_pci = rng.choice([True, False], size=n_patients, p=[0.05, 0.95])
    previous_cabg = rng.choice([True, False], size=n_patients, p=[0.03, 0.97])
    atherosclerotic_disease = rng.choice([True, False], size=n_patients, p=[0.1, 0.9])
    
    # QRISK3 comorbidities
    rheumatoid_arthritis = rng.choice([True, False], size=n_patients, p=[0.02, 0.98])
    renal_disease = rng.choice([True, False], size=n_patients, p=[0.05, 0.95])
    systemic_lupus = rng.choice([True, False], size=n_patients, p=[0.01, 0.99])
    severe_mental_illness = rng.choice([True, False], size=n_patients, p=[0.03, 0.97])
    migraine = rng.choice([True, False], size=n_patients, p=[0.1, 0.9])
    
    # Medications (QRISK3)
    atypical_antipsychotics = rng.choice([True, False], size=n_patients, p=[0.02, 0.98])
    corticosteroids = rng.choice([True, False], size=n_patients, p=[0.03, 0.97])
    
    # Erectile dysfunction (male only)
    impotence = np.where(
        sex == "male",
        rng.choice([True, False], size=n_patients, p=[0.15, 0.85]),
        False
    )
    
    # =========================================================================
    # ETHNICITY & RACE
    # =========================================================================
    ethnicity = rng.choice(
        ["white", "south_asian", "black", "chinese", "mixed", "other"],
        size=n_patients,
        p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]  # Approximate distribution
    )
    
    race = rng.choice(
        ["white", "black", "other"],
        size=n_patients,
        p=[0.7, 0.15, 0.15]
    )
    
    # =========================================================================
    # REGIONAL ASSIGNMENT (for SCORE2 / WHO models)
    # =========================================================================
    # SCORE2 regions
    score2_region = rng.choice(
        ["low", "moderate", "high", "very_high"],
        size=n_patients
    )
    
    # WHO epidemiological subregions
    who_region = rng.choice(
        ["AFR_D", "AFR_E", "AMR_A", "AMR_B", "AMR_D", 
         "EMR_B", "EMR_D", "EUR_A", "EUR_B", "EUR_C",
         "SEAR_B", "SEAR_D", "WPR_A", "WPR_B"],
        size=n_patients
    )
    
    # =========================================================================
    # SOCIOECONOMIC
    # =========================================================================
    # Townsend deprivation score (-7 to +11 in UK)
    townsend_deprivation = rng.uniform(low=-7.0, high=11.0, size=n_patients)
    
    # =========================================================================
    # SYSTOLIC BP VARIABILITY (QRISK3)
    # =========================================================================
    sbps5 = rng.uniform(low=0.0, high=30.0, size=n_patients)
    
    # =========================================================================
    # ASSEMBLE DATAFRAME
    # =========================================================================
    df = pd.DataFrame({
        # IDs
        "patient_id": [f"SYN_{i:06d}" for i in range(n_patients)],
        
        # Demographics
        "age": age,
        "sex": sex,
        "ethnicity": ethnicity,
        "race": race,
        
        # Cardiovascular
        "systolic_bp": np.round(systolic_bp, 1),
        "diastolic_bp": np.round(diastolic_bp, 1),
        "heart_rate": np.round(heart_rate, 1),
        
        # Lipids
        "total_cholesterol": np.round(total_cholesterol, 2),
        "hdl_cholesterol": np.round(hdl_cholesterol, 2),
        "ldl_cholesterol": np.round(ldl_cholesterol, 2),
        "non_hdl_cholesterol": np.round(non_hdl_cholesterol, 2),
        "triglycerides": np.round(triglycerides, 2),
        
        # Diabetes
        "diabetes": diabetes,
        "type1_diabetes": type1_diabetes,
        "type2_diabetes": type2_diabetes,
        "hba1c": np.round(hba1c, 1),
        "diabetes_age": diabetes_age,
        
        # Renal
        "egfr": np.round(egfr, 1),
        "creatinine": np.round(creatinine, 2),
        "acr": np.round(acr, 1),
        
        # Anthropometrics
        "bmi": np.round(bmi, 1),
        "height": np.round(height, 1),
        "weight": np.round(weight, 1),
        "waist_circumference": np.round(waist_circumference, 1),
        
        # Lifestyle
        "smoking": smoking,
        "smoking_category": smoking_category,
        "alcohol_use": alcohol_use,
        
        # Medications
        "antihypertensive": antihypertensive,
        "statin_use": statin_use,
        "aspirin_use": aspirin_use,
        "atypical_antipsychotics": atypical_antipsychotics,
        "corticosteroids": corticosteroids,
        
        # Comorbidities
        "family_history": family_history,
        "hypertension": hypertension,
        "treated_hypertension": treated_hypertension,
        "atrial_fibrillation": atrial_fibrillation,
        "previous_pci": previous_pci,
        "previous_cabg": previous_cabg,
        "atherosclerotic_disease": atherosclerotic_disease,
        "rheumatoid_arthritis": rheumatoid_arthritis,
        "renal_disease": renal_disease,
        "systemic_lupus": systemic_lupus,
        "severe_mental_illness": severe_mental_illness,
        "migraine": migraine,
        "impotence": impotence,
        
        # Regions
        "score2_region": score2_region,
        "who_region": who_region,
        
        # Socioeconomic
        "townsend_deprivation": np.round(townsend_deprivation, 2),
        
        # BP variability
        "sbps5": np.round(sbps5, 1),
    })
    
    return df


def save_dataset(df: pd.DataFrame, output_dir: Path, output_format: str = "csv"):
    """Save the synthetic dataset to file(s)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_format in ("csv", "both"):
        csv_path = output_dir / "synthetic_validation_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
    
    if output_format in ("parquet", "both"):
        parquet_path = output_dir / "synthetic_validation_data.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet: {parquet_path}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics of the generated dataset."""
    print("\n" + "=" * 70)
    print("SYNTHETIC VALIDATION DATASET SUMMARY")
    print("=" * 70)
    print(f"Total Patients: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print()
    
    print("CONTINUOUS VARIABLES (Uniform Sampling Ranges):")
    print("-" * 50)
    continuous_cols = [
        "age", "systolic_bp", "total_cholesterol", "hdl_cholesterol",
        "bmi", "egfr", "hba1c"
    ]
    for col in continuous_cols:
        if col in df.columns:
            print(f"  {col:25s}: [{df[col].min():8.2f}, {df[col].max():8.2f}]")
    
    print()
    print("CATEGORICAL DISTRIBUTIONS:")
    print("-" * 50)
    print(f"  Sex: {df['sex'].value_counts().to_dict()}")
    print(f"  Smoking: {df['smoking'].value_counts().to_dict()}")
    print(f"  Diabetes: {df['diabetes'].value_counts().to_dict()}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic validation dataset for PyCVDRisk"
    )
    parser.add_argument(
        "-n", "--n_patients",
        type=int,
        default=10000,
        help="Number of patients to generate (default: 10000)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["csv", "parquet", "both"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    print(f"Generating {args.n_patients:,} synthetic patients (seed={args.seed})...")
    df = generate_synthetic_validation_data(
        n_patients=args.n_patients,
        seed=args.seed
    )
    
    # Print summary
    print_summary(df)
    
    # Save
    save_dataset(df, Path(args.output_dir), args.format)
    
    print("\nDone!")
