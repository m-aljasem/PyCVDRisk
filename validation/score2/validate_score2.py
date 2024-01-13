"""
SCORE2 Validation Script.

This script validates the PyCVDRisk SCORE2 implementation by:
1. Loading the synthetic validation dataset
2. Running the SCORE2 model on applicable patients (age 40-69)
3. Exporting results for comparison with R implementation

Reference:
    SCORE2 Working Group and ESC Cardiovascular Risk Collaboration. (2021).
    SCORE2 risk prediction algorithms: new models to estimate 10-year risk
    of cardiovascular disease in Europe. European Heart Journal, 42(25), 2439-2454.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cvd_risk.models.score2 import SCORE2


def load_synthetic_data() -> pd.DataFrame:
    """Load and filter synthetic data for SCORE2 validation."""
    data_path = Path(__file__).parent.parent / "synthetic_validation_data.csv"
    df = pd.read_csv(data_path)
    
    # SCORE2 is valid for ages 40-69
    df_filtered = df[(df["age"] >= 40) & (df["age"] <= 69)].copy()
    
    # Rename columns to match SCORE2 expected format
    df_filtered = df_filtered.rename(columns={
        "score2_region": "region"
    })
    
    print(f"Loaded {len(df)} total patients")
    print(f"Filtered to {len(df_filtered)} patients (age 40-69)")
    
    return df_filtered


def run_single_patient_validation(model: SCORE2, df: pd.DataFrame, n_samples: int = 100):
    """
    Validate single-patient calculation method.
    
    Returns individual patient results for detailed comparison.
    """
    from cvd_risk.core.validation import PatientData
    
    results = []
    sample_df = df.head(n_samples)
    
    for idx, row in sample_df.iterrows():
        try:
            patient = PatientData(
                age=int(row["age"]),
                sex=row["sex"],
                systolic_bp=float(row["systolic_bp"]),
                total_cholesterol=float(row["total_cholesterol"]),
                hdl_cholesterol=float(row["hdl_cholesterol"]),
                smoking=bool(row["smoking"]),
                region=row["region"]
            )
            
            result = model.calculate(patient)
            
            results.append({
                "patient_id": row["patient_id"],
                "age": row["age"],
                "sex": row["sex"],
                "systolic_bp": row["systolic_bp"],
                "total_cholesterol": row["total_cholesterol"],
                "hdl_cholesterol": row["hdl_cholesterol"],
                "smoking": row["smoking"],
                "region": row["region"],
                "risk_score": result.risk_score,
                "risk_category": result.risk_category,
                "model_name": result.model_name,
                "model_version": result.model_version,
                "calculation_method": "single"
            })
        except Exception as e:
            results.append({
                "patient_id": row["patient_id"],
                "error": str(e),
                "calculation_method": "single"
            })
    
    return pd.DataFrame(results)


def run_batch_validation(model: SCORE2, df: pd.DataFrame):
    """
    Validate batch calculation method for performance and accuracy.
    """
    # Prepare data for batch calculation
    batch_df = df[["patient_id", "age", "sex", "systolic_bp", 
                   "total_cholesterol", "hdl_cholesterol", 
                   "smoking", "region"]].copy()
    
    # Time the batch calculation
    start_time = time.perf_counter()
    results = model.calculate_batch(batch_df)
    end_time = time.perf_counter()
    
    elapsed_ms = (end_time - start_time) * 1000
    patients_per_sec = len(batch_df) / (end_time - start_time)
    
    print(f"\nBatch Performance:")
    print(f"  Patients processed: {len(batch_df):,}")
    print(f"  Total time: {elapsed_ms:.2f} ms")
    print(f"  Throughput: {patients_per_sec:,.0f} patients/sec")
    
    results["calculation_method"] = "batch"
    
    return results


def validate_consistency(single_results: pd.DataFrame, batch_results: pd.DataFrame):
    """
    Validate that single and batch calculations produce identical results.
    """
    # Merge on patient_id
    single_subset = single_results[["patient_id", "risk_score"]].rename(
        columns={"risk_score": "risk_score_single"}
    )
    batch_subset = batch_results[["patient_id", "risk_score"]].rename(
        columns={"risk_score": "risk_score_batch"}
    )
    
    merged = single_subset.merge(batch_subset, on="patient_id", how="inner")
    
    # Calculate differences
    merged["diff"] = np.abs(merged["risk_score_single"] - merged["risk_score_batch"])
    max_diff = merged["diff"].max()
    mean_diff = merged["diff"].mean()
    
    print(f"\nConsistency Check (single vs batch):")
    print(f"  Patients compared: {len(merged)}")
    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")
    
    if max_diff < 1e-6:
        print("  ✓ PASS: Single and batch calculations are consistent")
    else:
        print("  ✗ FAIL: Significant differences detected")
    
    return merged


def generate_summary_statistics(results: pd.DataFrame):
    """Generate summary statistics for validation report."""
    print("\n" + "=" * 70)
    print("SCORE2 VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\nRisk Score Distribution:")
    print(f"  Min:    {results['risk_score'].min():.2f}%")
    print(f"  Max:    {results['risk_score'].max():.2f}%")
    print(f"  Mean:   {results['risk_score'].mean():.2f}%")
    print(f"  Median: {results['risk_score'].median():.2f}%")
    print(f"  Std:    {results['risk_score'].std():.2f}%")
    
    print(f"\nRisk Category Distribution:")
    cat_counts = results["risk_category"].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(results) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    print(f"\nBy Sex:")
    for sex in ["male", "female"]:
        subset = results[results["sex"] == sex]
        print(f"  {sex.capitalize()}: mean={subset['risk_score'].mean():.2f}%, "
              f"median={subset['risk_score'].median():.2f}%")
    
    print(f"\nBy Region:")
    for region in ["low", "moderate", "high", "very_high"]:
        subset = results[results["region"] == region]
        if len(subset) > 0:
            print(f"  {region}: mean={subset['risk_score'].mean():.2f}%, "
                  f"n={len(subset):,}")


def main():
    """Main validation routine."""
    print("=" * 70)
    print("SCORE2 MODEL VALIDATION")
    print("PyCVDRisk Package")
    print("=" * 70)
    
    # Initialize model
    model = SCORE2()
    print(f"\nModel: {model.model_name} v{model.model_version}")
    print(f"Supported regions: {model.supported_regions}")
    
    # Load data
    df = load_synthetic_data()
    
    # Run single patient validation
    print("\n" + "-" * 50)
    print("Running single-patient validation (n=100)...")
    single_results = run_single_patient_validation(model, df, n_samples=100)
    
    # Run batch validation
    print("\n" + "-" * 50)
    print("Running batch validation...")
    batch_results = run_batch_validation(model, df)
    
    # Validate consistency
    print("\n" + "-" * 50)
    consistency = validate_consistency(single_results, batch_results)
    
    # Generate summary statistics
    generate_summary_statistics(batch_results)
    
    # Export results for R comparison
    output_dir = Path(__file__).parent
    
    # Single patient results (for detailed comparison)
    single_output = output_dir / "score2_single_results.csv"
    single_results.to_csv(single_output, index=False)
    print(f"\nExported single results to: {single_output}")
    
    # Batch results (full dataset)
    batch_output = output_dir / "score2_batch_results.csv"
    batch_results.to_csv(batch_output, index=False)
    print(f"Exported batch results to: {batch_output}")
    
    # Export input data for R replication
    input_output = output_dir / "score2_input_data.csv"
    df.to_csv(input_output, index=False)
    print(f"Exported input data to: {input_output}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return batch_results


if __name__ == "__main__":
    results = main()
