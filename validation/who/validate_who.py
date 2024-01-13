"""
WHO CVD Risk Validation Script.

This script validates the PyCVDRisk WHO CVD Risk implementation by:
1. Loading the synthetic validation dataset
2. Running the WHO model on applicable patients (age 40-79)
3. Exporting results for comparison with R implementation

Reference:
    WHO CVD Risk Chart Working Group. (2019).
    World Health Organization cardiovascular disease risk charts:
    revised models to estimate risk in 21 global regions.
    The Lancet Global Health, 7(10), e1332-e1345.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cvd_risk.models.who import WHO


def load_synthetic_data() -> pd.DataFrame:
    """Load and filter synthetic data for WHO validation."""
    data_path = Path(__file__).parent.parent / "synthetic_validation_data.csv"
    df = pd.read_csv(data_path)
    
    # WHO model is typically valid for ages 40-79
    df_filtered = df[(df["age"] >= 40) & (df["age"] <= 79)].copy()
    
    # Rename columns to match expected format
    df_filtered = df_filtered.rename(columns={
        "who_region": "region"
    })
    
    print(f"Loaded {len(df)} total patients")
    print(f"Filtered to {len(df_filtered)} patients (age 40-79)")
    
    return df_filtered


def run_single_patient_validation(model: WHO, df: pd.DataFrame, n_samples: int = 100):
    """Validate single-patient calculation method."""
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
                diabetes=bool(row["diabetes"]),
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
                "diabetes": row["diabetes"],
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


def run_batch_validation(model: WHO, df: pd.DataFrame):
    """Validate batch calculation method for performance and accuracy."""
    batch_df = df[["patient_id", "age", "sex", "systolic_bp", 
                   "total_cholesterol", "hdl_cholesterol", 
                   "smoking", "diabetes", "region"]].copy()
    
    start_time = time.perf_counter()
    results = model.calculate_batch(batch_df)
    end_time = time.perf_counter()
    
    elapsed_ms = (end_time - start_time) * 1000
    patients_per_sec = len(batch_df) / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    print(f"\nBatch Performance:")
    print(f"  Patients processed: {len(batch_df):,}")
    print(f"  Total time: {elapsed_ms:.2f} ms")
    print(f"  Throughput: {patients_per_sec:,.0f} patients/sec")
    
    results["calculation_method"] = "batch"
    
    return results


def validate_consistency(single_results: pd.DataFrame, batch_results: pd.DataFrame):
    """Validate that single and batch calculations produce identical results."""
    single_valid = single_results[single_results["risk_score"].notna()]
    
    single_subset = single_valid[["patient_id", "risk_score"]].rename(
        columns={"risk_score": "risk_score_single"}
    )
    batch_subset = batch_results[["patient_id", "risk_score"]].rename(
        columns={"risk_score": "risk_score_batch"}
    )
    
    merged = single_subset.merge(batch_subset, on="patient_id", how="inner")
    merged["diff"] = np.abs(merged["risk_score_single"] - merged["risk_score_batch"])
    max_diff = merged["diff"].max() if len(merged) > 0 else 0
    mean_diff = merged["diff"].mean() if len(merged) > 0 else 0
    
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
    print("WHO CVD RISK VALIDATION SUMMARY")
    print("=" * 70)
    
    valid_results = results[results["risk_score"].notna()]
    
    print(f"\nRisk Score Distribution:")
    print(f"  Min:    {valid_results['risk_score'].min():.2f}%")
    print(f"  Max:    {valid_results['risk_score'].max():.2f}%")
    print(f"  Mean:   {valid_results['risk_score'].mean():.2f}%")
    print(f"  Median: {valid_results['risk_score'].median():.2f}%")
    print(f"  Std:    {valid_results['risk_score'].std():.2f}%")
    
    print(f"\nRisk Category Distribution:")
    cat_counts = valid_results["risk_category"].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(valid_results) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    print(f"\nBy Region (Top 5):")
    region_stats = valid_results.groupby("region")["risk_score"].agg(["mean", "count"])
    region_stats = region_stats.sort_values("count", ascending=False).head(5)
    for region, row in region_stats.iterrows():
        print(f"  {region}: mean={row['mean']:.2f}%, n={int(row['count']):,}")
    
    print(f"\nBy Diabetes Status:")
    for diabetes in [True, False]:
        subset = valid_results[valid_results["diabetes"] == diabetes]
        if len(subset) > 0:
            status = "Diabetic" if diabetes else "Non-diabetic"
            print(f"  {status}: mean={subset['risk_score'].mean():.2f}%, n={len(subset):,}")


def main():
    """Main validation routine."""
    print("=" * 70)
    print("WHO CVD RISK MODEL VALIDATION")
    print("PyCVDRisk Package")
    print("=" * 70)
    
    model = WHO()
    print(f"\nModel: {model.model_name} v{model.model_version}")
    
    df = load_synthetic_data()
    
    if len(df) == 0:
        print("\nNo patients meet age criteria (40-79). Validation skipped.")
        return None
    
    print("\n" + "-" * 50)
    print("Running single-patient validation...")
    n_samples = min(100, len(df))
    single_results = run_single_patient_validation(model, df, n_samples=n_samples)
    
    print("\n" + "-" * 50)
    print("Running batch validation...")
    batch_results = run_batch_validation(model, df)
    
    print("\n" + "-" * 50)
    consistency = validate_consistency(single_results, batch_results)
    
    generate_summary_statistics(batch_results)
    
    output_dir = Path(__file__).parent
    
    single_output = output_dir / "who_single_results.csv"
    single_results.to_csv(single_output, index=False)
    print(f"\nExported single results to: {single_output}")
    
    batch_output = output_dir / "who_batch_results.csv"
    batch_results.to_csv(batch_output, index=False)
    print(f"Exported batch results to: {batch_output}")
    
    input_output = output_dir / "who_input_data.csv"
    df.to_csv(input_output, index=False)
    print(f"Exported input data to: {input_output}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return batch_results


if __name__ == "__main__":
    results = main()
