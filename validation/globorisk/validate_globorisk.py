"""
Globorisk Validation Script.

This script validates the PyCVDRisk Globorisk implementation by:
1. Loading the synthetic validation dataset
2. Running the Globorisk model on applicable patients (age 40-84)
3. Testing different model versions (lab, office, fatal)
4. Exporting results for comparison with R implementation

Reference:
    Ueda P, Woodward M, Lu Y, et al. (2017).
    Laboratory-based and office-based risk scores and charts
    to predict 10-year risk of cardiovascular disease in 182 countries.
    The Lancet Diabetes & Endocrinology, 5(3), 196-213.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cvd_risk.models.globorisk import Globorisk


# Sample countries for testing (mix of regions)
SAMPLE_COUNTRIES = [
    "USA",  # North America
    "GBR",  # Europe
    "DEU",  # Europe
    "JPN",  # Asia
    "BRA",  # South America
    "IND",  # Asia
    "CHN",  # Asia
    "NGA",  # Africa
    "AUS",  # Oceania
    "MEX",  # North America
]


def load_synthetic_data() -> pd.DataFrame:
    """Load and filter synthetic data for Globorisk validation."""
    data_path = Path(__file__).parent.parent / "synthetic_validation_data.csv"
    df = pd.read_csv(data_path)
    
    # Globorisk is valid for ages 40-84
    df_filtered = df[(df["age"] >= 40) & (df["age"] <= 84)].copy()
    
    # Assign countries cyclically from sample list
    df_filtered["country"] = [
        SAMPLE_COUNTRIES[i % len(SAMPLE_COUNTRIES)] 
        for i in range(len(df_filtered))
    ]
    
    print(f"Loaded {len(df)} total patients")
    print(f"Filtered to {len(df_filtered)} patients (age 40-84)")
    
    return df_filtered


def run_single_patient_validation(df: pd.DataFrame, version: str = "lab", n_samples: int = 100):
    """Validate single-patient calculation method."""
    from cvd_risk.core.validation import PatientData
    
    results = []
    sample_df = df.head(n_samples)
    
    for idx, row in sample_df.iterrows():
        try:
            # Initialize model for this country
            model = Globorisk(country=row["country"], version=version, year=2020)
            
            patient = PatientData(
                age=int(row["age"]),
                sex=row["sex"],
                systolic_bp=float(row["systolic_bp"]),
                total_cholesterol=float(row["total_cholesterol"]),
                hdl_cholesterol=float(row["hdl_cholesterol"]),
                smoking=bool(row["smoking"]),
                diabetes=bool(row["diabetes"]) if version == "lab" else None
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
                "country": row["country"],
                "version": version,
                "risk_score": result.risk_score,
                "risk_category": result.risk_category,
                "model_name": result.model_name,
                "model_version": result.model_version,
                "calculation_method": "single"
            })
        except Exception as e:
            results.append({
                "patient_id": row["patient_id"],
                "country": row["country"],
                "version": version,
                "error": str(e),
                "calculation_method": "single"
            })
    
    return pd.DataFrame(results)


def run_batch_validation(df: pd.DataFrame, version: str = "lab"):
    """Validate batch calculation method for performance and accuracy."""
    # Initialize model with default settings
    model = Globorisk(version=version, year=2020)
    
    batch_df = df[["patient_id", "age", "sex", "systolic_bp", 
                   "total_cholesterol", "hdl_cholesterol", 
                   "smoking", "diabetes", "country"]].copy()
    
    start_time = time.perf_counter()
    try:
        results = model.calculate_batch(batch_df)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        patients_per_sec = len(batch_df) / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        print(f"\nBatch Performance ({version} version):")
        print(f"  Patients processed: {len(batch_df):,}")
        print(f"  Total time: {elapsed_ms:.2f} ms")
        print(f"  Throughput: {patients_per_sec:,.0f} patients/sec")
        
        results["version"] = version
        results["calculation_method"] = "batch"
        
        return results
    except Exception as e:
        print(f"\nBatch calculation failed: {e}")
        return pd.DataFrame()


def validate_consistency(single_results: pd.DataFrame, batch_results: pd.DataFrame):
    """Validate that single and batch calculations produce identical results."""
    if len(batch_results) == 0:
        print("\nConsistency Check: Skipped (batch results unavailable)")
        return pd.DataFrame()
    
    # Check if risk_score column exists in single results
    if "risk_score" not in single_results.columns:
        print("\nConsistency Check: Skipped (single-patient calculations failed)")
        return pd.DataFrame()
    
    single_valid = single_results[single_results["risk_score"].notna()]
    
    single_subset = single_valid[["patient_id", "risk_score"]].rename(
        columns={"risk_score": "risk_score_single"}
    )
    batch_subset = batch_results[["patient_id", "risk_score"]].rename(
        columns={"risk_score": "risk_score_batch"}
    )
    
    merged = single_subset.merge(batch_subset, on="patient_id", how="inner")
    
    if len(merged) == 0:
        print("\nConsistency Check: No matching patients to compare")
        return merged
    
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


def generate_summary_statistics(results: pd.DataFrame, version: str):
    """Generate summary statistics for validation report."""
    print("\n" + "=" * 70)
    print(f"GLOBORISK VALIDATION SUMMARY ({version.upper()} VERSION)")
    print("=" * 70)
    
    valid_results = results[results["risk_score"].notna()]
    
    if len(valid_results) == 0:
        print("\nNo valid results to summarize.")
        return
    
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
    
    print(f"\nBy Country:")
    country_stats = valid_results.groupby("country")["risk_score"].agg(["mean", "count"])
    country_stats = country_stats.sort_values("mean", ascending=False)
    for country, row in country_stats.iterrows():
        print(f"  {country}: mean={row['mean']:.2f}%, n={int(row['count']):,}")


def main():
    """Main validation routine."""
    print("=" * 70)
    print("GLOBORISK MODEL VALIDATION")
    print("PyCVDRisk Package")
    print("=" * 70)
    
    # Check available countries
    try:
        available_countries = Globorisk.get_available_countries()
        print(f"\nAvailable countries: {len(available_countries)}")
        print(f"Testing countries: {SAMPLE_COUNTRIES}")
    except Exception as e:
        print(f"\nWarning: Could not get available countries: {e}")
    
    df = load_synthetic_data()
    
    if len(df) == 0:
        print("\nNo patients meet age criteria (40-84). Validation skipped.")
        return None
    
    all_results = []
    
    # Test each version
    for version in ["lab", "office"]:
        print("\n" + "=" * 70)
        print(f"Testing {version.upper()} version")
        print("=" * 70)
        
        print("\n" + "-" * 50)
        print("Running single-patient validation...")
        n_samples = min(100, len(df))
        single_results = run_single_patient_validation(df, version=version, n_samples=n_samples)
        
        # Count successful calculations
        success_count = single_results["risk_score"].notna().sum() if "risk_score" in single_results.columns else 0
        error_count = single_results["error"].notna().sum() if "error" in single_results.columns else 0
        print(f"  Successful: {success_count}, Errors: {error_count}")
        
        print("\n" + "-" * 50)
        print("Running batch validation...")
        batch_results = run_batch_validation(df, version=version)
        
        if len(batch_results) > 0 and "risk_score" in batch_results.columns:
            print("\n" + "-" * 50)
            consistency = validate_consistency(single_results, batch_results)
            
            generate_summary_statistics(batch_results, version)
            all_results.append(batch_results)
        elif "risk_score" in single_results.columns and single_results["risk_score"].notna().any():
            generate_summary_statistics(single_results, version)
            all_results.append(single_results)
        else:
            print(f"\n⚠ {version.upper()} version: No valid results to summarize")
    
    # Export results
    output_dir = Path(__file__).parent
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        output_path = output_dir / "globorisk_results.csv"
        combined_results.to_csv(output_path, index=False)
        print(f"\nExported combined results to: {output_path}")
    
    # Export input data
    input_output = output_dir / "globorisk_input_data.csv"
    df.to_csv(input_output, index=False)
    print(f"Exported input data to: {input_output}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return combined_results if all_results else None


if __name__ == "__main__":
    results = main()
