"""
Python vs R Comparison Script.

This script compares the outputs from Python (PyCVDRisk) and R implementations
of CVD risk models, generating comparison statistics and concordance metrics.

Usage:
    python compare_python_r.py <model_name>
    
    model_name: score2, score2op, score2dm, who, or globorisk
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(model_dir: Path, model_name: str) -> tuple:
    """Load Python and R results for comparison."""
    # Python results
    if model_name == "globorisk":
        py_batch = model_dir / "globorisk_results.csv"
    else:
        py_batch = model_dir / f"{model_name}_batch_results.csv"
    if py_batch.exists():
        python_df = pd.read_csv(py_batch)
        # For Globorisk, filter to lab version only to match R output
        if model_name == "globorisk" and "version" in python_df.columns:
            python_df = python_df[python_df["version"] == "lab"]
    else:
        raise FileNotFoundError(f"Python results not found: {py_batch}")
    
    # R results
    r_results = model_dir / f"{model_name}_r_results.csv"
    if r_results.exists():
        r_df = pd.read_csv(r_results)
    else:
        raise FileNotFoundError(f"R results not found: {r_results}")
    
    return python_df, r_df


def compare_results(python_df: pd.DataFrame, r_df: pd.DataFrame, 
                    py_score_col: str = "risk_score",
                    r_score_col: str = "r_risk_score") -> pd.DataFrame:
    """Merge and compare Python vs R results."""
    # Merge on patient_id
    merged = python_df.merge(
        r_df[["patient_id", r_score_col]], 
        on="patient_id", 
        how="inner",
        suffixes=("_py", "_r")
    )
    
    # Rename columns for clarity
    merged = merged.rename(columns={
        py_score_col: "python_risk",
        r_score_col: "r_risk"
    })
    
    # Calculate differences
    merged["abs_diff"] = np.abs(merged["python_risk"] - merged["r_risk"])
    merged["rel_diff"] = merged["abs_diff"] / (merged["r_risk"] + 1e-9) * 100
    
    return merged


def calculate_concordance(merged: pd.DataFrame) -> dict:
    """Calculate concordance statistics."""
    valid = merged.dropna(subset=["python_risk", "r_risk"])
    
    if len(valid) == 0:
        return {"error": "No valid comparisons"}
    
    abs_diff = valid["abs_diff"]
    
    stats = {
        "n_compared": len(valid),
        "mean_abs_diff": abs_diff.mean(),
        "max_abs_diff": abs_diff.max(),
        "median_abs_diff": abs_diff.median(),
        "std_abs_diff": abs_diff.std(),
        "within_0.1pct": (abs_diff <= 0.1).sum() / len(valid) * 100,
        "within_0.5pct": (abs_diff <= 0.5).sum() / len(valid) * 100,
        "within_1.0pct": (abs_diff <= 1.0).sum() / len(valid) * 100,
        "within_2.0pct": (abs_diff <= 2.0).sum() / len(valid) * 100,
        "correlation": valid["python_risk"].corr(valid["r_risk"]),
    }
    
    return stats


def print_report(model_name: str, stats: dict, merged: pd.DataFrame):
    """Print comparison report."""
    print("=" * 70)
    print(f"{model_name.upper()} PYTHON vs R COMPARISON REPORT")
    print("=" * 70)
    
    print(f"\nPatients Compared: {stats['n_compared']:,}")
    print(f"Pearson Correlation: {stats['correlation']:.6f}")
    
    print("\n" + "-" * 50)
    print("ABSOLUTE DIFFERENCE (percentage points)")
    print("-" * 50)
    print(f"  Mean:   {stats['mean_abs_diff']:.4f}")
    print(f"  Median: {stats['median_abs_diff']:.4f}")
    print(f"  Max:    {stats['max_abs_diff']:.4f}")
    print(f"  Std:    {stats['std_abs_diff']:.4f}")
    
    print("\n" + "-" * 50)
    print("CONCORDANCE THRESHOLDS")
    print("-" * 50)
    print(f"  Within ±0.1%: {stats['within_0.1pct']:.1f}%")
    print(f"  Within ±0.5%: {stats['within_0.5pct']:.1f}%")
    print(f"  Within ±1.0%: {stats['within_1.0pct']:.1f}%")
    print(f"  Within ±2.0%: {stats['within_2.0pct']:.1f}%")
    
    # Show worst discrepancies
    print("\n" + "-" * 50)
    print("TOP 5 LARGEST DISCREPANCIES")
    print("-" * 50)
    worst = merged.nlargest(5, "abs_diff")
    for _, row in worst.iterrows():
        print(f"  {row['patient_id']}: Py={row['python_risk']:.2f}%, "
              f"R={row['r_risk']:.2f}%, Diff={row['abs_diff']:.4f}")
    
    # Verdict
    print("\n" + "=" * 70)
    if stats["within_1.0pct"] >= 95:
        print("✓ EXCELLENT CONCORDANCE: ≥95% within 1.0 percentage point")
    elif stats["within_2.0pct"] >= 95:
        print("✓ GOOD CONCORDANCE: ≥95% within 2.0 percentage points")
    elif stats["within_2.0pct"] >= 90:
        print("⚠ ACCEPTABLE CONCORDANCE: ≥90% within 2.0 percentage points")
    else:
        print("✗ POOR CONCORDANCE: <90% within 2.0 percentage points")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare Python vs R CVD risk results")
    parser.add_argument("model", choices=["score2", "score2op", "score2dm", "who", "globorisk"],
                        help="Model to compare")
    args = parser.parse_args()
    
    model_name = args.model
    model_dir = Path(__file__).parent / model_name
    
    print(f"Loading results for {model_name}...")
    
    try:
        python_df, r_df = load_results(model_dir, model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure to run both Python and R validation scripts first:")
        print(f"  python validation/{model_name}/validate_{model_name}.py")
        print(f"  Rscript validation/{model_name}/validate_{model_name}_r.R")
        sys.exit(1)
    
    # Determine score column names
    if model_name == "globorisk":
        r_score_col = "r_risk_score_lab"  # Use lab version for comparison
    else:
        r_score_col = "r_risk_score"
    
    merged = compare_results(python_df, r_df, r_score_col=r_score_col)
    stats = calculate_concordance(merged)
    
    print_report(model_name, stats, merged)
    
    # Export comparison
    output_path = model_dir / f"{model_name}_comparison.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nExported comparison to: {output_path}")


if __name__ == "__main__":
    main()
