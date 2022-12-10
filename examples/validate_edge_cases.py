"""
Edge Case Validation for CVD Risk Models

This script tests CVD risk models with edge case patient data to ensure
they handle boundary conditions, extreme values, and potential error cases
gracefully.

WARNING: Edge cases are designed to stress test models and may produce
unusual results. Use only for validation and testing purposes.

Usage:
    python validate_edge_cases.py

Requirements:
    - Edge case patient data (run generate_edge_case_patients.py first)
    - CVD risk models installed
"""

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cvd_risk_calculator.models.score2 import SCORE2


def load_edge_case_patients(csv_path: str = "data/edge_case_patients.csv") -> pd.DataFrame:
    """Load edge case patient data from CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Edge case data not found at {csv_path}. Run generate_edge_case_patients.py first.")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} edge case patients from {csv_path}")
    return df


def test_model_with_edge_cases(model_class, model_name: str, df: pd.DataFrame) -> Dict:
    """Test a specific model with edge case data."""
    print(f"\n{'='*60}")
    print(f"TESTING {model_name.upper()} WITH EDGE CASES")
    print('='*60)

    results = {
        "model_name": model_name,
        "success": False,
        "total_patients": len(df),
        "successful_calculations": 0,
        "failed_calculations": 0,
        "failures": [],
        "risk_range": None,
        "extreme_cases": []
    }

    try:
        model = model_class()

        # Try to calculate for all patients
        try:
            calculation_results = model.calculate_batch(df)
            results["success"] = True
            results["successful_calculations"] = len(calculation_results)
            results["risk_range"] = (
                calculation_results["risk_score"].min(),
                calculation_results["risk_score"].max()
            )

            # Identify most extreme cases
            top_risks = calculation_results.nlargest(3, "risk_score")
            for _, row in top_risks.iterrows():
                patient_desc = df[df["patient_id"] == row["patient_id"]]["description"].iloc[0]
                results["extreme_cases"].append({
                    "patient_id": row["patient_id"],
                    "risk_score": row["risk_score"],
                    "risk_category": row["risk_category"],
                    "description": patient_desc
                })

            print("✓ All edge cases processed successfully")
            print(f"Risk score range: {results['risk_range'][0]:.3f}% - {results['risk_range'][1]:.3f}%")

        except Exception as e:
            results["failures"].append({
                "type": "batch_calculation_failed",
                "error": str(e),
                "traceback": str(e.__traceback__)
            })
            print(f"✗ Batch calculation failed: {e}")

        # Test individual patient calculations
        failed_individual = 0
        for _, patient_row in df.iterrows():
            patient_data = patient_row.to_dict()
            # Remove edge case metadata for model input
            patient_data = {k: v for k, v in patient_data.items()
                          if not k.startswith('edge_case') and k != 'description'}

            try:
                # Convert to proper types
                patient_data["age"] = int(patient_data["age"])
                patient_data["smoking"] = bool(patient_data["smoking"])
                if pd.notna(patient_data.get("diabetes")):
                    patient_data["diabetes"] = bool(patient_data["diabetes"])
                if pd.notna(patient_data.get("family_history")):
                    patient_data["family_history"] = bool(patient_data["family_history"])
                if pd.notna(patient_data.get("antihypertensive")):
                    patient_data["antihypertensive"] = bool(patient_data["antihypertensive"])

                # Try individual calculation
                result = model.calculate(patient_data)
                results["successful_calculations"] = max(results["successful_calculations"],
                                                       results.get("successful_calculations", 0) + 1)

            except Exception as e:
                failed_individual += 1
                results["failures"].append({
                    "type": "individual_calculation_failed",
                    "patient_id": patient_data["patient_id"],
                    "error": str(e)
                })

        if failed_individual > 0:
            print(f"⚠️  {failed_individual} individual calculations failed")

    except Exception as e:
        results["failures"].append({
            "type": "model_initialization_failed",
            "error": str(e)
        })
        print(f"✗ Model initialization failed: {e}")

    results["failed_calculations"] = len(results["failures"])

    return results


def analyze_edge_case_performance(results: List[Dict]) -> None:
    """Analyze performance across all tested models."""
    print(f"\n{'='*60}")
    print("EDGE CASE VALIDATION SUMMARY")
    print('='*60)

    successful_models = sum(1 for r in results if r["success"])
    total_models = len(results)

    print(f"Models tested: {total_models}")
    print(f"Models handling edge cases successfully: {successful_models}")

    if successful_models > 0:
        print("\nSuccessful models:")
        for result in results:
            if result["success"]:
                risk_min, risk_max = result["risk_range"]
                print("4s: Risk range {:.3f}% - {:.3f}%")

    if any(not r["success"] for r in results):
        print("\nModels with failures:")
        for result in results:
            if not result["success"]:
                print(f"  {result['model_name']}: {len(result['failures'])} failures")

    # Analyze edge case categories
    print("\nEdge case handling:")
    edge_case_types = set()
    for result in results:
        if result["success"] and result["extreme_cases"]:
            for case in result["extreme_cases"]:
                edge_case_types.add(case["patient_id"].split("_")[0])

    print(f"  Edge case types tested: {len(edge_case_types)}")
    print(f"  Types: {', '.join(sorted(edge_case_types))}")


def main():
    """Main validation function."""
    print("CVD Risk Model Edge Case Validation")
    print("="*60)

    try:
        # Load edge case patient data
        df = load_edge_case_patients()

        # Test different models
        model_results = []

        # Test SCORE2 model
        score2_result = test_model_with_edge_cases(SCORE2, "SCORE2", df)
        model_results.append(score2_result)

        # TODO: Add other models here as they become available
        # from cvd_risk_calculator.models.framingham import Framingham
        # framingham_result = test_model_with_edge_cases(Framingham, "Framingham", df)
        # model_results.append(framingham_result)

        # Analyze overall performance
        analyze_edge_case_performance(model_results)

        print(f"\n{'='*60}")
        print("EDGE CASE VALIDATION COMPLETE")
        print('='*60)

        all_successful = all(r["success"] for r in model_results)
        if all_successful:
            print("✅ All tested models handled edge cases successfully!")
        else:
            print("⚠️  Some models failed edge case validation.")
            print("   Review the failures above and consider model improvements.")

        print("\nRecommendations:")
        print("1. Monitor for unusually high or low risk scores")
        print("2. Check biological implausibility (e.g., HDL > Total cholesterol)")
        print("3. Validate age-specific behavior (SCORE2 optimal range: 40-69)")
        print("4. Test with additional models as they are implemented")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
