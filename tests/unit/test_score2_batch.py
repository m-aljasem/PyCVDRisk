"""Additional tests for SCORE2 batch processing and edge cases."""

import pandas as pd
import pytest

from cvd_risk.core.validation import PatientData
from cvd_risk.models.score2 import SCORE2
from cvd_risk.models.score2.score2 import categorize_esc2021


class TestSCORE2Batch:
    """Test SCORE2 batch processing."""

    def test_batch_processing_success(self) -> None:
        """Test successful batch processing."""
        model = SCORE2()
        df = pd.DataFrame(
            {
                "age": [55, 60, 45],
                "sex": ["male", "female", "male"],
                "systolic_bp": [140.0, 130.0, 150.0],
                "total_cholesterol": [6.0, 5.5, 7.0],
                "hdl_cholesterol": [1.2, 1.5, 1.0],
                "smoking": [True, False, True],
                "region": ["moderate", "low", "high"],
            }
        )

        results = model.calculate_batch(df)

        assert len(results) == 3
        assert "risk_score" in results.columns
        assert "risk_category" in results.columns
        assert "model_name" in results.columns
        assert all(results["model_name"] == "SCORE2")
        assert all(results["risk_score"] >= 0)
        assert all(results["risk_score"] <= 100)

    def test_batch_missing_columns(self) -> None:
        """Test batch processing with missing columns."""
        model = SCORE2()
        df = pd.DataFrame(
            {
                "age": [55, 60],
                "sex": ["male", "female"],
                # Missing required columns
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            model.calculate_batch(df)

    def test_batch_with_invalid_data(self) -> None:
        """Test batch processing handles invalid data gracefully."""
        model = SCORE2()
        df = pd.DataFrame(
            {
                "age": [55, 15],  # Invalid age
                "sex": ["male", "female"],
                "systolic_bp": [140.0, 130.0],
                "total_cholesterol": [6.0, 5.5],
                "hdl_cholesterol": [1.2, 1.5],
                "smoking": [True, False],
                "region": ["moderate", "low"],
            }
        )

        results = model.calculate_batch(df)

        # First row should succeed
        assert not pd.isna(results.iloc[0]["risk_score"])
        # Second row should have error
        assert pd.isna(results.iloc[1]["risk_score"])
        assert results.iloc[1]["risk_category"] == "error"

    @pytest.mark.parametrize(
        "age, risk, expected",
        [
            # ESC 2021 thresholds, age < 50: 2.5% / 7.5%
            (45, 2.49, "low_to_moderate"),
            (45, 2.5, "high"),
            (45, 7.49, "high"),
            (45, 7.5, "very_high"),
            # age 50-69: 5% / 10%
            (55, 4.99, "low_to_moderate"),
            (55, 5.0, "high"),
            (55, 9.99, "high"),
            (55, 10.0, "very_high"),
            (55, 12.0, "very_high"),  # was mislabelled "high" before the fix
            # age >= 70: 7.5% / 15%
            (75, 7.49, "low_to_moderate"),
            (75, 7.5, "high"),
            (75, 15.0, "very_high"),
        ],
    )
    def test_risk_categorization_boundary_values(self, age: int, risk: float, expected: str) -> None:
        """ESC 2021 age-specific categories for the SCORE2 family."""
        assert str(categorize_esc2021(risk, age)) == expected

    def test_batch_flags_bad_region_and_sex(self) -> None:
        """Rows with unknown region or sex get an error, not a silent NaN."""
        df = pd.DataFrame(
            {
                "age": [55, 55, 55],
                "sex": ["male", "M", "male"],
                "systolic_bp": [140.0] * 3,
                "total_cholesterol": [6.0] * 3,
                "hdl_cholesterol": [1.2] * 3,
                "smoking": [True] * 3,
                "region": ["moderate", "moderate", "Moderate"],
            }
        )
        results = SCORE2().calculate_batch(df)
        assert results["error"].isna().tolist() == [True, False, False]
        assert results["risk_category"].tolist()[1:] == ["error", "error"]
        assert results["risk_score"].isna().tolist() == [False, True, True]

    def test_batch_matches_single(self, sample_patient: PatientData) -> None:
        """Batch and single-patient paths give identical results."""
        model = SCORE2()
        single = model.calculate(sample_patient)
        batch = model.calculate_batch(pd.DataFrame([sample_patient.model_dump()])).iloc[0]
        assert batch["risk_score"] == single.risk_score
        assert batch["risk_category"] == single.risk_category

    def test_all_regions(self) -> None:
        """Test calculation works for all supported regions."""
        model = SCORE2()
        patient_base = {
            "age": 55,
            "sex": "male",
            "systolic_bp": 140.0,
            "total_cholesterol": 6.0,
            "hdl_cholesterol": 1.2,
            "smoking": True,
        }

        for region in ["low", "moderate", "high", "very_high"]:
            patient = PatientData(**patient_base, region=region)
            result = model.calculate(patient)
            assert result.risk_score >= 0
            assert result.risk_score <= 100
            assert result.risk_category in ["low_to_moderate", "high", "very_high"]

    def test_extreme_values(self) -> None:
        """Test calculation with extreme but valid values."""
        model = SCORE2()

        # Very old patient
        patient_old = PatientData(
            age=69,  # Upper limit of optimal range
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )
        result_old = model.calculate(patient_old)
        assert result_old.risk_score >= 0

        # Very young patient
        patient_young = PatientData(
            age=40,  # Lower limit of optimal range
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )
        result_young = model.calculate(patient_young)
        assert result_young.risk_score >= 0
        assert result_young.risk_score < result_old.risk_score  # Younger should have lower risk

