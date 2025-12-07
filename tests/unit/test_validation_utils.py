"""Unit tests for validation utility functions."""

import pandas as pd
import pytest

from cvd_risk.core.validation import validate_dataframe


class TestValidateDataframe:
    """Test validate_dataframe utility function."""

    def test_validate_dataframe_success(self) -> None:
        """Test validation succeeds with all required columns."""
        df = pd.DataFrame(
            {
                "age": [55, 60],
                "sex": ["male", "female"],
                "systolic_bp": [140.0, 130.0],
            }
        )
        required = ["age", "sex", "systolic_bp"]

        result = validate_dataframe(df, required)

        pd.testing.assert_frame_equal(result, df)

    def test_validate_dataframe_missing_columns(self) -> None:
        """Test validation fails when columns are missing."""
        df = pd.DataFrame(
            {
                "age": [55, 60],
                "sex": ["male", "female"],
            }
        )
        required = ["age", "sex", "systolic_bp", "total_cholesterol"]

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, required)

    def test_validate_dataframe_partial_match(self) -> None:
        """Test validation with partial column match."""
        df = pd.DataFrame(
            {
                "age": [55, 60],
                "sex": ["male", "female"],
                "systolic_bp": [140.0, 130.0],
                "other_col": [1, 2],
            }
        )
        required = ["age", "sex"]

        result = validate_dataframe(df, required)
        # Should succeed even with extra columns
        pd.testing.assert_frame_equal(result, df)

