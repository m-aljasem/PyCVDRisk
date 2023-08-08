"""Performance benchmarks for CVD risk models using pytest-benchmark."""

import pandas as pd
import pytest

from cvd_risk.core.validation import PatientData
from cvd_risk.models import (
    ASCVD,
    Framingham,
    Globorisk,
    QRISK3,
    SCORE2,
    SMART2,
    WHO,
)


@pytest.fixture
def sample_patient() -> PatientData:
    """Create a sample patient for benchmarking."""
    return PatientData(
        age=55,
        sex="male",
        systolic_bp=140.0,
        total_cholesterol=6.0,
        hdl_cholesterol=1.2,
        smoking=True,
        region="moderate",
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for batch benchmarking."""
    return pd.DataFrame(
        {
            "age": [55] * 1000,
            "sex": ["male"] * 1000,
            "systolic_bp": [140.0] * 1000,
            "total_cholesterol": [6.0] * 1000,
            "hdl_cholesterol": [1.2] * 1000,
            "smoking": [True] * 1000,
            "region": ["moderate"] * 1000,
        }
    )


@pytest.mark.benchmark
class TestSCORE2Benchmark:
    """Benchmark SCORE2 model performance."""

    def test_score2_single_calculation(self, benchmark, sample_patient: PatientData) -> None:
        """Benchmark single SCORE2 calculation."""
        model = SCORE2()
        benchmark(model.calculate, sample_patient)

    def test_score2_batch_1000(self, benchmark, sample_dataframe: pd.DataFrame) -> None:
        """Benchmark SCORE2 batch processing of 1000 patients."""
        model = SCORE2()
        benchmark(model.calculate_batch, sample_dataframe)


@pytest.mark.benchmark
class TestFraminghamBenchmark:
    """Benchmark Framingham model performance."""

    def test_framingham_single_calculation(
        self, benchmark, sample_patient: PatientData
    ) -> None:
        """Benchmark single Framingham calculation."""
        model = Framingham()
        benchmark(model.calculate, sample_patient)

    def test_framingham_batch_1000(
        self, benchmark, sample_dataframe: pd.DataFrame
    ) -> None:
        """Benchmark Framingham batch processing of 1000 patients."""
        model = Framingham()
        benchmark(model.calculate_batch, sample_dataframe)


@pytest.mark.benchmark
class TestASCVDBenchmark:
    """Benchmark ASCVD model performance."""

    def test_ascvd_single_calculation(self, benchmark, sample_patient: PatientData) -> None:
        """Benchmark single ASCVD calculation."""
        model = ASCVD()
        benchmark(model.calculate, sample_patient)

    def test_ascvd_batch_1000(self, benchmark, sample_dataframe: pd.DataFrame) -> None:
        """Benchmark ASCVD batch processing of 1000 patients."""
        model = ASCVD()
        benchmark(model.calculate_batch, sample_dataframe)


@pytest.mark.benchmark
class TestQRISK3Benchmark:
    """Benchmark QRISK3 model performance."""

    def test_qrisk3_single_calculation(self, benchmark, sample_patient: PatientData) -> None:
        """Benchmark single QRISK3 calculation."""
        model = QRISK3()
        benchmark(model.calculate, sample_patient)

    def test_qrisk3_batch_1000(self, benchmark, sample_dataframe: pd.DataFrame) -> None:
        """Benchmark QRISK3 batch processing of 1000 patients."""
        model = QRISK3()
        benchmark(model.calculate_batch, sample_dataframe)


@pytest.mark.benchmark
class TestModelComparisonBenchmark:
    """Compare performance across all models."""

    def test_all_models_single(self, benchmark, sample_patient: PatientData) -> None:
        """Compare single calculation performance across models."""
        models = [
            SCORE2(),
            Framingham(),
            ASCVD(),
            QRISK3(),
            SMART2(),
            WHO(),
            Globorisk(),
        ]

        def run_all_models() -> None:
            for model in models:
                try:
                    model.calculate(sample_patient)
                except Exception:
                    pass  # Some models may need different inputs

        benchmark(run_all_models)

