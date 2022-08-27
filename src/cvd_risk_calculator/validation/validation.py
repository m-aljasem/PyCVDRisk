"""Reference data validation framework for model verification."""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData

logger = logging.getLogger(__name__)


class ValidationResult:
    """Results from model validation against reference data."""

    def __init__(
        self,
        model_name: str,
        correlation: float,
        rmse: float,
        mean_error: float,
        n_samples: int,
        reference_source: str,
    ) -> None:
        """
        Initialize validation result.

        Parameters
        ----------
        model_name : str
            Name of the validated model.
        correlation : float
            Pearson correlation coefficient.
        rmse : float
            Root mean squared error.
        mean_error : float
            Mean error (bias).
        n_samples : int
            Number of validation samples.
        reference_source : str
            Source of reference data (DOI, publication, etc.).
        """
        self.model_name = model_name
        self.correlation = correlation
        self.rmse = rmse
        self.mean_error = mean_error
        self.n_samples = n_samples
        self.reference_source = reference_source

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ValidationResult(model={self.model_name}, "
            f"r={self.correlation:.3f}, "
            f"RMSE={self.rmse:.2f}%, "
            f"bias={self.mean_error:.2f}%, "
            f"n={self.n_samples})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "correlation": self.correlation,
            "rmse": self.rmse,
            "mean_error": self.mean_error,
            "n_samples": self.n_samples,
            "reference_source": self.reference_source,
        }


class ValidationFramework:
    """Framework for validating risk models against reference data."""

    def __init__(self, data_dir: Path | str = "data/reference") -> None:
        """
        Initialize validation framework.

        Parameters
        ----------
        data_dir : Path or str
            Directory containing reference validation datasets.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def validate_model(
        self,
        model: RiskModel,
        reference_data: pd.DataFrame,
        reference_source: str = "unknown",
        risk_column: str = "reference_risk",
    ) -> ValidationResult:
        """
        Validate a model against reference data.

        Parameters
        ----------
        model : RiskModel
            Risk model to validate.
        reference_data : pd.DataFrame
            DataFrame with patient data and reference risk scores.
        reference_source : str
            Source of reference data (DOI, publication, etc.).
        risk_column : str
            Column name containing reference risk scores.

        Returns
        -------
        ValidationResult
            Validation results with statistics.
        """
        # Calculate predictions
        predictions = []
        reference_risks = []

        for _, row in reference_data.iterrows():
            try:
                # Create PatientData from row
                patient_dict = row.to_dict()
                reference_risk = patient_dict.pop(risk_column, None)

                if reference_risk is None:
                    continue

                patient = PatientData(**patient_dict)
                result = model.calculate(patient)
                predictions.append(result.risk_score)
                reference_risks.append(reference_risk)
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")
                continue

        if len(predictions) < 2:
            raise ValueError("Insufficient valid data for validation")

        predictions = np.array(predictions)
        reference_risks = np.array(reference_risks)

        # Calculate statistics
        correlation, _ = pearsonr(predictions, reference_risks)
        errors = predictions - reference_risks
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)

        return ValidationResult(
            model_name=model.model_name,
            correlation=correlation,
            rmse=rmse,
            mean_error=mean_error,
            n_samples=len(predictions),
            reference_source=reference_source,
        )

    def load_reference_data(self, filename: str) -> pd.DataFrame:
        """
        Load reference validation data from CSV.

        Parameters
        ----------
        filename : str
            Name of CSV file in data_dir.

        Returns
        -------
        pd.DataFrame
            Reference data with patient information and risk scores.
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Reference data file not found: {filepath}")

        return pd.read_csv(filepath)

    def compare_models(
        self,
        models: list[RiskModel],
        reference_data: pd.DataFrame,
        reference_source: str = "unknown",
    ) -> pd.DataFrame:
        """
        Compare multiple models against the same reference data.

        Parameters
        ----------
        models : list[RiskModel]
            List of models to compare.
        reference_data : pd.DataFrame
            DataFrame with patient data and reference risk scores.
        reference_source : str
            Source of reference data.

        Returns
        -------
        pd.DataFrame
            Comparison results with validation metrics for each model.
        """
        results = []
        for model in models:
            try:
                result = self.validate_model(model, reference_data, reference_source)
                results.append(result.to_dict())
            except Exception as e:
                logger.error(f"Error validating {model.model_name}: {e}")
                continue

        return pd.DataFrame(results)

