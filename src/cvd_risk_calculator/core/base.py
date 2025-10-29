"""Abstract base class for all CVD risk models."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from cvd_risk_calculator.core.validation import PatientData, RiskResult


class RiskModel(ABC):
    """
    Abstract base class for all cardiovascular risk prediction models.

    This class defines the interface that all risk models must implement,
    ensuring consistency across different algorithms. Subclasses must implement
    the `calculate` method for single patient calculations and may optionally
    override `calculate_batch` for optimized batch processing.

    Attributes
    ----------
    model_name : str
        Human-readable name of the risk model.
    model_version : str
        Version identifier for the algorithm implementation.
    supported_regions : list[str] or None
        List of supported geographic regions if applicable, else None.

    Examples
    --------
    >>> from cvd_risk_calculator.models.score2 import SCORE2
    >>> model = SCORE2()
    >>> patient = PatientData(...)
    >>> result = model.calculate(patient)
    >>> print(f"Risk: {result.risk_score}%")
    """

    model_name: str
    model_version: str
    supported_regions: Optional[list[str]] = None

    def __init__(self) -> None:
        """Initialize the risk model."""
        pass

    @abstractmethod
    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate CVD risk for a single patient.

        Parameters
        ----------
        patient : PatientData
            Validated patient data.

        Returns
        -------
        RiskResult
            Risk calculation result with metadata.

        Raises
        ------
        ValueError
            If patient data is invalid or outside model's applicable range.
        """
        pass

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CVD risk for a batch of patients (vectorized).

        This method provides optimized batch processing using vectorized
        operations. By default, it iterates over rows and calls `calculate`,
        but subclasses should override this for performance optimization.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with patient data. Must include columns required
            by the specific model.

        Returns
        -------
        pd.DataFrame
            DataFrame with original columns plus risk calculation results.
            Added columns typically include:
            - risk_score: Calculated risk percentage
            - risk_category: Risk classification
            - Other model-specific outputs

        Examples
        --------
        >>> model = SCORE2()
        >>> df = pd.DataFrame({...})
        >>> results = model.calculate_batch(df)
        >>> print(results[['risk_score', 'risk_category']])
        """
        results = []
        for _, row in df.iterrows():
            patient_dict = row.to_dict()
            try:
                # Convert row to PatientData
                patient = PatientData(**patient_dict)
                result = self.calculate(patient)
                results.append(
                    {
                        **patient_dict,
                        "risk_score": result.risk_score,
                        "risk_category": result.risk_category,
                        "model_name": result.model_name,
                    }
                )
            except Exception as e:
                # Handle validation errors or calculation failures
                results.append(
                    {
                        **patient_dict,
                        "risk_score": None,
                        "risk_category": "error",
                        "error": str(e),
                    }
                )
        return pd.DataFrame(results)

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input is within model's applicable range.

        Override this method to add model-specific validation.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is outside the model's applicable range.
        """
        # Base implementation does basic checks
        # Subclasses should add model-specific validation
        if patient.age < 18:
            raise ValueError("Age must be >= 18 years")
        if patient.systolic_bp < 50 or patient.systolic_bp > 250:
            raise ValueError("Systolic BP outside valid range [50, 250] mmHg")

    def _get_metadata(self) -> dict[str, Any]:
        """
        Get model metadata for inclusion in results.

        Returns
        -------
        dict
            Dictionary with model metadata.
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "supported_regions": self.supported_regions,
        }

