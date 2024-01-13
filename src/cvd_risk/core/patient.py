"""
Base Patient Data Schema.

This module defines the minimal set of patient attributes required by virtually
all CVD risk models (Age and Sex). Model-specific implementations should inherit
from `BasePatientData` and add their specific requirements.
"""

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class BasePatientData(BaseModel):
    """
    Base class for patient data.
    
    Contains only the absolute universal attributes.
    """
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    sex: Literal["male", "female"] = Field(..., description="Patient biological sex")

    model_config = ConfigDict(frozen=True, validate_assignment=True)
