"""Unit tests for HEART risk model."""

import pytest

from cvd_risk.core.validation import PatientData
from cvd_risk.models.heart import HEART


class TestHEART:
    """Test HEART model implementation."""

    def test_model_initialization(self) -> None:
        """Test that HEART model initializes correctly."""
        model = HEART()
        assert model.model_name == "HEART"
        assert model.model_version == "2013"
        assert model.supported_regions is None

    def test_calculate_basic(self, sample_heart_patient: PatientData) -> None:
        """Test basic HEART score calculation."""
        model = HEART()
        result = model.calculate(sample_heart_patient)

        assert result.model_name == "HEART"
        assert result.model_version == "2013"
        assert 0.0 <= result.risk_score <= 10.0
        assert result.risk_category in ["low", "moderate", "high"]
        assert "history_score" in result.calculation_metadata
        assert "ecg_score" in result.calculation_metadata
        assert "age_score" in result.calculation_metadata
        assert "risk_factors_score" in result.calculation_metadata
        assert "troponin_score" in result.calculation_metadata

    def test_calculate_missing_fields(self) -> None:
        """Test that calculation requires HEART-specific fields."""
        model = HEART()
        patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            # Missing HEART fields
        )

        with pytest.raises(ValueError, match="Missing required HEART fields"):
            model.calculate(patient)

    def test_history_score_calculation(self) -> None:
        """Test History component scoring."""
        model = HEART()

        # Test different symptom counts
        assert model._calculate_history_score(0) == 0  # 0-1: nonspecific
        assert model._calculate_history_score(1) == 0
        assert model._calculate_history_score(2) == 1  # 2-4: moderately suspicious
        assert model._calculate_history_score(3) == 1
        assert model._calculate_history_score(4) == 1
        assert model._calculate_history_score(5) == 2  # 5-6: mainly suspicious
        assert model._calculate_history_score(6) == 2

    def test_ecg_score_calculation(self) -> None:
        """Test ECG component scoring."""
        model = HEART()

        # Normal ECG = 0
        assert model._calculate_ecg_score(True, False, False) == 0

        # Abnormal repolarization only = 1
        assert model._calculate_ecg_score(False, True, False) == 1

        # ST depression = 2 (takes precedence)
        assert model._calculate_ecg_score(False, True, True) == 2
        assert model._calculate_ecg_score(False, False, True) == 2

        # Any other abnormality = 1
        assert model._calculate_ecg_score(False, False, False) == 1

    def test_age_score_calculation(self) -> None:
        """Test Age component scoring."""
        model = HEART()

        assert model._calculate_age_score(30) == 0  # < 45
        assert model._calculate_age_score(44) == 0
        assert model._calculate_age_score(45) == 1  # 45-64
        assert model._calculate_age_score(55) == 1
        assert model._calculate_age_score(64) == 1
        assert model._calculate_age_score(65) == 2  # 65+
        assert model._calculate_age_score(80) == 2

    def test_risk_factors_score_calculation(self) -> None:
        """Test Risk factors component scoring."""
        model = HEART()

        # No risk factors = 0
        patient_none = PatientData(
            age=50, sex="male", systolic_bp=120, total_cholesterol=5.0, hdl_cholesterol=1.5,
            smoking=False, diabetes=False, hypertension=False, hyperlipidaemia=False,
            family_history=False, atherosclerotic_disease=False,
            typical_symptoms_num=0, ecg_normal=True, abn_repolarisation=False,
            ecg_st_depression=False, presentation_hstni=10.0
        )
        assert model._calculate_risk_factors_score(patient_none) == 0

        # 1-2 risk factors = 1
        patient_two = PatientData(
            age=50, sex="male", systolic_bp=120, total_cholesterol=5.0, hdl_cholesterol=1.5,
            smoking=True, diabetes=True, hypertension=False, hyperlipidaemia=False,
            family_history=False, atherosclerotic_disease=False,
            typical_symptoms_num=0, ecg_normal=True, abn_repolarisation=False,
            ecg_st_depression=False, presentation_hstni=10.0
        )
        assert model._calculate_risk_factors_score(patient_two) == 1

        # 3+ risk factors = 2
        patient_three = PatientData(
            age=50, sex="male", systolic_bp=120, total_cholesterol=5.0, hdl_cholesterol=1.5,
            smoking=True, diabetes=True, hypertension=True, hyperlipidaemia=False,
            family_history=False, atherosclerotic_disease=False,
            typical_symptoms_num=0, ecg_normal=True, abn_repolarisation=False,
            ecg_st_depression=False, presentation_hstni=10.0
        )
        assert model._calculate_risk_factors_score(patient_three) == 2

        # Atherosclerotic disease = 2 (regardless of other factors)
        patient_athero = PatientData(
            age=50, sex="male", systolic_bp=120, total_cholesterol=5.0, hdl_cholesterol=1.5,
            smoking=False, diabetes=False, hypertension=False, hyperlipidaemia=False,
            family_history=False, atherosclerotic_disease=True,
            typical_symptoms_num=0, ecg_normal=True, abn_repolarisation=False,
            ecg_st_depression=False, presentation_hstni=10.0
        )
        assert model._calculate_risk_factors_score(patient_athero) == 2

    def test_troponin_score_calculation(self) -> None:
        """Test Troponin component scoring."""
        model = HEART()

        # Male thresholds (URL = 34 ng/L)
        assert model._calculate_troponin_score(10.0, "male") == 0  # < 34
        assert model._calculate_troponin_score(34.0, "male") == 1  # 34-101.99
        assert model._calculate_troponin_score(80.0, "male") == 1
        assert model._calculate_troponin_score(102.0, "male") == 2  # >= 102

        # Female thresholds (URL = 16 ng/L)
        assert model._calculate_troponin_score(10.0, "female") == 0  # < 16
        assert model._calculate_troponin_score(16.0, "female") == 1  # 16-47.99
        assert model._calculate_troponin_score(30.0, "female") == 1
        assert model._calculate_troponin_score(48.0, "female") == 2  # >= 48

    def test_full_score_calculation(self) -> None:
        """Test complete HEART score calculation."""
        model = HEART()

        # Create a patient with known scores
        patient = PatientData(
            age=50,  # age_score = 1
            sex="male",
            systolic_bp=120, total_cholesterol=5.0, hdl_cholesterol=1.5,
            smoking=True, diabetes=True, hypertension=False, hyperlipidaemia=False,
            family_history=False, atherosclerotic_disease=False,  # risk_factors_score = 1 (2 factors)
            typical_symptoms_num=3,  # history_score = 1
            ecg_normal=False, abn_repolarisation=True, ecg_st_depression=False,  # ecg_score = 1
            presentation_hstni=50.0  # troponin_score = 2 (male, > 3x URL)
        )

        result = model.calculate(patient)

        expected_total = 1 + 1 + 1 + 1 + 2  # 6
        assert result.risk_score == expected_total
        assert result.risk_category == "moderate"

        # Check component scores in metadata
        assert result.calculation_metadata["history_score"] == 1
        assert result.calculation_metadata["ecg_score"] == 1
        assert result.calculation_metadata["age_score"] == 1
        assert result.calculation_metadata["risk_factors_score"] == 1
        assert result.calculation_metadata["troponin_score"] == 2

    def test_risk_categories(self) -> None:
        """Test risk category boundaries."""
        model = HEART()

        # Low risk: 0-3
        for score in [0, 1, 2, 3]:
            patient = PatientData(
                age=30, sex="male", systolic_bp=120, total_cholesterol=5.0, hdl_cholesterol=1.5,
                smoking=False, diabetes=False, hypertension=False, hyperlipidaemia=False,
                family_history=False, atherosclerotic_disease=False,
                typical_symptoms_num=score, ecg_normal=True, abn_repolarisation=False,
                ecg_st_depression=False, presentation_hstni=10.0
            )
            result = model.calculate(patient)
            assert result.risk_category == "low"

        # Moderate risk: 4-6
        patient_mod = PatientData(
            age=50, sex="male", systolic_bp=120, total_cholesterol=5.0, hdl_cholesterol=1.5,
            smoking=True, diabetes=True, hypertension=True, hyperlipidaemia=True,
            family_history=True, atherosclerotic_disease=False,
            typical_symptoms_num=6, ecg_normal=False, abn_repolarisation=True,
            ecg_st_depression=True, presentation_hstni=200.0  # Should give score around 4-6
        )
        result_mod = model.calculate(patient_mod)
        assert result_mod.risk_category == "moderate"

        # High risk: 7+
        patient_high = PatientData(
            age=70, sex="male", systolic_bp=120, total_cholesterol=5.0, hdl_cholesterol=1.5,
            smoking=True, diabetes=True, hypertension=True, hyperlipidaemia=True,
            family_history=True, atherosclerotic_disease=True,
            typical_symptoms_num=6, ecg_normal=False, abn_repolarisation=True,
            ecg_st_depression=True, presentation_hstni=200.0  # Should give high score
        )
        result_high = model.calculate(patient_high)
        assert result_high.risk_category == "high"

    def test_calculate_batch_basic(self, sample_heart_patient: PatientData) -> None:
        """Test batch calculation."""
        model = HEART()

        # Create a small DataFrame
        import pandas as pd
        df = pd.DataFrame([{
            "age": sample_heart_patient.age,
            "sex": sample_heart_patient.sex,
            "systolic_bp": sample_heart_patient.systolic_bp,
            "total_cholesterol": sample_heart_patient.total_cholesterol,
            "hdl_cholesterol": sample_heart_patient.hdl_cholesterol,
            "smoking": sample_heart_patient.smoking,
            "diabetes": sample_heart_patient.diabetes,
            "hypertension": sample_heart_patient.hypertension,
            "hyperlipidaemia": sample_heart_patient.hyperlipidaemia,
            "family_history": sample_heart_patient.family_history,
            "atherosclerotic_disease": sample_heart_patient.atherosclerotic_disease,
            "typical_symptoms_num": sample_heart_patient.typical_symptoms_num,
            "ecg_normal": sample_heart_patient.ecg_normal,
            "abn_repolarisation": sample_heart_patient.abn_repolarisation,
            "ecg_st_depression": sample_heart_patient.ecg_st_depression,
            "presentation_hstni": sample_heart_patient.presentation_hstni,
        }])

        result_df = model.calculate_batch(df)

        assert "heart_score" in result_df.columns
        assert "risk_category" in result_df.columns
        assert "model_name" in result_df.columns
        assert len(result_df) == 1
        assert 0.0 <= result_df["heart_score"].iloc[0] <= 10.0
        assert result_df["risk_category"].iloc[0] in ["low", "moderate", "high"]
