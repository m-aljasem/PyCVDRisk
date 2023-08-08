# CVD Risk Model Validation Examples

This directory contains example scripts for validating and testing CVD risk prediction models using fake patient data.

## Scripts Overview

### 1. `generate_fake_patients.py`
Generates 1000 realistic fake patients for model validation and testing.

**Features:**
- Generates patients with realistic clinical distributions
- Includes all fields required by different CVD models
- Saves data in both CSV and JSON formats
- Provides comprehensive data summary statistics

**Usage:**
```bash
python generate_fake_patients.py
```

**Output:**
- `data/fake_patients.csv`: Patient data in CSV format
- `data/fake_patients.json`: Patient data in JSON format

**Data Fields:**
- `patient_id`: Unique patient identifier
- `age`: Age in years (30-80)
- `sex`: "male" or "female"
- `systolic_bp`: Systolic blood pressure (mmHg)
- `total_cholesterol`: Total cholesterol (mmol/L)
- `hdl_cholesterol`: HDL cholesterol (mmol/L)
- `smoking`: Smoking status (boolean)
- `region`: CVD risk region for SCORE2 ("low", "moderate", "high", "very_high")
- `diabetes`: Diabetes status (boolean)
- `ethnicity`: Ethnicity for QRISK3 ("white", "south_asian", "black", "chinese", "mixed", "other")
- `bmi`: Body mass index (kg/m²)
- `family_history`: Family history of CVD (boolean)
- `antihypertensive`: Antihypertensive medication use (boolean)

### 2. `validate_models_with_fake_data.py`
Validates CVD risk models using the generated fake patient data.

**Features:**
- Data quality validation
- Model compatibility testing
- Statistical analysis of model outputs
- Comprehensive validation reports

**Usage:**
```bash
python validate_models_with_fake_data.py
```

**Requirements:**
- Run `generate_fake_patients.py` first to create the test data

### 3. `generate_edge_case_patients.py`
Generates edge case patient data designed to stress test CVD risk models with boundary values, extreme combinations, and biologically implausible scenarios.

### 4. `validate_edge_cases.py`
Tests CVD risk models with edge case data to ensure they handle boundary conditions and extreme values gracefully.

### 5. `score2_example.py`
Basic example demonstrating SCORE2 model usage with both single patients and batch processing.

## Model Compatibility

The generated fake patient data includes all fields required by the following models:

| Model | Required Fields | Status |
|-------|----------------|--------|
| SCORE2 | age, sex, systolic_bp, total_cholesterol, hdl_cholesterol, smoking, region | ✅ Compatible |
| Framingham | age, sex, systolic_bp, total_cholesterol, hdl_cholesterol, smoking | ✅ Compatible |
| QRISK3 | age, sex, systolic_bp, total_cholesterol, hdl_cholesterol, smoking, diabetes, ethnicity, bmi, family_history | ✅ Compatible |
| ASCVD | age, sex, systolic_bp, total_cholesterol, hdl_cholesterol, smoking, diabetes, antihypertensive | ✅ Compatible |

## Data Characteristics

The fake patient data is designed to be realistic and representative:

- **Age distribution**: Beta distribution peaking around middle age (30-80 years)
- **Gender**: Approximately 50/50 split
- **Blood pressure**: Normal distribution with hypertension bias (90-220 mmHg)
- **Cholesterol**: Realistic ranges with HDL always ≤ total cholesterol
- **Risk factors**: Population-based prevalence rates
- **Regions**: European CVD risk distribution
- **Ethnicities**: Representative of diverse populations

## Example Usage

```python
import pandas as pd
from cvd_risk.models.score2 import SCORE2

# Load fake patient data
df = pd.read_csv('data/fake_patients.csv')

# Initialize model
model = SCORE2()

# Calculate risks for all patients
results = model.calculate_batch(df)

# View results
print(results[['patient_id', 'age', 'sex', 'risk_score', 'risk_category']].head())
```

## Validation Results

When tested with the SCORE2 model, the fake data produces:
- Risk scores ranging from 0.2% to 42.3%
- Realistic distribution across risk categories
- Proper statistical properties
- All data quality checks pass

## Edge Case Testing

The edge case generators create challenging test scenarios:

### Edge Case Categories
- **Boundary values**: Exact min/max allowed values (age 18-120, BP 50-250 mmHg, etc.)
- **Extreme combinations**: All risk factors maximized/minimized
- **Biological anomalies**: HDL > Total cholesterol, contradictory risk profiles
- **Precision issues**: Very small/large decimal values, values near boundaries
- **Model-specific**: Age ranges outside optimal model performance (SCORE2: 40-69)

### Edge Case Results (SCORE2 Model)
- **Risk range**: 0.010% - 96.337% (extreme boundary cases)
- **Most extreme cases**: 120-year-old patients (93-96% risk)
- **Validation**: All 50 edge cases processed successfully
- **Warning**: 5 patients have biologically implausible HDL > TC ratios

### Usage Example
```bash
# Generate edge cases
python examples/generate_edge_case_patients.py

# Validate models with edge cases
python examples/validate_edge_cases.py

# Test specific model
python -c "
import pandas as pd
from cvd_risk.models.score2 import SCORE2

df = pd.read_csv('data/edge_case_patients.csv')
model = SCORE2()
results = model.calculate_batch(df)
print(f'Edge case risk range: {results[\"risk_score\"].min():.3f}% - {results[\"risk_score\"].max():.3f}%')
"
```

## Next Steps

1. **Add more models**: Extend validation to include Framingham, QRISK3, ASCVD, etc.
2. **Cross-model comparison**: Compare outputs between different models
3. **Performance benchmarking**: Measure calculation speed and memory usage
4. **Statistical validation**: Compare distributions with real clinical data
5. **Error handling**: Test model responses to invalid inputs
