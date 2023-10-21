# PyCVDRisk 🫀

<div align="center">

### World's Most Comprehensive CVD Risk Calculator

*The world's most comprehensive CVD risk assessment library with global coverage*

[![PyPI version](https://badge.fury.io/py/cvd-risk.svg)](https://pypi.org/project/cvd-risk/)
[![Python versions](https://img.shields.io/pypi/pyversions/cvd-risk)](https://pypi.org/project/cvd-risk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/m-aljasem/PyCVDRisk/actions/workflows/ci.yml/badge.svg)](https://github.com/m-aljasem/PyCVDRisk/actions)

[📖 Documentation](https://pycvdrisk.aljasem.eu.org) • [🚀 Quick Start](#installation) • [💬 Discussions](https://github.com/m-aljasem/PyCVDRisk/discussions)

</div>

---

## ✨ What is PyCVDRisk?

PyCVDRisk is the **world's most comprehensive Python package** for cardiovascular disease risk prediction, implementing **46 validated models** from **35+ countries**. Whether you're a researcher analyzing global biobank data or a clinician integrating risk assessment into your workflow, PyCVDRisk provides accurate, validated, and easy-to-use CVD risk calculations with unmatched geographic and ethnic coverage.

### 🎯 Key Features

- **46 Production-Ready Models**: World's most comprehensive CVD risk library with global coverage across 35+ countries and regions
- **Global Coverage**: Models from every major continent (Europe, Americas, Asia, Oceania, Middle East)
- **Ethnic Diversity**: 15+ ethnic groups represented (European, African American, Asian, Hispanic, Maori, Pacific Islander, etc.)
- **Batch Processing**: Handle thousands of patients efficiently with vectorized operations
- **Type-Safe**: Full type hints and validation with Pydantic
- **Well-Tested**: Comprehensive test suite with high code coverage
- **Research-Grade**: Based on peer-reviewed algorithms and clinical guidelines

### 🌍 Global Coverage & Diversity

PyCVDRisk offers unparalleled geographic and ethnic coverage:

- **Europe**: 15+ models (SCORE2, PROCAM, REGICOR, Progetto CUORE, PRIME, DECODE, INTERHEART, etc.)
- **Americas**: 8+ models (ASCVD, Framingham, PREVENT, Reynolds, Brazilian CVD, Mexican CVD, etc.)
- **Asia**: 4+ models (Singapore, Malaysian CVD, QRISK2/3, SCORE2-Asia CKD)
- **Oceania**: 3+ models (New Zealand, PREDICT)
- **Middle East**: 1+ model (Gulf RACE)
- **Global**: 3+ models (WHO CVD, Globorisk, INTERHEART)

**Special Populations Covered:**
- HIV-positive patients (D:A:D Score)
- Diabetes patients (DIAL2, SCORE2-DM, DECODE)
- CKD patients (SCORE2-CKD)
- Elderly populations (SCORE2-OP, Rotterdam Study)
- Young adults (CARDIA)
- Atrial fibrillation (CHADS2, CHA2DS2-VASc)
- Anticoagulation bleeding risk (HAS-BLED)

---

## 🚀 Installation

```bash
pip install cvd-risk
```

That's it! PyCVDRisk is now ready to use.

---

## 📖 Quick Start

### Calculate Risk for One Patient

```python
from cvd_risk import SCORE2, PatientData

# Create a patient profile
patient = PatientData(
    age=55,
    sex='male',
    systolic_bp=140,
    total_cholesterol=6.0,
    hdl_cholesterol=1.2,
    smoking=True,
    region='moderate'
)

# Calculate 10-year CVD risk
model = SCORE2()
result = model.calculate(patient)

print(f"Risk Score: {result.risk_score:.1f}%")
print(f"Risk Category: {result.risk_category}")
```

**Output:**
```
Risk Score: 10.6%
Risk Category: Moderate risk
```

### Batch Processing for Multiple Patients

```python
import pandas as pd
from cvd_risk import SCORE2

# Load your patient data
df = pd.DataFrame({
    'age': [45, 60, 55, 70],
    'sex': ['female', 'male', 'male', 'female'],
    'systolic_bp': [120, 150, 135, 145],
    'total_cholesterol': [5.5, 7.2, 6.1, 4.8],
    'hdl_cholesterol': [1.8, 1.0, 1.3, 1.5],
    'smoking': [False, True, False, True],
    'region': ['low', 'high', 'moderate', 'moderate']
})

# Calculate risks for all patients
model = SCORE2()
results_df = model.calculate_batch(df)

# Results are added to the original dataframe
print(results_df[['age', 'sex', 'risk_score', 'risk_category']])
```

---

## 🏥 Available Models

### Primary Prevention Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **SCORE2** | European adults (40-69) | Primary prevention | ✅ Production Ready |
| **ASCVD** | US/International adults | Primary prevention | ✅ Production Ready |
| **Framingham** | US adults | Primary prevention | ✅ Production Ready |
| **QRISK2** | UK adults | Primary prevention | ✅ Production Ready |
| **QRISK3** | UK primary care | Primary prevention | ✅ Production Ready |
| **SCORE** | European adults | Primary prevention | ✅ Production Ready |
| **WHO CVD** | Global populations | Primary prevention | ✅ Production Ready |
| **Globorisk** | 182 countries | Primary prevention | ✅ Production Ready |
| **INTERHEART** | 52 countries | Primary prevention | ✅ Production Ready |
| **PREVENT** | US adults (40-79) | Primary prevention | ✅ Production Ready |
| **PROCAM** | German adults | Primary prevention | ✅ Production Ready |
| **Reynolds** | US adults | Primary prevention (hsCRP) | ✅ Production Ready |
| **FINRISK** | Finnish adults | Primary prevention | ✅ Production Ready |
| **REGICOR** | Spanish adults | Primary prevention | ✅ Production Ready |
| **Progetto CUORE** | Italian adults | Primary prevention | ✅ Production Ready |
| **PRIME** | France/Ireland adults | Primary prevention | ✅ Production Ready |
| **RISC** | German adults | Primary prevention | ✅ Production Ready |
| **ARIC Update** | US multi-ethnic | Primary prevention | ✅ Production Ready |
| **Jackson Heart** | US African American | Primary prevention | ✅ Production Ready |
| **CARDIA** | US young adults | Primary prevention | ✅ Production Ready |
| **Rotterdam** | Dutch elderly | Primary prevention | ✅ Production Ready |
| **Heinz Nixdorf** | German adults | Primary prevention | ✅ Production Ready |
| **EPIC-Norfolk** | UK adults | Primary prevention | ✅ Production Ready |
| **Singapore** | Singapore adults | Primary prevention | ✅ Production Ready |
| **PREDICT** | New Zealand adults | Primary prevention | ✅ Production Ready |
| **New Zealand** | New Zealand adults | Primary prevention | ✅ Production Ready |
| **Dundee** | Scottish adults | Primary prevention | ✅ Production Ready |
| **Cambridge** | UK adults | Primary prevention | ✅ Production Ready |
| **DECODE** | European adults | Primary prevention | ✅ Production Ready |
| **NHANES** | US population survey | Primary prevention | ✅ Production Ready |
| **Malaysian CVD** | Malaysian adults | Primary prevention | ✅ Production Ready |
| **Brazilian CVD** | Brazilian adults | Primary prevention | ✅ Production Ready |
| **Mexican CVD** | Mexican adults | Primary prevention | ✅ Production Ready |
| **Gulf RACE** | Gulf countries | Primary prevention | ✅ Production Ready |

### Secondary Prevention Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **SMART2** | Established CVD patients | Recurrent CVD risk | ✅ Production Ready |
| **SMART-REACH** | Established CVD patients | Recurrent CVD risk | ✅ Production Ready |

### Diabetes-Specific Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **DIAL2** | Type 2 diabetes patients | Lifetime CVD risk | ✅ Production Ready |
| **SCORE2-DM** | Diabetes patients | CVD risk assessment | ✅ Production Ready |

### HIV-Specific Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **D:A:D Score** | HIV-positive patients | CVD risk assessment | ✅ Production Ready |

### CKD-Specific Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **SCORE2-CKD** | CKD patients | CVD risk assessment | ✅ Production Ready |
| **SCORE2-OP** | Older persons | CVD risk assessment | ✅ Production Ready |

### Region-Specific Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **ASSIGN** | Scottish adults | Primary prevention | ✅ Production Ready |
| **SCORE2-Asia CKD** | Asian CKD patients | CVD risk assessment | ✅ Production Ready |
| **REGICOR** | Spanish adults | Primary prevention | ✅ Production Ready |
| **Progetto CUORE** | Italian adults | Primary prevention | ✅ Production Ready |
| **PRIME** | France/Ireland adults | Primary prevention | ✅ Production Ready |
| **RISC** | German adults | Primary prevention | ✅ Production Ready |
| **Heinz Nixdorf** | German adults | Primary prevention | ✅ Production Ready |
| **EPIC-Norfolk** | UK adults | Primary prevention | ✅ Production Ready |
| **Singapore** | Singapore adults | Primary prevention | ✅ Production Ready |
| **PREDICT** | New Zealand adults | Primary prevention | ✅ Production Ready |
| **New Zealand** | New Zealand adults | Primary prevention | ✅ Production Ready |
| **Dundee** | Scottish adults | Primary prevention | ✅ Production Ready |
| **Cambridge** | UK adults | Primary prevention | ✅ Production Ready |
| **DECODE** | European adults | Primary prevention | ✅ Production Ready |
| **NHANES** | US population survey | Primary prevention | ✅ Production Ready |
| **Malaysian CVD** | Malaysian adults | Primary prevention | ✅ Production Ready |
| **Brazilian CVD** | Brazilian adults | Primary prevention | ✅ Production Ready |
| **Mexican CVD** | Mexican adults | Primary prevention | ✅ Production Ready |
| **Gulf RACE** | Gulf countries | Primary prevention | ✅ Production Ready |

### Lifetime Risk Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **LifeCVD2** | General population | Lifetime CVD risk | ✅ Production Ready |

### Acute Coronary Syndrome Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **GRACE2** | ACS patients | 6-month mortality | ✅ Production Ready |
| **TIMI** | UA/NSTEMI patients | Short-term risk | ✅ Production Ready |

### Emergency Department Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **EDACS** | Chest pain patients | MACE risk stratification | ✅ Production Ready |
| **HEART** | Chest pain patients | MACE risk stratification | ✅ Production Ready |

---

## 🔬 Usage Examples

### Clinical Integration

```python
from cvd_risk import SCORE2, ASCVD, PROCAM, PatientData

# Compare multiple models for the same patient
patient = PatientData(
    age=65, sex='male', systolic_bp=150,
    total_cholesterol=6.5, hdl_cholesterol=1.1,
    smoking=False
)

# European models
score2_model = SCORE2()
procam_model = PROCAM()

# US models
ascvd_model = ASCVD()

score2_result = score2_model.calculate(patient)
procam_result = procam_model.calculate(patient)
ascvd_result = ascvd_model.calculate(patient)

print(f"SCORE2 (Europe): {score2_result.risk_score:.1f}%")
print(f"PROCAM (Germany): {procam_result.risk_score:.1f}%")
print(f"ASCVD (US): {ascvd_result.risk_score:.1f}%")
```

### Epidemiological Research

```python
import pandas as pd
from cvd_risk import SCORE2

# Process large cohorts efficiently
biobank_data = pd.read_csv('large_cohort.csv')  # 100K+ patients

# Calculate risks (takes ~1 second for large datasets)
model = SCORE2()
results_df = model.calculate_batch(biobank_data)

# Statistical analysis
high_risk = (results_df['risk_score'] > 10).sum()
total_patients = len(biobank_data)
high_risk_percentage = (high_risk / total_patients) * 100

print(f"High-risk patients: {high_risk:,} ({high_risk_percentage:.1f}%)")
```

### Model Comparison

```python
from cvd_risk import SCORE2, Framingham, PROCAM, Singapore, PREDICT
import pandas as pd
import numpy as np

# Compare international model predictions across a global population
np.random.seed(42)
n_patients = 100

# Generate synthetic global population data
population_df = pd.DataFrame({
    'age': np.random.normal(55, 10, n_patients).clip(40, 80).astype(int),
    'sex': np.random.choice(['male', 'female'], n_patients),
    'systolic_bp': np.random.normal(130, 20, n_patients).clip(90, 200),
    'total_cholesterol': np.random.normal(5.5, 1.2, n_patients).clip(3, 10),
    'hdl_cholesterol': np.random.normal(1.3, 0.4, n_patients).clip(0.5, 2.5),
    'smoking': np.random.choice([True, False], n_patients, p=[0.2, 0.8]),
    'region': np.random.choice(['low', 'moderate', 'high'], n_patients),
    'ethnicity': np.random.choice(['chinese', 'malay', 'indian', 'white'], n_patients)
})

# Calculate risks with international models
score2_model = SCORE2()
framingham_model = Framingham()
procam_model = PROCAM()
singapore_model = Singapore()
predict_model = PREDICT()

score2_results = score2_model.calculate_batch(population_df)
framingham_results = framingham_model.calculate_batch(population_df)
procam_results = procam_model.calculate_batch(population_df)
singapore_results = singapore_model.calculate_batch(population_df)
predict_results = predict_model.calculate_batch(population_df)

# Compare average risks across continents
avg_risks = {
    'SCORE2 (Europe)': score2_results['risk_score'].mean(),
    'Framingham (US)': framingham_results['risk_score'].mean(),
    'PROCAM (Germany)': procam_results['risk_score'].mean(),
    'Singapore (Asia)': singapore_results['risk_score'].mean(),
    'PREDICT (New Zealand)': predict_results['risk_score'].mean()
}

print("Average 10-year CVD risk by international models:")
for model, risk in avg_risks.items():
    print(f"{model}: {risk:.1f}%")
```

---

## 📚 Documentation

- **[Full Documentation](https://pycvdrisk.aljasem.eu.org)** - Complete API reference and guides
- **[Model Details](https://pycvdrisk.aljasem.eu.org/models)** - Clinical background for each algorithm
- **[Validation](https://pycvdrisk.aljasem.eu.org/validation)** - How we ensure accuracy
- **[Contributing Guide](CONTRIBUTING.md)** - Add new models or improve the package

---

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

### Ways to Contribute
- 🐛 **Report bugs** via [GitHub Issues](https://github.com/m-aljasem/PyCVDRisk/issues)
- 💡 **Suggest features** in [Discussions](https://github.com/m-aljasem/PyCVDRisk/discussions)
- 📝 **Improve documentation** by editing files
- 🧪 **Add test cases** or validation data
- 🔧 **Implement new risk models**

### Development Setup

```bash
# Clone the repository
git clone https://github.com/m-aljasem/PyCVDRisk.git
cd PyCVDRisk

# Install in development mode with all dependencies
pip install -e .[dev]

# Run tests
python build_script.py test

# Run all checks (linting + tests)
python build_script.py check
```

---

## 📄 License

**MIT License** - Free for academic, clinical, and commercial use.

Just cite us in your research! 📚

---

## 🙏 Acknowledgments

- **European Society of Cardiology** - SCORE2 algorithm
- **American Heart Association** - ASCVD and Framingham algorithms
- **National Health Services (UK)** - QRISK algorithms
- **World Health Organization** - Global CVD risk charts
- **International Cardiology Societies** - INTERHEART, PRIME, and other global studies
- **National Health Ministries** - Country-specific algorithms (Finland, Singapore, Thailand, Malaysia, Brazil, Mexico, etc.)
- **Scientific Python Community** - NumPy, Pandas, SciPy
- **Open Source Contributors** - Making global research software accessible

---

<div align="center">

**Built with ❤️ for global cardiovascular research and clinical practice**

[⭐ Star us on GitHub](https://github.com/m-aljasem/PyCVDRisk) • [📧 Contact](mailto:mohamad@aljasem.eu.org)

</div>
