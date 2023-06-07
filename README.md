# PyCVDRisk 🫀

<div align="center">

### Cardiovascular Disease Risk Calculator

*A comprehensive Python library for CVD risk assessment*

[![PyPI version](https://badge.fury.io/py/cvd-risk.svg)](https://pypi.org/project/cvd-risk/)
[![Python versions](https://img.shields.io/pypi/pyversions/cvd-risk)](https://pypi.org/project/cvd-risk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/m-aljasem/PyCVDRisk/workflows/CI/badge.svg)](https://github.com/m-aljasem/PyCVDRisk/actions)

[📖 Documentation](https://pycvdrisk.aljasem.eu.org) • [🚀 Quick Start](#installation) • [💬 Discussions](https://github.com/m-aljasem/PyCVDRisk/discussions)

</div>

---

## ✨ What is PyCVDRisk?

PyCVDRisk is a **production-ready Python package** that implements major cardiovascular disease risk prediction models. Whether you're a researcher analyzing biobank data or a clinician integrating risk assessment into your workflow, PyCVDRisk provides accurate, validated, and easy-to-use CVD risk calculations.

### 🎯 Key Features

- **7 Production-Ready Models**: SCORE2, ASCVD, Framingham, QRISK3, SMART2, WHO CVD, Globorisk
- **Batch Processing**: Handle thousands of patients efficiently with vectorized operations
- **Type-Safe**: Full type hints and validation with Pydantic
- **Well-Tested**: Comprehensive test suite with high code coverage
- **Research-Grade**: Based on peer-reviewed algorithms and clinical guidelines

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
from cvd_risk import SCORE2

# Create a patient profile
patient = {
    'age': 55,
    'sex': 'male',
    'systolic_bp': 140,
    'total_cholesterol': 6.0,
    'hdl_cholesterol': 1.2,
    'smoking': True,
    'region': 'moderate'
}

# Calculate 10-year CVD risk
result = SCORE2(**patient)

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
results = SCORE2.batch_calculate(**df)

# Add results to your dataframe
df['cvd_risk_10y'] = results['risk_score']
df['risk_category'] = results['risk_category']

print(df)
```

---

## 🏥 Available Models

| Model | Population | Primary Use | Status |
|-------|------------|-------------|--------|
| **SCORE2** | European adults (40-69) | Primary prevention | ✅ Production Ready |
| **ASCVD** | US/International adults | Primary prevention | ✅ Production Ready |
| **Framingham** | US adults | Primary prevention | ✅ Production Ready |
| **QRISK3** | UK primary care | Primary prevention | ✅ Production Ready |
| **SMART2** | Secondary prevention | Recurrent CVD risk | ✅ Production Ready |
| **WHO CVD** | Global populations | Primary prevention | ✅ Production Ready |
| **Globorisk** | 182 countries | Primary prevention | ✅ Production Ready |

*More models coming soon: SCORE2-OP, PREVENT, LifeCVD2, and others*

---

## 🔬 Usage Examples

### Clinical Integration

```python
from cvd_risk import SCORE2, ASCVD

# Compare multiple models for the same patient
patient = {
    'age': 65, 'sex': 'male', 'systolic_bp': 150,
    'total_cholesterol': 6.5, 'hdl_cholesterol': 1.1,
    'smoking': False, 'region': 'high'
}

score2_risk = SCORE2(**patient).risk_score
ascvd_risk = ASCVD(**patient).risk_score

print(f"SCORE2: {score2_risk:.1f}%")
print(f"ASCVD: {ascvd_risk:.1f}%")
```

### Epidemiological Research

```python
import pandas as pd
from cvd_risk import SCORE2

# Process large cohorts efficiently
biobank_data = pd.read_csv('large_cohort.csv')  # 100K+ patients

# Calculate risks (takes ~1 second for large datasets)
risks = SCORE2.batch_calculate(**biobank_data)

# Statistical analysis
high_risk = (risks['risk_score'] > 10).sum()
total_patients = len(biobank_data)
high_risk_percentage = (high_risk / total_patients) * 100

print(f"High-risk patients: {high_risk:,} ({high_risk_percentage:.1f}%)")
```

### Model Comparison

```python
from cvd_risk import SCORE2, Framingham, QRISK3
import numpy as np

# Compare model predictions across a population
ages = np.random.normal(55, 10, 1000)
# ... generate synthetic population data ...

results = {
    'SCORE2': SCORE2.batch_calculate(age=ages, ...),
    'Framingham': Framingham.batch_calculate(age=ages, ...),
    'QRISK3': QRISK3.batch_calculate(age=ages, ...)
}

# Analyze agreement between models
# Statistical analysis code here...
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
- **Scientific Python Community** - NumPy, Pandas, SciPy
- **Open Source Contributors** - Making research software accessible

---

<div align="center">

**Built with ❤️ for cardiovascular research and clinical practice**

[⭐ Star us on GitHub](https://github.com/m-aljasem/PyCVDRisk) • [📧 Contact](mailto:mohamad@aljasem.eu.org)

</div>
