Quick Start
===========

This guide will help you get started with CVD Risk Calculator.

Basic Usage
-----------

Calculate risk for a single patient:

.. code-block:: python

   from cvd_risk.models.score2 import SCORE2
   from cvd_risk.core.validation import PatientData

   # Create patient data
   patient = PatientData(
       age=55,
       sex="male",
       systolic_bp=140.0,
       total_cholesterol=6.0,
       hdl_cholesterol=1.2,
       smoking=True,
       region="moderate"
   )

   # Calculate risk
   model = SCORE2()
   result = model.calculate(patient)

   print(f"10-year CVD risk: {result.risk_score:.1f}%")
   print(f"Risk category: {result.risk_category}")

Batch Processing
----------------

Process multiple patients using pandas:

.. code-block:: python

   import pandas as pd
   from cvd_risk.models.score2 import SCORE2

   # Create DataFrame with patient data
   df = pd.DataFrame({
       'age': [55, 60, 45],
       'sex': ['male', 'female', 'male'],
       'systolic_bp': [140.0, 130.0, 150.0],
       'total_cholesterol': [6.0, 5.5, 7.0],
       'hdl_cholesterol': [1.2, 1.5, 1.0],
       'smoking': [True, False, True],
       'region': ['moderate', 'low', 'high']
   })

   # Calculate risks
   model = SCORE2()
   results = model.calculate_batch(df)

   # View results
   print(results[['age', 'sex', 'risk_score', 'risk_category']])

Risk Categories
---------------

Risk scores are categorized as:

- **Low**: <5% 10-year risk
- **Moderate**: 5-10% 10-year risk
- **High**: 10-20% 10-year risk
- **Very High**: ≥20% 10-year risk

Next Steps
----------

- See :doc:`api/index` for detailed API documentation
- Check the Jupyter notebooks for detailed examples
- Read model-specific documentation for clinical interpretation

