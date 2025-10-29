# Reference Data for Model Validation

This directory contains reference datasets for validating CVD risk models against published results.

## Data Format

Reference data files should be CSV format with the following structure:

- Patient demographic and clinical variables matching `PatientData` fields
- `reference_risk` column containing published risk scores (percentage)
- `source` column indicating publication/DOI (optional)
- `table_id` column indicating table/figure number in publication (optional)

## Example

```csv
age,sex,systolic_bp,total_cholesterol,hdl_cholesterol,smoking,region,reference_risk,source
55,male,140.0,6.0,1.2,True,moderate,6.8,10.1093/eurheartj/ehab309
60,female,130.0,5.5,1.5,False,low,2.3,10.1093/eurheartj/ehab309
```

## Adding Reference Data

1. Obtain validation data from original publications
2. Convert to CSV format matching the structure above
3. Add metadata file (see `metadata_template.json`)
4. Document source (DOI, table/figure number)

## Data Sources

- SCORE2: ESC 2021 guidelines
- Framingham: Original 1998 publication
- ASCVD: 2013 ACC/AHA guidelines
- QRISK3: BMJ 2017 publication
- SMART2: European Heart Journal 2014
- WHO: Lancet Global Health 2019
- Globorisk: Lancet Diabetes Endocrinol 2017

