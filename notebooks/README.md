# Jupyter Notebooks for CVD Risk Models

This directory contains interactive Jupyter notebooks demonstrating each cardiovascular risk model.

## Available Notebooks

1. **SCORE2** (`score2/SCORE2_Example.ipynb`)
   - Comprehensive examples with visualizations
   - Single patient calculations
   - Batch processing
   - Risk comparisons across regions
   - Age and sex risk profiles

2. **Framingham** (`framingham/Framingham_Example.ipynb`)
   - Framingham Risk Score point-based system
   - Risk calculation examples

3. **ASCVD** (`ascvd/ASCVD_Example.ipynb`)
   - ASCVD Pooled Cohort Equations
   - US-based risk prediction

4. **QRISK3** (`qrisk3/QRISK3_Example.ipynb`)
   - Enhanced QRISK3 model
   - Includes ethnicity, BMI, and family history

5. **SMART2** (`smart2/SMART2_Example.ipynb`)
   - Recurrent CVD events in secondary prevention
   - For patients with established cardiovascular disease

6. **WHO** (`who/WHO_Example.ipynb`)
   - WHO CVD Risk Charts
   - Global risk estimation

7. **Globorisk** (`globorisk/Globorisk_Example.ipynb`)
   - Country-specific risk estimates
   - Accounts for local mortality rates

## Usage

To run the notebooks:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then navigate to the appropriate notebook directory and open the `.ipynb` file.

## Requirements

All notebooks require:
- `cvd-risk-calculator` package installed
- `pandas`, `matplotlib`, `seaborn` for data manipulation and visualization
- `numpy` for numerical operations

Install dependencies:
```bash
pip install -e ".[dev,docs]"
```

## Structure

Each notebook includes:
- Model overview and clinical context
- Reference to original publication
- Example calculations
- Visualizations (where applicable)
- Batch processing examples

## Contributing

When adding new examples or models:
1. Follow the existing notebook structure
2. Include clear markdown explanations
3. Add visualizations where helpful
4. Document any assumptions or limitations

