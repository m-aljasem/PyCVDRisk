# CVD Risk Streamlit Demo

Interactive Streamlit UI showcasing all 46 cardiovascular risk calculators available in the `cvd_risk` package.

## Quick start

```bash
cd /home/m-aljasem/projects/PyCVDRisk
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r demo/requirements.txt
# run the app (add PYTHONPATH so the local package is importable)
PYTHONPATH=./src streamlit run demo/streamlit_app.py
```

## Using the app
- The form is pre-populated with a demo patient that satisfies the required inputs for every model. Click **Calculate all 46 models** to smoke-test the package in one go.
- Adjust patient values as needed; use **Clear optional fields** if you want to provide only the essentials.
- Region-specific models automatically fall back to sensible defaults (e.g., `WHO` uses `EUR_A`, SCORE2-family uses `moderate`) unless you supply your own.

## Files
- `streamlit_app.py` – main Streamlit UI.
- `.streamlit/config.toml` – theme and server defaults.
- `requirements.txt` – lightweight dependencies plus editable install of this package.

