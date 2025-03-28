# PyCVDRisk demo

A Streamlit app for trying every model in [`cvd-risk`](https://pypi.org/project/cvd-risk/): pick a model,
enter a patient, and see the risk together with the Python call that computes it.

```bash
pip install -r demo/requirements.txt
streamlit run demo/streamlit_app.py
```

- `forms.py` lists the inputs for each model (one entry per model). To add or change a model, edit it there.
- `streamlit_app.py` draws the page; the model descriptions come from `website/data/models.json`.
- `style.css` is the custom look.
- `test_forms.py` checks that every model gives a result with its default inputs (`pytest demo/test_forms.py`).

## Deploying on Streamlit Community Cloud

1. On share.streamlit.io choose **Create app**, then this repository, the branch, and `demo/streamlit_app.py`
   as the main file.
2. Under **App URL**, set the subdomain to `pycvdrisk`. The website embeds `https://pycvdrisk.streamlit.app`
   (`DEMO_URL` in `website/lib/site.ts`).
3. Deploy. `?model=<slug>` opens a model directly, e.g. `https://pycvdrisk.streamlit.app/?model=qrisk3`.
