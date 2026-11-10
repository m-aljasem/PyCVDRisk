"""PyCVDRisk demo: pick a model, enter a patient, get the risk from the cvd-risk package.

Run locally:  pip install -r demo/requirements.txt && streamlit run demo/streamlit_app.py
The inputs for every model live in forms.py; the model descriptions are the website's own
(website/data/models.json), so the demo and the site never disagree.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import cvd_risk
from forms import FORMS, Field, Variant

HERE = Path(__file__).parent
WEBSITE = HERE.parent / "website"
SITE_URL = "https://pycvdrisk.aljasem.eu.org"
REPO = "https://github.com/m-aljasem/PyCVDRisk"

st.set_page_config(page_title="PyCVDRisk demo", page_icon=str(WEBSITE / "public" / "favicon.ico"), layout="wide",
                   initial_sidebar_state="collapsed")


@st.cache_data
def load_site() -> tuple[dict, dict, dict]:
    data = json.loads((WEBSITE / "data" / "models.json").read_text())
    photos = json.loads((WEBSITE / "data" / "photos.json").read_text())
    return {m["slug"]: m for m in data["models"] if m["slug"] in FORMS}, data["groups"], photos


@st.cache_data
def photo_uri(file: str) -> str:
    path = WEBSITE / "public" / "images" / "regions" / file
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def pretty(v) -> str:
    """'very_high' -> 'Very high'; country codes and numbers stay as they are."""
    s = str(v)
    return s if s.isupper() or not s.replace("_", "").isalpha() else s.replace("_", " ").capitalize()


st.markdown((HERE / "style.css").read_text().join(("<style>", "</style>")), unsafe_allow_html=True)

# ?site=1: embedded in the website's /demo page. Blend into the page and tell it our height, so its frame
# fits the content exactly (no inner scrollbar). The script runs in a same-origin component frame.
if st.query_params.get("site") == "1":
    components.html("""<script>
      const doc = window.parent.document;
      doc.body.classList.add("site-mode");
      const block = doc.querySelector(".block-container");
      const send = () => window.top.postMessage({pycvdriskHeight: block.scrollHeight}, "*");
      new ResizeObserver(send).observe(block);
      send();
    </script>""", height=0)

models, groups, photos = load_site()

# Model picker; ?model=<slug> deep-links (the website's "Try it" buttons use it)
slug = st.query_params.get("model", "score2")
if slug not in models:
    slug = "score2"
SHORT_GROUP = {"primary": "Primary prevention", "conditions": "Specific conditions", "acute": "Acute care"}
top = st.columns([1.3, 1])
with top[0]:
    group = st.segmented_control("Category", list(groups), default=models[slug]["group"],
                                 format_func=lambda g: SHORT_GROUP.get(g, groups[g]),
                                 key="group", label_visibility="collapsed") or models[slug]["group"]
in_group = [s for s, m in models.items() if m["group"] == group]
with top[1]:
    slug = st.selectbox("Model", in_group, index=in_group.index(slug) if slug in in_group else 0,
                        format_func=lambda s: f"{models[s]['name']}  ·  {models[s]['region']}, {models[s]['year']}",
                        label_visibility="collapsed")
st.query_params["model"] = slug
model = models[slug]
photo = photos.get(model["photo"], {})

st.markdown(f"""
<div class="hero" style="background-image: linear-gradient(100deg, rgba(17,12,24,.92) 0%, rgba(17,12,24,.72) 45%,
     rgba(17,12,24,.25) 100%), url('{photo_uri(photo.get("file", ""))}')">
  <div class="chips"><span>{model['region']}</span><span>{model['year']}</span><span>{groups[model['group']]}</span></div>
  <h1>{model['name']}</h1>
  <p class="full">{model['fullName']}</p>
  <p class="outcome">{model['outcome']}</p>
</div>
""", unsafe_allow_html=True)

variants: list[Variant] = FORMS[slug]
variant = variants[0]
if len(variants) > 1:
    label = st.segmented_control("Variant", [v.label for v in variants], default=variants[0].label,
                                 key=f"{slug}:variant", label_visibility="collapsed") or variants[0].label
    variant = next(v for v in variants if v.label == label)


def widget(f: Field, key: str):
    label = f"{f.label} ({f.unit})" if f.unit else f.label
    if f.kind == "bool":
        return int(st.toggle(f.label, value=f.default, key=key, help=f.help or None))
    if f.kind == "choice":
        if len(f.options) <= 4:
            return st.segmented_control(label, f.options, default=f.default, format_func=pretty, key=key,
                                        help=f.help or None) or f.default
        return st.selectbox(label, f.options, index=f.options.index(f.default), format_func=pretty, key=key,
                            help=f.help or None)
    as_int = float(f.step).is_integer() and float(f.default).is_integer()
    cast = int if as_int else float
    return st.number_input(label, value=cast(f.default), min_value=None if f.lo is None else cast(f.lo),
                           max_value=None if f.hi is None else cast(f.hi), step=cast(f.step), key=key,
                           help=f.help or None, format=None if as_int else "%.1f")


form_col, result_col = st.columns([1.35, 1], gap="large")

with form_col:
    st.markdown('<p class="section">Patient</p>', unsafe_allow_html=True)
    inputs: dict = {}
    values = [f for f in variant.fields if f.kind != "bool"]
    flags = [f for f in variant.fields if f.kind == "bool"]
    with st.container(border=True):
        cols = st.columns(2)
        for i, f in enumerate(values):
            with cols[i % 2]:
                inputs[f.name] = widget(f, f"{slug}:{variant.label}:{f.name}")
    if flags:
        st.markdown('<p class="section">Conditions and treatment</p>', unsafe_allow_html=True)
        with st.container(border=True):
            cols = st.columns(2)
            for i, f in enumerate(flags):
                with cols[i % 2]:
                    inputs[f.name] = widget(f, f"{slug}:{variant.label}:{f.name}")

args = {**inputs, **variant.fixed}
error = None
try:
    value = float(np.asarray(variant.fn(**args), dtype=float).ravel()[0])
except ValueError as e:  # the functions raise only for wrong options, never for row-level data
    value, error = math.nan, str(e)


def ring(fraction: float) -> str:
    r, c = 70, 2 * math.pi * 70
    dash = c * min(max(fraction, 0.0), 1.0)
    return (f'<svg viewBox="0 0 160 160" class="ring"><circle cx="80" cy="80" r="{r}" class="track"/>'
            f'<circle cx="80" cy="80" r="{r}" class="bar" stroke-dasharray="{dash:.1f} {c:.1f}"/></svg>')


def result_html() -> str:
    if math.isnan(value):
        why = error or ("These inputs are outside what the model covers (often the age range, or a sex or "
                        "region it was not derived for). The library returns NaN instead of guessing.")
        return f'<div class="result na"><p class="kicker">Not applicable</p><p class="why">{why}</p></div>'
    if variant.output == "risk":
        big, unit, caption, frac = f"{value * 100:.1f}", "%", model["outcome"], value
    elif variant.output == "age":
        big, unit, caption, frac = f"{value:.1f}", "yrs", "Median age reached free of CVD", None
    else:
        big, unit = f"{value:g}", "pts" if variant.max_points is None else f"/ {variant.max_points}"
        frac = value / variant.max_points if variant.max_points else None
        caption = f"Equivalent 10-year risk: {variant.points_to_risk(value)}" if variant.points_to_risk else "Score points"
    gauge = ring(frac) if frac is not None else ""
    return (f'<div class="result"><div class="gauge">{gauge}<div class="value"><span class="big">{big}</span>'
            f'<span class="unit">{unit}</span></div></div><p class="caption">{caption}</p></div>')


def python_call() -> str:
    def lit(v):
        return f'"{v}"' if isinstance(v, str) else repr(v)
    body = ", ".join(f"{k}={lit(v)}" for k, v in args.items())
    return f"from cvd_risk import {variant.fn.__name__}\n\n{variant.fn.__name__}({body})\n# -> {value!r}"


with result_col:
    st.markdown('<p class="section">Result</p>', unsafe_allow_html=True)
    st.markdown(result_html(), unsafe_allow_html=True)
    st.markdown('<p class="section">The same call in Python</p>', unsafe_allow_html=True)
    st.code(python_call(), language="python", wrap_lines=True)

    evidence = ("Verified against a reference implementation" if model["evidence"] == "A"
                else "Checked against the published paper")
    st.markdown(f"""
<div class="about">
  <p class="kicker">About this model</p>
  <p><b>{evidence}.</b> {model['verification']}</p>
  <div class="links">
    <a href="{model['paperUrl']}" target="_blank">Original paper ↗</a>
    <a href="{model['report']}" target="_blank">Verification report ↗</a>
    <a href="{SITE_URL}/docs/models/{slug}" target="_blank">Documentation ↗</a>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="foot">
  <span>cvd-risk {cvd_risk.__version__} · <code>pip install cvd-risk</code></span>
  <span>For research and education; not a substitute for clinical judgement.</span>
  <a href="{REPO}" target="_blank">Source on GitHub ↗</a>
</div>
""", unsafe_allow_html=True)
