"""Streamlit demo for all 46 cardiovascular risk calculators in cvd_risk."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, get_args, get_origin

import pandas as pd
import streamlit as st

# Ensure the local src directory is on sys.path when running from the repo
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cvd_risk import models as cvd_models  # noqa: E402
from cvd_risk.core.validation import PatientData  # noqa: E402

MODEL_CLASSES: Dict[str, Any] = {name: getattr(cvd_models, name) for name in cvd_models.__all__}

# Region fallbacks to keep the demo running when a model needs a specific value
MODEL_DEFAULT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "WHO": {"region": "EUR_A"},
    "SCORE2": {"region": "moderate"},
    "SCORE2DM": {"region": "moderate"},
    "SCORE2CKD": {"region": "moderate"},
    "SCORE2OP": {"region": "moderate"},
    "SCORE2AsiaCKD": {"region": "moderate"},
}

CORE_FIELDS = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking"]

# Comprehensive demo defaults (valid ranges, meant to make all models runnable)
DEMO_PATIENT: Dict[str, Any] = {
    "age": 55,
    "sex": "male",
    "systolic_bp": 130.0,
    "total_cholesterol": 5.2,
    "hdl_cholesterol": 1.3,
    "smoking": False,
    "region": "moderate",
    "diabetes": False,
    "ethnicity": "white",
    "bmi": 27.0,
    "family_history": False,
    "antihypertensive": False,
    "statin_use": False,
    "race": "white",
    "hypertension": False,
    "hyperlipidaemia": False,
    "previous_pci": False,
    "previous_cabg": False,
    "aspirin_use": False,
    "angina_episodes_24h": 0,
    "ecg_st_depression": False,
    "troponin_level": 10.0,
    "atrial_fibrillation": False,
    "atypical_antipsychotics": False,
    "impotence2": False,
    "corticosteroids": False,
    "migraine": False,
    "rheumatoid_arthritis": False,
    "renal_disease": False,
    "systemic_lupus": False,
    "severe_mental_illness": False,
    "treated_hypertension": False,
    "type1_diabetes": False,
    "type2_diabetes": False,
    "diabetes_age": 40,
    "hba1c": 45.0,
    "egfr": 90.0,
    "acr": 30.0,
    "proteinuria_trace": "negative",
    "sbps5": 8.0,
    "smoking_category": 1,
    "townsend_deprivation": 0.0,
    "sweating": False,
    "pain_radiation": False,
    "pleuritic": False,
    "palpation": False,
    "ecg_twi": False,
    "presentation_hstni": 5.0,
    "second_hstni": 6.0,
    "typical_symptoms_num": 1,
    "ecg_normal": True,
    "abn_repolarisation": False,
    "atherosclerotic_disease": False,
    "heart_rate": 80.0,
    "creatinine": 1.0,
    "killip_class": 1,
    "cardiac_arrest": False,
}


def _is_bool_type(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return len(args) == 1 and _is_bool_type(args[0])
    return annotation is bool


def _is_number_type(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return len(args) == 1 and _is_number_type(args[0])
    return annotation in (int, float)


def _literal_choices(annotation: Any) -> Optional[List[Any]]:
    if get_origin(annotation) is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _literal_choices(args[0])
    if get_origin(annotation) is not None:
        origin = get_origin(annotation)
        if str(origin).endswith("Literal"):
            return [a for a in get_args(annotation) if a is not None]
    return None


def _friendly_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _parse_number(raw: str, as_int: bool) -> Optional[Union[int, float]]:
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    try:
        return int(value) if as_int else float(value)
    except ValueError:
        st.warning(f"Could not parse '{value}' as a number for {raw}")
        return None


def _apply_overrides(inputs: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    merged = dict(inputs)
    overrides = MODEL_DEFAULT_OVERRIDES.get(model_name, {})
    for key, val in overrides.items():
        if merged.get(key) in (None, "", "Not provided"):
            merged[key] = val
    return merged


@lru_cache(maxsize=None)
def _infer_required_fields(model_name: str) -> List[str]:
    """Best-effort detection of required fields for a given model.

    Strategy:
    - Build a PatientData with core fields filled and all others set to None.
    - Run model.validate_input to let the model surface missing requirements.
    - Parse any explicit missing fields from the error message.
    """
    model_cls = MODEL_CLASSES[model_name]
    # Core filled with safe values; everything else None
    base = {k: DEMO_PATIENT[k] for k in CORE_FIELDS}
    optional_nones = {k: None for k in PatientData.model_fields if k not in CORE_FIELDS}
    patient = PatientData(**base, **optional_nones)
    required = set(CORE_FIELDS)
    try:
        model = model_cls()
        model.validate_input(patient)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # Common patterns we use in the models
        for marker in ["requires the following fields:", "Missing required columns:", "Missing columns:", "Missing required fields:"]:
            if marker in msg:
                after = msg.split(marker, 1)[1]
                # try to extract items inside brackets if present
                if "[" in after and "]" in after:
                    fragment = after.split("[", 1)[1].split("]", 1)[0]
                    fragment = fragment.replace("'", "").replace('"', "")
                    parts = [p.strip() for p in fragment.split(",") if p.strip()]
                    required.update(parts)
                else:
                    # fallback: split by comma
                    parts = [p.strip() for p in after.split(",") if p.strip()]
                    required.update(parts)
    return sorted(required)


def _render_field(name: str, inputs: Dict[str, Any], required: bool, prefix: str = "") -> None:
    field = PatientData.model_fields[name]
    label = f"{prefix}{_friendly_label(name)}" + (" *" if required else "")
    annotation = field.annotation

    literal_choices = _literal_choices(annotation)
    if literal_choices:
        display_options: List[Any] = ["Not provided"] + list(literal_choices)
        current = inputs.get(name)
        index = 0
        if current in literal_choices:
            index = display_options.index(current)
        selection = st.selectbox(
            label, options=display_options, index=index, help=field.description, key=f"fld_{name}"
        )
        inputs[name] = None if selection == "Not provided" else selection
        return

    if _is_bool_type(annotation):
        choice = st.selectbox(
            label,
            options=["Not provided", "Yes", "No"],
            index={"Not provided": 0, True: 1, False: 2}.get(inputs.get(name), 0),
            help=field.description,
            key=f"fld_{name}",
        )
        inputs[name] = {"Not provided": None, "Yes": True, "No": False}[choice]
        return

    if _is_number_type(annotation):
        as_int = annotation in (int, Optional[int])
        placeholder = "" if inputs.get(name) is None else str(inputs[name])
        raw = st.text_input(label, value=placeholder, help=field.description, key=f"fld_{name}")
        inputs[name] = _parse_number(raw, as_int=as_int)
        return

    # Fallback to plain text
    inputs[name] = st.text_input(label, value=inputs.get(name) or "", help=field.description, key=f"fld_{name}") or None


def render_required_fields(inputs: Dict[str, Any], required_fields: List[str]) -> None:
    st.subheader("Required inputs for this model")
    cols = st.columns(3)
    for i, name in enumerate(required_fields):
        with cols[i % 3]:
            _render_field(name, inputs, required=True)


def render_optional_fields(inputs: Dict[str, Any], required_fields: List[str]) -> None:
    st.subheader("Optional inputs")
    optional_fields = [f for f in PatientData.model_fields.keys() if f not in required_fields]
    with st.expander("Show optional inputs"):
        cols = st.columns(3)
        for i, name in enumerate(optional_fields):
            with cols[i % 3]:
                _render_field(name, inputs, required=False)


def run_model(model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    model_cls = MODEL_CLASSES[model_name]
    model = model_cls()
    safe_inputs = _apply_overrides(inputs, model_name)
    patient = PatientData(**safe_inputs)
    result = model.calculate(patient)
    return {
        "model": model_name,
        "risk_score": result.risk_score,
        "risk_category": result.risk_category,
        "model_version": result.model_version,
    }


MODEL_REGION_LABELS: Dict[str, str] = {
    # Primary prevention
    "ASCVD": "US / International",
    "Framingham": "US",
    "Globorisk": "182 countries (country-specific)",
    "Prevent": "US",
    "PROCAM": "Germany",
    "Reynolds": "US",
    "FINRISK": "Finland",
    "REGICOR": "Spain",
    "ProgettoCUORE": "Italy",
    "RISC_Score": "Germany",
    "ARIC_Update": "US",
    "JacksonHeart": "US (African American)",
    "CARDIA": "US young adults",
    "RotterdamStudy": "Netherlands (elderly)",
    "HeinzNixdorf": "Germany",
    "EPIC_Norfolk": "UK",
    "Singapore": "Singapore",
    "PREDICT": "New Zealand",
    "NewZealand": "New Zealand",
    "Dundee": "Scotland",
    "Malaysian_CVD": "Malaysia",
    "Gulf_RACE": "Gulf region",
    "Cambridge": "UK",
    "QRISK2": "UK",
    "QRISK3": "UK",
    "SCORE": "Europe",
    "SCORE2": "Europe",
    "WHO": "Global (14 WHO regions)",
    "INTERHEART": "Global",
    # Secondary / diabetes / CKD / region
    "SMART2": "Europe",
    "SMART_REACH": "Europe",
    "DIAL2": "Diabetes cohorts (Europe)",
    "SCORE2DM": "Europe",
    "DAD_Score": "HIV cohorts (International)",
    "SCORE2CKD": "Europe",
    "SCORE2OP": "Europe (older persons)",
    "ASSIGN": "Scotland",
    "SCORE2AsiaCKD": "Asia",
    # Lifetime
    "LifeCVD2": "Netherlands / Europe",
    # ACS / ED
    "GRACE2": "Global ACS",
    "TIMI": "US (UA/NSTEMI)",
    "EDACS": "Emergency Dept (varied cohorts)",
    "HEART": "Emergency Dept (US/Europe)",
    # Atrial fibrillation / bleeding
    "CHADS2": "Atrial fibrillation",
    "CHA2DS2_VASc": "Atrial fibrillation",
    "HAS_BLED": "Bleeding risk (AF)",
}

CATEGORY_LABELS: Dict[str, str] = {
    # Primary prevention
    "ASCVD": "Primary prevention",
    "Framingham": "Primary prevention",
    "Globorisk": "Primary prevention",
    "Prevent": "Primary prevention",
    "PROCAM": "Primary prevention",
    "Reynolds": "Primary prevention",
    "FINRISK": "Primary prevention",
    "REGICOR": "Primary prevention",
    "ProgettoCUORE": "Primary prevention",
    "RISC_Score": "Primary prevention",
    "ARIC_Update": "Primary prevention",
    "JacksonHeart": "Primary prevention",
    "CARDIA": "Primary prevention",
    "RotterdamStudy": "Primary prevention",
    "HeinzNixdorf": "Primary prevention",
    "EPIC_Norfolk": "Primary prevention",
    "Singapore": "Primary prevention",
    "PREDICT": "Primary prevention",
    "NewZealand": "Primary prevention",
    "Dundee": "Primary prevention",
    "Malaysian_CVD": "Primary prevention",
    "Gulf_RACE": "Primary prevention",
    "Cambridge": "Primary prevention",
    "QRISK2": "Primary prevention",
    "QRISK3": "Primary prevention",
    "SCORE": "Primary prevention",
    "SCORE2": "Primary prevention",
    "WHO": "Primary prevention",
    "INTERHEART": "Primary prevention",
    # Specialized
    "SMART2": "Secondary prevention",
    "SMART_REACH": "Secondary prevention",
    "DIAL2": "Diabetes",
    "SCORE2DM": "Diabetes",
    "DAD_Score": "HIV",
    "SCORE2CKD": "CKD",
    "SCORE2OP": "Older adults",
    "ASSIGN": "Region-specific",
    "SCORE2AsiaCKD": "CKD",
    "LifeCVD2": "Lifetime",
    "GRACE2": "Acute coronary syndrome",
    "TIMI": "Acute coronary syndrome",
    "EDACS": "Emergency",
    "HEART": "Emergency",
    "CHADS2": "Atrial fibrillation",
    "CHA2DS2_VASc": "Atrial fibrillation",
    "HAS_BLED": "Bleeding risk",
}


def _placeholder_url(name: str) -> str:
    slug = name.lower().replace(" ", "-")
    return f"https://picsum.photos/seed/{slug}/600/340"


def _render_model_card(name: str) -> bool:
    region = MODEL_REGION_LABELS.get(name, "General / multiple regions")
    category = CATEGORY_LABELS.get(name, "Cardiovascular")
    with st.container(border=True):
        st.markdown(
            f"<img src='{_placeholder_url(name)}' class='card-img' loading='lazy' />",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='card-title'>{name}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='card-region'><span class='chip chip-region'>{region}</span> "
            f"<span class='chip chip-cat'>{category}</span></div>",
            unsafe_allow_html=True,
        )
        return st.button("Open calculator", key=f"btn_{name}")


def render_model_grid(search: str, category_filter: Optional[str]) -> Optional[str]:
    st.subheader("Pick a calculator")
    names = sorted(MODEL_CLASSES.keys())
    if search:
        names = [n for n in names if search.lower() in n.lower()]
    if category_filter and category_filter != "All":
        names = [n for n in names if CATEGORY_LABELS.get(n, "") == category_filter]

    cols = st.columns(3)
    chosen = None
    for idx, name in enumerate(names):
        with cols[idx % 3]:
            if _render_model_card(name):
                chosen = name
    return chosen


def render_calculator(model_choice: str, inputs: Dict[str, Any]) -> None:
    required_fields = _infer_required_fields(model_choice)

    calc_results = st.session_state.setdefault("calc_results", {})

    top_col1, top_col2, top_col3 = st.columns([3, 2, 2])
    with top_col1:
        region = MODEL_REGION_LABELS.get(model_choice, "General / multiple regions")
        st.markdown(f"### {model_choice}")
        st.caption(region)
    with top_col2:
        if st.button("Load demo patient"):
            for k, v in DEMO_PATIENT.items():
                inputs[k] = v
    with top_col3:
        if st.button("Clear all inputs"):
            for key in list(inputs.keys()):
                inputs[key] = None

    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"**{model_choice}** requires these fields:")
        render_required_fields(inputs, required_fields)
        render_optional_fields(inputs, required_fields)
        if st.button(f"Calculate {model_choice}", use_container_width=True):
            try:
                result = run_model(model_choice, inputs)
                calc_results[model_choice] = {"status": "ok", "data": result}
            except Exception as exc:  # noqa: BLE001
                calc_results[model_choice] = {"status": "error", "error": str(exc)}

    with right:
        pane = calc_results.get(model_choice)
        st.markdown("#### Result")
        if pane is None:
            st.info("Run the calculator to see results here.")
        elif pane["status"] == "error":
            st.error(pane["error"])
        else:
            res = pane["data"]
            st.metric("Risk score", f"{res['risk_score']:.2f}")
            st.write(f"Risk category: **{res['risk_category']}**")
            st.caption(f"Model version: {res['model_version']}")


def main() -> None:
    st.set_page_config(page_title="CVD Risk Calculator Demo", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --primary: linear-gradient(120deg, #fb7185, #f97316);
            --primary-solid: #f97316;
            --primary-dark: #ea580c;
        }
        .main {
            background: radial-gradient(circle at 20% 20%, #f0fdf4 0, #ecfeff 25%, #f8fafc 60%, #f8fafc 100%);
        }
        .block-container {
            padding-top: 1.5rem;
        }
        .card-title {
            font-weight: 700;
            font-size: 1.05rem;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }
        .card-region {
            color: #334155;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }
        .card-img {
            width: 100%;
            aspect-ratio: 16 / 9;
            border-radius: 10px;
            margin-bottom: 0.75rem;
            box-shadow: inset 0 1px 4px rgba(255,255,255,0.35), 0 8px 18px rgba(15, 23, 42, 0.08);
            object-fit: cover;
            display: block;
        }
        .stContainer {
            border-radius: 10px !important;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.55rem 0.75rem;
            font-weight: 600;
            box-shadow: 0 10px 20px rgba(15, 118, 110, 0.2);
            transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 26px rgba(251, 113, 133, 0.28);
            background: linear-gradient(120deg, #fb7185, #fdba74);
        }
        .stButton>button:active {
            transform: translateY(0);
            box-shadow: 0 6px 14px rgba(249, 115, 22, 0.25);
        }
        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            border: 1px solid #e2e8f0;
            background: rgba(255,255,255,0.7);
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            margin-right: 0.3rem;
        }
        .chip-region { color: #0f172a; }
        .chip-cat { color: #0f766e; }
        .search-row {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .glass {
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(226,232,240,0.6);
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, rgba(15,118,110,0.12), rgba(59,130,246,0.1));
            border: 1px solid #dbeafe;
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 1.2rem; font-weight: 700; color: #0f172a;">CVD Risk Calculator Demo</div>
            <div style="color: #334155; margin-top: 0.15rem;">Choose a calculator card to open its tailored form.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "patient_inputs" not in st.session_state:
        st.session_state["patient_inputs"] = dict(DEMO_PATIENT)
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = None
    if "applied_search" not in st.session_state:
        st.session_state["applied_search"] = ""
    if "applied_category" not in st.session_state:
        st.session_state["applied_category"] = "All"

    inputs = st.session_state["patient_inputs"]
    selected = st.session_state["selected_model"]

    if selected is None:
        st.markdown(
            """
            <div class="glass">
                <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">Find a calculator</div>
                <div class="search-row">
                    <div id="search-box"></div>
                    <div id="category-box"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        categories = ["All"] + sorted(set(CATEGORY_LABELS.values()))
        col_s, col_c, col_b = st.columns([2.5, 1.2, 0.8])
        with col_s:
            search_val = st.text_input("Search", key="search_box", placeholder="Type to filter by name...")
        with col_c:
            category_val = st.selectbox("Category", categories, index=categories.index(st.session_state["applied_category"]), key="category_box")
        with col_b:
            if st.button("Apply filters", use_container_width=True):
                st.session_state["applied_search"] = search_val
                st.session_state["applied_category"] = category_val
                st.experimental_rerun()

        choice = render_model_grid(st.session_state["applied_search"], st.session_state["applied_category"])
        if choice:
            st.session_state["selected_model"] = choice
            st.rerun()
    else:
        if st.button("← Back to model list"):
            st.session_state["selected_model"] = None
            st.rerun()
        render_calculator(selected, inputs)

    # Persist updates
    st.session_state["patient_inputs"] = inputs


if __name__ == "__main__":
    main()

