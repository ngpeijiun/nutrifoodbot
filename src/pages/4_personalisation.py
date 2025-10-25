from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from persistence.personalisation import PERSONALISATION_STORE

PROFILE_SENTINEL_KEY = "_profile_state_initialised_for"
PAGE_CLEANUP_REGISTRY_KEY = "_page_cleanup_registry"


def _register_profile_state_cleanup() -> None:
    """Ensure profile state reloads after visiting other pages."""

    ctx = get_script_run_ctx()
    page_hash = getattr(ctx, "page_script_hash", None)
    if page_hash is None:
        return

    registry: dict[str, Callable[[], None]] = st.session_state.setdefault(
        PAGE_CLEANUP_REGISTRY_KEY,
        {},
    )

    def _clear_profile_sentinel() -> None:
        st.session_state.pop(PROFILE_SENTINEL_KEY, None)

    registry[page_hash] = _clear_profile_sentinel

_register_profile_state_cleanup()


def _initialise_profile_state() -> None:
    """Reload the stored profile so widgets reflect the latest values."""
    user_id = st.session_state.get("user_id")
    stored: dict[str, Any] | None = None
    if user_id is not None:
        stored = PERSONALISATION_STORE.load_user_profile(str(user_id))

    if not isinstance(stored, dict):
        for key in PROFILE_STATE_KEYS:
            st.session_state.pop(key, None)
        return

    for key in PROFILE_STATE_KEYS:
        if key in stored:
            st.session_state[key] = deepcopy(stored[key])
        else:
            st.session_state.pop(key, None)


def _collect_profile_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in PROFILE_STATE_KEYS:
        if key in st.session_state:
            payload[key] = deepcopy(st.session_state[key])
    return payload


def _ensure_profile_state_initialised() -> None:
    """Initialise stored preferences once per user session."""

    current_user_id = st.session_state.get("user_id")
    if st.session_state.get(PROFILE_SENTINEL_KEY) == current_user_id:
        return

    _initialise_profile_state()
    st.session_state[PROFILE_SENTINEL_KEY] = current_user_id


def _slider_step(min_value: float, max_value: float) -> float:
    """Return a slider increment that divides the range into 100 steps."""

    return max((max_value - min_value) / 100, 0.01)


@st.cache_data(show_spinner=False)
def load_dietary_options() -> list[str]:
    """Retrieve the dietary compatibility labels from DuckDB."""

    column_names = PERSONALISATION_STORE.list_dietary_columns()
    options: list[str] = []
    for column in column_names:
        label = (
            column.removeprefix("is_")
            .removesuffix("_compatible")
            .replace("_", " ")
            .title()
        )
        options.append(label)

    return options


@st.cache_data(show_spinner=False)
def load_allergen_options() -> list[str]:
    """Retrieve allergen labels from DuckDB."""

    column_names = PERSONALISATION_STORE.list_allergen_columns()
    options: list[str] = []
    for column in column_names:
        label = column.removeprefix("contains_").replace("_", " ").title()
        options.append(label)

    return options


DIETARY_OPTIONS = load_dietary_options()

CULTURAL_OPTIONS = [
    "Vegan",
    "Vegetarian",
    "No beef",
    "No pork",
    "No alcohol",
]

ALLERGEN_OPTIONS = load_allergen_options()

HEALTH_GOALS = [
    "Weight management",
    "Heart health",
    "Diabetes support",
    "Muscle gain",
    "Digestive health",
    "Low sodium",
    "Energy boost",
]

MEDICAL_CONDITIONS = [
    "Hypertension",
    "Type 2 diabetes",
    "Gestational diabetes",
    "Celiac disease",
    "High cholesterol",
    "Kidney disease",
]

BUDGET_FOCUS_OPTIONS = [
    "Balanced value",
    "Prioritize affordability",
    "Prioritize nutrition quality",
]

PROFILE_STATE_KEYS = [
    "dietary_preferences",
    "cultural_requirements",
    "allergies",
    "additional_allergy_notes",
    "health_goals",
    "medical_conditions",
    "preferred_cuisines",
    "max_price_per_item",
    "budget_focus",
    "use_budget_filter",
    "max_sugar",
    "use_sugar_filter",
    "max_fat",
    "use_fat_filter",
    "max_sodium",
    "use_sodium_filter",
    "min_protein",
    "use_protein_filter",
]

_ensure_profile_state_initialised()


st.title("Personalisation")
st.write(
    "Set the preferences and guardrails NutriFoodBot should respect when recommending food products."
)

st.subheader("Dietary & Cultural Profile")
st.multiselect(
    "Dietary preferences",
    DIETARY_OPTIONS,
    key="dietary_preferences",
)
st.multiselect(
    "Cultural or religious constraints",
    CULTURAL_OPTIONS,
    key="cultural_requirements",
)
st.text_input(
    "Preferred cuisines or ingredients",
    key="preferred_cuisines",
    help="Add cues like 'Mediterranean lunches' or 'no artificial sweeteners'.",
)

st.subheader("Allergies & Intolerances")
st.multiselect(
    "Allergens to avoid",
    ALLERGEN_OPTIONS,
    key="allergies",
)
st.text_input(
    "Other allergens or notes",
    key="additional_allergy_notes",
    help="List uncommon ingredients or cross-contamination warnings to surface in chat.",
)

st.subheader("Health Goals & Conditions")
st.multiselect(
    "Health goals",
    HEALTH_GOALS,
    key="health_goals",
)
st.multiselect(
    "Medical considerations",
    MEDICAL_CONDITIONS,
    key="medical_conditions",
)

st.subheader("Budget & Shopping Preferences")
st.slider(
    "Max price per item (SGD)",
    1.0,
    50.0,
    step=_slider_step(1.0, 50.0),
    key="max_price_per_item",
    help="Upper spend that the recommender should try to stay within when possible.",
)
st.selectbox(
    "Budget focus",
    BUDGET_FOCUS_OPTIONS,
    key="budget_focus",
)
st.checkbox(
    "Highlight budget considerations in chat",
    key="use_budget_filter",
)

st.subheader("Nutritional Guardrails")
col_left, col_right = st.columns(2)
with col_left:
    st.slider(
        "Max sugar (g per 100g)",
        0.0,
        100.0,
        step=_slider_step(0.0, 100.0),
        key="max_sugar",
        help="Caps high-sugar items when enabled.",
    )
    st.checkbox(
        "Use sugar limit",
        key="use_sugar_filter",
    )
    st.slider(
        "Max sodium (mg per 100g)",
        0.0,
        2000.0,
        step=_slider_step(0.0, 2000.0),
        key="max_sodium",
        help="Keep sodium-sensitive recommendations under control.",
    )
    st.checkbox(
        "Use sodium limit",
        key="use_sodium_filter",
    )
with col_right:
    st.slider(
        "Max fat (g per 100g)",
        0.0,
        100.0,
        step=_slider_step(0.0, 100.0),
        key="max_fat",
        help="Reduces high-fat options when the limit is active.",
    )
    st.checkbox(
        "Use fat limit",
        key="use_fat_filter",
    )
    st.slider(
        "Min protein (g per 100g)",
        0.0,
        50.0,
        step=_slider_step(0.0, 50.0),
        key="min_protein",
        help="Surface higher-protein products when the floor is active.",
    )
    st.checkbox(
        "Use protein floor",
        key="use_protein_filter",
    )

save_col, status_col = st.columns([1, 3])
with save_col:
    save_clicked = st.button("Save")
status_placeholder = status_col.empty()
user_id = st.session_state.get("user_id")

if save_clicked:
    if user_id is None:
        status_placeholder.error("Unable to save without a user ID. Please sign in again.")
    else:
        profile_payload = _collect_profile_payload()
        did_save = PERSONALISATION_STORE.save_user_profile(str(user_id), profile_payload)
        if did_save:
            status_placeholder.success("Preferences saved.")
        else:
            status_placeholder.error("Failed to save preferences. Please try again later.")
