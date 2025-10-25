from __future__ import annotations

import csv
from functools import lru_cache
import html
from pathlib import Path
from typing import Sequence
from urllib.parse import quote_plus

import streamlit as st

from hybrid_recommender import (
    EXPLICIT_MAP_AT_10,
    EXPLICIT_WEIGHT,
    IMPLICIT_MAP_AT_10,
    IMPLICIT_WEIGHT,
    HybridRecommender,
    RecommendationResult,
    UserSummary,
    USERS_PATH,
)


@st.cache_resource(show_spinner=False)
def load_recommender() -> HybridRecommender:
    """Initialise the hybrid recommender once per Streamlit session."""

    return HybridRecommender()


@lru_cache(maxsize=1)
def load_user_profiles(path: str) -> dict[str, str]:
    """Return user profile names keyed by user identifier."""

    file_path = Path(path)
    if not file_path.exists():
        return {}

    profiles: dict[str, str] = {}
    with file_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user_id = (row.get("UserID") or "").strip()
            if not user_id:
                continue
            profiles[user_id] = (row.get("UserProfile") or "").strip()
    return profiles


def format_user_option(user: UserSummary) -> str:
    """Return the display label for the user selectbox."""

    if user.display_name and user.display_name != user.user_id:
        return f"{user.display_name} ({user.user_id})"
    return user.user_id

def render_recommendations(results: Sequence[RecommendationResult]) -> None:
    """Render the recommendations in a Streamlit dataframe."""

    if not results:
        st.info("No recommendations found for the selected user.")
        return

    if not st.session_state.get("_recommendations_table_css"):
        st.markdown(
            """
            <style>
            .recommendations-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 1.5rem;
            }
            .recommendations-table thead th {
                text-align: left;
                padding: 0.6rem 0.75rem;
                border-bottom: 1px solid rgba(49, 51, 63, 0.2);
                background-color: rgba(49, 51, 63, 0.04);
                font-weight: 600;
                font-size: 0.9rem;
            }
            .recommendations-table tbody td {
                padding: 0.55rem 0.75rem;
                border-bottom: 1px solid rgba(49, 51, 63, 0.12);
            }
            .recommendations-table tbody tr:hover {
                background-color: rgba(49, 51, 63, 0.06);
            }
            .recommendations-table tbody td a {
                color: inherit;
                text-decoration: none;
            }
            .recommendations-table tbody td a:hover {
                text-decoration: underline;
            }
            .recommendations-table tbody td:first-child a {
                font-weight: 600;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_recommendations_table_css"] = True

    def format_score(value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.4f}"

    table_rows: list[str] = []
    for result in results:
        product_code = result.product_code
        product_label = result.product_name or product_code
        detail_url = f"./?page=6_product_details&product_code={quote_plus(product_code)}"
        escaped_url = html.escape(detail_url, quote=True)
        link_attrs = f'href="{escaped_url}"'
        product_cell = f"<td><a {link_attrs}>{html.escape(product_label)}</a></td>"
        code_cell = f"<td>{html.escape(product_code)}</td>"
        combined_cell = f"<td>{html.escape(format_score(result.combined_score))}</td>"
        explicit_cell = f"<td>{html.escape(format_score(result.explicit_score))}</td>"
        implicit_cell = f"<td>{html.escape(format_score(result.implicit_score))}</td>"
        allergens_text = ", ".join(result.allergens)
        allergens_cell = (
            f"<td>{html.escape(allergens_text)}</td>" if allergens_text else "<td></td>"
        )
        table_rows.append(
            "<tr>"
            f"{product_cell}"
            f"{code_cell}"
            f"{combined_cell}"
            f"{explicit_cell}"
            f"{implicit_cell}"
            f"{allergens_cell}"
            "</tr>"
        )

    header_html = (
        "<thead><tr>"
        "<th>Product</th>"
        "<th>Product Code</th>"
        "<th>Combined Score</th>"
        "<th>Explicit Score</th>"
        "<th>Implicit Score</th>"
        "<th>Allergens</th>"
        "</tr></thead>"
    )
    body_html = "<tbody>" + "".join(table_rows) + "</tbody>"
    table_html = f'<table class="recommendations-table">{header_html}{body_html}</table>'
    st.markdown(table_html, unsafe_allow_html=True)


recommender = load_recommender()

st.title("Food Recommendations")
st.caption(
    "Hybrid recommendations combine explicit (ratings) and implicit (behaviour) models "
    "weighted by their MAP@10 test performance."
)

weight_caption = (
    f"Explicit MAP@10: {EXPLICIT_MAP_AT_10:.4f} (weight {EXPLICIT_WEIGHT:.2f}) · "
    f"Implicit MAP@10: {IMPLICIT_MAP_AT_10:.4f} (weight {IMPLICIT_WEIGHT:.2f})"
)
st.caption(weight_caption)

user_options = recommender.list_users()
if not user_options:
    st.warning("No users available in the recommender datasets.")
    st.stop()

default_user = st.session_state.get("user_id")
default_index = 0

if default_user is not None:
    for index, user in enumerate(user_options):
        if user.user_id == default_user:
            default_index = index
            break

user_profiles = load_user_profiles(str(USERS_PATH))
selected_user = user_options[default_index]
st.session_state["user_id"] = selected_user.user_id

profile_name = user_profiles.get(selected_user.user_id)
st.write(f"Current user: {selected_user.display_name}")
st.write(f"Profile: {profile_name}" if profile_name else "Profile: Not available")

# selected_user = st.selectbox(
#     "Select a user",
#     user_options,
#     index=default_index,
#     format_func=format_user_option,
# )

top_k = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=30,
    value=10,
    step=1,
)

with st.spinner("Computing recommendations..."):
    print(f"recommending for {selected_user.user_id}")
    recommendations = recommender.recommend(selected_user.user_id, top_k=top_k)

render_recommendations(recommendations)
