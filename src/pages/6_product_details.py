"""Product details page for recommended foods."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover - streamlit runtime should provide duckdb
    duckdb = None  # type: ignore

from utils import get_image_url

DB_PATH = "db/open_food.duckdb"
PRODUCT_IMAGE_WIDTH = 320

def _fetch_single_row(query: str, params: list[Any]) -> dict[str, Any]:
    with duckdb.connect(DB_PATH, read_only=True) as connection:  # type: ignore[attr-defined]
        cursor = connection.execute(query, params)
        row = cursor.fetchone()
        if row is None:
            return {}
        columns = [col[0] for col in cursor.description or []]
        return dict(zip(columns, row, strict=False))


@st.cache_data(show_spinner=False)
def load_product(code: str) -> dict[str, Any]:
    """Fetch core product metadata."""

    query = """
        SELECT
            *
        FROM products
        WHERE code = ?
        LIMIT 1
    """
    return _fetch_single_row(query, [code])


@st.cache_data(show_spinner=False)
def load_ingredients_profile(code: str) -> dict[str, Any]:
    """Fetch ingredient and allergen metadata."""

    query = """
        SELECT *
        FROM product_ingredients
        WHERE code = ?
        LIMIT 1
    """
    return _fetch_single_row(query, [code])


@st.cache_data(show_spinner=False)
def load_nutrients(code: str) -> dict[str, Any]:
    """Fetch nutrient profile for the product."""

    query = """
        SELECT *
        FROM product_nutrients
        WHERE code = ?
        LIMIT 1
    """
    return _fetch_single_row(query, [code])


def _prettify_label(label: str) -> str:
    cleaned = label.replace("_", " ").replace("-", " ").strip()
    cleaned = " ".join(part for part in cleaned.split() if part)
    return cleaned.title()


def _format_nutrient_rows(nutrient_record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in nutrient_record.items():
        if key == "code":
            continue
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, (int, float)):
            rows.append(
                {
                    "Nutrient": _prettify_label(key),
                    "Value": round(float(value), 4),
                }
            )
    rows.sort(key=lambda item: item["Nutrient"])
    return rows


def _extract_image_url(product: dict[str, Any]) -> str | None:
    for key in (
        "image_url",
        "image_front_url",
        "image_small_url",
        "image_front_small_url",
        "image_thumb_url",
    ):
        candidate = str(product.get(key) or "").strip()
        if candidate:
            return candidate

    return get_image_url(product.get("code"), product.get("image_key"), product.get("image_rev"))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            else:
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
        separators = [",", ";"]
        for separator in separators:
            if separator in raw:
                parts = [part.strip() for part in raw.split(separator)]
                return [part for part in parts if part]
        return [raw]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _collect_ingredient_texts(
    product: dict[str, Any], ingredients_profile: dict[str, Any]
) -> list[str]:
    texts: list[str] = []
    for record in (product, ingredients_profile):
        if not record:
            continue
        for key in (
            "ingredients_text",
        ):
            value = record.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    texts.append(cleaned)
        if texts:
            break
    if not texts and ingredients_profile:
        for key, value in ingredients_profile.items():
            if key.startswith("ingredients_text") and isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    texts.append(cleaned)
    if not texts:
        for record in (ingredients_profile, product):
            if not record:
                continue
            tag_text = ", ".join(_as_list(record.get("ingredients_tags")))
            if tag_text:
                texts.append(tag_text)
    deduped: dict[str, None] = {text: None for text in texts}
    return list(deduped.keys())


def _collect_allergen_labels(
    product: dict[str, Any], ingredients_profile: dict[str, Any]
) -> list[str]:
    labels: list[str] = []
    for record in (product, ingredients_profile):
        if not record:
            continue
        for key, value in record.items():
            if key.startswith("contains_"):
                if value:
                    raw = key.removeprefix("contains_")
                    labels.append(_prettify_label(raw))
            elif key.startswith("allergens") and value:
                for item in _as_list(value):
                    cleaned = item.split(":", 1)[-1] if ":" in item else item
                    cleaned = cleaned.replace("_", " ").strip()
                    if cleaned:
                        labels.append(_prettify_label(cleaned))
    deduped: dict[str, None] = {label: None for label in labels}
    return list(deduped.keys())


def _resolve_product_code() -> str:
    possible_keys = ("product_code", "code")
    if hasattr(st, "query_params"):
        # Streamlit 1.32+ exposes typed query params
        params = st.query_params  # type: ignore[attr-defined]
        for key in possible_keys:
            value = params.get(key)
            if value:
                return str(value)

    params_legacy = st.experimental_get_query_params()
    for key in possible_keys:
        values = params_legacy.get(key)
        if values:
            return str(values[0])

    return str(st.session_state.get("selected_product_code") or "")


params = st.query_params
code = page_param = params.get("product_code", None)
if not code:
    st.error("No product code provided.")
    st.stop()

if duckdb is None:
    st.error("DuckDB dependency is unavailable; unable to load product details.")
    st.stop()

product = load_product(code)
if not product:
    st.error("Product not found in the catalogue.")
    st.stop()

title = product.get("product_name") or code
st.title(str(title))
st.caption(f"Product code: {code}")

brands = product.get("brands")
if brands:
    st.write(f"Brands: {brands}")

categories = product.get("categories")
if categories:
    st.write(f"Categories: {categories}")

image_url = _extract_image_url(product)
if image_url:
    st.image(image_url, caption=title, width=PRODUCT_IMAGE_WIDTH)

nutrients = load_nutrients(code)
nutrient_rows = _format_nutrient_rows(nutrients)
if nutrient_rows:
    st.subheader("Nutrients (per 100g)")
    st.table(nutrient_rows)
else:
    st.info("No nutrient information available for this product.")

ingredients_profile = load_ingredients_profile(code)
allergen_labels = _collect_allergen_labels(product, ingredients_profile)
if allergen_labels:
    st.subheader("Allergens")
    st.write(", ".join(allergen_labels))
else:
    st.info("No allergen information available for this product.")

#ingredient_texts = _collect_ingredient_texts(product, ingredients_profile)
#if ingredient_texts:
#    st.subheader("Ingredients")
#    for text in ingredient_texts:
#        st.write(text)
#else:
#    st.info("No ingredient information available for this product.")
