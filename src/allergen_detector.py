"""Rule-based allergen detection module.

This module provides a simple function to check if a food product contains
allergens that the user wants to avoid based on their profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import duckdb
from persistence.personalisation import PERSONALISATION_STORE  

# Minimal synonym map to make user-provided labels more robust.
# Canonical labels should already be aligned with contains_* columns via UI,
# but this helps for free-form profiles or external calls.
_ALLERGEN_SYNONYMS: dict[str, str] = {
    "dairy": "milk",
    "sulphites": "sulfites",  # UK spelling → DB column spelling
    "lupine": "lupin",
    "mollusc": "molluscs",
    "tree nut": "tree nuts",
    "tree-nut": "tree nuts",
    "peanut": "peanuts",
    "egg": "eggs",
}


def _to_allergen_column(label: str) -> str:
    """Normalize a user allergen label into a contains_* column name.

    Keeps behavior stable for canonical labels while adding a few safe
    synonyms. Does not widen matching beyond existing DB columns.
    """

    a = label.strip().lower()
    if not a:
        return ""
    # Normalize simple punctuation/hyphenation variants
    a = a.replace("-", " ")
    a = _ALLERGEN_SYNONYMS.get(a, a)
    return "contains_" + a.replace(" ", "_")

# Use the same DB_PATH as the personalisation module
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "open_food.duckdb"


def is_containing_allergens(user_id: str, food_code: str) -> bool:
    """Check if a food product contains allergens the user needs to avoid.

    This is a rule-based allergen detector that:
    1. Loads the user's allergen preferences from their profile
    2. Queries the product_ingredients table for allergen flags
    3. Returns True if any of the user's allergens are present in the food

    Args:
        user_id: The user's unique identifier
        food_code: The product code to check for allergens

    Returns:
        True if the food contains any allergens the user wants to avoid,
        False otherwise (including cases where user has no allergen preferences
        or the food product is not found)

    Example:
        >>> is_containing_allergens("user_123", "3017620422003")
        True  # if product contains user's allergens
    """
    # Load user profile to get allergen preferences
    profile = PERSONALISATION_STORE.load_user_profile(user_id)
    if not profile:
        # No profile found, assume no restrictions
        return False

    # Get user's allergen list from profile
    user_allergens = profile.get("allergies", [])
    if not user_allergens:
        # No allergen preferences set
        return False

    # Normalize allergen names to match database column names
    # User allergens are typically like: ["Milk", "Peanuts", "Gluten"]
    # Database columns are like: contains_milk, contains_peanuts, contains_gluten
    allergen_columns: list[str] = []
    for allergen in user_allergens:
        # Convert labels like "Milk" → "contains_milk", handle a few synonyms.
        column_name = _to_allergen_column(str(allergen))
        if column_name:
            allergen_columns.append(column_name)

    if not allergen_columns:
        return False

    # Query the database to check if the food contains any of these allergens
    try:
        with duckdb.connect(DB_PATH.as_posix(), read_only=True) as conn:
            # Get all available allergen columns in the product_ingredients table
            available_columns_query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'main'
                  AND table_name = 'product_ingredients'
                  AND column_name LIKE 'contains_%'
            """
            available_columns = {
                row[0] for row in conn.execute(available_columns_query).fetchall()
            }

            # Filter to only check columns that exist in the database
            valid_columns = [
                col for col in allergen_columns if col in available_columns
            ]

            if not valid_columns:
                # None of the user's allergens have corresponding columns
                return False

            # Build a dynamic query to check if any allergen flag is TRUE
            # Example: SELECT (contains_milk OR contains_peanuts) AS has_allergen
            #          FROM product_ingredients WHERE code = ?
            conditions = " OR ".join(valid_columns)
            query = f"""
                SELECT ({conditions}) AS has_allergen
                FROM product_ingredients
                WHERE code = ?
                LIMIT 1
            """

            result = conn.execute(query, [food_code]).fetchone()

            if result is None:
                # Product not found in database
                return False

            # Return True if any allergen flag is set
            has_allergen = result[0]
            return bool(has_allergen) if has_allergen is not None else False

    except duckdb.Error as e:
        # Log error but don't raise - fail safe by returning False
        print(f"Database error while checking allergens: {e}")
        return False


def get_allergen_details(user_id: str, food_code: str) -> dict[str, Any]:
    """Get detailed allergen information for a food product.

    This function provides more detailed information than is_containing_allergens,
    including which specific allergens are present.

    Args:
        user_id: The user's unique identifier
        food_code: The product code to check for allergens

    Returns:
        A dictionary containing:
        - has_allergens (bool): Whether the food contains user's allergens
        - user_allergens (list[str]): Allergens the user wants to avoid
        - detected_allergens (list[str]): Which of the user's allergens are in the food
        - all_allergens (dict[str, bool]): All allergen flags for the product

    Example:
        >>> get_allergen_details("user_123", "3017620422003")
        {
            "has_allergens": True,
            "user_allergens": ["Milk", "Peanuts"],
            "detected_allergens": ["Milk"],
            "all_allergens": {
                "contains_milk": True,
                "contains_peanuts": False,
                "contains_wheat": True,
                ...
            }
        }
    """
    result: dict[str, Any] = {
        "has_allergens": False,
        "user_allergens": [],
        "detected_allergens": [],
        "all_allergens": {},
    }

    # Load user profile
    profile = PERSONALISATION_STORE.load_user_profile(user_id)
    if not profile:
        return result

    user_allergens = profile.get("allergies", [])
    result["user_allergens"] = user_allergens

    if not user_allergens:
        return result

    # Query database for all allergen information
    try:
        with duckdb.connect(DB_PATH.as_posix(), read_only=True) as conn:
            # Get all allergen columns
            columns_query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'main'
                  AND table_name = 'product_ingredients'
                  AND column_name LIKE 'contains_%'
                ORDER BY column_name
            """
            allergen_columns = [
                row[0] for row in conn.execute(columns_query).fetchall()
            ]

            if not allergen_columns:
                return result

            # Query for the product
            columns_sql = ", ".join(allergen_columns)
            query = f"""
                SELECT {columns_sql}
                FROM product_ingredients
                WHERE code = ?
                LIMIT 1
            """

            row = conn.execute(query, [food_code]).fetchone()
            if row is None:
                return result

            # Build allergen dictionary
            all_allergens = dict(zip(allergen_columns, row))
            result["all_allergens"] = all_allergens

            # Check which user allergens are detected
            detected = []
            for allergen_name in user_allergens:
                column_name = _to_allergen_column(str(allergen_name))
                if column_name in all_allergens and all_allergens[column_name]:
                    detected.append(allergen_name)

            result["detected_allergens"] = detected
            result["has_allergens"] = len(detected) > 0

    except duckdb.Error as e:
        print(f"Database error while getting allergen details: {e}")

    return result


__all__ = ["is_containing_allergens", "get_allergen_details"]
