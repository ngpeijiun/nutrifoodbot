from pathlib import Path

import duckdb


class IngestProductIngredientsJob:
    """
    Ingest ingredient features and compute allergen flags from ingredients text.

    Root cause fixed: previous export defaulted missing allergen columns to False.
    Here we derive booleans from ingredients_text and OR them with any existing flags.
    """

    def __init__(self):
        self.db_path = Path("db/open_food.duckdb")
        # Source feature set (may contain placeholder contains_* columns)
        self.features_path = Path("data_mining/data/ingredient_features.parquet")
        # Original raw products data with ingredients_text
        self.food_path = Path("eda/food.parquet")
        self.table_name = "product_ingredients"

    def run(self) -> None:
        if not self.features_path.exists():
            raise FileNotFoundError(f"File not found: {self.features_path}")
        if not self.food_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.food_path} (required for allergen detection)"
            )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with duckdb.connect(self.db_path.as_posix()) as conn:
            # Derive allergen booleans from ingredients_text; OR with existing flags if present
            # Note: Regex patterns include common derivatives/synonyms
            sql = f"""
            CREATE OR REPLACE TABLE {self.table_name} AS
            WITH src AS (
                SELECT 
                    CAST(f.code AS VARCHAR) AS code,
                    -- Normalize ingredients_text: handle LIST<STRUCT(lang, text)> to a single lowercased string
                    lower(
                        trim(
                            concat_ws(' ',
                                coalesce(array_to_string(list_transform(p.ingredients_text, x -> x."text"), ' '), ''),
                                -- include ingredients_tags (LIST<VARCHAR>) if available
                                coalesce(array_to_string(p.ingredients_tags, ' '), ''),
                                -- include allergens_tags (LIST<VARCHAR>) if available
                                coalesce(array_to_string(p.allergens_tags, ' '), ''),
                                -- include ingredients (LIST<VARCHAR> or simple) if available
                                coalesce(
                                    CASE WHEN typeof(p.ingredients) LIKE 'LIST%'
                                         THEN array_to_string(p.ingredients, ' ')
                                         ELSE CAST(p.ingredients AS VARCHAR)
                                    END,
                                    ''
                                )
                            )
                        )
                    ) AS ingredients_text,
                    -- existing flags (may be missing); coalesce to FALSE for OR logic
                    coalesce(f.contains_milk, FALSE)        AS contains_milk_src,
                    coalesce(f.contains_eggs, FALSE)        AS contains_eggs_src,
                    coalesce(f.contains_peanuts, FALSE)     AS contains_peanuts_src,
                    coalesce(f.contains_tree_nuts, FALSE)   AS contains_tree_nuts_src,
                    coalesce(f.contains_fish, FALSE)        AS contains_fish_src,
                    coalesce(f.contains_shellfish, FALSE)   AS contains_shellfish_src,
                    coalesce(f.contains_soy, FALSE)         AS contains_soy_src,
                    coalesce(f.contains_wheat, FALSE)       AS contains_wheat_src,
                    coalesce(f.contains_gluten, FALSE)      AS contains_gluten_src,
                    coalesce(f.contains_sesame, FALSE)      AS contains_sesame_src,
                    coalesce(f.contains_sulfites, FALSE)    AS contains_sulfites_src,
                    coalesce(f.contains_celery, FALSE)      AS contains_celery_src,
                    coalesce(f.contains_mustard, FALSE)     AS contains_mustard_src,
                    coalesce(f.contains_lupin, FALSE)       AS contains_lupin_src,
                    coalesce(f.contains_molluscs, FALSE)    AS contains_molluscs_src
                                                FROM read_parquet('{self.features_path.as_posix()}') f
                                                LEFT JOIN read_parquet('{self.food_path.as_posix()}') p
                                                    ON TRIM(CAST(f.code AS VARCHAR)) = TRIM(CAST(p.code AS VARCHAR))
            ),
            detected AS (
                SELECT
                    code,
                    ingredients_text,
                    (length(ingredients_text) > 0) AS has_ingredients_text,
                    -- Dairy / Milk (substring checks on lowercased text)
                    (
                        contains_milk_src OR (
                            position('milk' in ingredients_text) > 0 OR
                            position('dairy' in ingredients_text) > 0 OR
                            position('lactose' in ingredients_text) > 0 OR
                            position('casein' in ingredients_text) > 0 OR
                            position('whey' in ingredients_text) > 0 OR
                            position('butter' in ingredients_text) > 0 OR
                            position('cheese' in ingredients_text) > 0 OR
                            position('cream' in ingredients_text) > 0 OR
                            position('yogurt' in ingredients_text) > 0 OR
                            position('yoghurt' in ingredients_text) > 0 OR
                            position('lait' in ingredients_text) > 0 OR
                            position('leche' in ingredients_text) > 0 OR
                            position('milch' in ingredients_text) > 0 OR
                            position('latte' in ingredients_text) > 0 OR
                            position('leite' in ingredients_text) > 0
                        )
                    ) AS contains_milk,
                    -- Eggs (singular/plural & derivatives)
                    (
                        contains_eggs_src OR (
                            position('egg' in ingredients_text) > 0 OR
                            position('eggs' in ingredients_text) > 0 OR
                            position('albumin' in ingredients_text) > 0 OR
                            position('lecithin' in ingredients_text) > 0 OR
                            position('mayonnaise' in ingredients_text) > 0 OR
                            position('oeuf' in ingredients_text) > 0 OR
                            position('oeufs' in ingredients_text) > 0 OR
                            position('huevo' in ingredients_text) > 0 OR
                            position('huevos' in ingredients_text) > 0 OR
                            position('ei' in ingredients_text) > 0 OR
                            position('eier' in ingredients_text) > 0 OR
                            position('uovo' in ingredients_text) > 0 OR
                            position('uova' in ingredients_text) > 0 OR
                            position('ovo' in ingredients_text) > 0 OR
                            position('ovos' in ingredients_text) > 0 OR
                            position('ägg' in ingredients_text) > 0
                        )
                    ) AS contains_eggs,
                    -- Peanuts
                    (
                        contains_peanuts_src OR (
                            position('peanut' in ingredients_text) > 0 OR
                            position('peanuts' in ingredients_text) > 0 OR
                            position('groundnut' in ingredients_text) > 0 OR
                            position('arachis' in ingredients_text) > 0
                        )
                    ) AS contains_peanuts,
                    -- Tree nuts (common types)
                    (
                        contains_tree_nuts_src OR (
                            position('almond' in ingredients_text) > 0 OR
                            position('walnut' in ingredients_text) > 0 OR
                            position('hazelnut' in ingredients_text) > 0 OR
                            position('cashew' in ingredients_text) > 0 OR
                            position('pistachio' in ingredients_text) > 0 OR
                            position('pecan' in ingredients_text) > 0 OR
                            position('brazil nut' in ingredients_text) > 0 OR
                            position('macadamia' in ingredients_text) > 0 OR
                            position('pine nut' in ingredients_text) > 0
                        )
                    ) AS contains_tree_nuts,
                    -- Fish
                    (
                        contains_fish_src OR (
                            position('fish' in ingredients_text) > 0 OR
                            position('salmon' in ingredients_text) > 0 OR
                            position('tuna' in ingredients_text) > 0 OR
                            position('cod' in ingredients_text) > 0 OR
                            position('trout' in ingredients_text) > 0 OR
                            position('sardine' in ingredients_text) > 0 OR
                            position('anchovy' in ingredients_text) > 0 OR
                            position('mackerel' in ingredients_text) > 0 OR
                            position('haddock' in ingredients_text) > 0
                        )
                    ) AS contains_fish,
                    -- Shellfish / Crustacea
                    (
                        contains_shellfish_src OR (
                            position('shrimp' in ingredients_text) > 0 OR
                            position('prawn' in ingredients_text) > 0 OR
                            position('crab' in ingredients_text) > 0 OR
                            position('lobster' in ingredients_text) > 0 OR
                            position('crayfish' in ingredients_text) > 0 OR
                            position('shellfish' in ingredients_text) > 0
                        )
                    ) AS contains_shellfish,
                    -- Soy
                    (
                        contains_soy_src OR (
                            position('soy' in ingredients_text) > 0 OR
                            position('soya' in ingredients_text) > 0 OR
                            position('tofu' in ingredients_text) > 0 OR
                            position('tempeh' in ingredients_text) > 0 OR
                            position('soybean' in ingredients_text) > 0
                        )
                    ) AS contains_soy,
                    -- Wheat (explicit)
                    (
                        contains_wheat_src OR (
                            position('wheat' in ingredients_text) > 0 OR
                            position('wholewheat' in ingredients_text) > 0 OR
                            position('farina' in ingredients_text) > 0 OR
                            position('semolina' in ingredients_text) > 0
                        )
                    ) AS contains_wheat,
                    -- Gluten (grains containing gluten)
                    (
                        contains_gluten_src OR (
                            position('gluten' in ingredients_text) > 0 OR
                            position('wheat' in ingredients_text) > 0 OR
                            position('barley' in ingredients_text) > 0 OR
                            position('rye' in ingredients_text) > 0 OR
                            position('malt' in ingredients_text) > 0 OR
                            position('spelt' in ingredients_text) > 0 OR
                            position('kamut' in ingredients_text) > 0
                        )
                    ) AS contains_gluten,
                    -- Sesame
                    (
                        contains_sesame_src OR (
                            position('sesame' in ingredients_text) > 0 OR
                            position('tahini' in ingredients_text) > 0
                        )
                    ) AS contains_sesame,
                    -- Sulfites (approximate: match sulfite keywords or E220-E229)
                    (
                        contains_sulfites_src OR (
                            position('sulfite' in ingredients_text) > 0 OR
                            position('sulphite' in ingredients_text) > 0 OR
                            position('e220' in ingredients_text) > 0 OR
                            position('e221' in ingredients_text) > 0 OR
                            position('e222' in ingredients_text) > 0 OR
                            position('e223' in ingredients_text) > 0 OR
                            position('e224' in ingredients_text) > 0 OR
                            position('e225' in ingredients_text) > 0 OR
                            position('e226' in ingredients_text) > 0 OR
                            position('e227' in ingredients_text) > 0 OR
                            position('e228' in ingredients_text) > 0 OR
                            position('e229' in ingredients_text) > 0
                        )
                    ) AS contains_sulfites,
                    -- Celery
                    (
                        contains_celery_src OR (
                            position('celery' in ingredients_text) > 0 OR
                            position('celeriac' in ingredients_text) > 0
                        )
                    ) AS contains_celery,
                    -- Mustard
                    (
                        contains_mustard_src OR (
                            position('mustard' in ingredients_text) > 0 OR
                            position('dijon' in ingredients_text) > 0
                        )
                    ) AS contains_mustard,
                    -- Lupin
                    (
                        contains_lupin_src OR (
                            position('lupin' in ingredients_text) > 0 OR
                            position('lupine' in ingredients_text) > 0
                        )
                    ) AS contains_lupin,
                    -- Molluscs
                    (
                        contains_molluscs_src OR (
                            position('mussel' in ingredients_text) > 0 OR
                            position('clam' in ingredients_text) > 0 OR
                            position('oyster' in ingredients_text) > 0 OR
                            position('scallop' in ingredients_text) > 0 OR
                            position('octopus' in ingredients_text) > 0 OR
                            position('squid' in ingredients_text) > 0 OR
                            position('cuttlefish' in ingredients_text) > 0 OR
                            position('snail' in ingredients_text) > 0 OR
                            position('mollusc' in ingredients_text) > 0 OR
                            position('mollusk' in ingredients_text) > 0
                        )
                    ) AS contains_molluscs
                FROM src
            )
            SELECT * FROM detected;
            """
            conn.execute(sql)


if __name__ == "__main__":
    IngestProductIngredientsJob().run()
