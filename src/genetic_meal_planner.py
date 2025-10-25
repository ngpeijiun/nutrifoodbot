"""Genetic Algorithm based Meal Planner.

This module provides:
- RealFoodDatabase: loads real food nutrient profiles from DuckDB (with robust fallbacks)
- GAConstraints: tunable constraints and GA hyper-parameters
- GeneticMealPlanner: end-to-end GA optimizer that produces a multi-day meal plan

Data requirements (from DuckDB table `product_nutrient_profiles`):
- code (VARCHAR): product code identifier
- energy_density (DOUBLE): kcal per 100g
- protein_ratio, fat_ratio, carb_ratio (DOUBLE): share of calories by macro (sum ~= 1)
- fiber_g (DOUBLE): grams fiber per 100g
- sugar_to_carb_ratio (DOUBLE): fraction of carbs that are sugars (0..1)
- salt_g (DOUBLE): grams salt per 100g

If any field is missing or invalid, the item is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import math

import duckdb
from allergen_detector import is_containing_allergens  


DB_PATH = "db/product_nutrients.duckdb"
TABLE_NAME = "product_nutrient_profiles"


# -----------------------------
# Data models
# -----------------------------


@dataclass
class Food:
    food_code: str
    name: str
    calories_per_100g: float
    protein_g_per_100g: float
    carbs_g_per_100g: float
    fat_g_per_100g: float
    fiber_g_per_100g: float
    sugar_g_per_100g: float
    salt_g_per_100g: float
    macro_profile: Optional[str] = None
    saturated_fat_g_per_100g: Optional[float] = None  # newly loaded if source column exists
    nutriscore_score_actual: Optional[int] = None
    nutriscore_grade_actual: Optional[str] = None


@dataclass
class MealItem:
    food: Food
    portion_grams: float

    def get_nutrition(self) -> Dict[str, float]:
        factor = max(self.portion_grams, 0.0) / 100.0
        return {
            "calories": self.food.calories_per_100g * factor,
            "protein": self.food.protein_g_per_100g * factor,
            "carbs": self.food.carbs_g_per_100g * factor,
            "fat": self.food.fat_g_per_100g * factor,
            "fiber": self.food.fiber_g_per_100g * factor,
            "sugar": self.food.sugar_g_per_100g * factor,
            "salt": self.food.salt_g_per_100g * factor,
            "cost": 0.0,  # cost not available from DB; keep 0.0 to retain API shape
        }


@dataclass
class Meal:
    items: List[MealItem] = field(default_factory=list)

    def get_total_nutrition(self) -> Dict[str, float]:
        totals = {k: 0.0 for k in [
            "calories", "protein", "carbs", "fat", "fiber", "sugar", "salt", "cost"
        ]}
        for it in self.items:
            n = it.get_nutrition()
            for k in totals:
                totals[k] += n.get(k, 0.0)
        return totals


@dataclass
class Day:
    meals: Dict[str, Meal]  # keys: "breakfast", "lunch", "dinner"

    def get_daily_totals(self) -> Dict[str, float]:
        totals = {k: 0.0 for k in [
            "calories", "protein", "carbs", "fat", "fiber", "sugar", "salt", "cost"
        ]}
        for meal in self.meals.values():
            n = meal.get_total_nutrition()
            for k in totals:
                totals[k] += n.get(k, 0.0)
        return totals


@dataclass
class MealPlan:
    days: List[Day]
    fitness: float
    # Optional Nutri-Score for the overall meal plan (approximated to a single composite product)
    nutri_score: Optional[int] = None
    nutri_score_letter: Optional[str] = None
    nutri_score_source: str = "heuristic"  # 'actual' if derived from actual food scores

    def get_average_daily_nutrition(self) -> Dict[str, float]:
        if not self.days:
            return {k: 0.0 for k in [
                "calories", "protein", "carbs", "fat", "fiber", "sugar", "salt", "cost"
            ]}
        totals = {k: 0.0 for k in [
            "calories", "protein", "carbs", "fat", "fiber", "sugar", "salt", "cost"
        ]}
        for d in self.days:
            dn = d.get_daily_totals()
            for k in totals:
                totals[k] += dn[k]
        return {k: totals[k] / len(self.days) for k in totals}


# -----------------------------
# Food database loader
# -----------------------------


class RealFoodDatabase:
    """Loads food profiles from DuckDB and provides simple accessors.

    If DuckDB is unavailable or table missing, the database will be empty.
    """

    def __init__(self, db_path: str = DB_PATH, table_name: str = TABLE_NAME, user_id: Optional[str] = None):
        self.db_path = db_path
        self.table_name = table_name
        self.user_id = user_id
        self.foods_cache: Dict[str, Food] = {}
        self.foods_by_category: Dict[str, List[Food]] = {
            "breakfast": [],
            "lunch": [],
            "dinner": [],
        }
        self._load_foods()
        # Apply optional allergen filtering based on user's profile
        if self.user_id:
            self._apply_allergen_filter()

    def _apply_allergen_filter(self) -> None:
        """Remove foods that contain user's allergens from caches.

        Uses the rule-based detector; if user has no allergen preferences,
        the detector returns False and no filtering is applied.
        """
        try:
            # Determine safe codes
            safe_codes: Dict[str, Food] = {}
            for code, food in self.foods_cache.items():
                try:
                    unsafe = is_containing_allergens(self.user_id or "", code)
                except Exception:
                    # Fail-open (treat as safe) to avoid emptying DB due to transient errors
                    unsafe = False
                if not unsafe:
                    safe_codes[code] = food

            # Replace foods_cache with safe subset
            self.foods_cache = safe_codes

            # Rebuild category pools with safe foods only
            for cat, foods in list(self.foods_by_category.items()):
                self.foods_by_category[cat] = [f for f in foods if f.food_code in self.foods_cache]

            # If any category becomes empty but we still have safe foods, backfill with all safe foods
            if self.foods_cache:
                pool_all = list(self.foods_cache.values())
                for cat, foods in self.foods_by_category.items():
                    if not foods:
                        self.foods_by_category[cat] = pool_all.copy()
        except Exception:
            # Best-effort filter; keep original pools on any unexpected error
            pass

    def _load_foods(self, limit_per_profile: int = 400) -> None:
        try:
            with duckdb.connect(self.db_path, read_only=True) as con:
                # Verify table exists
                row = con.execute(
                    """
                    SELECT COUNT(*)>0 FROM information_schema.tables
                    WHERE table_name = ?
                    """,
                    [self.table_name],
                ).fetchone()
                exists = bool(row[0]) if row else False
                if not exists:
                    return

                # Pull a diverse sample by macro_profile when present
                # Fallback to a simple random sample
                # Detect optional saturated fat column
                try:
                    has_sat = bool(
                        con.execute(
                            """
                            SELECT COUNT(*)>0 FROM information_schema.columns
                            WHERE table_name = ? AND column_name = 'saturated_fat_g'
                            """,
                            [self.table_name],
                        ).fetchone()[0]
                    )
                except Exception:
                    has_sat = False

                cols = [
                    "code", "macro_profile", "energy_density", "protein_ratio", "fat_ratio", "carb_ratio",
                    "fiber_g", "sugar_to_carb_ratio", "salt_g"
                ]
                if has_sat:
                    cols.append("saturated_fat_g")
                # Optional actual Nutri-Score columns (score/grade) if present
                try:
                    has_ns_score = bool(
                        con.execute(
                            """
                            SELECT COUNT(*)>0 FROM information_schema.columns
                            WHERE table_name = ? AND column_name = 'nutriscore_score'
                            """,
                            [self.table_name],
                        ).fetchone()[0]
                    )
                except Exception:
                    has_ns_score = False
                try:
                    has_ns_grade = bool(
                        con.execute(
                            """
                            SELECT COUNT(*)>0 FROM information_schema.columns
                            WHERE table_name = ? AND column_name = 'nutriscore_grade'
                            """,
                            [self.table_name],
                        ).fetchone()[0]
                    )
                except Exception:
                    has_ns_grade = False
                if has_ns_score:
                    cols.append("nutriscore_score")
                if has_ns_grade:
                    cols.append("nutriscore_grade")
                cols_sql = ", ".join(cols)
                try:
                    df = con.execute(
                        f"""
                        WITH base AS (
                            SELECT {cols_sql}
                            FROM {self.table_name}
                            WHERE energy_density IS NOT NULL AND energy_density > 0
                        )
                        SELECT * FROM base
                        USING SAMPLE 2000 ROWS
                        """
                    ).df()
                except Exception:
                    df = con.execute(
                        f"SELECT {cols_sql} FROM {self.table_name} LIMIT 2000"
                    ).df()

                for _, row in df.iterrows():
                    try:
                        code = str(row.get("code"))
                        if not code or code == "nan":
                            continue
                        ed = float(row.get("energy_density", 0) or 0)
                        pr = float(row.get("protein_ratio", 0) or 0)
                        fr = float(row.get("fat_ratio", 0) or 0)
                        cr = float(row.get("carb_ratio", 0) or 0)
                        fiber = float(row.get("fiber_g", 0) or 0)
                        sugar_to_carb = float(row.get("sugar_to_carb_ratio", 0) or 0)
                        salt = float(row.get("salt_g", 0) or 0)
                        sat_fat = None
                        if "saturated_fat_g" in row:
                            try:
                                sat_fat_val = row.get("saturated_fat_g")
                                if sat_fat_val is not None and not (isinstance(sat_fat_val, float) and math.isnan(sat_fat_val)):
                                    sat_fat = float(sat_fat_val)
                            except Exception:
                                sat_fat = None

                        if ed <= 0 or (pr + fr + cr) <= 0:
                            continue

                        # Convert macro calorie ratios to grams per 100g
                        protein_g = (ed * pr) / 4.0
                        carbs_g = (ed * cr) / 4.0
                        fat_g = (ed * fr) / 9.0
                        sugar_g = max(0.0, carbs_g * max(0.0, min(1.0, sugar_to_carb)))

                        food = Food(
                            food_code=code,
                            name=f"Product {code}",
                            calories_per_100g=ed,
                            protein_g_per_100g=protein_g,
                            carbs_g_per_100g=carbs_g,
                            fat_g_per_100g=fat_g,
                            fiber_g_per_100g=fiber,
                            sugar_g_per_100g=sugar_g,
                            salt_g_per_100g=salt,
                            macro_profile=(row.get("macro_profile") or None),
                            saturated_fat_g_per_100g=sat_fat,
                            nutriscore_score_actual=(int(row.get("nutriscore_score")) if "nutriscore_score" in row and str(row.get("nutriscore_score")) not in ("", "nan", "None") else None),
                            nutriscore_grade_actual=(str(row.get("nutriscore_grade")).upper() if "nutriscore_grade" in row and str(row.get("nutriscore_grade")) not in ("", "nan", "None") else None),
                        )
                        self.foods_cache[code] = food
                    except Exception:
                        # Skip malformed rows
                        continue

                # Naive categorization by macro profile / energy density
                for food in self.foods_cache.values():
                    profile = (food.macro_profile or "").lower()
                    if "high_protein" in profile:
                        self.foods_by_category["lunch"].append(food)
                        self.foods_by_category["dinner"].append(food)
                    elif "high_carb" in profile or food.calories_per_100g < 200:
                        self.foods_by_category["breakfast"].append(food)
                    elif "high_fat" in profile:
                        self.foods_by_category["dinner"].append(food)
                    else:
                        # fallback: add to all
                        for k in self.foods_by_category:
                            self.foods_by_category[k].append(food)
        except Exception:
            # If DB not available, leave caches empty
            return

    def get_food_by_code(self, code: str) -> Optional[Food]:
        """Return a Food by product code if present in cache.

        If not present yet, try to fetch a single row from DuckDB and add it
        to cache. Returns None if unavailable.
        """
        if code in self.foods_cache:
            return self.foods_cache[code]
        try:
            with duckdb.connect(self.db_path, read_only=True) as con:
                # Attempt to also fetch saturated_fat_g if present
                try:
                    has_sat = bool(
                        con.execute(
                            """
                            SELECT COUNT(*)>0 FROM information_schema.columns
                            WHERE table_name = ? AND column_name = 'saturated_fat_g'
                            """,
                            [self.table_name],
                        ).fetchone()[0]
                    )
                except Exception:
                    has_sat = False
                # Detect nutriscore columns
                def _col_exists(col: str) -> bool:
                    try:
                        return bool(
                            con.execute(
                                "SELECT COUNT(*)>0 FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
                                [self.table_name, col],
                            ).fetchone()[0]
                        )
                    except Exception:
                        return False
                has_ns_score = _col_exists('nutriscore_score')
                has_ns_grade = _col_exists('nutriscore_grade')
                select_cols = [
                    "code", "macro_profile", "energy_density", "protein_ratio", "fat_ratio", "carb_ratio",
                    "fiber_g", "sugar_to_carb_ratio", "salt_g"
                ]
                if has_sat:
                    select_cols.append("saturated_fat_g")
                if has_ns_score:
                    select_cols.append("nutriscore_score")
                if has_ns_grade:
                    select_cols.append("nutriscore_grade")
                select_cols_sql = ", ".join(select_cols)
                row = con.execute(
                    f"SELECT {select_cols_sql} FROM {self.table_name} WHERE code = ? LIMIT 1",
                    [code],
                ).fetchone()
                if not row:
                    return None
                # Unpack dynamically
                idx = 0
                code_v = row[idx]; idx += 1
                macro_profile = row[idx]; idx += 1
                energy_density = row[idx]; idx += 1
                protein_ratio = row[idx]; idx += 1
                fat_ratio = row[idx]; idx += 1
                carb_ratio = row[idx]; idx += 1
                fiber_g = row[idx]; idx += 1
                sugar_to_carb_ratio = row[idx]; idx += 1
                salt_g = row[idx]; idx += 1
                saturated_fat_g = None
                if has_sat:
                    saturated_fat_g = row[idx]; idx += 1
                ns_score_val = None
                ns_grade_val = None
                if has_ns_score:
                    ns_score_val = row[idx]; idx += 1
                if has_ns_grade:
                    ns_grade_val = row[idx]; idx += 1
                ed = float(energy_density or 0)
                if ed <= 0:
                    return None
                pr = float(protein_ratio or 0)
                fr = float(fat_ratio or 0)
                cr = float(carb_ratio or 0)
                protein_g = (ed * pr) / 4.0
                carbs_g = (ed * cr) / 4.0
                fat_g = (ed * fr) / 9.0
                sugar_g = max(0.0, carbs_g * max(0.0, min(1.0, float(sugar_to_carb_ratio or 0))))
                food = Food(
                    food_code=str(code_v),
                    name=f"Product {code_v}",
                    calories_per_100g=ed,
                    protein_g_per_100g=protein_g,
                    carbs_g_per_100g=carbs_g,
                    fat_g_per_100g=fat_g,
                    fiber_g_per_100g=float(fiber_g or 0),
                    sugar_g_per_100g=sugar_g,
                    salt_g_per_100g=float(salt_g or 0),
                    macro_profile=(macro_profile or None),
                    saturated_fat_g_per_100g=(float(saturated_fat_g) if has_sat and saturated_fat_g is not None else None),
                    nutriscore_score_actual=(int(ns_score_val) if (has_ns_score and ns_score_val is not None and str(ns_score_val) not in ("", "nan", "None")) else None),
                    nutriscore_grade_actual=(str(ns_grade_val).upper() if (has_ns_grade and ns_grade_val is not None and str(ns_grade_val) not in ("", "nan", "None")) else None),
                )
                self.foods_cache[str(code_v)] = food
                # also add to generic categories as fallback
                for k in self.foods_by_category:
                    self.foods_by_category[k].append(food)
                return food
        except Exception:
            return None

    def get_random_food(self, category: Optional[str] = None) -> Food:
        pool: List[Food]
        if category and category in self.foods_by_category and self.foods_by_category[category]:
            pool = self.foods_by_category[category]
        else:
            pool = list(self.foods_cache.values())
        if not pool:
            # Fallback to a simple mock food to avoid crashes
            return Food(
                food_code="00000000",
                name="Mock Food",
                calories_per_100g=200.0,
                protein_g_per_100g=10.0,
                carbs_g_per_100g=20.0,
                fat_g_per_100g=5.0,
                fiber_g_per_100g=3.0,
                sugar_g_per_100g=5.0,
                salt_g_per_100g=0.5,
            )
        return random.choice(pool)


# -----------------------------
# Genetic Algorithm
# -----------------------------


@dataclass
class GAConstraints:
    daily_calories: int = 2000
    num_days: int = 3
    meal_types: Tuple[str, str, str] = ("breakfast", "lunch", "dinner")
    calorie_distribution: Dict[str, float] = field(default_factory=lambda: {
        "breakfast": 0.25,
        "lunch": 0.35,
        "dinner": 0.40,
    })
    min_protein: float = 50.0
    min_fiber: float = 25.0
    max_sugar: float = 50.0
    max_fat: float = 70.0

    population_size: int = 20
    generations: int = 30
    mutation_rate: float = 0.2
    tournament_size: int = 3
    # Early stopping: stop if best fitness hasn't improved by at least
    # `min_improvement` over `patience` consecutive generations.
    patience: int = 0  # 0 disables early stopping
    min_improvement: float = 1e-4


class GeneticMealPlanner:
    def __init__(
        self,
        constraints: GAConstraints,
        food_db: Optional[RealFoodDatabase] = None,
        seed_food_codes: Optional[Dict[str, List[str]]] = None,
        user_id: Optional[str] = None,
    ):
        self.constraints = constraints
        self.food_db = food_db or RealFoodDatabase(user_id=user_id)
        # Pre-resolve seed foods by meal type
        self.seed_foods: Dict[str, List[Food]] = {}
        if seed_food_codes:
            for mt, codes in seed_food_codes.items():
                foods: List[Food] = []
                for code in codes or []:
                    f = self.food_db.get_food_by_code(str(code))
                    if f:
                        foods.append(f)
                if foods:
                    self.seed_foods[mt] = foods

    # ---- GA plumbing ----
    def plan_meals(self) -> MealPlan:
        pop = [self._random_plan() for _ in range(self.constraints.population_size)]
        fitnesses = [self._fitness(p) for p in pop]

        best_so_far = max(fitnesses) if fitnesses else 0.0
        stagnant = 0
        for gen in range(self.constraints.generations):
            new_pop: List[MealPlan] = []
            # Elitism: carry the best
            best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
            # Deep copy elite to avoid downstream mutation altering it
            elite_days = [
                Day(meals={mt: Meal(items=[MealItem(food=it.food, portion_grams=it.portion_grams) for it in meal.items]) for mt, meal in d.meals.items()})
                for d in pop[best_idx].days
            ]
            new_pop.append(MealPlan(days=elite_days, fitness=fitnesses[best_idx]))

            while len(new_pop) < self.constraints.population_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1_days, c2_days = self._crossover(p1.days, p2.days)
                c1 = MealPlan(days=self._mutate(c1_days), fitness=0.0)
                c2 = MealPlan(days=self._mutate(c2_days), fitness=0.0)
                new_pop.extend([c1, c2])
            pop = new_pop[: self.constraints.population_size]
            fitnesses = [self._fitness(p) for p in pop]
            current_best = max(fitnesses)
            if current_best > best_so_far + self.constraints.min_improvement:
                best_so_far = current_best
                stagnant = 0
            else:
                stagnant += 1
            if self.constraints.patience > 0 and stagnant >= self.constraints.patience:
                break

        best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
        best = pop[best_idx]
        result = MealPlan(days=best.days, fitness=fitnesses[best_idx])
        # Compute Nutri-Score approximation
        ns_score, ns_letter, ns_source = self._compute_plan_nutri_score(result)
        result.nutri_score = ns_score
        result.nutri_score_letter = ns_letter
        result.nutri_score_source = ns_source
        return result

    def plan_meals_with_history(self) -> Tuple[MealPlan, List[float]]:
        """Run GA and also return best fitness per generation.

        Returns a tuple of (best_plan, history) where history[i] is the best
        fitness after generation i (0-indexed). This does not change the
        behavior of ``plan_meals`` and is safe for callers who do not need
        plotting/analytics.
        """
        pop = [self._random_plan() for _ in range(self.constraints.population_size)]
        fitnesses = [self._fitness(p) for p in pop]

        history: List[float] = []
        # record initial best
        history.append(max(fitnesses) if fitnesses else 0.0)

        best_so_far = max(fitnesses) if fitnesses else 0.0
        stagnant = 0
        for gen in range(self.constraints.generations):
            new_pop: List[MealPlan] = []
            # Elitism: carry the best
            best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
            elite_days = [
                Day(meals={mt: Meal(items=[MealItem(food=it.food, portion_grams=it.portion_grams) for it in meal.items]) for mt, meal in d.meals.items()})
                for d in pop[best_idx].days
            ]
            new_pop.append(MealPlan(days=elite_days, fitness=fitnesses[best_idx]))

            while len(new_pop) < self.constraints.population_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1_days, c2_days = self._crossover(p1.days, p2.days)
                c1 = MealPlan(days=self._mutate(c1_days), fitness=0.0)
                c2 = MealPlan(days=self._mutate(c2_days), fitness=0.0)
                new_pop.extend([c1, c2])
            pop = new_pop[: self.constraints.population_size]
            fitnesses = [self._fitness(p) for p in pop]
            history.append(max(fitnesses) if fitnesses else 0.0)
            current_best = history[-1]
            if current_best > best_so_far + self.constraints.min_improvement:
                best_so_far = current_best
                stagnant = 0
            else:
                stagnant += 1
            if self.constraints.patience > 0 and stagnant >= self.constraints.patience:
                break

        best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
        best = pop[best_idx]
        result = MealPlan(days=best.days, fitness=fitnesses[best_idx])
        ns_score, ns_letter, ns_source = self._compute_plan_nutri_score(result)
        result.nutri_score = ns_score
        result.nutri_score_letter = ns_letter
        result.nutri_score_source = ns_source
        return result, history

    def _random_plan(self) -> MealPlan:
        days: List[Day] = []
        for _ in range(self.constraints.num_days):
            meals: Dict[str, Meal] = {}
            for mt in self.constraints.meal_types:
                meals[mt] = self._random_meal(mt)
            days.append(Day(meals=meals))
        return MealPlan(days=days, fitness=0.0)

    def _random_meal(self, meal_type: str) -> Meal:
        # Choose 1-3 items; assign portions to roughly hit calorie target
        target_cals = self.constraints.daily_calories * self.constraints.calorie_distribution.get(meal_type, 1/len(self.constraints.meal_types))
        num_items = random.choice([1, 2, 3])
        items: List[MealItem] = []
        # Random proportion split for calories per item
        weights = [random.random() for _ in range(num_items)]
        total_w = sum(weights) or 1.0
        # Prefer seed foods for at least one item if available
        seed_pool = self.seed_foods.get(meal_type, [])
        seed_used = False
        for i, w in enumerate(weights):
            f: Food
            if seed_pool and (not seed_used or random.random() < 0.5):
                f = random.choice(seed_pool)
                seed_used = True
            else:
                f = self.food_db.get_random_food(meal_type)
            # grams ~= (target_cals * proportion) / (kcal per 100g) * 100
            portion_g = max(30.0, (target_cals * (w / total_w)) / max(1.0, f.calories_per_100g) * 100.0)
            portion_g = min(portion_g, 600.0)  # cap to reasonable single-portion
            items.append(MealItem(food=f, portion_grams=portion_g))
        return Meal(items=items)

    def _tournament_select(self, pop: List[MealPlan], fitnesses: List[float]) -> MealPlan:
        k = min(self.constraints.tournament_size, len(pop))
        idxs = random.sample(range(len(pop)), k)
        best = max(idxs, key=lambda i: fitnesses[i])
        return pop[best]

    def _crossover(self, d1: List[Day], d2: List[Day]) -> Tuple[List[Day], List[Day]]:
        if len(d1) <= 1:
            return d1, d2
        point = random.randint(1, len(d1) - 1)
        c1 = d1[:point] + d2[point:]
        c2 = d2[:point] + d1[point:]
        return c1, c2

    def _mutate(self, days: List[Day]) -> List[Day]:
        if random.random() > self.constraints.mutation_rate:
            # Return a deep copy to keep immutability assumptions
            return [
                Day(meals={mt: Meal(items=[MealItem(food=it.food, portion_grams=it.portion_grams) for it in meal.items]) for mt, meal in d.meals.items()})
                for d in days
            ]
        mutated: List[Day] = []
        for d in days:
            new_meals: Dict[str, Meal] = {}
            for mt, meal in d.meals.items():
                new_items: List[MealItem] = []
                for idx, old_item in enumerate(meal.items):
                    item = MealItem(food=old_item.food, portion_grams=old_item.portion_grams)
                    # 20% chance to replace this item
                    if random.random() < 0.2:
                        item.food = self.food_db.get_random_food(mt)
                    # 20% chance to tweak portion
                    if random.random() < 0.2:
                        delta = random.uniform(-0.15, 0.15)
                        item.portion_grams = min(600.0, max(30.0, item.portion_grams * (1.0 + delta)))
                    new_items.append(item)
                new_meals[mt] = Meal(items=new_items)
            mutated.append(Day(meals=new_meals))
        return mutated

    def plan_meals_with_stats(self) -> Tuple[MealPlan, List[float], List[float]]:
        """Extended run returning (best_plan, best_history, mean_history)."""
        pop = [self._random_plan() for _ in range(self.constraints.population_size)]
        fitnesses = [self._fitness(p) for p in pop]
        best_hist: List[float] = [max(fitnesses) if fitnesses else 0.0]
        mean_hist: List[float] = [sum(fitnesses)/len(fitnesses) if fitnesses else 0.0]
        best_so_far = max(fitnesses) if fitnesses else 0.0
        stagnant = 0
        for gen in range(self.constraints.generations):
            new_pop: List[MealPlan] = []
            best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
            elite_days = [
                Day(meals={mt: Meal(items=[MealItem(food=it.food, portion_grams=it.portion_grams) for it in meal.items]) for mt, meal in d.meals.items()})
                for d in pop[best_idx].days
            ]
            new_pop.append(MealPlan(days=elite_days, fitness=fitnesses[best_idx]))
            while len(new_pop) < self.constraints.population_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1_days, c2_days = self._crossover(p1.days, p2.days)
                c1 = MealPlan(days=self._mutate(c1_days), fitness=0.0)
                c2 = MealPlan(days=self._mutate(c2_days), fitness=0.0)
                new_pop.extend([c1, c2])
            pop = new_pop[: self.constraints.population_size]
            fitnesses = [self._fitness(p) for p in pop]
            best_hist.append(max(fitnesses) if fitnesses else 0.0)
            mean_hist.append(sum(fitnesses)/len(fitnesses) if fitnesses else 0.0)
            current_best = best_hist[-1]
            if current_best > best_so_far + self.constraints.min_improvement:
                best_so_far = current_best
                stagnant = 0
            else:
                stagnant += 1
            if self.constraints.patience > 0 and stagnant >= self.constraints.patience:
                break
        best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
        best = pop[best_idx]
        result = MealPlan(days=best.days, fitness=fitnesses[best_idx])
        ns_score, ns_letter, ns_source = self._compute_plan_nutri_score(result)
        result.nutri_score = ns_score
        result.nutri_score_letter = ns_letter
        result.nutri_score_source = ns_source
        return result, best_hist, mean_hist

    # ---- Fitness ----
    def _fitness(self, plan: MealPlan) -> float:
        if not plan.days:
            return 0.0
        c = self.constraints
        penalties = []

        # Average daily totals over plan horizon
        avg = plan.get_average_daily_nutrition()

        # Calorie proximity (as relative error)
        cal_rel_err = abs(avg["calories"] - c.daily_calories) / max(1.0, c.daily_calories)
        penalties.append(4.0 * cal_rel_err)

        # Protein and fiber shortfall
        if c.min_protein > 0:
            penalties.append(max(0.0, (c.min_protein - avg["protein"]) / c.min_protein))
        if c.min_fiber > 0:
            penalties.append(max(0.0, (c.min_fiber - avg["fiber"]) / c.min_fiber))

        # Sugar and fat overage
        if c.max_sugar > 0:
            penalties.append(max(0.0, (avg["sugar"] - c.max_sugar) / c.max_sugar))
        if c.max_fat > 0:
            penalties.append(max(0.0, (avg["fat"] - c.max_fat) / c.max_fat))

        # Variety bonus: proportion of unique foods
        items = [it.food.food_code for d in plan.days for m in d.meals.values() for it in m.items]
        unique = len(set(items))
        total_items = max(1, len(items))
        variety_bonus = 0.05 * (unique / total_items)  # up to +0.05

        total_penalty = sum(penalties)
        # Map penalty to fitness in (0, 1]; higher is better
        fitness_base = 1.0 / (1.0 + total_penalty)
        fitness = (fitness_base * (1.0 + variety_bonus))  # multiplicative scaling
        # Clamp into [0,1]; previous code referenced undefined 'cap'
        if fitness > 1.0:
            fitness = 1.0
        elif fitness < 0.0:
            fitness = 0.0

        print({
            "cal_err": cal_rel_err,
            "protein_short": max(0, (c.min_protein - avg['protein']) / c.min_protein),
            "fiber_short": max(0, (c.min_fiber - avg['fiber']) / c.min_fiber),
            "sugar_over": max(0, (avg['sugar'] - c.max_sugar) / c.max_sugar),
            "fat_over": max(0, (avg['fat'] - c.max_fat) / c.max_fat),
            "variety_bonus": variety_bonus,
            "fitness": fitness,
        })
        
        return fitness

    # ---- Nutri-Score (approximate) ----
    def _compute_plan_nutri_score(self, plan: MealPlan) -> Tuple[int, str, str]:
        """Approximate an overall Nutri-Score for the meal plan.

        Nutri-Score is formally defined per 100g of a single food product. Here we
        approximate the *composite* of all foods in the meal plan by computing a
        weighted average of nutrient densities across all consumed grams. This is
        ONLY an heuristic summary and should not be used for labeling.

        Assumptions / limitations:
        - Uses actual saturated fat if column `saturated_fat_g` exists and at least 
          one food provides a non-null value; otherwise falls back to 35% of total fat.
        - Sodium derived from salt_g (1 g salt ≈ 400 mg sodium; precise factor 393.4).
        - Fruits/vegetables/nuts percentage unknown -> counted as 0 (disallows
          protein points when negative points >= 11).
        - Applies *solid food* Nutri-Score thresholds.
        - If the source table provides product-level Nutri-Score (columns
          `nutriscore_score` / `nutriscore_grade`), we compute a weighted average
          of the numeric score across items (portion-weighted) and take the *sign*
          nearest integer of that average, mapping to the average grade if all
          grades agree; otherwise we map the averaged numeric score to a grade.
        """
        # First pass: if any food has actual nutriscore data, use that path
        total_grams = 0.0
        actual_score_weighted = 0.0
        actual_any = False
        all_grades = set()
        for day in plan.days:
            for meal in day.meals.values():
                for item in meal.items:
                    g = item.portion_grams
                    if g <= 0:
                        continue
                    f = item.food
                    total_grams += g
                    if f.nutriscore_score_actual is not None:
                        actual_any = True
                        actual_score_weighted += f.nutriscore_score_actual * g
                    if f.nutriscore_grade_actual:
                        all_grades.add(f.nutriscore_grade_actual.upper())
        if actual_any and total_grams > 0:
            avg_score = actual_score_weighted / total_grams
            # Round to nearest int for composite (Nutri-Score integers)
            composite_score = int(round(avg_score))
            # If all grades identical, use that; else derive from numeric
            if len(all_grades) == 1:
                grade = next(iter(all_grades))
            else:
                # Derive grade from numeric thresholds
                if composite_score <= -1:
                    grade = "A"
                elif composite_score <= 2:
                    grade = "B"
                elif composite_score <= 10:
                    grade = "C"
                elif composite_score <= 18:
                    grade = "D"
                else:
                    grade = "E"
            return composite_score, grade, "actual"

        # Collect weighted totals (heuristic path)
        total_grams = 0.0
        energy_kcal_total = 0.0
        sugar_total = 0.0
        fat_total = 0.0
        fiber_total = 0.0
        protein_total = 0.0
        salt_total = 0.0
        any_sat_available = False
        sat_fat_total = 0.0  # gram-weighted saturated fat total if available
        for day in plan.days:
            for meal in day.meals.values():
                for item in meal.items:
                    g = item.portion_grams
                    if g <= 0:
                        continue
                    f = item.food
                    total_grams += g
                    factor = g / 100.0
                    energy_kcal_total += f.calories_per_100g * factor
                    sugar_total += f.sugar_g_per_100g * factor
                    fat_total += f.fat_g_per_100g * factor
                    fiber_total += f.fiber_g_per_100g * factor
                    protein_total += f.protein_g_per_100g * factor
                    salt_total += f.salt_g_per_100g * factor
                    if f.saturated_fat_g_per_100g is not None:
                        any_sat_available = True
                        sat_fat_total += f.saturated_fat_g_per_100g * factor
        if total_grams <= 0:
            return 0, "?"
        # Convert to per 100g composite
        scale = 100.0 / total_grams
        energy_kcal = energy_kcal_total * scale
        energy_kj = energy_kcal * 4.184
        sugar_g = sugar_total * scale
        if any_sat_available:
            sat_fat_g = sat_fat_total * scale
        else:
            # Fallback approximation
            sat_fat_g = (fat_total * scale) * 0.35
        fiber_g = fiber_total * scale
        protein_g = protein_total * scale
        sodium_mg = (salt_total * scale) * 400.0

        def pts_energy(kj: float) -> int:
            bounds = [335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350]
            for i, b in enumerate(bounds):
                if kj <= b:
                    return i
            return 10

        def pts_sugar(g: float) -> int:
            bounds = [4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45]
            for i, b in enumerate(bounds):
                if g <= b:
                    return i
            return 10

        def pts_satfat(g: float) -> int:
            bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for i, b in enumerate(bounds):
                if g <= b:
                    return i
            return 10

        def pts_sodium(mg: float) -> int:
            bounds = [90, 180, 270, 360, 450, 540, 630, 720, 810, 900]
            for i, b in enumerate(bounds):
                if mg <= b:
                    return i
            return 10

        def pts_fiber(g: float) -> int:
            bounds = [0.9, 1.9, 2.8, 3.7, 4.7]
            for i, b in enumerate(bounds):
                if g <= b:
                    return i
            return 5

        def pts_protein(g: float) -> int:
            bounds = [1.6, 3.2, 4.8, 6.4, 8.0]
            for i, b in enumerate(bounds):
                if g <= b:
                    return i
            return 5

        # Negative points
        neg = pts_energy(energy_kj) + pts_sugar(sugar_g) + pts_satfat(sat_fat_g) + pts_sodium(sodium_mg)
        # Positive points (fruit/veg/nuts unknown -> 0)
        fiber_p = pts_fiber(fiber_g)
        protein_p = pts_protein(protein_g)
        pos = fiber_p
        if neg < 11:  # include protein only if negative < 11 (since F/V% = 0)
            pos += protein_p
        score = neg - pos

        # Letter mapping for foods
        if score <= -1:
            letter = "A"
        elif score <= 2:
            letter = "B"
        elif score <= 10:
            letter = "C"
        elif score <= 18:
            letter = "D"
        else:
            letter = "E"
        return int(score), letter, "heuristic"


# Convenience glue for UI/tests
def plan_with_ga(daily_calories: int = 2000, num_days: int = 3, user_id: Optional[str] = None) -> MealPlan:
    c = GAConstraints(daily_calories=daily_calories, num_days=num_days)
    return GeneticMealPlanner(c, user_id=user_id).plan_meals()


# -----------------------------
# Plotting utilities (optional)
# -----------------------------

def plot_ga_fitness(history: List[float], save_path: Optional[str] = None) -> Optional[str]:
    """Plot GA best-fitness history.

    Lazy-imports matplotlib to avoid hard dependency at import time. No return
    value; shows a simple line chart of best fitness per generation.
    """
    if not history:
        return None
    import matplotlib.pyplot as plt  # lazy import

    gens = list(range(len(history)))
    plt.figure(figsize=(6, 3.5))
    plt.plot(gens, history, marker="o", lw=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("GA convergence (best fitness per generation)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_mealplan_vs_targets(
    plan: MealPlan, constraints: GAConstraints, save_path: Optional[str] = None
) -> Optional[str]:
    """Plot average daily nutrition of a plan vs target constraints.

    Compares the plan's average daily totals against the key targets:
    - calories (target exact)
    - protein (minimum)
    - fiber (minimum)
    - sugar (maximum)
    - fat (maximum)
    """
    import matplotlib.pyplot as plt  # lazy import

    avg = plan.get_average_daily_nutrition()

    metrics = [
        ("calories", float(constraints.daily_calories), avg.get("calories", 0.0), "exact"),
        ("protein", float(constraints.min_protein), avg.get("protein", 0.0), "min"),
        ("fiber", float(constraints.min_fiber), avg.get("fiber", 0.0), "min"),
        ("sugar", float(constraints.max_sugar), avg.get("sugar", 0.0), "max"),
        ("fat", float(constraints.max_fat), avg.get("fat", 0.0), "max"),
    ]

    labels = [m[0] for m in metrics]
    targets = [m[1] for m in metrics]
    actuals = [m[2] for m in metrics]

    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(7.5, 4))
    plt.bar([i - width / 2 for i in x], actuals, width=width, label="Actual")
    plt.bar([i + width / 2 for i in x], targets, width=width, label="Target")
    plt.xticks(x, labels)
    plt.ylabel("Value (per day)")
    plt.title("Meal plan vs targets (avg per day)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def run_ga_and_plot(daily_calories: int = 2000, num_days: int = 3, user_id: Optional[str] = None) -> MealPlan:
    """Convenience runner: execute GA with history and display plots.

    Returns the best meal plan. Intended for quick manual exploration.
    """
    c = GAConstraints(daily_calories=daily_calories, num_days=num_days)
    planner = GeneticMealPlanner(c, user_id=user_id)
    plan, history = planner.plan_meals_with_history()
    try:
        plot_ga_fitness(history)
        plot_mealplan_vs_targets(plan, c)
    except Exception:
        # plotting is optional; ignore if environment has no display
        pass
    return plan
