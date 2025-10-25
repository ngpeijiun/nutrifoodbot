import io
from typing import cast

import streamlit as st

from genetic_meal_planner import (
    GAConstraints,
    GeneticMealPlanner,
    plot_ga_fitness,
    plot_mealplan_vs_targets,
)


st.title("Genetic Algorithm Meal Planner")
st.caption("Tune constraints, run the GA, and inspect results visually.")
st.set_page_config(layout="wide")

def _default_constraints() -> GAConstraints:
    return GAConstraints()


with st.sidebar:
    st.subheader("GA Settings")
    daily_calories = st.number_input("Daily calories", 1200, 4000, 2000, 50)
    num_days = st.number_input("Number of days", 1, 14, 3, 1)
    population_size = st.number_input("Population size", 4, 200, 20, 1)
    generations = st.number_input("Generations", 1, 200, 30, 1)
    mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.2, 0.01)
    tournament_size = st.number_input("Tournament size", 2, 10, 3, 1)
    patience = st.number_input("Early stop patience (0=off)", 0, 200, 0, 1, help="Stop if no sufficient best fitness improvement across this many generations.")
    min_improvement = st.number_input("Min improvement", 0.0, 0.1, 0.0001, 0.0001, format="%.4f", help="Minimum delta in best fitness to reset patience.")
    show_mean = st.checkbox("Show mean fitness", value=True)

    st.markdown("---")
    st.subheader("Nutrition Targets")
    min_protein = st.number_input("Min protein (g)", 0, 300, 50, 1)
    min_fiber = st.number_input("Min fiber (g)", 0, 150, 25, 1)
    max_sugar = st.number_input("Max sugar (g)", 0, 300, 50, 1)
    max_fat = st.number_input("Max fat (g)", 0, 300, 70, 1)

    st.markdown("---")
    st.subheader("Meal Calories Distribution")
    b = st.number_input("Breakfast share", 0.0, 1.0, 0.25, 0.01)
    l = st.number_input("Lunch share", 0.0, 1.0, 0.35, 0.01)
    d = st.number_input("Dinner share", 0.0, 1.0, 0.40, 0.01)

    if abs((b + l + d) - 1.0) > 1e-6:
        st.warning("Calorie distribution does not sum to 1. It will be normalized.")

    run = st.button("Generate Meal Plan", use_container_width=True)


col_left, col_right = st.columns([1, 1])

if run:
    # Build constraints
    c = GAConstraints(
        daily_calories=int(daily_calories),
        num_days=int(num_days),
        min_protein=float(min_protein),
        min_fiber=float(min_fiber),
        max_sugar=float(max_sugar),
        max_fat=float(max_fat),
        population_size=int(population_size),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        tournament_size=int(tournament_size),
        patience=int(patience),
        min_improvement=float(min_improvement),
    )
    # Normalize distribution
    total = max(1e-6, b + l + d)
    c.calorie_distribution = {"breakfast": b / total, "lunch": l / total, "dinner": d / total}

    with st.spinner("Running genetic algorithm..."):
        planner = GeneticMealPlanner(c)
        if show_mean:
            plan, best_hist, mean_hist = planner.plan_meals_with_stats()
        else:
            plan, best_hist = planner.plan_meals_with_history()

    with col_left:
        st.subheader("Convergence")
        # Render fitness plot inline using a buffer
        import matplotlib.pyplot as plt  # noqa: E402

        import matplotlib.pyplot as plt  # noqa: E402
        import numpy as np  # noqa: E402
        fig_buf = io.BytesIO()
        plt.figure(figsize=(6,3.5))
        plt.plot(best_hist, marker='o', label='Best')
        if show_mean:
            plt.plot(mean_hist, marker='.', alpha=0.7, label='Mean')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        title_extra = ''
        if c.patience > 0 and len(best_hist) - 1 < c.generations:
            title_extra = f" (stopped @ {len(best_hist)-1})"
        plt.title(f'GA convergence{title_extra}')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_buf, dpi=150)
        fig_buf.seek(0)
        st.image(fig_buf, caption="Fitness history", width=500)

    with col_right:
        st.subheader("Plan vs Targets")
        fig_buf2 = io.BytesIO()
        plot_mealplan_vs_targets(plan, c, save_path=fig_buf2)
        fig_buf2.seek(0)
        st.image(fig_buf2, caption="Average daily nutrition vs targets", width=500)

    st.markdown("---")
    st.subheader("Meal Plan Details")
    st.write(
        f"Best fitness: **{plan.fitness:.3f}**  | Generations executed: **{len(best_hist)-1}**"
    )
    # Nutri-Score summary (actual if sourced from DB, else heuristic composite)
    if getattr(plan, 'nutri_score', None) is not None:
        letter = plan.nutri_score_letter or "?"
        score = plan.nutri_score
        source = getattr(plan, 'nutri_score_source', 'heuristic')
        ns_colors = {
            "A": "#1e8e3e",  # deep green
            "B": "#34a853",  # green
            "C": "#fbbc05",  # yellow
            "D": "#f57c00",  # orange
            "E": "#d93025",  # red,
        }
        color = ns_colors.get(letter, "#5f6368")
        source_label = "actual (weighted)" if source == "actual" else "heuristic composite"
        badge_html = f"""
        <div style='display:flex;align-items:center;gap:14px;'>
          <div style=\"background:{color};color:#fff;padding:10px 22px;border-radius:12px;font-weight:700;font-size:2.2rem;line-height:1;min-width:70px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.2);\" aria-label=\"Nutri-Score {letter}\">{letter}</div>
          <div style='font-size:0.95rem;'>
            <div><strong>Nutri-Score</strong> ({source_label})</div>
            <div style='margin-top:2px;'>Numeric score: <code style='font-weight:600;font-size:0.9rem;'>{score}</code></div>
            <div style='font-size:0.75rem;color:#666;margin-top:4px;'>Lower (better) numeric scores map to letters A (best) → E (least healthy).</div>
          </div>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)
    if c.patience > 0 and len(best_hist)-1 < c.generations:
        st.success("Early stopping triggered.")

    for day_idx, day in enumerate(plan.days, start=1):
        st.markdown(f"### Day {day_idx}")
        for meal_name, meal in day.meals.items():
            with st.expander(meal_name.capitalize(), expanded=False):
                for item in meal.items:
                    st.write(
                        f"- {item.food.name} (code: {item.food.food_code}) — {item.portion_grams:.0f} g"
                    )
                totals = meal.get_total_nutrition()
                st.caption(
                    f"Totals: {totals['calories']:.0f} kcal, P {totals['protein']:.1f} g, C {totals['carbs']:.1f} g, "
                    f"F {totals['fat']:.1f} g, fiber {totals['fiber']:.1f} g, sugar {totals['sugar']:.1f} g, salt {totals['salt']:.1f} g"
                )
else:
    st.info("Adjust settings in the sidebar and click 'Generate Meal Plan' to generate a plan and charts.")
