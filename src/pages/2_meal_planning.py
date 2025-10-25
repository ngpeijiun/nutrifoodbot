import streamlit as st

MEAL_PLANNING_DEFAULTS = {
    "planner_mode": "Rule-based",
    "dietary_prefs": [],
    "calories": 2000,
}

for key, value in MEAL_PLANNING_DEFAULTS.items():
    st.session_state.setdefault(key, value)


st.title("Meal Planning")
st.write("Use quick shortcuts to generate plans tailored to your goals.")

planner_options = ["Rule-based", "Genetic Algorithm"]

with st.container():
    st.subheader("🍽️ Meal Planning")
    st.caption("Quick meal planning shortcuts")
    st.radio(
        "Planner Engine",
        planner_options,
        help="Choose the engine used for generating meal plans",
        key="planner_mode",
    )
    dietary_prefs = st.multiselect(
        "Dietary Preferences",
        [
            "Vegetarian",
            "Vegan",
            "Gluten-Free",
            "Dairy-Free",
            "Keto",
            "Paleo",
            "Low-Carb",
        ],
        help="Select your dietary preferences for meal planning",
        key="dietary_prefs",
    )
    calories = st.number_input(
        "Daily Calories Target",
        min_value=1200,
        max_value=4000,
        step=50,
        help="Your daily calorie target for meal planning",
        key="calories",
    )
    quick_plan_cols = st.columns(2)
    with quick_plan_cols[0]:
        if st.button("📅 3-Day Plan", use_container_width=True):
            dietary_text = " ".join(dietary_prefs).lower() if dietary_prefs else ""
            meal_plan_prompt = (
                f"Create a 3-day meal plan for {calories} calories per day with {dietary_text} preferences "
                "and balanced nutrition"
            ).strip()
            st.session_state.quick_prompt = meal_plan_prompt
    with quick_plan_cols[1]:
        if st.button("🗓️ Weekly Plan", use_container_width=True):
            dietary_text = " ".join(dietary_prefs).lower() if dietary_prefs else ""
            meal_plan_prompt = (
                f"Create a 7-day weekly meal plan for {calories} calories per day with {dietary_text} preferences "
                "and variety"
            ).strip()
            st.session_state.quick_prompt = meal_plan_prompt

if st.button("🎯 Generate Custom Plan", use_container_width=True):
    dietary_text = " ".join(dietary_prefs).lower() if dietary_prefs else ""
    custom_prompt = f"Create a meal plan for {calories} calories per day with {dietary_text} preferences"
    st.session_state.quick_prompt = custom_prompt
