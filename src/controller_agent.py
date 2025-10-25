import traceback
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from food_query_agent import query_food
from meal_planner_agent import plan_meals

_controller_agent: CompiledStateGraph | None = None


def _query_food_tool(prompt: str) -> list[dict[str, Any]]:
    """
    Fetch food information based on a user query.

    Args:
        prompt (str): A string containing the user's food query or request.

    Returns:
        list[dict[str, Any]]: A list of product records matching the query.
    """

    try:
        return query_food(prompt).result()
    except Exception as e:
        traceback.print_exc()
        raise Exception("Error querying food data") from e


def _meal_planning_tool(prompt: str) -> str:
    """Generate meal plans based on user requirements.

    Args:
        prompt (str): A string containing the user's meal planning request,
            including preferences, dietary restrictions, calorie targets, etc.

    Returns:
        str: A formatted meal plan or error message.
    """

    try:
        result = plan_meals(prompt)
        if result["success"]:
            return result["meal_plan"]
        else:
            return f"I couldn't create your meal plan: {result.get('fallback_message', result.get('error', 'Unknown error'))}"
    except Exception as e:
        traceback.print_exc()
        return "I'm sorry, I encountered an error while creating your meal plan. Please try again with a simpler request."


def get_controller_agent() -> CompiledStateGraph:

    global _controller_agent

    if _controller_agent is None:
        _controller_agent = create_react_agent(
            model="gpt-4.1-2025-04-14",
            tools=[_query_food_tool, _meal_planning_tool],
            prompt=(
                "In markdown format, display list of food names with clickable href URL to `/?page=6_product_details&product_code=<code>`. "
                "If allergen exists, display behind food name, the list of allergens in brackets."
            ),
#            prompt="""You are NutriFoodBot, a helpful food and nutrition assistant. You can:
#
#1. Answer food and nutrition questions using the food query tool
#2. Create personalized meal plans using the meal planning tool
#
#When users ask for meal plans, use the meal_planning_tool. Look for keywords like:
#- "meal plan", "plan meals", "weekly meal plan"
#- "what should I eat", "help me plan my meals"
#- requests mentioning days, calories, dietary restrictions
#
#For general food questions, use the query_food_tool.
#
#Always be helpful, friendly, and focused on providing accurate nutritional guidance.""",
        )

    return _controller_agent
