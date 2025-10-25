"""
Meal Planner Agent for NutriFoodBot

This agent handles meal planning requests by:
1. Analyzing user dietary preferences and constraints
2. Generating balanced meal plans using food query capabilities
3. Ensuring nutritional targets are met
4. Providing meal variety and balance
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from food_query_agent import query_food


@dataclass
class MealPlanConstraints:
    """Structure for meal planning constraints"""
    daily_calories: int = 2000
    num_days: int = 3
    meals_per_day: int = 3
    dietary_restrictions: Optional[List[str]] = None
    max_sugar: float = 50.0
    max_fat: float = 65.0
    min_protein: float = 50.0
    min_fiber: float = 25.0
    budget_per_day: Optional[float] = None
    
    def __post_init__(self):
        if self.dietary_restrictions is None:
            self.dietary_restrictions = []


@dataclass
class Meal:
    """Structure for individual meal"""
    name: str
    foods: List[Dict]
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    total_fiber: float
    total_sugar: float
    meal_type: str  # breakfast, lunch, dinner, snack
    estimated_cost: float = 0.0


@dataclass
class DayPlan:
    """Structure for a single day's meal plan"""
    date: str
    meals: List[Meal]
    daily_totals: Dict[str, float]
    meets_targets: bool


@dataclass
class MealPlanResult:
    """Complete meal plan result"""
    days: List[DayPlan]
    plan_summary: Dict[str, Any]
    recommendations: List[str]
    success: bool
    error_message: str = ""


class MealPlannerAgent:
    """Agent responsible for generating meal plans"""
    
    def __init__(self, use_genetic_algorithm: bool = False):
        self.meal_types = ["breakfast", "lunch", "dinner"]
        self.calorie_distribution = {
            "breakfast": 0.25,  # 25% of daily calories
            "lunch": 0.35,      # 35% of daily calories
            "dinner": 0.40      # 40% of daily calories
        }
        self.use_genetic_algorithm = use_genetic_algorithm
    
    def plan_meals(self, constraints: MealPlanConstraints) -> MealPlanResult:
        """
        Generate a complete meal plan based on constraints
        
        Args:
            constraints: MealPlanConstraints object with user preferences
            
        Returns:
            MealPlanResult with generated meal plan
        """
        try:
            if self.use_genetic_algorithm:
                return self._plan_with_ga(constraints)
            days = []
            
            for day_num in range(constraints.num_days):
                date = (datetime.now() + timedelta(days=day_num)).strftime("%Y-%m-%d")
                day_plan = self._generate_day_plan(date, constraints)
                days.append(day_plan)
            
            plan_summary = self._calculate_plan_summary(days, constraints)
            recommendations = self._generate_recommendations(days, constraints)
            
            return MealPlanResult(
                days=days,
                plan_summary=plan_summary,
                recommendations=recommendations,
                success=True
            )
            
        except Exception as e:
            return MealPlanResult(
                days=[],
                plan_summary={},
                recommendations=[],
                success=False,
                error_message=str(e)
            )

    def _plan_with_ga(self, constraints: MealPlanConstraints) -> MealPlanResult:
        """Use the GA-based planner and convert its output into this agent's structures."""
        try:
            # Local import to avoid hard dependency for non-GA path
            from genetic_meal_planner import GAConstraints, GeneticMealPlanner

            ga_c = GAConstraints(
                daily_calories=constraints.daily_calories,
                num_days=constraints.num_days,
                min_protein=constraints.min_protein,
                min_fiber=constraints.min_fiber,
                max_sugar=constraints.max_sugar,
                max_fat=constraints.max_fat,
            )
            # NLQ seeding: query foods per meal type and parse codes
            seed_codes: Dict[str, List[str]] = {}
            try:
                for mt in self.meal_types:
                    q = self._build_food_query(mt, constraints.daily_calories * self.calorie_distribution[mt], constraints)
                    data_str = query_food(q).result()
                    # Expect JSON array of records with at least 'code' field
                    codes: List[str] = []
                    try:
                        arr = json.loads(data_str) if isinstance(data_str, str) else data_str
                        if isinstance(arr, list):
                            for rec in arr:
                                code = (rec.get("code") if isinstance(rec, dict) else None)
                                if code:
                                    codes.append(str(code))
                    except Exception:
                        pass
                    if codes:
                        seed_codes[mt] = codes
            except Exception:
                # seeding is best-effort; ignore failures
                seed_codes = {}

            planner = GeneticMealPlanner(ga_c, seed_food_codes=seed_codes or None)
            best_plan = planner.plan_meals()

            # Convert to DayPlan/Meal
            days: List[DayPlan] = []
            base_date = datetime.now()
            for i, day in enumerate(best_plan.days):
                date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
                meals: List[Meal] = []
                daily_totals = {
                    "calories": 0.0,
                    "protein": 0.0,
                    "carbs": 0.0,
                    "fat": 0.0,
                    "fiber": 0.0,
                    "sugar": 0.0,
                    "cost": 0.0,
                }
                for meal_type in self.meal_types:
                    m = day.meals.get(meal_type)
                    if not m:
                        continue
                    foods = []
                    n = m.get_total_nutrition()
                    for it in m.items:
                        foods.append({
                            "name": it.food.name,
                            "portion": f"{it.portion_grams:.0f} g",
                            "calories": it.get_nutrition()["calories"],
                            "protein": it.get_nutrition()["protein"],
                            "carbs": it.get_nutrition()["carbs"],
                            "fat": it.get_nutrition()["fat"],
                            "fiber": it.get_nutrition()["fiber"],
                            "sugar": it.get_nutrition()["sugar"],
                            "cost": 0.0,
                        })
                    meals.append(Meal(
                        name=f"{meal_type.title()} - GA",
                        foods=foods,
                        total_calories=n["calories"],
                        total_protein=n["protein"],
                        total_carbs=n["carbs"],
                        total_fat=n["fat"],
                        total_fiber=n["fiber"],
                        total_sugar=n["sugar"],
                        meal_type=meal_type,
                        estimated_cost=n.get("cost", 0.0),
                    ))
                    for k in daily_totals:
                        daily_totals[k] += n.get(k, 0.0)

                meets = self._check_daily_targets(daily_totals, constraints)
                days.append(DayPlan(date=date, meals=meals, daily_totals=daily_totals, meets_targets=meets))

            plan_summary = self._calculate_plan_summary(days, constraints)
            # Add GA fitness metric
            plan_summary["ga_fitness"] = round(best_plan.fitness, 4)
            recs = self._generate_recommendations(days, constraints)
            return MealPlanResult(days=days, plan_summary=plan_summary, recommendations=recs, success=True)
        except Exception as e:
            return MealPlanResult(days=[], plan_summary={}, recommendations=[], success=False, error_message=str(e))
    
    def _generate_day_plan(self, date: str, constraints: MealPlanConstraints) -> DayPlan:
        """Generate meal plan for a single day"""
        meals = []
        daily_totals = {
            "calories": 0.0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "fiber": 0.0,
            "sugar": 0.0,
            "cost": 0.0
        }
        
        for meal_type in self.meal_types:
            target_calories = constraints.daily_calories * self.calorie_distribution[meal_type]
            meal = self._generate_meal(meal_type, target_calories, constraints)
            meals.append(meal)
            
            # Update daily totals
            daily_totals["calories"] += meal.total_calories
            daily_totals["protein"] += meal.total_protein
            daily_totals["carbs"] += meal.total_carbs
            daily_totals["fat"] += meal.total_fat
            daily_totals["fiber"] += meal.total_fiber
            daily_totals["sugar"] += meal.total_sugar
            daily_totals["cost"] += meal.estimated_cost
        
        meets_targets = self._check_daily_targets(daily_totals, constraints)
        
        return DayPlan(
            date=date,
            meals=meals,
            daily_totals=daily_totals,
            meets_targets=meets_targets
        )
    
    def _generate_meal(self, meal_type: str, target_calories: float, constraints: MealPlanConstraints) -> Meal:
        """Generate a single meal"""
        # Build query for food search
        query = self._build_food_query(meal_type, target_calories, constraints)
        
        # Query foods using existing food_query_agent
        try:
            food_query_result = query_food(query)
            food_results = food_query_result.result() if hasattr(food_query_result, 'result') else []
            if isinstance(food_results, str):
                food_results = [food_results]  # Convert single string to list
            foods = self._select_foods_for_meal(food_results, target_calories, constraints)
        except Exception as e:
            # Fallback to basic foods if query fails
            foods = self._get_fallback_foods(meal_type, target_calories)
        
        # Calculate meal totals
        totals = self._calculate_meal_totals(foods)
        
        return Meal(
            name=f"{meal_type.title()} - {datetime.now().strftime('%m/%d')}",
            foods=foods,
            total_calories=totals["calories"],
            total_protein=totals["protein"],
            total_carbs=totals["carbs"],
            total_fat=totals["fat"],
            total_fiber=totals["fiber"],
            total_sugar=totals["sugar"],
            meal_type=meal_type,
            estimated_cost=totals["cost"]
        )
    
    def _build_food_query(self, meal_type: str, target_calories: float, constraints: MealPlanConstraints) -> str:
        """Build a natural language query for food search"""
        query_parts = []
        
        # Add meal type context
        if meal_type == "breakfast":
            query_parts.append("healthy breakfast foods")
        elif meal_type == "lunch":
            query_parts.append("nutritious lunch options")
        elif meal_type == "dinner":
            query_parts.append("balanced dinner meals")
        
        # Add calorie target
        query_parts.append(f"around {int(target_calories)} calories")
        
        # Add dietary restrictions
        if constraints.dietary_restrictions:
            restrictions = " ".join(constraints.dietary_restrictions)
            query_parts.append(f"{restrictions}")
        
        # Add nutritional constraints
        if constraints.max_sugar < 50:
            query_parts.append(f"low sugar (under {constraints.max_sugar}g)")
        
        if constraints.max_fat < 65:
            query_parts.append(f"moderate fat (under {constraints.max_fat}g)")
        
        if constraints.min_protein > 20:
            query_parts.append(f"high protein (at least {constraints.min_protein}g)")
        
        return " ".join(query_parts)
    
    def _select_foods_for_meal(self, food_results: List[str], target_calories: float, constraints: MealPlanConstraints) -> List[Dict]:
        """Select appropriate foods from query results to meet calorie target"""
        # This is a simplified implementation
        # In a real scenario, you'd parse the food_results and calculate portions
        
        selected_foods = []
        current_calories = 0
        
        # Mock food selection logic
        for i, food_description in enumerate(food_results[:3]):  # Limit to 3 foods per meal
            # Estimate calories per food item (this would need real data)
            estimated_calories = target_calories / 3
            portion_size = "1 serving"
            
            food_item = {
                "name": food_description,
                "portion": portion_size,
                "calories": estimated_calories,
                "protein": estimated_calories * 0.15 / 4,  # Rough estimate
                "carbs": estimated_calories * 0.45 / 4,
                "fat": estimated_calories * 0.30 / 9,
                "fiber": 3.0,  # Default estimate
                "sugar": 5.0,  # Default estimate
                "cost": 3.50   # Default estimate
            }
            
            selected_foods.append(food_item)
            current_calories += estimated_calories
            
            if current_calories >= target_calories * 0.9:  # Within 90% of target
                break
        
        return selected_foods
    
    def _get_fallback_foods(self, meal_type: str, target_calories: float) -> List[Dict]:
        """Provide fallback foods if query fails"""
        fallback_meals = {
            "breakfast": [
                {"name": "Oatmeal with berries", "calories": 300, "protein": 8, "carbs": 54, "fat": 6, "fiber": 8, "sugar": 12, "cost": 2.50},
                {"name": "Greek yogurt", "calories": 150, "protein": 15, "carbs": 8, "fat": 5, "fiber": 0, "sugar": 8, "cost": 1.75},
                {"name": "Banana", "calories": 100, "protein": 1, "carbs": 27, "fat": 0, "fiber": 3, "sugar": 14, "cost": 0.50}
            ],
            "lunch": [
                {"name": "Grilled chicken salad", "calories": 350, "protein": 30, "carbs": 15, "fat": 18, "fiber": 5, "sugar": 8, "cost": 5.00},
                {"name": "Quinoa", "calories": 150, "protein": 6, "carbs": 27, "fat": 2, "fiber": 3, "sugar": 1, "cost": 1.25},
                {"name": "Mixed vegetables", "calories": 50, "protein": 2, "carbs": 12, "fat": 0, "fiber": 4, "sugar": 6, "cost": 2.00}
            ],
            "dinner": [
                {"name": "Grilled salmon", "calories": 300, "protein": 25, "carbs": 0, "fat": 20, "fiber": 0, "sugar": 0, "cost": 6.00},
                {"name": "Brown rice", "calories": 150, "protein": 3, "carbs": 30, "fat": 1, "fiber": 2, "sugar": 1, "cost": 0.75},
                {"name": "Steamed broccoli", "calories": 50, "protein": 4, "carbs": 8, "fat": 0, "fiber": 4, "sugar": 2, "cost": 1.50}
            ]
        }
        
        return fallback_meals.get(meal_type, fallback_meals["lunch"])
    
    def _calculate_meal_totals(self, foods: List[Dict]) -> Dict[str, float]:
        """Calculate nutritional totals for a meal"""
        totals = {
            "calories": 0.0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "fiber": 0.0,
            "sugar": 0.0,
            "cost": 0.0
        }
        
        for food in foods:
            for key in totals.keys():
                totals[key] += float(food.get(key, 0))
        
        return totals
    
    def _check_daily_targets(self, daily_totals: Dict[str, float], constraints: MealPlanConstraints) -> bool:
        """Check if daily totals meet the specified constraints"""
        checks = [
            daily_totals["calories"] >= constraints.daily_calories * 0.9,  # Within 90% of target
            daily_totals["calories"] <= constraints.daily_calories * 1.1,  # Within 110% of target
            daily_totals["protein"] >= constraints.min_protein,
            daily_totals["fiber"] >= constraints.min_fiber,
            daily_totals["sugar"] <= constraints.max_sugar,
            daily_totals["fat"] <= constraints.max_fat
        ]
        
        if constraints.budget_per_day:
            checks.append(daily_totals["cost"] <= constraints.budget_per_day)
        
        return all(checks)
    
    def _calculate_plan_summary(self, days: List[DayPlan], constraints: MealPlanConstraints) -> Dict[str, Any]:
        """Calculate summary statistics for the meal plan"""
        total_days = len(days)
        days_meeting_targets = sum(1 for day in days if day.meets_targets)
        
        avg_calories = sum(day.daily_totals["calories"] for day in days) / total_days
        avg_protein = sum(day.daily_totals["protein"] for day in days) / total_days
        avg_cost = sum(day.daily_totals["cost"] for day in days) / total_days
        
        return {
            "total_days": total_days,
            "days_meeting_targets": days_meeting_targets,
            "target_compliance_rate": days_meeting_targets / total_days * 100,
            "average_daily_calories": round(avg_calories, 1),
            "average_daily_protein": round(avg_protein, 1),
            "average_daily_cost": round(avg_cost, 2),
            "total_estimated_cost": round(sum(day.daily_totals["cost"] for day in days), 2)
        }
    
    def _generate_recommendations(self, days: List[DayPlan], constraints: MealPlanConstraints) -> List[str]:
        """Generate helpful recommendations based on the meal plan"""
        recommendations = []
        
        # Check for common issues and suggest improvements
        avg_fiber = sum(day.daily_totals["fiber"] for day in days) / len(days)
        if avg_fiber < constraints.min_fiber:
            recommendations.append("Consider adding more high-fiber foods like beans, vegetables, and whole grains to meet your daily fiber goals.")
        
        avg_protein = sum(day.daily_totals["protein"] for day in days) / len(days)
        if avg_protein < constraints.min_protein:
            recommendations.append("Try incorporating more lean proteins like chicken, fish, tofu, or legumes to meet your protein targets.")
        
        days_over_budget = sum(1 for day in days if constraints.budget_per_day and day.daily_totals["cost"] > constraints.budget_per_day)
        if days_over_budget > 0:
            recommendations.append("Consider batch cooking, buying in bulk, or choosing more affordable protein sources to stay within budget.")
        
        if not recommendations:
            recommendations.append("Great job! Your meal plan looks well-balanced and meets your nutritional goals.")
        
        return recommendations


# Main function to be called by controller agent
def plan_meals(prompt: str) -> Dict[str, Any]:
    """
    Main function to handle meal planning requests
    
    Args:
        prompt: Natural language meal planning request
        
    Returns:
        Dictionary containing meal plan results
    """
    try:
        # Parse the prompt to extract constraints
        constraints = _parse_meal_plan_request(prompt)

        # Create meal planner and generate plan
        lower_prompt = prompt.lower()
        use_ga = ("[ga" in lower_prompt) or ("genetic" in lower_prompt)
        planner = MealPlannerAgent(use_genetic_algorithm=use_ga)
        result = planner.plan_meals(constraints)
        
        if result.success:
            return {
                "success": True,
                "meal_plan": _format_meal_plan_for_display(result),
                "summary": result.plan_summary,
                "recommendations": result.recommendations
            }
        else:
            return {
                "success": False,
                "error": result.error_message,
                "fallback_message": "I encountered an issue generating your meal plan. Please try with simpler requirements or contact support."
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback_message": "I'm sorry, I couldn't process your meal planning request. Please try again with a simpler request."
        }


def _parse_meal_plan_request(prompt: str) -> MealPlanConstraints:
    """Parse natural language prompt to extract meal planning constraints"""
    prompt_lower = prompt.lower()
    
    # Extract number of days
    num_days = 3  # default
    if "week" in prompt_lower or "7 day" in prompt_lower:
        num_days = 7
    elif "day" in prompt_lower:
        # Try to extract number before "day"
        import re
        days_match = re.search(r'(\d+)\s*day', prompt_lower)
        if days_match:
            num_days = int(days_match.group(1))
    
    # Extract calorie target
    daily_calories = 2000  # default
    calorie_patterns = [r'(\d+)\s*calorie', r'(\d+)\s*cal']
    for pattern in calorie_patterns:
        import re
        cal_match = re.search(pattern, prompt_lower)
        if cal_match:
            daily_calories = int(cal_match.group(1))
            break
    
    # Extract dietary restrictions
    dietary_restrictions = []
    restrictions_map = {
        "vegetarian": "vegetarian",
        "vegan": "vegan",
        "gluten-free": "gluten-free",
        "gluten free": "gluten-free",
        "dairy-free": "dairy-free",
        "dairy free": "dairy-free",
        "keto": "ketogenic",
        "ketogenic": "ketogenic",
        "paleo": "paleo",
        "low-carb": "low-carb",
        "low carb": "low-carb"
    }
    
    for key, value in restrictions_map.items():
        if key in prompt_lower:
            dietary_restrictions.append(value)
    
    # Extract budget if mentioned
    budget_per_day = None
    import re
    budget_match = re.search(r'\$(\d+(?:\.\d{2})?)', prompt)
    if budget_match:
        budget_per_day = float(budget_match.group(1))
    
    return MealPlanConstraints(
        daily_calories=daily_calories,
        num_days=num_days,
        dietary_restrictions=dietary_restrictions,
        budget_per_day=budget_per_day if budget_per_day is not None else None
    )


def _format_meal_plan_for_display(result: MealPlanResult) -> str:
    """Format meal plan result for chat display"""
    output = []
    output.append(f"# 🍽️ Your {len(result.days)}-Day Meal Plan\n")
    
    for day in result.days:
        output.append(f"## 📅 {day.date}")
        output.append(f"**Daily Totals:** {int(day.daily_totals['calories'])} calories, {int(day.daily_totals['protein'])}g protein, ${day.daily_totals['cost']:.2f}")
        output.append("")
        
        for meal in day.meals:
            output.append(f"### {meal.meal_type.title()} - {int(meal.total_calories)} calories")
            for food in meal.foods:
                output.append(f"- {food['name']} ({food['portion']}) - {int(food['calories'])} cal")
            output.append("")
        
        if not day.meets_targets:
            output.append("⚠️ *This day doesn't fully meet your nutritional targets*")
        output.append("---")
    
    # Add summary
    summary = result.plan_summary
    output.append("## 📊 Plan Summary")
    output.append(f"- **Target Compliance:** {summary['target_compliance_rate']:.1f}%")
    if 'ga_fitness' in summary:
        output.append(f"- **GA Fitness:** {summary['ga_fitness']}")
    output.append(f"- **Average Daily Calories:** {summary['average_daily_calories']}")
    output.append(f"- **Average Daily Protein:** {summary['average_daily_protein']}g")
    output.append(f"- **Total Estimated Cost:** ${summary['total_estimated_cost']}")
    
    # Add recommendations
    if result.recommendations:
        output.append("\n## 💡 Recommendations")
        for rec in result.recommendations:
            output.append(f"- {rec}")
    
    return "\n".join(output)
