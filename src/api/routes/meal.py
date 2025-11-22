"""
Meal recommendation endpoint
"""
from fastapi import APIRouter, HTTPException
import logging

from ...models.schemas import (
    MealRecommendationRequest,
    MealRecommendationResponse,
    MealPlan
)
from ...recommendation.meal_recommender import MealRecommender
from ...prediction.bmr_calculator import calculate_bmr, calculate_tdee
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize meal recommender (lazy loading)
_meal_recommender: MealRecommender = None


def get_meal_recommender() -> MealRecommender:
    """Get or initialize meal recommender"""
    global _meal_recommender
    if _meal_recommender is None:
        _meal_recommender = MealRecommender()
        try:
            _meal_recommender.load_meals(settings.MEAL_DATABASE_PATH)
            logger.info("Meal recommender initialized")
        except Exception as e:
            logger.warning(f"Error loading meal database: {e}")
    return _meal_recommender


@router.post("/recommend/meal", response_model=MealRecommendationResponse)
async def recommend_meal(request: MealRecommendationRequest):
    """
    Recommend meal plan based on user profile, goals, and calorie needs
    
    - **goal**: loss, gain, or maintain
    - **activity_level**: low, moderate, or high
    - **target_calories**: Target daily calories (optional, will be calculated if not provided)
    - **meals_per_day**: Number of meals per day (1-6)
    """
    try:
        meal_recommender = get_meal_recommender()
        
        # Calculate target calories if not provided
        if request.target_calories is None:
            bmr = calculate_bmr(
                age=request.age,
                gender=request.gender.value,
                weight=request.weight,
                height=request.height
            )
            tdee = calculate_tdee(bmr, request.activity_level.value)
            
            # Adjust based on goal
            if request.goal.value == "loss":
                target_calories = tdee * 0.85  # 15% deficit
            elif request.goal.value == "gain":
                target_calories = tdee * 1.15  # 15% surplus
            else:  # maintain
                target_calories = tdee
        else:
            target_calories = request.target_calories
        
        # Get meal plan
        meal_plan = meal_recommender.recommend_meal_plan(
            target_calories=target_calories,
            goal=request.goal.value,
            meals_per_day=request.meals_per_day or 3
        )
        
        return MealRecommendationResponse(
            success=True,
            meal_plan=meal_plan,
            goal=request.goal.value,
            message=f"Generated meal plan with {len(meal_plan.meals)} meals"
        )
        
    except Exception as e:
        logger.error(f"Error in meal recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating meal recommendations: {str(e)}"
        )

