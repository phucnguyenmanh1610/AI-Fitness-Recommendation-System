"""
Workout recommendation endpoint
"""
from fastapi import APIRouter, HTTPException
import logging

from ...models.schemas import (
    WorkoutRecommendationRequest,
    WorkoutRecommendationResponse,
    WorkoutItem
)
from ...recommendation.hybrid_recommender import HybridRecommender
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize recommender (lazy loading)
_recommender: HybridRecommender = None


def get_recommender() -> HybridRecommender:
    """Get or initialize recommender"""
    global _recommender
    if _recommender is None:
        _recommender = HybridRecommender()
        try:
            _recommender.load_items(settings.WORKOUT_ITEMS_PATH)
            logger.info("Workout recommender initialized")
        except Exception as e:
            logger.warning(f"Error loading workout items: {e}")
    return _recommender


@router.post("/recommend/workout", response_model=WorkoutRecommendationResponse)
async def recommend_workout(request: WorkoutRecommendationRequest):
    """
    Recommend workout plans based on user profile and goals
    
    - **goal**: loss, gain, or maintain
    - **experience_level**: 1=beginner, 2=intermediate, 3=advanced
    - **workout_frequency**: Workouts per week (1-7)
    - **preferred_duration**: Preferred workout duration in minutes
    """
    try:
        recommender = get_recommender()
        
        # Create user profile
        user_profile = {
            "age": request.age,
            "gender": request.gender.value,
            "height": request.height,
            "weight": request.weight,
            "goal": request.goal.value,
            "experience_level": request.experience_level or 2,
            "workout_frequency": request.workout_frequency or 3,
            "preferred_duration": request.preferred_duration or 30
        }
        
        # Get recommendations
        recommendations = recommender.recommend(
            user_profile=user_profile,
            top_n=5,
            include_score=True
        )
        
        # Convert to response format
        workout_items = []
        for idx, rec in recommendations.iterrows():
            workout_items.append(WorkoutItem(
                plan_id=int(rec.get('plan_id', idx)),
                name=str(rec.get('name', 'Unknown')),
                difficulty=int(rec.get('difficulty', 2)),
                duration_min=int(rec.get('duration_min', 30)),
                focus=str(rec.get('focus', 'general')),
                calories_burned=float(rec.get('calories_burned', 0)) if 'calories_burned' in rec else None,
                description=rec.get('description'),
                score=float(rec.get('score', 0)) if 'score' in rec else None
            ))
        
        return WorkoutRecommendationResponse(
            success=True,
            recommendations=workout_items,
            goal=request.goal.value,
            message=f"Found {len(workout_items)} workout recommendations"
        )
        
    except Exception as e:
        logger.error(f"Error in workout recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating workout recommendations: {str(e)}"
        )

