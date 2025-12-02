"""
Health prediction endpoint
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from ...models.schemas import (
    HealthPredictionRequest,
    HealthPredictionResponse,
    HealthMetrics,
    ErrorResponse
)
from ...prediction.bmr_calculator import calculate_bmr, calculate_tdee
from ...prediction.models.calorie_predictor import CaloriePredictor
from ...prediction.models.bmi_predictor import BMIPredictor
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize models (lazy loading)
_calorie_predictor: Optional[CaloriePredictor] = None
_bmi_predictor: Optional[BMIPredictor] = None


def get_calorie_predictor() -> CaloriePredictor:
    """Get or load calorie predictor"""
    global _calorie_predictor
    if _calorie_predictor is None:
        _calorie_predictor = CaloriePredictor()
        try:
            _calorie_predictor.load()
            logger.info("Calorie predictor model loaded")
        except FileNotFoundError:
            logger.warning("Calorie model not found. Train model first.")
            raise HTTPException(
                status_code=503,
                detail="Calorie prediction model not available. Please train the model first."
            )
    return _calorie_predictor


def get_bmi_predictor() -> BMIPredictor:
    """Get or load BMI predictor"""
    global _bmi_predictor
    if _bmi_predictor is None:
        _bmi_predictor = BMIPredictor()
        try:
            _bmi_predictor.load()
            logger.info("BMI predictor model loaded")
        except FileNotFoundError:
            logger.warning("BMI model not found. Using calculated BMI.")
    return _bmi_predictor


@router.post("/predict/health", response_model=HealthPredictionResponse)
async def predict_health(request: HealthPredictionRequest):
    """
    Predict health metrics: BMI, BMR, and calories burned
    
    - **age**: Age in years (1-120)
    - **gender**: Male or Female
    - **height**: Height in meters
    - **weight**: Weight in kilograms
    - **steps**: Daily steps (optional)
    - **heart_rate**: Heart rate in bpm (optional)
    - **sleep_hours**: Sleep hours per day (optional)
    - **activity_level**: low, moderate, or high (optional)
    """
    try:
        # Calculate BMI (use model if available, otherwise calculate)
        bmi_predictor = get_bmi_predictor()
        try:
            bmi = bmi_predictor.predict({
                "age": request.age,
                "gender": request.gender.value,
                "height": request.height,
                "weight": request.weight
            })
        except:
            # Fallback to calculated BMI
            bmi = request.weight / (request.height ** 2)
        
        # Calculate BMR using Harris-Benedict formula
        bmr = calculate_bmr(
            age=request.age,
            gender=request.gender.value,
            weight=request.weight,
            height=request.height
        )
        
        # Predict calories burned
        activity_level = request.activity_level.value if request.activity_level else "moderate"
        
        # Prepare features for calorie prediction
        features = {
            "age": request.age,
            "gender": request.gender.value,
            "height": request.height,
            "weight": request.weight,
            "activity_level": activity_level,
            "steps": request.steps or 0,
            "heart_rate": request.heart_rate or 0,
            "sleep_hours": request.sleep_hours or 0
        }
        
        try:
            calorie_predictor = get_calorie_predictor()
            calories_burned = calorie_predictor.predict(features)
        except HTTPException:
            # Fallback: use TDEE if model not available
            calories_burned = calculate_tdee(bmr, activity_level)
            logger.warning("Using TDEE as fallback for calories burned")
        
        # Calculate TDEE
        tdee = calculate_tdee(bmr, activity_level)
        
        metrics = HealthMetrics(
            bmi=round(bmi, 2),
            bmr=round(bmr, 2),
            calories_burned=round(calories_burned, 2),
            tdee=round(tdee, 2)
        )
        
        return HealthPredictionResponse(
            success=True,
            metrics=metrics,
            message="Health metrics predicted successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in health prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting health metrics: {str(e)}"
        )

