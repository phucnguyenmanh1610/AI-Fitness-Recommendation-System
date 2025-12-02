"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from enum import Enum


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class ActivityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Goal(str, Enum):
    LOSS = "loss"
    GAIN = "gain"
    MAINTAIN = "maintain"


# ========== Request Schemas ==========

class HealthPredictionRequest(BaseModel):
    """Request schema for health prediction"""
    age: int = Field(..., ge=1, le=120, description="Age in years")
    gender: Gender = Field(..., description="Gender")
    height: float = Field(..., gt=0, le=3.0, description="Height in meters")
    weight: float = Field(..., gt=0, le=300, description="Weight in kilograms")
    steps: Optional[int] = Field(None, ge=0, description="Daily steps")
    heart_rate: Optional[int] = Field(None, ge=30, le=220, description="Heart rate (bpm)")
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Sleep hours per day")
    activity_level: Optional[ActivityLevel] = Field(None, description="Activity level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 30,
                "gender": "Male",
                "height": 1.75,
                "weight": 75.0,
                "steps": 8000,
                "heart_rate": 72,
                "sleep_hours": 7.5,
                "activity_level": "moderate"
            }
        }


class WorkoutRecommendationRequest(BaseModel):
    """Request schema for workout recommendation"""
    age: int = Field(..., ge=1, le=120)
    gender: Gender
    height: float = Field(..., gt=0, le=3.0)
    weight: float = Field(..., gt=0, le=300)
    goal: Goal = Field(..., description="Fitness goal: loss, gain, or maintain")
    experience_level: Optional[int] = Field(1, ge=1, le=3, description="Experience level: 1=beginner, 2=intermediate, 3=advanced")
    workout_frequency: Optional[int] = Field(3, ge=1, le=7, description="Workouts per week")
    preferred_duration: Optional[int] = Field(30, ge=15, le=120, description="Preferred workout duration in minutes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 30,
                "gender": "Male",
                "height": 1.75,
                "weight": 75.0,
                "goal": "loss",
                "experience_level": 2,
                "workout_frequency": 4,
                "preferred_duration": 45
            }
        }


class MealRecommendationRequest(BaseModel):
    """Request schema for meal recommendation"""
    age: int = Field(..., ge=1, le=120)
    gender: Gender
    height: float = Field(..., gt=0, le=3.0)
    weight: float = Field(..., gt=0, le=300)
    goal: Goal = Field(..., description="Fitness goal")
    activity_level: ActivityLevel = Field(..., description="Activity level")
    target_calories: Optional[float] = Field(None, gt=0, description="Target daily calories (if not provided, will be calculated)")
    meals_per_day: Optional[int] = Field(3, ge=1, le=6, description="Number of meals per day")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 30,
                "gender": "Male",
                "height": 1.75,
                "weight": 75.0,
                "goal": "loss",
                "activity_level": "moderate",
                "meals_per_day": 3
            }
        }


# ========== Response Schemas ==========

class HealthMetrics(BaseModel):
    """Health metrics response"""
    bmi: float = Field(..., description="Body Mass Index")
    bmr: float = Field(..., description="Basal Metabolic Rate (kcal/day)")
    calories_burned: float = Field(..., description="Estimated calories burned per day")
    tdee: Optional[float] = Field(None, description="Total Daily Energy Expenditure")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bmi": 24.5,
                "bmr": 1800.5,
                "calories_burned": 2200.0,
                "tdee": 2200.0
            }
        }


class HealthPredictionResponse(BaseModel):
    """Response schema for health prediction"""
    success: bool = True
    metrics: HealthMetrics
    message: Optional[str] = None


class WorkoutItem(BaseModel):
    """Workout item schema"""
    plan_id: int
    name: str
    difficulty: int
    duration_min: int
    focus: str
    calories_burned: Optional[float] = None
    description: Optional[str] = None
    score: Optional[float] = None


class WorkoutRecommendationResponse(BaseModel):
    """Response schema for workout recommendation"""
    success: bool = True
    recommendations: List[WorkoutItem]
    goal: str
    message: Optional[str] = None


class MealItem(BaseModel):
    """Meal item schema"""
    meal_id: int
    name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    meal_type: str  # breakfast, lunch, dinner, snack
    description: Optional[str] = None


class MealPlan(BaseModel):
    """Daily meal plan"""
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    meals: List[MealItem]


class MealRecommendationResponse(BaseModel):
    """Response schema for meal recommendation"""
    success: bool = True
    meal_plan: MealPlan
    goal: str
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    message: Optional[str] = None

