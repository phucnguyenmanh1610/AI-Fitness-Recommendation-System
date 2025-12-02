"""
User Profile model
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Goal(str, Enum):
    LOSS = "loss"
    GAIN = "gain"
    MAINTAIN = "maintain"


@dataclass
class UserProfile:
    """User profile data model"""
    age: int
    gender: str  # "Male" or "Female"
    height: float  # in meters
    weight: float  # in kilograms
    goal: Goal
    bmi: Optional[float] = None
    bmr: Optional[float] = None
    activity_level: Optional[str] = None
    experience_level: Optional[int] = None
    workout_frequency: Optional[int] = None
    
    def __post_init__(self):
        """Calculate BMI if not provided"""
        if self.bmi is None:
            self.bmi = self.weight / (self.height ** 2)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "age": self.age,
            "gender": self.gender,
            "height": self.height,
            "weight": self.weight,
            "goal": self.goal.value if isinstance(self.goal, Goal) else self.goal,
            "bmi": self.bmi,
            "bmr": self.bmr,
            "activity_level": self.activity_level,
            "experience_level": self.experience_level,
            "workout_frequency": self.workout_frequency
        }


@dataclass
class ActivityData:
    """Activity data model"""
    steps: Optional[int] = None
    heart_rate: Optional[int] = None
    sleep_hours: Optional[float] = None
    calories_intake: Optional[float] = None
    workout_type: Optional[str] = None
    session_duration: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "steps": self.steps,
            "heart_rate": self.heart_rate,
            "sleep_hours": self.sleep_hours,
            "calories_intake": self.calories_intake,
            "workout_type": self.workout_type,
            "session_duration": self.session_duration
        }

