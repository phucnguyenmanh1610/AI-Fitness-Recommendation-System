"""
BMR (Basal Metabolic Rate) calculation using Harris-Benedict formula
"""
from typing import Literal


def calculate_bmr(age: int, gender: Literal["Male", "Female"], weight: float, height: float) -> float:
    """
    Calculate BMR using Harris-Benedict equation
    
    For Men: BMR = 88.362 + (13.397 × weight in kg) + (4.799 × height in cm) - (5.677 × age in years)
    For Women: BMR = 447.593 + (9.247 × weight in kg) + (3.098 × height in cm) - (4.330 × age in years)
    
    Args:
        age: Age in years
        gender: "Male" or "Female"
        weight: Weight in kilograms
        height: Height in meters (will be converted to cm)
    
    Returns:
        BMR in kcal/day
    """
    height_cm = height * 100  # Convert meters to cm
    
    if gender.lower() == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height_cm) - (5.677 * age)
    else:  # Female
        bmr = 447.593 + (9.247 * weight) + (3.098 * height_cm) - (4.330 * age)
    
    return round(bmr, 2)


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """
    Calculate TDEE (Total Daily Energy Expenditure) based on activity level
    
    Activity multipliers:
    - Sedentary (little/no exercise): BMR × 1.2
    - Lightly active (light exercise 1-3 days/week): BMR × 1.375
    - Moderately active (moderate exercise 3-5 days/week): BMR × 1.55
    - Very active (hard exercise 6-7 days/week): BMR × 1.725
    - Extremely active (very hard exercise, physical job): BMR × 1.9
    
    Args:
        bmr: Basal Metabolic Rate in kcal/day
        activity_level: "low", "moderate", or "high"
    
    Returns:
        TDEE in kcal/day
    """
    activity_multipliers = {
        "low": 1.2,
        "moderate": 1.55,
        "high": 1.725
    }
    
    multiplier = activity_multipliers.get(activity_level.lower(), 1.55)
    tdee = bmr * multiplier
    
    return round(tdee, 2)

