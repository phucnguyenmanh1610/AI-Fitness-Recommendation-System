"""
Meal recommendation system
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Optional
from ..models.schemas import MealPlan, MealItem

logger = logging.getLogger(__name__)


class MealRecommender:
    """Meal recommendation system based on calorie and macronutrient needs"""
    
    def __init__(self):
        self.meals_df: Optional[pd.DataFrame] = None
    
    def load_meals(self, meals_path: str):
        """
        Load meal database from file
        
        Args:
            meals_path: Path to meals CSV file
        """
        if isinstance(meals_path, str) and os.path.exists(meals_path):
            self.meals_df = pd.read_csv(meals_path)
            logger.info(f"Loaded {len(self.meals_df)} meals from {meals_path}")
        else:
            # Create synthetic meal database
            logger.warning("Meals file not found. Creating synthetic meal database.")
            self._create_synthetic_meals()
    
    def _create_synthetic_meals(self):
        """Create synthetic meal database"""
        meals_data = [
            # Breakfast
            {"meal_id": 1, "name": "Oatmeal with Berries", "meal_type": "breakfast", 
             "calories": 350, "protein": 12, "carbs": 55, "fat": 8, 
             "description": "Whole grain oatmeal with fresh berries and nuts"},
            {"meal_id": 2, "name": "Greek Yogurt Parfait", "meal_type": "breakfast",
             "calories": 280, "protein": 20, "carbs": 35, "fat": 6,
             "description": "Greek yogurt with granola and fruits"},
            {"meal_id": 3, "name": "Scrambled Eggs with Toast", "meal_type": "breakfast",
             "calories": 320, "protein": 18, "carbs": 30, "fat": 12,
             "description": "2 eggs scrambled with whole grain toast"},
            {"meal_id": 4, "name": "Smoothie Bowl", "meal_type": "breakfast",
             "calories": 400, "protein": 15, "carbs": 60, "fat": 10,
             "description": "Acai smoothie bowl with toppings"},
            
            # Lunch
            {"meal_id": 5, "name": "Grilled Chicken Salad", "meal_type": "lunch",
             "calories": 450, "protein": 35, "carbs": 25, "fat": 20,
             "description": "Grilled chicken breast with mixed greens"},
            {"meal_id": 6, "name": "Quinoa Bowl", "meal_type": "lunch",
             "calories": 500, "protein": 20, "carbs": 65, "fat": 15,
             "description": "Quinoa with vegetables and protein"},
            {"meal_id": 7, "name": "Turkey Wrap", "meal_type": "lunch",
             "calories": 380, "protein": 25, "carbs": 40, "fat": 12,
             "description": "Whole wheat wrap with turkey and vegetables"},
            {"meal_id": 8, "name": "Salmon with Rice", "meal_type": "lunch",
             "calories": 550, "protein": 40, "carbs": 50, "fat": 18,
             "description": "Grilled salmon with brown rice and vegetables"},
            
            # Dinner
            {"meal_id": 9, "name": "Lean Beef Steak", "meal_type": "dinner",
             "calories": 600, "protein": 50, "carbs": 10, "fat": 35,
             "description": "Lean beef steak with sweet potato and vegetables"},
            {"meal_id": 10, "name": "Baked Chicken Breast", "meal_type": "dinner",
             "calories": 450, "protein": 45, "carbs": 30, "fat": 15,
             "description": "Baked chicken with quinoa and roasted vegetables"},
            {"meal_id": 11, "name": "Vegetable Stir Fry", "meal_type": "dinner",
             "calories": 400, "protein": 15, "carbs": 55, "fat": 12,
             "description": "Mixed vegetables with tofu and brown rice"},
            {"meal_id": 12, "name": "Fish Tacos", "meal_type": "dinner",
             "calories": 500, "protein": 30, "carbs": 45, "fat": 18,
             "description": "Grilled fish tacos with vegetables"},
            
            # Snacks
            {"meal_id": 13, "name": "Protein Shake", "meal_type": "snack",
             "calories": 200, "protein": 25, "carbs": 15, "fat": 3,
             "description": "Whey protein shake with banana"},
            {"meal_id": 14, "name": "Mixed Nuts", "meal_type": "snack",
             "calories": 250, "protein": 8, "carbs": 10, "fat": 20,
             "description": "Almonds, walnuts, and cashews"},
            {"meal_id": 15, "name": "Apple with Peanut Butter", "meal_type": "snack",
             "calories": 220, "protein": 6, "carbs": 25, "fat": 12,
             "description": "Apple slices with natural peanut butter"},
        ]
        
        self.meals_df = pd.DataFrame(meals_data)
        logger.info(f"Created {len(self.meals_df)} synthetic meals")
    
    def recommend_meal_plan(self, target_calories: float, goal: str = "maintain",
                           meals_per_day: int = 3) -> MealPlan:
        """
        Recommend a daily meal plan
        
        Args:
            target_calories: Target daily calories
            goal: "loss", "gain", or "maintain"
            meals_per_day: Number of meals per day (3-6)
            
        Returns:
            MealPlan object
        """
        if self.meals_df is None:
            raise ValueError("Meals not loaded. Call load_meals() first.")
        
        # Adjust calories based on goal
        if goal == "loss":
            actual_target = target_calories * 0.95  # Slight deficit
        elif goal == "gain":
            actual_target = target_calories * 1.05  # Slight surplus
        else:
            actual_target = target_calories
        
        # Calculate calories per meal
        calories_per_meal = actual_target / meals_per_day
        
        # Select meals based on meal type distribution
        meal_distribution = self._get_meal_distribution(meals_per_day)
        
        selected_meals = []
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        for meal_type, count in meal_distribution.items():
            # Filter meals by type
            type_meals = self.meals_df[self.meals_df['meal_type'] == meal_type].copy()
            
            if type_meals.empty:
                continue
            
            # Select meals closest to target calories for this meal type
            type_meals['cal_diff'] = abs(type_meals['calories'] - calories_per_meal)
            type_meals = type_meals.sort_values('cal_diff')
            
            for _ in range(count):
                if len(type_meals) > 0:
                    meal = type_meals.iloc[0].to_dict()
                    # Handle meal_id - can be string (M1, M131) or int
                    meal_id = meal.get('meal_id', '')
                    if isinstance(meal_id, str):
                        meal_id_value = meal_id
                    else:
                        try:
                            meal_id_value = int(meal_id)
                        except (ValueError, TypeError):
                            meal_id_value = str(meal_id)
                    
                    selected_meals.append(MealItem(
                        meal_id=meal_id_value,
                        name=str(meal['name']),
                        calories=float(meal['calories']),
                        protein=float(meal['protein']),
                        carbs=float(meal['carbs']),
                        fat=float(meal['fat']),
                        meal_type=str(meal['meal_type']),
                        description=meal.get('description')
                    ))
                    
                    total_calories += meal['calories']
                    total_protein += meal['protein']
                    total_carbs += meal['carbs']
                    total_fat += meal['fat']
                    
                    # Remove selected meal to avoid duplicates
                    type_meals = type_meals.iloc[1:]
        
        # If we need more meals, add snacks
        while len(selected_meals) < meals_per_day:
            snacks = self.meals_df[self.meals_df['meal_type'] == 'snack']
            if len(snacks) > 0:
                snack = snacks.sample(1).iloc[0].to_dict()
                # Handle meal_id - can be string (M1, M131) or int
                snack_id = snack.get('meal_id', '')
                if isinstance(snack_id, str):
                    snack_id_value = snack_id
                else:
                    try:
                        snack_id_value = int(snack_id)
                    except (ValueError, TypeError):
                        snack_id_value = str(snack_id)
                
                selected_meals.append(MealItem(
                    meal_id=snack_id_value,
                    name=str(snack['name']),
                    calories=float(snack['calories']),
                    protein=float(snack['protein']),
                    carbs=float(snack['carbs']),
                    fat=float(snack['fat']),
                    meal_type=str(snack['meal_type']),
                    description=snack.get('description')
                ))
                total_calories += snack['calories']
                total_protein += snack['protein']
                total_carbs += snack['carbs']
                total_fat += snack['fat']
            else:
                break
        
        return MealPlan(
            total_calories=round(total_calories, 2),
            total_protein=round(total_protein, 2),
            total_carbs=round(total_carbs, 2),
            total_fat=round(total_fat, 2),
            meals=selected_meals
        )
    
    def _get_meal_distribution(self, meals_per_day: int) -> Dict[str, int]:
        """Get meal type distribution based on number of meals per day"""
        if meals_per_day == 3:
            return {"breakfast": 1, "lunch": 1, "dinner": 1}
        elif meals_per_day == 4:
            return {"breakfast": 1, "lunch": 1, "dinner": 1, "snack": 1}
        elif meals_per_day == 5:
            return {"breakfast": 1, "lunch": 1, "dinner": 1, "snack": 2}
        elif meals_per_day == 6:
            return {"breakfast": 1, "lunch": 1, "dinner": 1, "snack": 3}
        else:
            # Default to 3 meals
            return {"breakfast": 1, "lunch": 1, "dinner": 1}

