"""
Content-based filtering for workout recommendations
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based filtering using cosine similarity (rule-based)"""
    
    def __init__(self, use_ml_model: bool = False):
        """
        Initialize content-based recommender
        
        Args:
            use_ml_model: If True, use trained neural model. If False, use cosine similarity (default).
        """
        self.use_ml_model = use_ml_model
        self.items_df: pd.DataFrame = None
        self.feature_matrix: np.ndarray = None
        self.feature_names: List[str] = []
        self.ml_model = None
        
        if use_ml_model:
            try:
                from .models.neural_content_based import NeuralContentBased
                model_path = "models/neural_content_based.pkl"
                if os.path.exists(model_path):
                    self.ml_model = NeuralContentBased(model_path)
                    self.ml_model.load()
                    logger.info("Loaded trained Neural Content-Based model")
                else:
                    logger.warning("ML model not found. Falling back to cosine similarity.")
                    self.use_ml_model = False
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}. Using cosine similarity.")
                self.use_ml_model = False
    
    def load_items(self, items_df: pd.DataFrame):
        """Load items dataframe"""
        self.items_df = items_df.copy()
        self._build_feature_matrix()
    
    def _build_feature_matrix(self):
        """Build feature matrix from items"""
        if self.items_df is None or self.items_df.empty:
            raise ValueError("Items dataframe is empty")
        
        # Extract features for content-based filtering
        features = []
        
        # Normalize difficulty (1-5 scale)
        if 'difficulty' in self.items_df.columns:
            features.append(self.items_df['difficulty'].values / 5.0)
        
        # Normalize duration (assume max 120 minutes)
        if 'duration_min' in self.items_df.columns:
            features.append(self.items_df['duration_min'].values / 120.0)
        
        # Encode focus type
        if 'focus' in self.items_df.columns:
            focus_types = self.items_df['focus'].unique()
            for focus_type in focus_types:
                features.append((self.items_df['focus'] == focus_type).astype(float).values)
        
        # Calories burned (if available)
        if 'calories_burned' in self.items_df.columns:
            max_cal = self.items_df['calories_burned'].max() or 1
            features.append(self.items_df['calories_burned'].values / max_cal)
        
        if not features:
            raise ValueError("No features available for content-based filtering")
        
        # Stack features
        self.feature_matrix = np.column_stack(features)
        logger.info(f"Built feature matrix with shape {self.feature_matrix.shape}")
    
    def _create_user_profile(self, user_profile: Dict) -> np.ndarray:
        """Create user profile vector from user preferences"""
        profile_vector = []
        
        # Difficulty preference (from experience_level: 1->2, 2->3, 3->4)
        exp_level = user_profile.get('experience_level', 2)
        difficulty_pref = (exp_level + 1) / 5.0
        profile_vector.append(difficulty_pref)
        
        # Duration preference
        pref_duration = user_profile.get('preferred_duration', 30)
        duration_pref = pref_duration / 120.0
        profile_vector.append(duration_pref)
        
        # Focus preferences based on goal
        goal = user_profile.get('goal', 'maintain')
        focus_prefs = {
            'loss': {'cardio': 0.8, 'hiit': 0.7, 'strength': 0.3, 'yoga': 0.4},
            'gain': {'strength': 0.9, 'hiit': 0.6, 'cardio': 0.2, 'yoga': 0.3},
            'maintain': {'cardio': 0.5, 'strength': 0.5, 'yoga': 0.6, 'hiit': 0.4}
        }
        
        focus_map = focus_prefs.get(goal, focus_prefs['maintain'])
        if self.items_df is not None and 'focus' in self.items_df.columns:
            focus_types = self.items_df['focus'].unique()
            for focus_type in focus_types:
                focus_lower = focus_type.lower()
                score = 0.3  # default
                for key, value in focus_map.items():
                    if key in focus_lower:
                        score = value
                        break
                profile_vector.append(score)
        
        # Calories preference (based on goal)
        if goal == 'loss':
            cal_pref = 0.7  # Higher calorie burn
        elif goal == 'gain':
            cal_pref = 0.4  # Lower calorie burn
        else:
            cal_pref = 0.5  # Moderate
        
        # Add calories preference if we have calories in items
        if self.items_df is not None and 'calories_burned' in self.items_df.columns:
            profile_vector.append(cal_pref)
        
        return np.array(profile_vector)
    
    def recommend(self, user_profile: Dict, top_n: int = 5) -> pd.DataFrame:
        """
        Recommend items based on content similarity
        
        Args:
            user_profile: User profile dictionary
            top_n: Number of recommendations
            
        Returns:
            DataFrame with recommendations and scores
        """
        if self.items_df is None:
            raise ValueError("Items not loaded. Call load_items() first.")
        
        result_df = self.items_df.copy()
        
        if self.use_ml_model and self.ml_model is not None:
            # Use ML model for prediction
            scores = []
            for _, item in self.items_df.iterrows():
                # Prepare user features
                user_feat = {
                    'age': user_profile.get('age', 30),
                    'gender': 0 if user_profile.get('gender', 'Male') == 'Male' else 1,
                    'height': user_profile.get('height', 1.7),
                    'weight': user_profile.get('weight', 70),
                    'activity_level': {'low': 0, 'moderate': 1, 'high': 2}.get(
                        str(user_profile.get('activity_level', 'moderate')).lower(), 1
                    ),
                    'experience_level': user_profile.get('experience_level', 2),
                    'workout_frequency': user_profile.get('workout_frequency', 3),
                }
                
                # Goal encoding
                goal = user_profile.get('goal', 'maintain')
                user_feat['goal_loss'] = 1 if goal == 'loss' else 0
                user_feat['goal_gain'] = 1 if goal == 'gain' else 0
                user_feat['goal_maintain'] = 1 if goal == 'maintain' else 0
                
                # Prepare item features
                item_feat = {
                    'difficulty': item.get('difficulty', 3),
                    'duration_min': item.get('duration_min', 30),
                    'calories_burned': item.get('calories_burned', 200),
                }
                
                # Focus one-hot encoding
                focus = str(item.get('focus', '')).lower()
                for focus_type in ['cardio', 'strength', 'yoga', 'hiit', 'core', 'flexibility', 'full-body']:
                    item_feat[f'focus_{focus_type}'] = 1 if focus_type in focus else 0
                
                # Predict rating
                score = self.ml_model.predict(user_feat, item_feat)
                scores.append(score / 5.0)  # Normalize to 0-1
            
            result_df['content_score'] = scores
        else:
            # Fallback to cosine similarity
            if self.feature_matrix is None:
                self._build_feature_matrix()
            
            user_vector = self._create_user_profile(user_profile)
            
            if len(user_vector) != self.feature_matrix.shape[1]:
                if len(user_vector) < self.feature_matrix.shape[1]:
                    user_vector = np.pad(user_vector, (0, self.feature_matrix.shape[1] - len(user_vector)), 'constant')
                else:
                    user_vector = user_vector[:self.feature_matrix.shape[1]]
            
            user_vector = user_vector.reshape(1, -1)
            similarities = cosine_similarity(user_vector, self.feature_matrix)[0]
            result_df['content_score'] = similarities
        
        # Sort by score
        result_df = result_df.sort_values('content_score', ascending=False)
        
        # Return top N
        return result_df.head(top_n)

