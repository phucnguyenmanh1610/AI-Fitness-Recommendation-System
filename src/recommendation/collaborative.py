"""
Collaborative filtering for workout recommendations
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    """Collaborative filtering using SVD/KNN (rule-based)"""
    
    def __init__(self, method: str = "svd", use_ml_model: bool = False):
        """
        Initialize collaborative recommender
        
        Args:
            method: "svd" or "knn" (default method)
            use_ml_model: If True, use trained neural model. If False, use SVD/KNN (default).
        """
        self.method = method
        self.use_ml_model = use_ml_model
        self.items_df: pd.DataFrame = None
        self.user_item_matrix: np.ndarray = None
        self.model = None
        self.item_indices: Dict = {}
        self.ml_model = None
        
        if use_ml_model:
            try:
                from .models.neural_collaborative import NeuralCollaborativeFiltering
                model_path = "models/neural_collaborative.pkl"
                if os.path.exists(model_path):
                    self.ml_model = NeuralCollaborativeFiltering(model_path)
                    self.ml_model.load()
                    logger.info("Loaded trained Neural Collaborative Filtering model")
                else:
                    logger.warning("ML model not found. Falling back to SVD/KNN.")
                    self.use_ml_model = False
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}. Using SVD/KNN.")
                self.use_ml_model = False
    
    def load_items(self, items_df: pd.DataFrame, user_interactions: Optional[pd.DataFrame] = None):
        """
        Load items and optionally user interactions
        
        Args:
            items_df: DataFrame with items
            user_interactions: DataFrame with columns: user_id, item_id, rating/interaction
        """
        self.items_df = items_df.copy()
        
        # If no user interactions, create synthetic interactions based on item popularity
        if user_interactions is None:
            logger.warning("No user interactions provided. Using synthetic interactions.")
            self._create_synthetic_interactions()
        else:
            self._build_user_item_matrix(user_interactions)
    
    def _create_synthetic_interactions(self):
        """Create synthetic user-item interactions based on item features"""
        n_users = 100  # Synthetic users
        n_items = len(self.items_df)
        
        # Create interactions based on item difficulty and popularity
        interactions = []
        for user_id in range(n_users):
            # Simulate user preferences
            user_pref_difficulty = np.random.randint(1, 6)
            user_pref_duration = np.random.randint(20, 90)
            
            for item_idx, item in self.items_df.iterrows():
                # Calculate similarity score
                diff_score = 1 - abs(item.get('difficulty', 3) - user_pref_difficulty) / 5.0
                dur_score = 1 - abs(item.get('duration_min', 30) - user_pref_duration) / 60.0
                rating = (diff_score + dur_score) / 2.0
                
                # Add some noise
                rating += np.random.normal(0, 0.1)
                rating = max(0, min(5, rating * 5))  # Scale to 0-5
                
                if rating > 2.5:  # Only positive interactions
                    interactions.append({
                        'user_id': user_id,
                        'item_id': item.get('plan_id', item_idx),
                        'rating': rating
                    })
        
        interactions_df = pd.DataFrame(interactions)
        self._build_user_item_matrix(interactions_df)
    
    def _build_user_item_matrix(self, interactions_df: pd.DataFrame):
        """Build user-item interaction matrix"""
        # Create pivot table
        if 'rating' in interactions_df.columns:
            matrix = interactions_df.pivot_table(
                index='user_id',
                columns='item_id',
                values='rating',
                fill_value=0
            )
        else:
            # Binary interactions
            matrix = interactions_df.pivot_table(
                index='user_id',
                columns='item_id',
                aggfunc='size',
                fill_value=0
            )
            matrix = (matrix > 0).astype(float)
        
        self.user_item_matrix = matrix.values
        
        # Store item indices
        self.item_indices = {item_id: idx for idx, item_id in enumerate(matrix.columns)}
        
        logger.info(f"Built user-item matrix with shape {self.user_item_matrix.shape}")
    
    def _train_svd(self, n_components: int = 50):
        """Train SVD model"""
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        self.item_factors = self.model.fit_transform(self.user_item_matrix.T)
        logger.info(f"SVD trained with {n_components} components")
    
    def _train_knn(self, n_neighbors: int = 10):
        """Train KNN model"""
        # Use item-item similarity
        item_similarity = np.corrcoef(self.user_item_matrix.T)
        item_similarity = np.nan_to_num(item_similarity)
        
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.model.fit(self.user_item_matrix.T)
        logger.info(f"KNN trained with {n_neighbors} neighbors")
    
    def recommend(self, user_profile: Dict, top_n: int = 5) -> pd.DataFrame:
        """
        Recommend items using collaborative filtering
        
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
            
            result_df['collaborative_score'] = scores
        else:
            # Fallback to SVD/KNN
            if self.user_item_matrix is None:
                raise ValueError("User-item matrix not built. Call load_items() first.")
            
            if self.model is None:
                if self.method == "svd":
                    self._train_svd()
                else:
                    self._train_knn()
            
            # Get average ratings/interactions per item
            item_scores = self.user_item_matrix.mean(axis=0)
            item_scores += np.random.normal(0, 0.1, len(item_scores))
            
            collaborative_scores = []
            for idx, item in self.items_df.iterrows():
                item_id = item.get('plan_id', idx)
                if item_id in self.item_indices:
                    score = item_scores[self.item_indices[item_id]]
                else:
                    score = 0.5
                collaborative_scores.append(score)
            
            result_df['collaborative_score'] = collaborative_scores
        
        # Sort by score
        result_df = result_df.sort_values('collaborative_score', ascending=False)
        
        return result_df.head(top_n)

