"""
Hybrid recommender system combining content-based and collaborative filtering
"""
import pandas as pd
import logging
from typing import Dict, Optional
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid recommender: score = 0.6 * content + 0.4 * collaborative
    """
    
    def __init__(self, content_weight: float = 0.6, collaborative_weight: float = 0.4):
        """
        Initialize hybrid recommender
        
        Args:
            content_weight: Weight for content-based score (default 0.6)
            collaborative_weight: Weight for collaborative score (default 0.4)
        """
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender(method="svd")
        
        self.items_df: Optional[pd.DataFrame] = None
    
    def load_items(self, items_path: str, user_interactions: Optional[pd.DataFrame] = None):
        """
        Load items from file or DataFrame
        
        Args:
            items_path: Path to items CSV file
            user_interactions: Optional user interaction data
        """
        import os
        if isinstance(items_path, str) and os.path.exists(items_path):
            self.items_df = pd.read_csv(items_path)
            logger.info(f"Loaded {len(self.items_df)} items from {items_path}")
        elif isinstance(items_path, pd.DataFrame):
            self.items_df = items_path.copy()
            logger.info(f"Loaded {len(self.items_df)} items from DataFrame")
        else:
            # Create synthetic items
            logger.warning("Items file not found. Creating synthetic items.")
            self._create_synthetic_items()
        
        # Initialize recommenders
        self.content_recommender.load_items(self.items_df)
        self.collaborative_recommender.load_items(self.items_df, user_interactions)
    
    def _create_synthetic_items(self):
        """Create synthetic workout items"""
        import numpy as np
        
        workout_types = [
            {"name": "Yoga", "difficulty": 2, "duration_min": 45, "focus": "flexibility", "calories_burned": 150},
            {"name": "Cardio", "difficulty": 3, "duration_min": 30, "focus": "cardio", "calories_burned": 300},
            {"name": "Strength Training", "difficulty": 4, "duration_min": 60, "focus": "strength", "calories_burned": 250},
            {"name": "HIIT", "difficulty": 5, "duration_min": 20, "focus": "hiit", "calories_burned": 400},
            {"name": "Pilates", "difficulty": 2, "duration_min": 50, "focus": "core", "calories_burned": 180},
            {"name": "Running", "difficulty": 3, "duration_min": 40, "focus": "cardio", "calories_burned": 350},
            {"name": "Cycling", "difficulty": 3, "duration_min": 45, "focus": "cardio", "calories_burned": 280},
            {"name": "Swimming", "difficulty": 4, "duration_min": 30, "focus": "full-body", "calories_burned": 320},
            {"name": "Weight Lifting", "difficulty": 4, "duration_min": 60, "focus": "strength", "calories_burned": 200},
            {"name": "Dance Fitness", "difficulty": 3, "duration_min": 45, "focus": "cardio", "calories_burned": 270}
        ]
        
        items = []
        for idx, workout in enumerate(workout_types):
            items.append({
                "plan_id": idx + 1,
                "name": workout["name"],
                "difficulty": workout["difficulty"],
                "duration_min": workout["duration_min"],
                "focus": workout["focus"],
                "calories_burned": workout["calories_burned"]
            })
        
        self.items_df = pd.DataFrame(items)
        logger.info(f"Created {len(self.items_df)} synthetic workout items")
    
    def recommend(self, user_profile: Dict, top_n: int = 5, include_score: bool = False) -> pd.DataFrame:
        """
        Get hybrid recommendations
        
        Args:
            user_profile: User profile dictionary
            top_n: Number of recommendations
            include_score: Whether to include score in results
            
        Returns:
            DataFrame with recommendations
        """
        if self.items_df is None:
            raise ValueError("Items not loaded. Call load_items() first.")
        
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend(user_profile, top_n=top_n * 2)
        
        # Get collaborative recommendations
        collaborative_recs = self.collaborative_recommender.recommend(user_profile, top_n=top_n * 2)
        
        # Merge recommendations
        all_items = self.items_df.copy()
        all_items['content_score'] = 0.0
        all_items['collaborative_score'] = 0.0
        
        # Map content scores
        for idx, row in content_recs.iterrows():
            item_id = row.get('plan_id', idx)
            mask = all_items['plan_id'] == item_id
            if mask.any():
                all_items.loc[mask, 'content_score'] = row.get('content_score', 0)
        
        # Map collaborative scores
        for idx, row in collaborative_recs.iterrows():
            item_id = row.get('plan_id', idx)
            mask = all_items['plan_id'] == item_id
            if mask.any():
                all_items.loc[mask, 'collaborative_score'] = row.get('collaborative_score', 0)
        
        # Normalize scores to 0-1 range
        if all_items['content_score'].max() > 0:
            all_items['content_score'] = all_items['content_score'] / all_items['content_score'].max()
        if all_items['collaborative_score'].max() > 0:
            all_items['collaborative_score'] = all_items['collaborative_score'] / all_items['collaborative_score'].max()
        
        # Calculate hybrid score
        all_items['score'] = (
            self.content_weight * all_items['content_score'] +
            self.collaborative_weight * all_items['collaborative_score']
        )
        
        # Sort by hybrid score
        result = all_items.sort_values('score', ascending=False).head(top_n)
        
        # Remove score columns if not requested
        if not include_score:
            result = result.drop(columns=['content_score', 'collaborative_score', 'score'], errors='ignore')
        
        return result

