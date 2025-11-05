import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_items() -> pd.DataFrame:
    """Load recommendation items (exercises/meals) from CSV/JSON."""
    # Placeholder: Create synthetic items
    items = pd.DataFrame({
        'name': ['Running', 'Weight Lifting', 'Salad', 'Protein Shake'],
        'type': ['exercise', 'exercise', 'meal', 'meal'],
        'vector': [[0.8, 0.2, 0.5], [0.3, 0.9, 0.4], [0.1, 0.1, 0.9], [0.4, 0.7, 0.6]]  # Feature vectors
    })
    return items

def recommend_plans(user_profile: np.ndarray, items_df: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """
    Hybrid recommender: Cosine similarity for content-based.
    TODO: Add collaborative filtering.
    :param user_profile: User vector (np array)
    :param items_df: DF with 'vector' column
    :param top_k: Number of recommendations
    :return: List of dicts [{'name': str, 'type': str, ...}]
    """
    sim = cosine_similarity([user_profile], np.stack(items_df['vector']))
    top_indices = sim[0].argsort()[-top_k:][::-1]
    recommendations = items_df.iloc[top_indices].to_dict(orient='records')
    logger.info(f"Generated {top_k} recommendations")
    return recommendations