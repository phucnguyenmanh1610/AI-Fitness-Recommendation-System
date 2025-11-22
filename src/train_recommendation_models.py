"""
Script to train recommendation ML models
"""
import os
import sys
import logging
import numpy as np
import pandas as pd

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_input.input import load_data
from src.data_input.preprocess import preprocess_data
from src.recommendation.models.neural_collaborative import NeuralCollaborativeFiltering
from src.recommendation.models.neural_content_based import NeuralContentBased
from src.api.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_training_data(df: pd.DataFrame, items_df: pd.DataFrame) -> tuple:
    """
    Generate training data for recommendation models from fitness data
    
    Args:
        df: Fitness data with user information
        items_df: Workout items dataframe
        
    Returns:
        user_item_data, user_features, item_features
    """
    logger.info("Generating training data for recommendation models...")
    
    # Create user features from fitness data
    user_features_list = []
    user_item_interactions = []
    
    # Process each user (row in fitness data)
    for idx, row in df.iterrows():
        user_id = idx
        
        # Extract user features
        user_feat = {
            'user_id': user_id,
            'age': row.get('age', 30),
            'gender': 0 if row.get('gender', 'Male') == 'Male' else 1,
            'height': row.get('height', 1.7),
            'weight': row.get('weight', 70),
            'bmi': row.get('bmi', 22),
            'activity_level': {'low': 0, 'moderate': 1, 'high': 2}.get(
                str(row.get('activity_level', 'moderate')).lower(), 1
            ),
            'experience_level': row.get('experience_level', 2),
            'workout_frequency': row.get('workout_frequency', 3),
        }
        
        # Determine goal based on BMI
        bmi = user_feat['bmi']
        if bmi < 18.5:
            goal = 'gain'  # Underweight
        elif bmi > 25:
            goal = 'loss'  # Overweight
        else:
            goal = 'maintain'  # Normal
        
        user_feat['goal_loss'] = 1 if goal == 'loss' else 0
        user_feat['goal_gain'] = 1 if goal == 'gain' else 0
        user_feat['goal_maintain'] = 1 if goal == 'maintain' else 0
        
        user_features_list.append(user_feat)
        
        # Generate interactions for this user
        # Simulate user preferences based on their profile
        preferred_difficulty = user_feat['experience_level']  # 1-3
        preferred_duration = 30 + (user_feat['workout_frequency'] - 1) * 10  # 30-90 min
        
        for _, item in items_df.iterrows():
            item_id = item.get('plan_id', 0)
            
            # Calculate compatibility score
            diff_score = 1 - abs(item.get('difficulty', 3) - (preferred_difficulty + 1)) / 5.0
            dur_score = 1 - abs(item.get('duration_min', 30) - preferred_duration) / 60.0
            
            # Goal-based preference
            focus = str(item.get('focus', '')).lower()
            goal_score = 0.5
            if goal == 'loss' and ('cardio' in focus or 'hiit' in focus):
                goal_score = 0.9
            elif goal == 'gain' and 'strength' in focus:
                goal_score = 0.9
            elif goal == 'maintain':
                goal_score = 0.7
            
            # Calculate final rating
            rating = (diff_score * 0.3 + dur_score * 0.3 + goal_score * 0.4) * 5
            rating += np.random.normal(0, 0.3)  # Add noise
            rating = max(1, min(5, rating))  # Clamp to 1-5
            
            # Only include positive interactions (rating >= 3)
            if rating >= 3.0:
                user_item_interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating
                })
    
    # Create DataFrames
    user_features_df = pd.DataFrame(user_features_list)
    user_item_df = pd.DataFrame(user_item_interactions)
    
    # Create item features
    item_features_df = items_df[['plan_id', 'difficulty', 'duration_min', 'calories_burned']].copy()
    item_features_df.rename(columns={'plan_id': 'item_id'}, inplace=True)
    
    # One-hot encode focus type
    focus_types = items_df['focus'].unique() if 'focus' in items_df.columns else []
    for focus_type in focus_types:
        item_features_df[f'focus_{focus_type}'] = (items_df['focus'] == focus_type).astype(int)
    
    logger.info(f"Generated {len(user_features_df)} users")
    logger.info(f"Generated {len(user_item_df)} interactions")
    logger.info(f"Generated {len(item_features_df)} items")
    
    return user_item_df, user_features_df, item_features_df


def train_recommendation_models():
    """Train all recommendation ML models"""
    logger.info("=" * 60)
    logger.info("Training Recommendation ML Models")
    logger.info("=" * 60)
    
    # Load fitness data
    logger.info("\n1. Loading fitness data...")
    try:
        df_raw = load_data(settings.RAW_DATA_PATH)
        logger.info(f"Loaded {len(df_raw)} records from {settings.RAW_DATA_PATH}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {settings.RAW_DATA_PATH}")
        return
    
    # Preprocess
    df_processed = preprocess_data(df_raw)
    logger.info(f"Preprocessed data shape: {df_processed.shape}")
    
    # Load workout items
    logger.info("\n2. Loading workout items...")
    import os
    items_path = settings.WORKOUT_ITEMS_PATH
    if os.path.exists(items_path):
        items_df = pd.read_csv(items_path)
        logger.info(f"Loaded {len(items_df)} workout items")
    else:
        logger.warning("Items file not found. Creating synthetic items...")
        # Create synthetic items
        items_df = pd.DataFrame({
            'plan_id': range(1, 11),
            'name': ['Yoga', 'Cardio', 'Strength Training', 'HIIT', 'Pilates',
                    'Running', 'Cycling', 'Swimming', 'Weight Lifting', 'Dance Fitness'],
            'difficulty': [2, 3, 4, 5, 2, 3, 3, 4, 4, 3],
            'duration_min': [45, 30, 60, 20, 50, 40, 45, 30, 60, 45],
            'focus': ['flexibility', 'cardio', 'strength', 'hiit', 'core',
                     'cardio', 'cardio', 'full-body', 'strength', 'cardio'],
            'calories_burned': [150, 300, 250, 400, 180, 350, 280, 320, 200, 270]
        })
    
    # Generate training data
    logger.info("\n3. Generating training data...")
    user_item_data, user_features, item_features = generate_training_data(
        df_processed.head(1000),  # Use first 1000 users for training
        items_df
    )
    
    if len(user_item_data) == 0:
        logger.error("No training data generated!")
        return
    
    # Train Neural Collaborative Filtering
    logger.info("\n4. Training Neural Collaborative Filtering...")
    ncf_model = NeuralCollaborativeFiltering("models/neural_collaborative.pkl")
    try:
        metrics = ncf_model.train(
            user_item_data,
            user_features,
            item_features,
            test_size=0.2
        )
        logger.info(f"✓ Neural Collaborative Filtering trained successfully!")
        logger.info(f"  Test MAE: {metrics['test_mae']:.4f}")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
    except Exception as e:
        logger.error(f"✗ Error training Neural Collaborative Filtering: {e}")
        import traceback
        traceback.print_exc()
    
    # Train Neural Content-Based
    logger.info("\n5. Training Neural Content-Based...")
    ncb_model = NeuralContentBased("models/neural_content_based.pkl")
    try:
        metrics = ncb_model.train(
            user_item_data,
            user_features,
            item_features,
            test_size=0.2
        )
        logger.info(f"✓ Neural Content-Based trained successfully!")
        logger.info(f"  Test MAE: {metrics['test_mae']:.4f}")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
    except Exception as e:
        logger.error(f"✗ Error training Neural Content-Based: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 60)
    logger.info("Recommendation model training completed!")
    logger.info("=" * 60)
    logger.info(f"\nModels saved to: {settings.MODEL_DIR}/")
    logger.info("You can now use ML-based recommendation system.")


if __name__ == "__main__":
    train_recommendation_models()

