"""
Script to train all ML models
"""
import os
import sys
import logging

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_input.input import load_data, get_synthetic_data
from src.data_input.preprocess import preprocess_data
from src.prediction.models.calorie_predictor import CaloriePredictor
from src.prediction.models.bmi_predictor import BMIPredictor
from src.api.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_all_models():
    """Train all ML models"""
    logger.info("=" * 60)
    logger.info("Training AI Fitness Models")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\n1. Loading data...")
    try:
        df_raw = load_data(settings.RAW_DATA_PATH)
        logger.info(f"Loaded {len(df_raw)} records from {settings.RAW_DATA_PATH}")
    except FileNotFoundError:
        logger.warning(f"Data file not found at {settings.RAW_DATA_PATH}")
        logger.info("Generating synthetic data...")
        df_raw = get_synthetic_data(1000)
        logger.info(f"Generated {len(df_raw)} synthetic records")
    
    # Preprocess data
    logger.info("\n2. Preprocessing data...")
    df_processed = preprocess_data(df_raw)
    logger.info(f"Preprocessed data shape: {df_processed.shape}")
    
    # Ensure required columns exist
    if 'calories_burned' not in df_processed.columns:
        logger.warning("'calories_burned' not found. Creating from steps...")
        if 'steps' in df_processed.columns:
            df_processed['calories_burned'] = df_processed['steps'] * 0.04
        else:
            df_processed['calories_burned'] = 500  # Default
    
    if 'bmi' not in df_processed.columns:
        logger.warning("'bmi' not found. Calculating from height and weight...")
        if 'height' in df_processed.columns and 'weight' in df_processed.columns:
            df_processed['bmi'] = df_processed['weight'] / (df_processed['height'] ** 2)
        else:
            df_processed['bmi'] = 22.0  # Default
    
    # Train Calorie Predictor (XGBoost)
    logger.info("\n3. Training Calorie Predictor (XGBoost)...")
    calorie_predictor = CaloriePredictor(settings.CALORIE_MODEL_PATH)
    try:
        metrics = calorie_predictor.train(df_processed, target_col='calories_burned')
        logger.info(f"✓ Calorie model trained successfully!")
        logger.info(f"  Test MAE: {metrics['test_mae']:.2f} kcal")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.2f} kcal")
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
        
        if metrics['test_mae'] > 50:
            logger.warning(f"  ⚠ MAE ({metrics['test_mae']:.2f}) exceeds target of 50 kcal")
        else:
            logger.info(f"  ✓ MAE meets target (< 50 kcal)")
    except Exception as e:
        logger.error(f"✗ Error training calorie model: {e}")
    
    # Train BMI Predictor (Random Forest)
    logger.info("\n4. Training BMI Predictor (Random Forest)...")
    bmi_predictor = BMIPredictor(settings.BMI_MODEL_PATH)
    try:
        metrics = bmi_predictor.train(df_processed, target_col='bmi')
        logger.info(f"✓ BMI model trained successfully!")
        logger.info(f"  Test MAE: {metrics['test_mae']:.2f}")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
    except Exception as e:
        logger.error(f"✗ Error training BMI model: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Model training completed!")
    logger.info("=" * 60)
    logger.info(f"\nModels saved to: {settings.MODEL_DIR}/")
    logger.info("You can now start the API server.")


if __name__ == "__main__":
    train_all_models()
    logger.info("\n" + "=" * 60)
    logger.info("To train recommendation models, run:")
    logger.info("python src/train_recommendation_models.py")
    logger.info("=" * 60)

