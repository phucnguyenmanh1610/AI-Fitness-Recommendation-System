"""
Script to train all models (prediction + recommendation)
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 60)
    
    # Train prediction models
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Training Prediction Models")
    logger.info("=" * 60)
    from src.train_models import train_all_models
    train_all_models()
    
    # Train recommendation models
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Training Recommendation Models")
    logger.info("=" * 60)
    from src.train_recommendation_models import train_recommendation_models
    train_recommendation_models()
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL MODELS TRAINED SUCCESSFULLY!")
    logger.info("=" * 60)

