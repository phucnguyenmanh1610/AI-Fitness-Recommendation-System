"""
Neural Collaborative Filtering model for workout recommendations
"""
import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering model using MLP"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[MLPRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_path = model_path or "models/neural_collaborative.pkl"
        self.scaler_path = "models/neural_collaborative_scaler.pkl"
        self.user_features: list = []
        self.item_features: list = []
        
    def _prepare_features(self, user_item_data: pd.DataFrame, 
                         user_features: pd.DataFrame,
                         item_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training
        
        Args:
            user_item_data: DataFrame with user_id, item_id, rating
            user_features: DataFrame with user features
            item_features: DataFrame with item features
            
        Returns:
            X (features), y (ratings)
        """
        # Merge user and item features
        merged = user_item_data.merge(
            user_features, on='user_id', how='left'
        ).merge(
            item_features, on='item_id', how='left', suffixes=('_user', '_item')
        )
        
        # Select feature columns (exclude user_id, item_id, rating)
        feature_cols = [col for col in merged.columns 
                       if col not in ['user_id', 'item_id', 'rating']]
        
        X = merged[feature_cols].values
        y = merged['rating'].values
        
        # Store feature names
        self.user_features = [col for col in feature_cols if '_user' in col]
        self.item_features = [col for col in feature_cols if '_item' in col]
        
        return X, y
    
    def train(self, user_item_data: pd.DataFrame,
              user_features: pd.DataFrame,
              item_features: pd.DataFrame,
              test_size: float = 0.2,
              random_state: int = 42) -> Dict:
        """
        Train Neural Collaborative Filtering model
        
        Args:
            user_item_data: DataFrame with user_id, item_id, rating
            user_features: DataFrame with user features (age, gender, goal, etc.)
            item_features: DataFrame with item features (difficulty, duration, etc.)
            test_size: Test split ratio
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training Neural Collaborative Filtering on {len(user_item_data)} interactions")
        
        # Prepare features
        X, y = self._prepare_features(user_item_data, user_features, item_features)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        if len(user_item_data) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
        
        # Train MLP model
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_r2": r2_score(y_test, y_pred_test)
        }
        
        logger.info(f"Training completed. Test MAE: {metrics['test_mae']:.4f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        
        # Save model
        self.save()
        
        return metrics
    
    def predict(self, user_features: Dict, item_features: Dict) -> float:
        """
        Predict rating for user-item pair
        
        Args:
            user_features: Dictionary with user features
            item_features: Dictionary with item features
            
        Returns:
            Predicted rating
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Combine features in correct order
        feature_vector = []
        
        # Add user features
        for feat in self.user_features:
            base_feat = feat.replace('_user', '')
            feature_vector.append(user_features.get(base_feat, 0))
        
        # Add item features
        for feat in self.item_features:
            base_feat = feat.replace('_item', '')
            feature_vector.append(item_features.get(base_feat, 0))
        
        # Scale and predict
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return max(0, min(5, prediction))  # Clamp to 0-5 range
    
    def save(self):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Scaler saved to {self.scaler_path}")
    
    def load(self):
        """Load model and scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"Model loaded from {self.model_path}")
        return self

