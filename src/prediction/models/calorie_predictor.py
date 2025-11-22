"""
Calorie prediction model using XGBoost
"""
import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Categorical columns that need encoding
CATEGORICAL_COLS = ["gender", "activity_level", "workout_type"]


class CaloriePredictor:
    """XGBoost-based calorie prediction model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[XGBRegressor] = None
        self.mappings: Dict = {}
        self.model_path = model_path or "models/calorie_model.pkl"
        self.mappings_path = "models/calorie_mappings.pkl"
        self.feature_names: List[str] = []
        
    def encode_categoricals(self, df: pd.DataFrame, mappings: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical columns"""
        df_encoded = df.copy()
        new_mappings = {} if mappings is None else mappings.copy()
        
        for col in CATEGORICAL_COLS:
            if col in df_encoded.columns:
                if mappings and col in mappings:
                    # Use existing mapping
                    inv_map = {v: k for k, v in mappings[col].items()}
                    df_encoded[col] = df_encoded[col].map(inv_map).fillna(-1)
                else:
                    # Create new mapping
                    df_encoded[col] = df_encoded[col].astype("category")
                    categories = df_encoded[col].cat.categories
                    new_mappings[col] = {i: cat for i, cat in enumerate(categories)}
                    inv_map = {cat: i for i, cat in enumerate(categories)}
                    df_encoded[col] = df_encoded[col].map(inv_map).fillna(-1)
        
        return df_encoded, new_mappings
    
    def train(self, df: pd.DataFrame, target_col: str = "calories_burned", 
              test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train XGBoost model for calorie prediction
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training calorie prediction model on {len(df)} samples")
        
        # Encode categoricals
        df_encoded, mappings = self.encode_categoricals(df)
        self.mappings = mappings
        
        # Prepare features and target
        if target_col not in df_encoded.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df_encoded.drop(columns=[target_col], errors='ignore')
        y = df_encoded[target_col]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Remove any remaining non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        self.feature_names = list(X.columns)
        
        # Split data
        if len(df) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train XGBoost model
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
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
        
        logger.info(f"Training completed. Test MAE: {metrics['test_mae']:.2f} kcal")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.2f} kcal")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        
        # Save model
        self.save()
        
        return metrics
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict calories burned from features
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Predicted calories burned
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        # Encode categorical features
        encoded_features = features.copy()
        for col in CATEGORICAL_COLS:
            if col in encoded_features and col in self.mappings:
                inv_map = {v: k for k, v in self.mappings[col].items()}
                value = encoded_features[col]
                encoded_features[col] = inv_map.get(value, -1)
        
        # Create DataFrame with correct feature order
        df_input = pd.DataFrame([encoded_features])
        
        # Select only numeric columns that match training features
        df_input = df_input.select_dtypes(include=[np.number])
        
        # Ensure all training features are present
        for feat in self.feature_names:
            if feat not in df_input.columns:
                df_input[feat] = 0
        
        # Reorder columns to match training
        df_input = df_input[self.feature_names]
        
        # Predict
        prediction = self.model.predict(df_input)[0]
        
        return max(0, round(prediction, 2))  # Ensure non-negative
    
    def save(self):
        """Save model and mappings"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.mappings, self.mappings_path)
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Mappings saved to {self.mappings_path}")
    
    def load(self):
        """Load model and mappings"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.mappings_path):
            raise FileNotFoundError(f"Mappings file not found at {self.mappings_path}")
        
        self.model = joblib.load(self.model_path)
        self.mappings = joblib.load(self.mappings_path)
        
        # Try to get feature names from model
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        
        logger.info(f"Model loaded from {self.model_path}")
        return self

