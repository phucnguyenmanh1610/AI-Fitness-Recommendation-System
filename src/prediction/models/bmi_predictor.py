"""
BMI prediction model using Random Forest
"""
import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class BMIPredictor:
    """Random Forest-based BMI prediction model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[RandomForestRegressor] = None
        self.model_path = model_path or "models/bmi_model.pkl"
        self.feature_names: list = []
        
    def train(self, df: pd.DataFrame, target_col: str = "bmi",
              test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train Random Forest model for BMI prediction
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training BMI prediction model on {len(df)} samples")
        
        # Prepare features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Select relevant features
        feature_cols = ["age", "gender", "height", "weight", "activity_level"]
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].copy()
        y = df[target_col]
        
        # Encode categorical features
        if "gender" in X.columns:
            X["gender"] = X["gender"].map({"Male": 0, "Female": 1}).fillna(-1)
        if "activity_level" in X.columns:
            activity_map = {"low": 0, "moderate": 1, "high": 2}
            X["activity_level"] = X["activity_level"].map(activity_map).fillna(1)
        
        # Ensure numeric
        X = X.select_dtypes(include=[np.number])
        self.feature_names = list(X.columns)
        
        # Split data
        if len(df) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
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
        
        logger.info(f"Training completed. Test MAE: {metrics['test_mae']:.2f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.2f}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        
        # Save model
        self.save()
        
        return metrics
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict BMI from features
        
        Args:
            features: Dictionary with age, gender, height, weight, activity_level
            
        Returns:
            Predicted BMI
        """
        if self.model is None:
            # Fallback to calculated BMI
            if "height" in features and "weight" in features:
                return features["weight"] / (features["height"] ** 2)
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        # Prepare features
        feature_dict = {
            "age": features.get("age", 30),
            "gender": 0 if features.get("gender", "Male") == "Male" else 1,
            "height": features.get("height", 1.7),
            "weight": features.get("weight", 70),
            "activity_level": 1  # default moderate
        }
        
        if "activity_level" in features:
            activity_map = {"low": 0, "moderate": 1, "high": 2}
            feature_dict["activity_level"] = activity_map.get(
                features["activity_level"].lower() if isinstance(features["activity_level"], str) else "moderate",
                1
            )
        
        # Create DataFrame
        df_input = pd.DataFrame([feature_dict])
        
        # Select only features used in training
        df_input = df_input[[col for col in self.feature_names if col in df_input.columns]]
        
        # Ensure all features are present
        for feat in self.feature_names:
            if feat not in df_input.columns:
                df_input[feat] = 0
        
        # Reorder columns
        df_input = df_input[self.feature_names]
        
        # Predict
        prediction = self.model.predict(df_input)[0]
        
        return max(10, min(50, round(prediction, 2)))  # Clamp to reasonable range
    
    def save(self):
        """Save model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.model_path.replace(".pkl", "_features.pkl"))
        logger.info(f"Model saved to {self.model_path}")
    
    def load(self):
        """Load model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        
        # Try to load feature names
        features_path = self.model_path.replace(".pkl", "_features.pkl")
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
        elif hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        
        logger.info(f"Model loaded from {self.model_path}")
        return self

