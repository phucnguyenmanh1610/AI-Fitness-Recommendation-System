# TÀI LIỆU KIẾN TRÚC HỆ THỐNG

## 📋 MỤC LỤC
1. [Tổng Quan Kiến Trúc](#tổng-quan-kiến-trúc)
2. [Lớp API (API Layer)](#lớp-api-api-layer)
3. [Lớp Dịch Vụ (Service Layer)](#lớp-dịch-vụ-service-layer)
4. [Lớp Mô Hình (Model Layer)](#lớp-mô-hình-model-layer)
5. [Lớp Dữ Liệu (Data Layer)](#lớp-dữ-liệu-data-layer)
6. [Models & Schemas](#models--schemas)
7. [Configuration](#configuration)

---

## 🏗️ TỔNG QUAN KIẾN TRÚC

### Kiến Trúc Phân Lớp (Layered Architecture)

```
┌─────────────────────────────────────────────────────────┐
│                    API LAYER                            │
│  FastAPI Application, Routes, Request/Response         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  SERVICE LAYER                           │
│  Business Logic, Recommenders, Meal Planner             │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                   MODEL LAYER                           │
│  ML Models (XGBoost, Random Forest), BMR Calculator    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                   DATA LAYER                             │
│  Data Processing, Preprocessing, Normalization            │
└──────────────────────────────────────────────────────────┘
```

### Dependency Flow

```
API Layer → Service Layer → Model Layer → Data Layer
```

---

## 🌐 LỚP API (API LAYER)

### Mục Đích
Xử lý HTTP requests, validation, routing, và response formatting.

### Components

#### 1. `src/api/main.py` - FastAPI Application

**Vai trò:** Entry point của API server, khởi tạo FastAPI app và cấu hình middleware.

**Chức năng:**
- Tạo FastAPI application instance
- Setup CORS middleware
- Include routers
- Global exception handling
- Lifespan events (startup/shutdown)

**Cách sử dụng:**
```python
from src.api.main import app

# App được khởi tạo tự động với:
# - Title: "AI Fitness & Health Recommendation System"
# - Version: "1.0.0"
# - Docs: /docs (Swagger UI)
# - ReDoc: /redoc
```

**Endpoints:**
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc documentation

**Dependencies:**
- FastAPI
- Uvicorn
- CORS middleware

---

#### 2. `src/api/config.py` - Configuration Settings

**Vai trò:** Quản lý cấu hình ứng dụng (settings, paths, environment variables).

**Chức năng:**
- API settings (title, version, prefix)
- Server settings (host, port, debug)
- Model paths
- Data paths
- Database settings
- Performance settings

**Cách sử dụng:**
```python
from src.api.config import settings

# Access settings
print(settings.API_TITLE)
print(settings.PORT)
print(settings.CALORIE_MODEL_PATH)
```

**Cấu hình:**
```python
class Settings(BaseSettings):
    API_TITLE: str = "AI Fitness & Health Recommendation System"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    MODEL_DIR: str = "models"
    # ... more settings
```

**Environment Variables:**
- `DEBUG` - Enable debug mode
- `DATABASE_URL` - Database connection string
- `MAX_WORKERS` - Number of workers

---

#### 3. `src/api/routes/health.py` - Health Prediction Endpoint

**Vai trò:** Xử lý requests dự đoán sức khỏe (BMI, BMR, calories).

**Chức năng:**
- Validate request với Pydantic
- Load ML models (lazy loading)
- Tính toán BMI, BMR, calories
- Format response

**Endpoint:** `POST /api/v1/predict/health`

**Request Schema:** `HealthPredictionRequest`
```python
{
    "age": int (1-120),
    "gender": "Male" | "Female",
    "height": float (>0, <=3.0),
    "weight": float (>0, <=300),
    "steps": int (optional),
    "heart_rate": int (optional, 30-220),
    "sleep_hours": float (optional, 0-24),
    "activity_level": "low" | "moderate" | "high" (optional)
}
```

**Response Schema:** `HealthPredictionResponse`
```python
{
    "success": bool,
    "metrics": {
        "bmi": float,
        "bmr": float,
        "calories_burned": float,
        "tdee": float
    },
    "message": str
}
```

**Flow:**
1. Validate request
2. Load BMI predictor (fallback to calculation)
3. Calculate BMR (Harris-Benedict)
4. Predict calories (XGBoost, fallback to TDEE)
5. Calculate TDEE
6. Return response

**Dependencies:**
- `BMIPredictor` - BMI prediction model
- `CaloriePredictor` - Calorie prediction model
- `calculate_bmr`, `calculate_tdee` - BMR calculator

---

#### 4. `src/api/routes/workout.py` - Workout Recommendation Endpoint

**Vai trò:** Đề xuất bài tập phù hợp với user profile và mục tiêu.

**Chức năng:**
- Validate workout request
- Load workout items
- Initialize hybrid recommender
- Generate recommendations
- Format response

**Endpoint:** `POST /api/v1/recommend/workout`

**Request Schema:** `WorkoutRecommendationRequest`
```python
{
    "age": int,
    "gender": "Male" | "Female",
    "height": float,
    "weight": float,
    "goal": "loss" | "gain" | "maintain",
    "experience_level": int (1-3, optional),
    "workout_frequency": int (1-7, optional),
    "preferred_duration": int (15-120, optional)
}
```

**Response Schema:** `WorkoutRecommendationResponse`
```python
{
    "success": bool,
    "recommendations": [
        {
            "plan_id": int,
            "name": str,
            "difficulty": int,
            "duration_min": int,
            "focus": str,
            "calories_burned": float,
            "score": float
        }
    ],
    "goal": str,
    "message": str
}
```

**Dependencies:**
- `HybridRecommender` - Hybrid recommendation system

---

#### 5. `src/api/routes/meal.py` - Meal Recommendation Endpoint

**Vai trò:** Đề xuất thực đơn theo nhu cầu calo và macronutrients.

**Chức năng:**
- Validate meal request
- Calculate target calories
- Load meal database
- Generate meal plan
- Format response

**Endpoint:** `POST /api/v1/recommend/meal`

**Request Schema:** `MealRecommendationRequest`
```python
{
    "age": int,
    "gender": "Male" | "Female",
    "height": float,
    "weight": float,
    "goal": "loss" | "gain" | "maintain",
    "activity_level": "low" | "moderate" | "high",
    "target_calories": float (optional),
    "meals_per_day": int (1-6, optional, default: 3)
}
```

**Response Schema:** `MealRecommendationResponse`
```python
{
    "success": bool,
    "meal_plan": {
        "total_calories": float,
        "total_protein": float,
        "total_carbs": float,
        "total_fat": float,
        "meals": [
            {
                "meal_id": int,
                "name": str,
                "calories": float,
                "protein": float,
                "carbs": float,
                "fat": float,
                "meal_type": str
            }
        ]
    },
    "goal": str,
    "message": str
}
```

**Dependencies:**
- `MealRecommender` - Meal recommendation system
- `calculate_bmr`, `calculate_tdee` - BMR calculator

---

## 🔧 LỚP DỊCH VỤ (SERVICE LAYER)

### Mục Đích
Xử lý business logic, recommendation algorithms, và meal planning.

### Components

#### 1. `src/recommendation/content_based.py` - Content-Based Filtering

**Vai trò:** Đề xuất dựa trên similarity giữa user profile và item features.

**Chức năng:**
- Build feature matrix từ items
- Create user profile vector
- Calculate cosine similarity
- Rank items by similarity

**Cách sử dụng:**
```python
from src.recommendation.content_based import ContentBasedRecommender

recommender = ContentBasedRecommender()
recommender.load_items(items_df)

recommendations = recommender.recommend(
    user_profile={
        "experience_level": 2,
        "preferred_duration": 45,
        "goal": "loss"
    },
    top_n=5
)
```

**Features:**
- Difficulty (normalized)
- Duration (normalized)
- Focus type (one-hot encoded)
- Calories burned (normalized)

**Algorithm:**
- Cosine similarity giữa user vector và item feature matrix
- Score = cosine(user_vector, item_features)

**Dependencies:**
- scikit-learn (cosine_similarity)
- pandas, numpy

---

#### 2. `src/recommendation/collaborative.py` - Collaborative Filtering

**Vai trò:** Đề xuất dựa trên user-item interactions (collaborative approach).

**Chức năng:**
- Build user-item interaction matrix
- Train SVD hoặc KNN model
- Generate recommendations từ item popularity/similarity

**Cách sử dụng:**
```python
from src.recommendation.collaborative import CollaborativeRecommender

recommender = CollaborativeRecommender(method="svd")
recommender.load_items(items_df, user_interactions_df)

recommendations = recommender.recommend(
    user_profile={},
    top_n=5
)
```

**Methods:**
- **SVD**: TruncatedSVD để reduce dimensions
- **KNN**: NearestNeighbors với cosine similarity

**Fallback:**
- Nếu không có user interactions, tạo synthetic interactions dựa trên item features

**Dependencies:**
- scikit-learn (TruncatedSVD, NearestNeighbors)
- pandas, numpy

---

#### 3. `src/recommendation/hybrid_recommender.py` - Hybrid Recommender

**Vai trò:** Kết hợp content-based và collaborative filtering.

**Chức năng:**
- Combine scores từ cả 2 methods
- Weighted scoring: `0.6 * content + 0.4 * collaborative`
- Rank và select top N

**Cách sử dụng:**
```python
from src.recommendation.hybrid_recommender import HybridRecommender

recommender = HybridRecommender(
    content_weight=0.6,
    collaborative_weight=0.4
)
recommender.load_items("data/train/items.csv")

recommendations = recommender.recommend(
    user_profile={
        "age": 30,
        "gender": "Male",
        "goal": "loss",
        "experience_level": 2
    },
    top_n=5,
    include_score=True
)
```

**Scoring Formula:**
```python
hybrid_score = (
    content_weight * content_score +
    collaborative_weight * collaborative_score
)
```

**Dependencies:**
- `ContentBasedRecommender`
- `CollaborativeRecommender`

---

#### 4. `src/recommendation/meal_recommender.py` - Meal Recommender

**Vai trò:** Tạo meal plan dựa trên calorie và macronutrient needs.

**Chức năng:**
- Load meal database
- Calculate target calories
- Distribute meals theo meal type
- Select meals closest to target calories
- Calculate totals (calories, protein, carbs, fat)

**Cách sử dụng:**
```python
from src.recommendation.meal_recommender import MealRecommender

recommender = MealRecommender()
recommender.load_meals("data/train/meals.csv")

meal_plan = recommender.recommend_meal_plan(
    target_calories=2000.0,
    goal="loss",
    meals_per_day=3
)
```

**Meal Distribution:**
- 3 meals: breakfast, lunch, dinner
- 4 meals: + 1 snack
- 5 meals: + 2 snacks
- 6 meals: + 3 snacks

**Dependencies:**
- pandas, numpy

---

## 🤖 LỚP MÔ HÌNH (MODEL LAYER)

### Mục Đích
ML models cho prediction và calculation functions.

### Components

#### 1. `src/prediction/models/calorie_predictor.py` - Calorie Predictor

**Vai trò:** Dự đoán calories burned sử dụng XGBoost Regressor.

**Chức năng:**
- Train XGBoost model
- Encode categorical features
- Predict calories từ user features
- Save/load model

**Cách sử dụng:**
```python
from src.prediction.models.calorie_predictor import CaloriePredictor

# Training
predictor = CaloriePredictor()
metrics = predictor.train(df_processed, target_col='calories_burned')
predictor.save()

# Prediction
predictor.load()
calories = predictor.predict({
    "age": 30,
    "gender": "Male",
    "height": 1.75,
    "weight": 75.0,
    "activity_level": "moderate",
    "steps": 8000,
    "heart_rate": 72,
    "sleep_hours": 7.5
})
```

**Model:**
- **Algorithm**: XGBoost Regressor
- **Parameters**: n_estimators=100, max_depth=6, learning_rate=0.1
- **Target**: calories_burned
- **Metrics**: MAE (target: < 50 kcal), RMSE, R²

**Dependencies:**
- XGBoost
- scikit-learn
- joblib (model persistence)

---

#### 2. `src/prediction/models/bmi_predictor.py` - BMI Predictor

**Vai trò:** Dự đoán BMI sử dụng Random Forest Regressor.

**Chức năng:**
- Train Random Forest model
- Predict BMI từ user features
- Fallback to calculated BMI nếu model không có

**Cách sử dụng:**
```python
from src.prediction.models.bmi_predictor import BMIPredictor

# Training
predictor = BMIPredictor()
metrics = predictor.train(df_processed, target_col='bmi')
predictor.save()

# Prediction
predictor.load()
bmi = predictor.predict({
    "age": 30,
    "gender": "Male",
    "height": 1.75,
    "weight": 75.0,
    "activity_level": "moderate"
})
```

**Model:**
- **Algorithm**: Random Forest Regressor
- **Parameters**: n_estimators=100, max_depth=10
- **Target**: bmi
- **Features**: age, gender, height, weight, activity_level

**Dependencies:**
- scikit-learn
- joblib

---

#### 3. `src/prediction/bmr_calculator.py` - BMR Calculator

**Vai trò:** Tính toán BMR và TDEE sử dụng Harris-Benedict formula.

**Chức năng:**
- Calculate BMR (Basal Metabolic Rate)
- Calculate TDEE (Total Daily Energy Expenditure)
- Support cho cả Male và Female

**Cách sử dụng:**
```python
from src.prediction.bmr_calculator import calculate_bmr, calculate_tdee

# Calculate BMR
bmr = calculate_bmr(
    age=30,
    gender="Male",
    weight=75.0,  # kg
    height=1.75   # meters
)

# Calculate TDEE
tdee = calculate_tdee(
    bmr=bmr,
    activity_level="moderate"  # "low", "moderate", "high"
)
```

**Formulas:**
- **Male BMR**: `88.362 + (13.397 × weight) + (4.799 × height_cm) - (5.677 × age)`
- **Female BMR**: `447.593 + (9.247 × weight) + (3.098 × height_cm) - (4.330 × age)`
- **TDEE**: `BMR × activity_multiplier`
  - low: 1.2
  - moderate: 1.55
  - high: 1.725

**Dependencies:**
- None (pure Python)

---

## 📊 LỚP DỮ LIỆU (DATA LAYER)

### Mục Đích
Xử lý và preprocessing dữ liệu.

### Components

#### 1. `src/data_input/input.py` - Data Loading

**Vai trò:** Load dữ liệu từ CSV files hoặc generate synthetic data.

**Chức năng:**
- Load CSV files
- Merge multiple CSV files
- Generate synthetic data nếu không có file
- Standardize values

**Cách sử dụng:**
```python
from src.data_input.input import load_data, get_synthetic_data

# Load from file
df = load_data("data/raw/fitness.csv")

# Generate synthetic
df = get_synthetic_data(n_samples=1000)
```

**Dependencies:**
- pandas
- `normalize.py` (standardize_values)

---

#### 2. `src/data_input/normalize.py` - Data Normalization

**Vai trò:** Chuẩn hóa và standardize dữ liệu.

**Chức năng:**
- Standardize column names
- Normalize gender values
- Normalize workout types
- Fill missing numeric values

**Cách sử dụng:**
```python
from src.data_input.normalize import standardize_values

df_normalized = standardize_values(df)
```

**Transformations:**
- Gender: "M"/"F" → "Male"/"Female"
- Workout Type: Capitalize và standardize
- Numeric: Fill NaN với median

**Dependencies:**
- pandas

---

#### 3. `src/data_input/preprocess.py` - Data Preprocessing

**Vai trò:** Preprocessing cuối cùng trước khi train models.

**Chức năng:**
- Map categorical to numeric
- Fill missing values
- Convert to numeric types
- Prepare data cho ML

**Cách sử dụng:**
```python
from src.data_input.preprocess import preprocess_data

df_processed = preprocess_data(df_raw)
```

**Transformations:**
- Gender: "Male"/"Female" → 0/1
- Workout Type: "None"/"Cardio"/"Strength"/"Yoga" → 0/1/2/3
- Fill NaN với median cho numeric columns

**Dependencies:**
- pandas
- `normalize.py`

---

#### 4. `src/data_input/pipeline.py` - Data Pipeline

**Vai trò:** Pipeline tổng hợp để load và preprocess data.

**Chức năng:**
- Combine load và preprocess
- Single function call

**Cách sử dụng:**
```python
from src.data_input.pipeline import get_processed_data

df = get_processed_data("data/raw/fitness.csv")
```

**Dependencies:**
- `input.py`, `preprocess.py`

---

## 📝 MODELS & SCHEMAS

### Mục Đích
Data models và Pydantic schemas cho validation.

### Components

#### 1. `src/models/schemas.py` - Pydantic Schemas

**Vai trò:** Request/response validation và serialization.

**Schemas:**
- `HealthPredictionRequest` - Input cho health prediction
- `HealthPredictionResponse` - Output từ health prediction
- `WorkoutRecommendationRequest` - Input cho workout recommendation
- `WorkoutRecommendationResponse` - Output từ workout recommendation
- `MealRecommendationRequest` - Input cho meal recommendation
- `MealRecommendationResponse` - Output từ meal recommendation
- `HealthMetrics` - Health metrics structure
- `WorkoutItem` - Workout item structure
- `MealItem` - Meal item structure
- `MealPlan` - Meal plan structure

**Enums:**
- `Gender`: "Male", "Female"
- `ActivityLevel`: "low", "moderate", "high"
- `Goal`: "loss", "gain", "maintain"

**Cách sử dụng:**
```python
from src.models.schemas import HealthPredictionRequest

request = HealthPredictionRequest(
    age=30,
    gender="Male",
    height=1.75,
    weight=75.0
)
# Automatic validation
```

**Dependencies:**
- Pydantic

---

#### 2. `src/models/user_profile.py` - User Profile Models

**Vai trò:** Data classes cho user profile và activity data.

**Classes:**
- `UserProfile`: User profile với BMI auto-calculation
- `ActivityData`: Activity tracking data

**Cách sử dụng:**
```python
from src.models.user_profile import UserProfile, ActivityData

profile = UserProfile(
    age=30,
    gender="Male",
    height=1.75,
    weight=75.0,
    goal="loss"
)
# BMI calculated automatically

activity = ActivityData(
    steps=8000,
    heart_rate=72,
    sleep_hours=7.5
)
```

**Dependencies:**
- dataclasses
- enum (Goal)

---

## ⚙️ CONFIGURATION

### `src/api/config.py` - Application Configuration

**Vai trò:** Centralized configuration management.

**Settings:**
- API settings (title, version, prefix)
- Server settings (host, port, debug)
- Model paths
- Data paths
- Database settings
- Performance settings

**Cách sử dụng:**
```python
from src.api.config import settings

# Access any setting
print(settings.API_TITLE)
print(settings.PORT)
print(settings.CALORIE_MODEL_PATH)
```

**Environment Variables:**
- `DEBUG` - Enable debug mode
- `DATABASE_URL` - Database connection
- `MAX_WORKERS` - Number of workers

**Dependencies:**
- Pydantic Settings

---

## 🔗 DEPENDENCY GRAPH

```
API Layer
  ├─► Routes
  │   ├─► health.py → Model Layer
  │   ├─► workout.py → Service Layer
  │   └─► meal.py → Service Layer
  │
  └─► Config → Settings

Service Layer
  ├─► Hybrid Recommender
  │   ├─► Content-Based → Items
  │   └─► Collaborative → Items
  │
  └─► Meal Recommender → Meal Database

Model Layer
  ├─► Calorie Predictor → XGBoost
  ├─► BMI Predictor → Random Forest
  └─► BMR Calculator → Formulas

Data Layer
  ├─► Input → CSV Files
  ├─► Normalize → Standardization
  └─► Preprocess → ML Ready
```

---

## 📦 PACKAGE STRUCTURE

```
src/
├── api/                    # API Layer
│   ├── main.py            # FastAPI app
│   ├── config.py          # Configuration
│   └── routes/            # API endpoints
│       ├── health.py
│       ├── workout.py
│       └── meal.py
│
├── recommendation/         # Service Layer
│   ├── content_based.py
│   ├── collaborative.py
│   ├── hybrid_recommender.py
│   └── meal_recommender.py
│
├── prediction/            # Model Layer
│   ├── models/
│   │   ├── calorie_predictor.py
│   │   └── bmi_predictor.py
│   └── bmr_calculator.py
│
├── data_input/            # Data Layer
│   ├── input.py
│   ├── normalize.py
│   ├── preprocess.py
│   └── pipeline.py
│
└── models/                # Data Models
    ├── schemas.py
    └── user_profile.py
```

---

## 🎯 TÓM TẮT VAI TRÒ CÁC LỚP

| Lớp | Vai Trò | Components | Dependencies |
|-----|---------|------------|--------------|
| **API Layer** | HTTP handling, routing, validation | FastAPI, Routes, Config | FastAPI, Pydantic |
| **Service Layer** | Business logic, recommendations | Recommenders, Meal Planner | scikit-learn, pandas |
| **Model Layer** | ML predictions, calculations | XGBoost, Random Forest, BMR | XGBoost, scikit-learn |
| **Data Layer** | Data processing, preprocessing | Input, Normalize, Preprocess | pandas, numpy |
| **Models** | Data validation, schemas | Pydantic schemas, User models | Pydantic |

---

**Lưu ý:** Tất cả các lớp được thiết kế để độc lập và có thể test riêng biệt. Dependency injection được sử dụng để dễ dàng mock và test.

