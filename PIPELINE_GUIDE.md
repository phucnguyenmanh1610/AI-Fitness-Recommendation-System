# PIPELINE HƯỚNG DẪN CHẠY VÀ TRIỂN KHAI DỰ ÁN

##  MỤC LỤC
1. [Pipeline Tổng Quan](#pipeline-tổng-quan)
2. [Pipeline Training Models](#pipeline-training-models)
3. [Pipeline API Request](#pipeline-api-request)
4. [Pipeline Health Prediction](#pipeline-health-prediction)
5. [Pipeline Workout Recommendation](#pipeline-workout-recommendation)
6. [Pipeline Meal Recommendation](#pipeline-meal-recommendation)
7. [Hướng Dẫn Triển Khai](#hướng-dẫn-triển-khai)

---

##  PIPELINE TỔNG QUAN

### Luồng Dữ Liệu Tổng Thể

```
┌─────────────────┐
│  Raw Data CSV  │
│ (fitness.csv)  │
└────────┬───────┘
         │
         ▼
┌─────────────────┐
│ Data Processing │
│  (Preprocessing)│
└────────┬───────┘
         │
         ▼
┌─────────────────┐
│  Model Training │
│ (XGBoost, RF)   │
└────────┬───────┘
         │
         ▼
┌─────────────────┐
│  Saved Models   │
│   (.pkl files)  │
└────────┬───────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
│   (Endpoints)   │
└────────┬───────┘
         │
         ▼
┌─────────────────┐
│  User Request   │
│   (JSON Input)  │
└────────┬───────┘
         │
         ▼
┌─────────────────┐
│  Processing     │
│  (Prediction/   │
│  Recommendation)│
└────────┬───────┘
         │
         ▼
┌─────────────────┐
│  JSON Response  │
│   (Results)     │
└─────────────────┘
```

---

##  PIPELINE TRAINING MODELS

### Training Prediction Models

#### Bước 1: Chuẩn Bị Dữ Liệu

**Input:** `data/raw/fitness.csv` (10,002 records)

**Quá trình:**
```
Raw CSV
  │
  ├─► Load Data (input.py)
  │   └─► DataFrame với columns: age, height, weight, bmi, gender, 
  │       activity_level, heart_rate, steps, calories_burned, ...
  │
  ├─► Standardize Values (normalize.py)
  │   ├─► Gender: "Male"/"Female" → 0/1
  │   ├─► Workout Type: "None"/"Cardio"/"Strength"/"Yoga" → 0/1/2/3
  │   └─► Numeric columns: Fill NaN với median
  │
  └─► Preprocess (preprocess.py)
      ├─► Map categorical to numeric
      ├─► Fill missing values
      └─► Convert to numeric types
```

**Output:** `df_processed` - DataFrame đã được xử lý sẵn sàng cho training

### Bước 2: Train Calorie Prediction Model (XGBoost)

**Input:** `df_processed` với target column `calories_burned`

**Quá trình:**
```
df_processed
  │
  ├─► Encode Categorical Features
  │   ├─► gender → numeric (0/1)
  │   ├─► activity_level → numeric (0/1/2)
  │   └─► workout_type → numeric (0/1/2/3)
  │
  ├─► Feature Selection
  │   └─► Drop target column, keep features
  │
  ├─► Train/Test Split (80/20)
  │
  ├─► Train XGBoost Regressor
  │   ├─► n_estimators: 100
  │   ├─► max_depth: 6
  │   ├─► learning_rate: 0.1
  │   └─► Fit on training data
  │
  ├─► Evaluate
  │   ├─► Calculate MAE (target: < 50 kcal)
  │   ├─► Calculate RMSE
  │   └─► Calculate R²
  │
  └─► Save Model
      ├─► models/calorie_model.pkl
      └─► models/calorie_mappings.pkl
```

**Output:**
- `calorie_model.pkl` - Trained XGBoost model
- `calorie_mappings.pkl` - Categorical mappings
- Metrics: MAE, RMSE, R²

### Bước 3: Train BMI Prediction Model (Random Forest)

**Input:** `df_processed` với target column `bmi`

**Quá trình:**
```
df_processed
  │
  ├─► Feature Selection
  │   └─► Select: age, gender, height, weight, activity_level
  │
  ├─► Encode Categorical
  │   ├─► gender → 0/1
  │   └─► activity_level → 0/1/2
  │
  ├─► Train/Test Split (80/20)
  │
  ├─► Train Random Forest Regressor
  │   ├─► n_estimators: 100
  │   ├─► max_depth: 10
  │   └─► Fit on training data
  │
  ├─► Evaluate
  │   ├─► Calculate MAE
  │   ├─► Calculate RMSE
  │   └─► Calculate R²
  │
  └─► Save Model
      └─► models/bmi_model.pkl
```

**Output:**
- `bmi_model.pkl` - Trained Random Forest model
- Metrics: MAE, RMSE, R²

### Chạy Training Pipeline cho Prediction Models

```bash
python src/train_models.py
```

---

### Recommendation System (Rule-Based)

**Lưu ý:** Recommendation system sử dụng rule-based approach và **KHÔNG CẦN TRAINING**.

**Content-Based Filtering:**
- Sử dụng cosine similarity
- Không cần train model
- Tính toán trực tiếp từ user profile và item features

**Collaborative Filtering:**
- Sử dụng SVD (TruncatedSVD) hoặc KNN
- Train SVD model on-the-fly từ user-item matrix
- Không cần pre-training

**Hybrid Recommender:**
- Combine content-based và collaborative scores
- Weights: 0.6 * content + 0.4 * collaborative
- Không cần training

**Optional: ML Models cho Recommendation**

Nếu muốn sử dụng ML models cho recommendation (optional), có thể train:
```bash
python src/train_recommendation_models.py
```

Sau đó set `use_ml_model=True` khi khởi tạo recommenders. Mặc định hệ thống sử dụng rule-based approach.

**Kết quả:**
```
Training AI Fitness Models
============================================================

1. Loading data...
   Loaded 10002 records from data/raw/fitness.csv

2. Preprocessing data...
   Preprocessed data shape: (10002, 19)

3. Training Calorie Predictor (XGBoost)...
   ✓ Calorie model trained successfully!
     Test MAE: 45.23 kcal
     Test RMSE: 58.67 kcal
     Test R²: 0.8234
     ✓ MAE meets target (< 50 kcal)

4. Training BMI Predictor (Random Forest)...
   ✓ BMI model trained successfully!
     Test MAE: 1.45
     Test RMSE: 2.12
     Test R²: 0.9123

============================================================
Model training completed!
Models saved to: models/
```

---

## PIPELINE API REQUEST

### Khởi Động Server

```bash
# Option 1: Direct
python3 run_api.py

# Option 2: Uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Docker
docker-compose up
```

**Server khởi động:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Available Endpoints:**
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `POST /api/v1/predict/health` - Health prediction
- `POST /api/v1/recommend/workout` - Workout recommendation
- `POST /api/v1/recommend/meal` - Meal recommendation

---

## PIPELINE HEALTH PREDICTION

### Endpoint: `POST /api/v1/predict/health`

### Input Flow

```
User Request (JSON)
  │
  ├─► {
  │     "age": 30,
  │     "gender": "Male",
  │     "height": 1.75,
  │     "weight": 75.0,
  │     "steps": 8000,
  │     "heart_rate": 72,
  │     "sleep_hours": 7.5,
  │     "activity_level": "moderate"
  │   }
  │
  └─► Pydantic Validation
      ├─► age: 1-120 ✓
      ├─► gender: "Male"/"Female" ✓
      ├─► height: > 0, <= 3.0 ✓
      ├─► weight: > 0, <= 300 ✓
      └─► Optional fields validated
```

### Processing Flow

```
Validated Request
  │
  ├─► Calculate BMI
  │   │
  │   ├─► Try: BMI Predictor (Random Forest)
  │   │   ├─► Load model from models/bmi_model.pkl
  │   │   ├─► Prepare features: age, gender, height, weight, activity_level
  │   │   ├─► Encode categoricals
  │   │   └─► Predict BMI
  │   │
  │   └─► Fallback: Calculated BMI
  │       └─► BMI = weight / (height²)
  │
  ├─► Calculate BMR (Harris-Benedict)
  │   │
  │   ├─► Convert height: meters → cm
  │   │
  │   ├─► If gender == "Male":
  │   │   └─► BMR = 88.362 + (13.397 × weight) + (4.799 × height_cm) - (5.677 × age)
  │   │
  │   └─► If gender == "Female":
  │       └─► BMR = 447.593 + (9.247 × weight) + (3.098 × height_cm) - (4.330 × age)
  │
  ├─► Predict Calories Burned
  │   │
  │   ├─► Try: Calorie Predictor (XGBoost)
  │   │   ├─► Load model from models/calorie_model.pkl
  │   │   ├─► Prepare features: age, gender, height, weight, activity_level,
  │   │   │                     steps, heart_rate, sleep_hours
  │   │   ├─► Encode categoricals using mappings
  │   │   ├─► Ensure feature order matches training
  │   │   └─► Predict calories_burned
  │   │
  │   └─► Fallback: TDEE Calculation
  │       └─► TDEE = BMR × activity_multiplier
  │
  └─► Calculate TDEE
      ├─► activity_multiplier:
      │   ├─► "low": 1.2
      │   ├─► "moderate": 1.55
      │   └─► "high": 1.725
      └─► TDEE = BMR × multiplier
```

### Output Flow

```
Processed Results
  │
  └─► Build Response
      ├─► HealthMetrics
      │   ├─► bmi: 24.49
      │   ├─► bmr: 1800.5
      │   ├─► calories_burned: 2200.0
      │   └─► tdee: 2790.78
      │
      └─► HealthPredictionResponse
          ├─► success: true
          ├─► metrics: HealthMetrics
          └─► message: "Health metrics predicted successfully"
```

### Example Request/Response

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/health" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "gender": "Male",
    "height": 1.75,
    "weight": 75.0,
    "steps": 8000,
    "heart_rate": 72,
    "sleep_hours": 7.5,
    "activity_level": "moderate"
  }'
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "bmi": 24.49,
    "bmr": 1800.5,
    "calories_burned": 2200.0,
    "tdee": 2790.78
  },
  "message": "Health metrics predicted successfully"
}
```

---

## PIPELINE WORKOUT RECOMMENDATION

### Endpoint: `POST /api/v1/recommend/workout`

### Input Flow

```
User Request (JSON)
  │
  └─► {
        "age": 30,
        "gender": "Male",
        "height": 1.75,
        "weight": 75.0,
        "goal": "loss",
        "experience_level": 2,
        "workout_frequency": 4,
        "preferred_duration": 45
      }
```

### Processing Flow

```
Validated Request
  │
  ├─► Load Workout Items
  │   └─► data/train/items.csv
  │       ├─► plan_id, name, difficulty, duration_min, focus, calories_burned
  │       └─► 10 workout items
  │
  ├─► Initialize Hybrid Recommender
  │   │
  │   ├─► Content-Based Recommender
  │   │   ├─► Build feature matrix from items
  │   │   │   ├─► difficulty (normalized 0-1)
  │   │   │   ├─► duration_min (normalized 0-1)
  │   │   │   ├─► focus (one-hot encoded)
  │   │   │   └─► calories_burned (normalized)
  │   │   │
  │   │   ├─► Create user profile vector
  │   │   │   ├─► difficulty_pref from experience_level
  │   │   │   ├─► duration_pref from preferred_duration
  │   │   │   ├─► focus_prefs from goal (loss/gain/maintain)
  │   │   │   └─► calories_pref from goal
  │   │   │
  │   │   └─► Calculate cosine similarity
  │   │       └─► content_score = cosine(user_vector, item_features)
  │   │
  │   └─► Collaborative Recommender
  │       ├─► Build user-item matrix (synthetic if no real data)
  │       ├─► Train SVD model
  │       └─► Calculate collaborative_score
  │
  ├─► Hybrid Scoring
  │   └─► score = 0.6 × content_score + 0.4 × collaborative_score
  │
  └─► Rank & Select Top N
      └─► Sort by score, return top 5
```

### Output Flow

```
Recommendations
  │
  └─► Build Response
      ├─► WorkoutItem[]
      │   ├─► plan_id: 2
      │   ├─► name: "Cardio"
      │   ├─► difficulty: 3
      │   ├─► duration_min: 30
      │   ├─► focus: "cardio"
      │   ├─► calories_burned: 300.0
      │   └─► score: 0.85
      │
      └─► WorkoutRecommendationResponse
          ├─► success: true
          ├─► recommendations: WorkoutItem[]
          ├─► goal: "loss"
          └─► message: "Found 5 workout recommendations"
```

### Example Request/Response

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/recommend/workout" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "gender": "Male",
    "height": 1.75,
    "weight": 75.0,
    "goal": "loss",
    "experience_level": 2,
    "workout_frequency": 4,
    "preferred_duration": 45
  }'
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "plan_id": 2,
      "name": "Cardio",
      "difficulty": 3,
      "duration_min": 30,
      "focus": "cardio",
      "calories_burned": 300.0,
      "score": 0.85
    },
    {
      "plan_id": 4,
      "name": "HIIT",
      "difficulty": 5,
      "duration_min": 20,
      "focus": "hiit",
      "calories_burned": 400.0,
      "score": 0.82
    }
  ],
  "goal": "loss",
  "message": "Found 5 workout recommendations"
}
```

---

##  PIPELINE MEAL RECOMMENDATION

### Endpoint: `POST /api/v1/recommend/meal`

### Input Flow

```
User Request (JSON)
  │
  └─► {
        "age": 30,
        "gender": "Male",
        "height": 1.75,
        "weight": 75.0,
        "goal": "loss",
        "activity_level": "moderate",
        "meals_per_day": 3
      }
```

### Processing Flow

```
Validated Request
  │
  ├─► Calculate Target Calories
  │   │
  │   ├─► Calculate BMR (Harris-Benedict)
  │   │   └─► BMR = 1800.5 kcal
  │   │
  │   ├─► Calculate TDEE
  │   │   └─► TDEE = BMR × activity_multiplier = 2790.78 kcal
  │   │
  │   └─► Adjust for Goal
  │       ├─► "loss": target = TDEE × 0.85 (15% deficit)
  │       ├─► "gain": target = TDEE × 1.15 (15% surplus)
  │       └─► "maintain": target = TDEE
  │
  ├─► Load Meal Database
  │   └─► data/train/meals.csv (or synthetic)
  │       ├─► meal_id, name, meal_type, calories, protein, carbs, fat
  │       └─► 15+ meal items
  │
  ├─► Distribute Meals
  │   └─► Based on meals_per_day:
  │       ├─► 3 meals: breakfast, lunch, dinner
  │       ├─► 4 meals: + 1 snack
  │       ├─► 5 meals: + 2 snacks
  │       └─► 6 meals: + 3 snacks
  │
  ├─► Select Meals
  │   │
  │   ├─► For each meal type:
  │   │   ├─► Filter meals by type
  │   │   ├─► Calculate target calories per meal
  │   │   ├─► Find meals closest to target
  │   │   └─► Select meal
  │   │
  │   └─► Calculate Totals
  │       ├─► total_calories
  │       ├─► total_protein
  │       ├─► total_carbs
  │       └─► total_fat
  │
  └─► Build Meal Plan
      └─► MealPlan object with meals and totals
```

### Output Flow

```
Meal Plan
  │
  └─► Build Response
      ├─► MealPlan
      │   ├─► total_calories: 2100.0
      │   ├─► total_protein: 150.0
      │   ├─► total_carbs: 200.0
      │   ├─► total_fat: 80.0
      │   └─► meals: MealItem[]
      │       ├─► MealItem (breakfast)
      │       ├─► MealItem (lunch)
      │       └─► MealItem (dinner)
      │
      └─► MealRecommendationResponse
          ├─► success: true
          ├─► meal_plan: MealPlan
          ├─► goal: "loss"
          └─► message: "Generated meal plan with 3 meals"
```

### Example Request/Response

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/recommend/meal" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "gender": "Male",
    "height": 1.75,
    "weight": 75.0,
    "goal": "loss",
    "activity_level": "moderate",
    "meals_per_day": 3
  }'
```

**Response:**
```json
{
  "success": true,
  "meal_plan": {
    "total_calories": 2100.0,
    "total_protein": 150.0,
    "total_carbs": 200.0,
    "total_fat": 80.0,
    "meals": [
      {
        "meal_id": 1,
        "name": "Oatmeal with Berries",
        "calories": 350.0,
        "protein": 12.0,
        "carbs": 55.0,
        "fat": 8.0,
        "meal_type": "breakfast"
      },
      {
        "meal_id": 5,
        "name": "Grilled Chicken Salad",
        "calories": 450.0,
        "protein": 35.0,
        "carbs": 25.0,
        "fat": 20.0,
        "meal_type": "lunch"
      },
      {
        "meal_id": 10,
        "name": "Baked Chicken Breast",
        "calories": 450.0,
        "protein": 45.0,
        "carbs": 30.0,
        "fat": 15.0,
        "meal_type": "dinner"
      }
    ]
  },
  "goal": "loss",
  "message": "Generated meal plan with 3 meals"
}
```

---

## HƯỚNG DẪN TRIỂN KHAI

### Development Environment

```bash
# 1. Clone repository
git clone <repository-url>
cd AI-Fitness-Recommendation-System

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models
python src/train_models.py

# 5. Run API server
python run_api.py
```

### Production Deployment với Docker

```bash
# 1. Build Docker image
docker build -t ai-fitness-api:latest .

# 2. Run container
docker run -d \
  --name ai-fitness-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  ai-fitness-api:latest

# 3. Hoặc dùng docker-compose
docker-compose up -d

# 4. Check logs
docker logs -f ai-fitness-api

# 5. Health check
curl http://localhost:8000/health
```

### Testing Pipeline

```bash
# 1. Test health prediction
curl -X POST "http://localhost:8000/api/v1/predict/health" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "gender": "Male", "height": 1.75, "weight": 75.0}'

# 2. Test workout recommendation
curl -X POST "http://localhost:8000/api/v1/recommend/workout" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "gender": "Male", "height": 1.75, "weight": 75.0, "goal": "loss"}'

# 3. Test meal recommendation
curl -X POST "http://localhost:8000/api/v1/recommend/meal" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "gender": "Male", "height": 1.75, "weight": 75.0, "goal": "loss", "activity_level": "moderate"}'
```

### Performance Monitoring

```bash
# Check API response time
time curl -X POST "http://localhost:8000/api/v1/predict/health" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "gender": "Male", "height": 1.75, "weight": 75.0}'

# Load testing (cần install Apache Bench hoặc wrk)
ab -n 1000 -c 10 -p request.json -T application/json \
  http://localhost:8000/api/v1/predict/health
```

---

## TÓM TẮT PIPELINE

### Data Flow Summary

1. **Training Phase:**
   - Raw CSV → Preprocessing → Feature Engineering → Model Training → Saved Models

2. **Inference Phase:**
   - User Request → Validation → Feature Preparation → Model Prediction → Response

3. **Recommendation Phase:**
   - User Profile → Content-Based Scoring → Collaborative Scoring → Hybrid Scoring → Top N Results

### Key Components

- **Data Layer**: CSV files, preprocessing pipeline
- **Model Layer**: XGBoost, Random Forest, BMR calculator
- **Service Layer**: Recommenders, meal planner
- **API Layer**: FastAPI endpoints, validation, response formatting

### Performance Targets

- **API Response Time**: < 150ms (NFR1)
- **Calorie Prediction MAE**: < 50 kcal (NFR2)
- **Throughput**: ≥ 500 requests/second (NFR4)

---

## DEBUGGING PIPELINE

### Check Model Files
```bash
ls -lh models/
# Should see:
# - calorie_model.pkl
# - calorie_mappings.pkl
# - bmi_model.pkl
```

### Check Logs
```bash
# API logs
tail -f logs/api.log

# Training logs
python src/train_models.py 2>&1 | tee training.log
```

### Test Individual Components
```python
# Test BMR calculation
from src.prediction.bmr_calculator import calculate_bmr
bmr = calculate_bmr(30, "Male", 1.75, 75.0)
print(f"BMR: {bmr}")

# Test model loading
from src.prediction.models.calorie_predictor import CaloriePredictor
predictor = CaloriePredictor()
predictor.load()
print("Model loaded successfully")
```

---

## CHECKLIST TRIỂN KHAI

- [ ] Install dependencies
- [ ] Train models
- [ ] Verify model files exist
- [ ] Start API server
- [ ] Test health endpoint
- [ ] Test prediction endpoint
- [ ] Test recommendation endpoints
- [ ] Check response times
- [ ] Verify accuracy metrics
- [ ] Setup Docker (if needed)
- [ ] Configure production settings
- [ ] Setup monitoring
- [ ] Load testing

---

**Lưu ý:** Đảm bảo models đã được train trước khi chạy API. Nếu không có models, API sẽ sử dụng fallback calculations.

