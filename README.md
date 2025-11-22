# AI Fitness & Health Recommendation System

## Overview

Hệ thống AI dự đoán sức khỏe và khuyến nghị cá nhân hóa dựa trên Machine Learning. Hệ thống cung cấp:
- **Dự đoán chỉ số sức khỏe**: BMI, BMR, calories burned
- **Đề xuất bài tập**: Dựa trên mục tiêu và profile người dùng
- **Đề xuất thực đơn**: Theo nhu cầu calo và macronutrients

## System Architecture

Hệ thống được thiết kế theo kiến trúc phân lớp:
- **API Layer**: FastAPI với REST endpoints
- **Service Layer**: Business logic và xử lý
- **Model Layer**: ML models (XGBoost, Random Forest, Neural Networks)
- **Data Layer**: Data processing và storage

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd AI-Fitness-Recommendation-System

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

Trước khi chạy API, cần train các ML models:

```bash
# Train tất cả models (prediction + recommendation)
python train_all.py

# Hoặc train riêng:
python src/train_models.py              # Prediction models
python src/train_recommendation_models.py  # Recommendation models
```

**Prediction Models:**
- Load dữ liệu từ `data/raw/fitness.csv`
- Train XGBoost model cho calorie prediction
- Train Random Forest model cho BMI prediction

**Recommendation Models:**
- Generate training data từ fitness data
- Train Neural Collaborative Filtering model
- Train Neural Content-Based model

Tất cả models được lưu vào thư mục `models/`

### 3. Run API Server

#### Option 1: Run directly
```bash
python run_api.py
```

#### Option 2: Using uvicorn
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option 3: Using Docker
```bash
# Build image
docker build -t ai-fitness-api .

# Run container
docker run -p 8000:8000 ai-fitness-api

# Or use docker-compose
docker-compose up
```

### 4. Access API Documentation

Sau khi server chạy, truy cập:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### 1. Health Prediction
**POST** `/api/v1/predict/health`

Dự đoán các chỉ số sức khỏe: BMI, BMR, calories burned

**Request Body:**
```json
{
  "age": 30,
  "gender": "Male",
  "height": 1.75,
  "weight": 75.0,
  "steps": 8000,
  "heart_rate": 72,
  "sleep_hours": 7.5,
  "activity_level": "moderate"
}
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

### 2. Workout Recommendation
**POST** `/api/v1/recommend/workout`

Đề xuất bài tập phù hợp với mục tiêu

**Request Body:**
```json
{
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
    }
  ],
  "goal": "loss"
}
```

### 3. Meal Recommendation
**POST** `/api/v1/recommend/meal`

Đề xuất thực đơn theo nhu cầu calo

**Request Body:**
```json
{
  "age": 30,
  "gender": "Male",
  "height": 1.75,
  "weight": 75.0,
  "goal": "loss",
  "activity_level": "moderate",
  "meals_per_day": 3
}
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
    "meals": [...]
  },
  "goal": "loss"
}
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_prediction_model.py

# Run with coverage
pytest --cov=src tests/
```

## Project Structure

```
AI-Fitness-Recommendation-System/
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Processed data
│   └── train/            # Training data (workouts, meals)
├── models/               # Trained ML models
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # FastAPI app
│   │   ├── config.py     # Configuration
│   │   └── routes/       # API routes
│   ├── prediction/       # ML prediction models
│   │   ├── models/       # XGBoost, Random Forest
│   │   └── bmr_calculator.py
│   ├── recommendation/   # Recommender systems
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   ├── hybrid_recommender.py
│   │   └── meal_recommender.py
│   ├── data_input/       # Data processing
│   ├── models/           # Data models (schemas, user profile)
│   └── output/           # Dashboard (Streamlit)
├── test/                 # Test files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Cấu hình trong `src/api/config.py` hoặc environment variables:

- `API_TITLE`: API title
- `API_VERSION`: API version
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: False)
- `DATABASE_URL`: Database connection string
- `MAX_WORKERS`: Number of workers (default: 4)

##  Features

### Implemented
- [x] FastAPI REST API
- [x] Health prediction (BMI, BMR, calories)
- [x] XGBoost for calorie prediction
- [x] Random Forest for BMI prediction
- [x] BMR calculation (Harris-Benedict)
- [x] Hybrid recommender system
- [x] Content-based filtering
- [x] Collaborative filtering (SVD)
- [x] Workout recommendations (ML-based)
- [x] Meal recommendations
- [x] Neural Collaborative Filtering
- [x] Neural Content-Based recommendation
- [x] Docker support
- [x] API documentation (Swagger/ReDoc)

### Future Enhancements
- [ ] Database layer (SQLAlchemy)
- [ ] User authentication
- [ ] Prediction history storage
- [ ] Model versioning (MLflow)
- [ ] Caching (Redis)
- [ ] CI/CD pipeline
- [ ] Performance monitoring

##  Model Performance

### Calorie Prediction (XGBoost)
- **Target**: MAE < 50 kcal
- **Current**: Check training logs

### BMI Prediction (Random Forest)
- **Metrics**: MAE, RMSE, R²
- **Performance**: Check training logs

##  Docker Deployment

```bash
# Build
docker build -t ai-fitness-api .

# Run
docker run -p 8000:8000 ai-fitness-api

# With docker-compose
docker-compose up -d
```

##  Development

### Code Structure
- **API Layer**: FastAPI routes và handlers
- **Service Layer**: Business logic
- **Model Layer**: ML models và predictions
- **Data Layer**: Data processing

### Adding New Features
1. Create feature branch
2. Implement feature
3. Add tests
4. Update documentation
5. Create pull request

