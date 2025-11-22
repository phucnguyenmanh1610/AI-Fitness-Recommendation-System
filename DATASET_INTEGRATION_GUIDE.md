# HƯỚNG DẪN TÍCH HỢP DATASET TỪ KAGGLE VÀ TRAIN RECOMMENDATION MODELS

## PHẦN 1: TÍCH HỢP DATASET TỪ KAGGLE

### 1.1 Quy Trình Tổng Quan

```
Kaggle Dataset
    │
    ├─► Download Dataset
    │   └─► Kaggle API hoặc manual download
    │
    ├─► Data Exploration
    │   ├─► Kiểm tra structure, columns
    │   ├─► Phân tích missing values
    │   └─► Kiểm tra data quality
    │
    ├─► Data Mapping
    │   └─► Map columns từ Kaggle → format của hệ thống
    │
    ├─► Data Transformation
    │   ├─► Standardize column names
    │   ├─► Convert data types
    │   └─► Handle missing values
    │
    └─► Integration
        └─► Save vào data/raw/ hoặc data/train/
```

### 1.2 Các Loại Dataset Cần Thiết

#### A. Fitness/Health Dataset (cho Prediction Models)
**Mục đích:** Train calorie prediction và BMI prediction models

**Các dataset phù hợp trên Kaggle:**
- **Fitness Tracker Data**: Steps, heart rate, calories burned, sleep
- **Health & Fitness Survey Data**: BMI, age, gender, activity levels
- **Exercise Dataset**: Workout types, duration, calories burned
- **Body Metrics Dataset**: Height, weight, BMI, body fat percentage

**Các columns cần có:**
- **Input features**: age, gender, height, weight, activity_level, steps, heart_rate, sleep_hours
- **Target variables**: calories_burned, bmi (nếu có)

**Quy trình tích hợp:**
1. Download dataset từ Kaggle (CSV format)
2. Kiểm tra columns có sẵn
3. Map columns từ Kaggle format → format của hệ thống:
   - Ví dụ: Kaggle có "sex" → map thành "gender" (Male/Female)
   - Kaggle có "height_cm" → convert thành "height" (meters)
   - Kaggle có "calories" → map thành "calories_burned"
4. Handle missing values:
   - Drop rows với quá nhiều missing values
   - Fill missing với median/mean cho numeric
   - Fill missing với mode cho categorical
5. Validate data ranges:
   - Age: 1-120
   - Height: 0.5-3.0 meters
   - Weight: 20-300 kg
   - Calories: > 0
6. Save vào `data/raw/fitness_kaggle.csv` hoặc merge với data hiện có
7. Update `train_models.py` để load dataset mới

**Lưu ý:**
- Nếu dataset lớn (>100k records), có thể sample để train nhanh hơn
- Nếu có nhiều datasets, merge lại thành một file
- Đảm bảo data quality trước khi train

---

#### B. Workout/Exercise Dataset (cho Recommendation)
**Mục đích:** Xây dựng database bài tập và train recommendation models

**Các dataset phù hợp:**
- **Exercise Database**: Tên bài tập, difficulty, duration, calories, muscle groups
- **Workout Plans Dataset**: Workout routines, exercises included, target goals
- **Fitness App Data**: User workouts, ratings, preferences

**Các columns cần có:**
- **Item features**: plan_id, name, difficulty (1-5), duration_min, focus (cardio/strength/yoga/etc), calories_burned
- **Optional**: muscle_groups, equipment_needed, description, instructions

**Quy trình tích hợp:**
1. Download workout/exercise dataset
2. Extract và standardize:
   - Tên bài tập → name
   - Độ khó → difficulty (normalize về 1-5 scale)
   - Thời gian → duration_min
   - Loại → focus (map về: cardio, strength, yoga, hiit, core, flexibility)
   - Calories → calories_burned (nếu có)
3. Tạo synthetic data nếu thiếu:
   - Nếu không có calories_burned, estimate dựa trên difficulty và duration
   - Nếu không có focus, categorize dựa trên tên bài tập
4. Save vào `data/train/items.csv` (hoặc merge với file hiện có)
5. Update `hybrid_recommender.py` để load items mới

---

#### C. Meal/Nutrition Dataset (cho Meal Recommendation)
**Mục đích:** Xây dựng database thực đơn và meal recommendation

**Các dataset phù hợp:**
- **Nutrition Database**: Food items, calories, macronutrients (protein, carbs, fat)
- **Recipe Dataset**: Recipes với nutrition info
- **Meal Plan Dataset**: Pre-made meal plans

**Các columns cần có:**
- **Meal features**: meal_id, name, meal_type (breakfast/lunch/dinner/snack), calories, protein, carbs, fat
- **Optional**: description, ingredients, preparation_time

**Quy trình tích hợp:**
1. Download nutrition/meal dataset
2. Categorize meals:
   - Phân loại thành breakfast, lunch, dinner, snack
   - Có thể dựa trên meal name hoặc time of day
3. Standardize nutrition info:
   - Calories: per serving
   - Protein, carbs, fat: grams per serving
4. Filter và clean:
   - Loại bỏ items với missing nutrition info
   - Đảm bảo calories > 0
5. Save vào `data/train/meals.csv`
6. Update `meal_recommender.py` để load meals mới

---

### 1.3 User Interaction Dataset (cho Collaborative Filtering)

**Mục đích:** Train collaborative filtering model tốt hơn

**Các dataset phù hợp:**
- **Fitness App User Data**: User workouts, ratings, preferences
- **Exercise Log Dataset**: User exercise history, completion rates
- **Workout Rating Dataset**: User ratings cho workouts

**Các columns cần có:**
- **user_id**: ID người dùng
- **item_id** hoặc **plan_id**: ID bài tập/workout
- **rating** hoặc **interaction**: Rating (1-5) hoặc binary (completed/not completed)
- **Optional**: timestamp, goal, experience_level

**Quy trình tích hợp:**
1. Download user interaction dataset
2. Map columns:
   - user_id → user_id
   - workout_id/exercise_id → item_id
   - rating/interaction → rating (normalize về 0-5 scale)
3. Filter data:
   - Loại bỏ users với quá ít interactions (< 3)
   - Loại bỏ items với quá ít interactions (< 5)
4. Save vào `data/train/user_interactions.csv`
5. Update `collaborative.py` để load real interactions thay vì synthetic

**Lợi ích:**
- Collaborative filtering sẽ chính xác hơn
- Có thể recommend dựa trên user behavior thực tế
- Hybrid recommender sẽ tốt hơn với real data

---

## PHẦN 2: TRAIN RECOMMENDATION MODELS

### 2.1 Hiện Trạng Recommendation System

**Hiện tại hệ thống đang dùng:**
- **Content-Based**: Rule-based với cosine similarity (không cần train)
- **Collaborative**: SVD với synthetic interactions (không train từ real data)
- **Hybrid**: Combine 2 methods với weights cố định

**Vấn đề:**
- Không có ML model được train
- Weights (0.6, 0.4) là hard-coded
- Không học từ user behavior

---

### 2.2 Cách Train Recommendation Models

#### A. Train Content-Based Model (Nâng Cấp)

**Hiện tại:** Cosine similarity đơn giản

**Có thể nâng cấp thành:**
- **Neural Collaborative Filtering**: Deep learning model
- **Matrix Factorization**: SVD, NMF
- **Feature Learning**: Autoencoder để học features tốt hơn

**Quy trình train (nếu dùng Neural Network):**

1. **Chuẩn bị data:**
   - User features: age, gender, height, weight, goal, experience_level
   - Item features: difficulty, duration, focus, calories
   - User-item interactions: ratings hoặc completion (0/1)

2. **Build model:**
   - Input: User features + Item features
   - Output: Predicted rating/score (0-1 hoặc 1-5)
   - Architecture: Embedding layers → Dense layers → Output

3. **Train:**
   - Loss function: MSE (cho regression) hoặc Binary Crossentropy (cho classification)
   - Optimizer: Adam
   - Metrics: MAE, RMSE, Precision@K, Recall@K

4. **Evaluate:**
   - Split train/test (80/20)
   - Calculate metrics trên test set
   - Tune hyperparameters

5. **Deploy:**
   - Save trained model
   - Load trong `content_based.py`
   - Predict scores thay vì cosine similarity

---

#### B. Train Collaborative Filtering Model (Nâng Cấp)

**Hiện tại:** SVD với synthetic data

**Có thể nâng cấp thành:**
- **Deep Collaborative Filtering**: Neural network
- **Wide & Deep Learning**: Combine linear và deep models
- **Neural Matrix Factorization**: Neural network cho matrix factorization

**Quy trình train:**

1. **Chuẩn bị data:**
   - User-item interaction matrix
   - User features (optional)
   - Item features (optional)

2. **Build model:**
   - **Option 1: Matrix Factorization**
     - User embedding + Item embedding
     - Dot product → predicted rating
   - **Option 2: Neural Collaborative Filtering**
     - User embedding + Item embedding
     - Concatenate → Dense layers → Output

3. **Train:**
   - Loss: MSE hoặc BPR (Bayesian Personalized Ranking)
   - Negative sampling (cho implicit feedback)
   - Train trên user-item pairs

4. **Evaluate:**
   - Precision@K: Top K recommendations có bao nhiêu relevant
   - Recall@K: Bao nhiêu relevant items được recommend
   - NDCG@K: Normalized Discounted Cumulative Gain

5. **Deploy:**
   - Save embeddings và model
   - Load trong `collaborative.py`
   - Predict scores cho user-item pairs

---

#### C. Train Hybrid Recommender Model

**Hiện tại:** Weighted combination (0.6, 0.4) cố định

**Có thể nâng cấp thành:**
- **Learning to Rank**: Train model để học weights tốt nhất
- **Ensemble Model**: Combine multiple recommenders với learned weights
- **Meta-Learning**: Model học cách combine recommendations

**Quy trình train:**

1. **Chuẩn bị data:**
   - User features
   - Content-based scores (từ content-based model)
   - Collaborative scores (từ collaborative model)
   - Ground truth: User actual preferences/ratings

2. **Build model:**
   - Input: Content score + Collaborative score + User features
   - Output: Final recommendation score
   - Architecture: Dense layers → Output
   - Hoặc: Linear combination với learned weights

3. **Train:**
   - Loss: Ranking loss (ListNet, LambdaRank) hoặc MSE
   - Optimize để maximize relevance của top K recommendations

4. **Evaluate:**
   - Precision@K, Recall@K, NDCG@K
   - Compare với fixed weights (0.6, 0.4)

5. **Deploy:**
   - Save learned weights hoặc model
   - Load trong `hybrid_recommender.py`
   - Use learned combination thay vì fixed weights

---

### 2.3 Training Pipeline cho Recommendation

**Quy trình tổng thể:**

```
1. Data Preparation
   ├─► Load user interactions
   ├─► Load item features
   ├─► Create train/test split
   └─► Prepare features

2. Train Content-Based Model (Optional)
   ├─► Build neural network
   ├─► Train on user-item pairs
   └─► Save model

3. Train Collaborative Model
   ├─► Build matrix factorization/neural model
   ├─► Train on interactions
   └─► Save embeddings + model

4. Train Hybrid Model
   ├─► Get scores từ content-based và collaborative
   ├─► Train combination model
   └─► Save hybrid model

5. Evaluate
   ├─► Test on held-out data
   ├─► Calculate metrics
   └─► Compare với baseline

6. Deploy
   ├─► Update recommendation code
   ├─► Load models
   └─► Serve recommendations
```

---

### 2.4 Metrics để Đánh Giá Recommendation

**Ranking Metrics:**
- **Precision@K**: Trong top K recommendations, bao nhiêu là relevant
- **Recall@K**: Bao nhiêu relevant items được recommend trong top K
- **NDCG@K**: Normalized Discounted Cumulative Gain (quality của ranking)
- **MAP**: Mean Average Precision

**Diversity Metrics:**
- **Coverage**: Bao nhiêu items được recommend
- **Diversity**: Độ đa dạng của recommendations

**Business Metrics:**
- **Click-through rate**: User click vào recommendations
- **Conversion rate**: User thực hiện workout/meal
- **User satisfaction**: Ratings từ users

---

### 2.5 So Sánh: Rule-Based vs ML-Based

| Aspect | Rule-Based (Hiện tại) | ML-Based (Nâng cấp) |
|--------|----------------------|---------------------|
| **Training** | Không cần | Cần dataset và training |
| **Accuracy** | Tốt với rules tốt | Tốt hơn nếu có data tốt |
| **Personalization** | Limited | Tốt hơn, học từ behavior |
| **Scalability** | Dễ scale | Cần optimize model |
| **Interpretability** | Dễ hiểu | Khó hơn (black box) |
| **Maintenance** | Dễ maintain | Cần retrain định kỳ |

**Khi nào dùng ML-based:**
- Có dataset user interactions lớn (>10k interactions)
- Muốn personalization tốt hơn
- Có resources để train và maintain models

**Khi nào dùng Rule-based:**
- Dataset nhỏ hoặc không có interactions
- Cần interpretability
- Muốn deploy nhanh

---

## TÓM TẮT

### Để tích hợp Kaggle Dataset:

1. **Cho Prediction Models:**
   - Download fitness/health dataset
   - Map columns về format của hệ thống
   - Preprocess và validate
   - Save vào `data/raw/`
   - Retrain models với data mới

2. **Cho Workout Recommendation:**
   - Download exercise/workout dataset
   - Extract và standardize item features
   - Save vào `data/train/items.csv`
   - Update recommender để load items mới

3. **Cho Meal Recommendation:**
   - Download nutrition/meal dataset
   - Categorize và standardize
   - Save vào `data/train/meals.csv`
   - Update meal recommender

4. **Cho Collaborative Filtering:**
   - Download user interaction dataset
   - Map và filter data
   - Save vào `data/train/user_interactions.csv`
   - Update collaborative recommender

### Để train Recommendation Models:

1. **Content-Based:**
   - Có thể train neural network nếu có user-item interactions
   - Hoặc giữ nguyên cosine similarity (đơn giản, hiệu quả)

2. **Collaborative:**
   - Train matrix factorization hoặc neural model
   - Cần user-item interaction data
   - Evaluate với ranking metrics

3. **Hybrid:**
   - Train model để học weights tốt nhất
   - Combine content và collaborative scores
   - Optimize cho ranking quality

**Lưu ý:** Hiện tại hệ thống đã hoạt động tốt với rule-based approach. Chỉ nên train ML models nếu:
- Có dataset interactions đủ lớn
- Muốn cải thiện accuracy đáng kể
- Có resources để maintain models

