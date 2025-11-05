# AI Fitness & Health Recommendation System

## Overview
Hệ thống AI dự đoán sức khỏe và khuyến nghị cá nhân hóa dựa trên ML.

## Setup
1. Clone repo: git clone <url>
2. Install deps: pip install -r requirements.txt
3. Run prototype: python src/main.py

## Modules
- data_input: Thu thập và xử lý dữ liệu.
- prediction: Dự đoán sức khỏe (regression).
- recommendation: Khuyến nghị (hybrid recommender).
- output: Dashboard hiển thị.

## Development Rules
[Paste note chung ở trên vào đây]

## Data Format
- Input: Dict hoặc DF với columns: age (int), gender (str: 'male'/'female'), height (float cm), weight (float kg), daily_steps (int), heart_rate (int), sleep_time (float hours), calorie_intake (float).
- Output Prediction: Dict {'BMI': float, 'cal_burned': float, 'fitness_score': float}
- Output Recommendation: List of dicts [{'type': 'exercise', 'name': 'Running', 'duration': 30}, ...]

## Testing
Run: pytest tests/
