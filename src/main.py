import os
import sys
import logging
import streamlit as st
import pandas as pd

# --- ROOT DIR SETUP ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- IMPORTS ---
from src.data_input.input import load_data, get_synthetic_data
from src.data_input.preprocess import preprocess_data
from src.prediction.model import train_prediction_model, predict_health
from src.recommendation.recommender import load_items, recommend_plans
from src.output.dashboard import display_dashboard

# --- CONFIG ---
logging.basicConfig(level=logging.INFO)
st.title("AI Fitness Dashboard (Multi-Output Prediction Model)")

TARGET_COLS = [
    "calories_burned",
    "bmi",
    "heart_rate",
    "sleep_hours",
    "water_intake",
]

# =========================================================
# STEP 1 — LOAD DATA
# =========================================================
try:
    df_raw = load_data("data/raw/fitness.csv")
except FileNotFoundError:
    logging.warning("CSV file not found → generating synthetic dataset...")
    df_raw = get_synthetic_data(500)

# =========================================================
# STEP 2 — PREPROCESS
# =========================================================
df_processed = preprocess_data(df_raw)

# =========================================================
# STEP 3 — ENSURE daily_steps EXISTS
# =========================================================
if "daily_steps" not in df_processed.columns:
    if "steps" in df_processed.columns:
        df_processed["daily_steps"] = df_processed["steps"]
        logging.info("Mapped steps → daily_steps")
    else:
        logging.warning("No steps column found → creating default daily_steps=5000")
        df_processed["daily_steps"] = 5000

# =========================================================
# STEP 4 — TRAIN MULTI-OUTPUT MODEL
# =========================================================
model_data = train_prediction_model(
    df_processed,
    target_cols=TARGET_COLS,
)

# =========================================================
# STEP 5 — USER INPUT FORM
# =========================================================
st.subheader("Enter User Information for Prediction")

with st.form("user_form"):
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    height = st.number_input("Height (cm)", 120, 220, 170)
    age = st.number_input("Age", 10, 100, 25)

    gender = st.selectbox("Gender", ["male", "female"])
    activity_level = st.selectbox("Activity Level", ["low", "medium", "high"])
    workout_type = st.selectbox("Workout Type", ["cardio", "strength", "yoga"])

    daily_steps = st.number_input("Daily Steps", 0, 40000, 6000)

    submitted = st.form_submit_button("Predict")

# =========================================================
# STEP 6 — RUN AI PREDICTION
# =========================================================
if submitted:
    st.success("Running AI prediction...")

    sample_features = {
        "weight": weight,
        "height": height,
        "age": age,
        "gender": gender,
        "activity_level": activity_level,
        "workout_type": workout_type,
        "daily_steps": daily_steps,
    }

    # AI prediction
    predictions = predict_health(model_data, sample_features)

    # User info
    user_info = {
        "weight": weight,
        "height": height,
        "age": age,
        "true_values": {},  # no actual ground truth
    }

    # =========================================================
    # STEP 7 — RECOMMENDATION ENGINE
    # =========================================================
    items_df = load_items()
    user_profile = [3, daily_steps / 10000]  # difficulty + normalized duration

    recommendations_df = recommend_plans(
        user_profile,
        items_df,
        top_n=3,
        include_score=True,
    )

    # =========================================================
    # STEP 8 — DISPLAY DASHBOARD
    # =========================================================
    display_dashboard(predictions, recommendations_df, user_info)
