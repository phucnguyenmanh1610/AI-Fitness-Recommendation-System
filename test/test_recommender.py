import numpy as np
import pandas as pd
import pytest

# Import đúng module của bạn
from src.recommendation.recommender import (
    load_items,
    recommend_plans
)

# -----------------------------
# FIXTURE: Load items
# -----------------------------
@pytest.fixture
def items_df():
    df = load_items("data/train/items.csv")
    return df


# -----------------------------
# TEST 1: Load items hoạt động
# -----------------------------
def test_load_items(items_df):
    assert len(items_df) > 0
    assert "plan_id" in items_df.columns
    assert "name" in items_df.columns
    assert "difficulty" in items_df.columns
    assert "duration_min" in items_df.columns
    assert "focus" in items_df.columns


# -----------------------------
# TEST 2: Recommend trả về đúng số lượng
# -----------------------------
def test_recommend_basic(items_df):
    user_profile = np.array([3, 0.6])   # độ khó 3, duration ~ 36m
    result = recommend_plans(user_profile, items_df.copy(), top_n=3)

    assert len(result) == 3
    assert list(result.columns) == ["plan_id", "name", "focus"]


# -----------------------------
# TEST 3: Score phải sorted giảm dần
# -----------------------------
def test_score_sorted(items_df):
    user_profile = np.array([2, 0.5])
    result = recommend_plans(user_profile, items_df.copy(), top_n=10, include_score=True)
    scores = result["score"].values

    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# -----------------------------
# TEST 4: Kiểm tra với nhiều giá trị top_n
# -----------------------------
@pytest.mark.parametrize("n", [1, 5, 10, 50])
def test_recommend_top_n(items_df, n):
    user_profile = np.array([4, 1.0])
    result = recommend_plans(user_profile, items_df.copy(), top_n=n)

    assert len(result) == n


# -----------------------------
# TEST 5: Input lỗi → phải raise
# -----------------------------
def test_invalid_input(items_df):
    with pytest.raises(Exception):
        recommend_plans("invalid", items_df, top_n=3)
