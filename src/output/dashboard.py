import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def display_dashboard(predictions, recommendations):
    st.title("🏋️ AI Fitness Dashboard")

    # --- Dự đoán sức khỏe ---
    st.subheader("📊 Dự đoán sức khỏe")

    bmi = predictions.get("BMI", 0) if isinstance(predictions, dict) else 0
    cal_burned = predictions.get("cal_burned", 0) if isinstance(predictions, dict) else 0
    cal_intake = predictions.get("cal_intake", 0) if isinstance(predictions, dict) else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("BMI", f"{bmi:.2f}")
    col2.metric("Calo tiêu thụ", f"{cal_burned:.0f}")
    col3.metric("Calo hấp thụ", f"{cal_intake:.0f}")

    # --- Biểu đồ năng lượng ---
    fig, ax = plt.subplots()
    ax.bar(["Hấp thụ", "Tiêu thụ"], [cal_intake, cal_burned], color=["#4CAF50", "#FF7043"])
    ax.set_ylabel("Calories")
    ax.set_title("So sánh năng lượng vào - ra")
    st.pyplot(fig)

    # --- Gợi ý luyện tập / dinh dưỡng ---
    st.subheader("🔥 Gợi ý luyện tập / dinh dưỡng")

    if isinstance(recommendations, pd.DataFrame):
        if not recommendations.empty:
            st.table(recommendations)
        else:
            st.info("Không có khuyến nghị nào được tạo ra.")
    elif isinstance(recommendations, list):
        if recommendations:
            st.table(pd.DataFrame(recommendations))
        else:
            st.info("Không có khuyến nghị nào được tạo ra.")
    else:
        st.warning("Dữ liệu khuyến nghị không hợp lệ.")

    logger.info("Dashboard rendered successfully")
