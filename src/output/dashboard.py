import streamlit as st
import matplotlib.pyplot as plt
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def display_dashboard(predictions: dict, recommendations):
    """
    Hiển thị bảng điều khiển (dashboard) cho dự đoán và khuyến nghị fitness.
    Có thể chạy bằng:
        streamlit run src/main.py
    hoặc chỉ chạy python src/main.py (hiển thị console)
    """
    st.title("🏋️ AI Fitness Dashboard")

    # ----- PHẦN DỰ ĐOÁN -----
    st.subheader("🔹 Dự Đoán Sức Khỏe")

    # Lấy giá trị BMI và cal_burned, tránh lỗi kiểu dữ liệu
    bmi = predictions.get("BMI")
    cal_burned = predictions.get("cal_burned")
    cal_intake = predictions.get("cal_intake", 0)

    # Hiển thị an toàn
    if isinstance(bmi, (int, float)):
        st.write(f"BMI: {bmi:.2f}")
    else:
        st.write(f"BMI: {bmi if bmi is not None else 'N/A'}")

    if isinstance(cal_burned, (int, float)):
        st.write(f"Calo Tiêu Thụ: {cal_burned:.2f}")
    else:
        st.write(f"Calo Tiêu Thụ: {cal_burned if cal_burned is not None else 'N/A'}")

    # ----- BIỂU ĐỒ -----
    fig, ax = plt.subplots()
    ax.bar(["Intake", "Burned"], [cal_intake, cal_burned if isinstance(cal_burned, (int, float)) else 0])
    ax.set_ylabel("Calories")
    ax.set_title("Calo Nạp vs Calo Tiêu Thụ")
    st.pyplot(fig)

    # ----- KHUYẾN NGHỊ -----
    st.subheader("💡 Khuyến Nghị")

    if isinstance(recommendations, pd.DataFrame):
        st.dataframe(recommendations)
    elif isinstance(recommendations, list):
        for rec in recommendations:
            if isinstance(rec, dict):
                rec_type = rec.get("type", "Kế hoạch")
                name = rec.get("name", "Không rõ")
                st.write(f"- {rec_type}: {name}")
            else:
                st.write(f"- {rec}")
    else:
        st.write("Không có khuyến nghị phù hợp.")

    logger.info("Dashboard rendered successfully")
