import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def display_dashboard(predictions: Dict[str, float], recommendations: List[Dict]):
    """
    Render dashboard using Streamlit.
    :param predictions: Dict from prediction
    :param recommendations: List from recommender
    """
    st.title("AI Fitness Dashboard")

    # Display predictions
    st.subheader("Dự Đoán Sức Khỏe")
    st.write(f"BMI: {predictions.get('BMI', 'N/A'):.2f}")
    st.write(f"Calo Tiêu Thụ: {predictions.get('cal_burned', 'N/A'):.2f}")

    # Chart example
    fig, ax = plt.subplots()
    ax.bar(['Intake', 'Burned'], [predictions.get('cal_intake', 0), predictions.get('cal_burned', 0)])
    ax.set_ylabel('Calories')
    st.pyplot(fig)

    # Recommendations
    st.subheader("Khuyến Nghị")
    for rec in recommendations:
        st.write(f"- {rec['type']}: {rec['name']}")

    logger.info("Dashboard rendered")