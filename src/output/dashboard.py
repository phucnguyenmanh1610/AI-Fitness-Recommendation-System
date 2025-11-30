import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------
# Premium UI CSS
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>

body {
    font-family: 'Inter', sans-serif;
}

/* Smooth fade */
.block {
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn { 
    from {opacity: 0; transform: translateY(6px);} 
    to   {opacity: 1; transform: translateY(0);} 
}

.metric-card {
    background-color: #11141A;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #272B35;
    margin-bottom: 14px;
    transition: 0.25s ease;
}
.metric-card:hover {
    border-color: #6366F1;
    background-color: #171B22;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #E5E7EB;
    border-left: 4px solid #6366F1;
    padding-left: 10px;
}

.subtext {
    color: #9CA3AF;
    font-size: 14px;
}

.metric-val {
    font-size: 22px;
    font-weight: 600;
    color: white;
}

.divider-line {
    border-bottom: 1px solid #2F333D;
    margin: 28px 0 22px 0;
}

.stPlotlyChart {
    animation: fadeIn 0.8s ease-in-out;
}

</style>
"""


# ----------------------------------------------------------------------
# MAIN DASHBOARD FUNCTION
# ----------------------------------------------------------------------
def display_dashboard(predictions, recommendations, user_info=None):

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="block">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">AI Health & Fitness Dashboard</div>', unsafe_allow_html=True)


    # =====================================================================
    # USER PROFILE
    # =====================================================================
    st.markdown('<div class="section-title">User Profile</div>', unsafe_allow_html=True)

    if user_info:
        weight = user_info.get("weight")
        height_cm = user_info.get("height")
        height_m = height_cm / 100 if height_cm else None
        age = user_info.get("age")
        gender = user_info.get("gender", "male")
        activity = user_info.get("activity_level", "medium")

        # User info cards
        col1, col2, col3 = st.columns(3)
        for col, label, val in [
            (col1, "Weight (kg)", weight),
            (col2, "Height (cm)", height_cm),
            (col3, "Age", age),
        ]:
            col.markdown(f"""
                <div class="metric-card">
                    <div class="subtext">{label}</div>
                    <div class="metric-val">{val}</div>
                </div>
            """, unsafe_allow_html=True)

        # Advanced metrics
        bmi = round(weight / (height_m ** 2), 2)
        bmr = round(10 * weight + 6.25 * height_cm - 5 * age + (5 if gender == "male" else -161), 2)
        activity_map = {"low": 1.2, "medium": 1.55, "high": 1.725}
        tdee = round(bmr * activity_map.get(activity, 1.55), 2)
        body_fat = round(1.20 * bmi + 0.23 * age - 10.8 * (1 if gender == "male" else 0) - 5.4, 2)

        st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Advanced Metrics</div>', unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        metrics = [
            (colA, "BMI", bmi),
            (colB, "BMR (kcal/day)", bmr),
            (colC, "TDEE (kcal/day)", tdee),
        ]
        for col, label, val in metrics:
            col.markdown(f"""
                <div class="metric-card">
                    <div class="subtext">{label}</div>
                    <div class="metric-val">{val}</div>
                </div>
            """, unsafe_allow_html=True)

        colD, colE = st.columns(2)
        colD.markdown(f"""
            <div class="metric-card">
                <div class="subtext">Estimated Body Fat %</div>
                <div class="metric-val">{body_fat}%</div>
            </div>
        """, unsafe_allow_html=True)

        colE.markdown(f"""
            <div class="metric-card">
                <div class="subtext">Recommended Water Intake</div>
                <div class="metric-val">{round(weight * 0.035, 2)} L/day</div>
            </div>
        """, unsafe_allow_html=True)


    # =====================================================================
    # AI PREDICTIONS
    # =====================================================================
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Predicted Health Values</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    i = 0
    for key, value in predictions.items():
        with cols[i % 3]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="subtext">{key.replace('_', ' ').title()}</div>
                    <div class="metric-val">{round(value, 3)}</div>
                </div>
            """, unsafe_allow_html=True)
        i += 1

    pred_df = pd.DataFrame(predictions, index=[0]).T
    pred_df.columns = ["Predicted Value"]
    st.dataframe(pred_df, use_container_width=True)

    # =====================================================================
    # RADAR CHART PRO EDITION (2 LAYERS + GRADIENT + GLOW)
    # =====================================================================

    # =====================================================================
    #   CÁCH 3 — CHIA 2 NHÓM RADAR CHART SIÊU ĐẸP
    # =====================================================================

    def create_radar(categories, values, title):
        # Normalize 0–1
        min_val = min(values)
        max_val = max(values)
        scaled = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in values]

        # Close curve
        theta = categories + [categories[0]]
        scaled_r = scaled + [scaled[0]]
        real_r = values + [values[0]]

        fig = go.Figure()

        # Outer real values
        fig.add_trace(go.Scatterpolar(
            r=real_r,
            theta=theta,
            mode="lines+markers",
            line=dict(color="rgba(99,102,241,0.9)", width=4),
            marker=dict(size=7, color="rgba(255,255,255,0.95)",
                        line=dict(width=2, color="rgba(99,102,241,1)")),
            fill="toself",
            fillcolor="rgba(99,102,241,0.22)",
            hovertemplate="<b>%{theta}</b><br>Value: %{r}<extra></extra>",
            name="Real Value",
        ))

        # Inner scaled layer
        fig.add_trace(go.Scatterpolar(
            r=scaled_r,
            theta=theta,
            mode="lines",
            line=dict(color="rgba(147,197,253,0.8)", width=2),
            fill="toself",
            fillcolor="rgba(147,197,253,0.12)",
            hovertemplate="<b>%{theta}</b><br>Scaled: %{r:.2f}<extra></extra>",
            name="Scaled (0–1)",
        ))

        fig.update_layout(
            template="plotly_dark",
            title=f"<b>{title}</b>",
            title_x=0.5,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(showline=False, gridcolor="rgba(120,120,120,0.3)"),
                angularaxis=dict(gridcolor="rgba(120,120,120,0.3)")
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )

        return fig

    # ===========================================================
    # CREATE 2 RADAR CHARTS
    # ===========================================================

    colR1, colR2 = st.columns(2)

    # Radar 1 – Body Metrics
    body_keys = ["bmi", "heart_rate", "sleep_hours"]
    body_vals = [predictions[k] for k in body_keys]
    fig_body = create_radar(body_keys, body_vals, "Body Metrics Radar")

    # Radar 2 – Daily Activity
    activity_keys = ["calories_burned", "water_intake"]
    activity_vals = [predictions[k] for k in activity_keys]
    fig_activity = create_radar(activity_keys, activity_vals, "Daily Activity Radar")

    colR1.plotly_chart(fig_body, use_container_width=True)
    colR2.plotly_chart(fig_activity, use_container_width=True)

    # =====================================================================
    # RECOMMENDATIONS + BAR CHART
    # =====================================================================
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Personalized Recommendations</div>', unsafe_allow_html=True)

    if len(recommendations) > 0:

        rec_df = recommendations.rename(columns={
            "name": "Workout Name",
            "focus": "Focus Area",
            "score": "Match Score",
        })

        st.dataframe(rec_df, use_container_width=True)

        # BAR CHART
        fig_rec = px.bar(
            rec_df,
            x="Workout Name",
            y="Match Score",
            text="Match Score",
            title="Recommendation Match Score",
            template="plotly_dark",
        )
        fig_rec.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_rec.update_layout(yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig_rec, use_container_width=True)

    else:
        st.info("No recommendations available.")


    st.markdown("</div>", unsafe_allow_html=True)
