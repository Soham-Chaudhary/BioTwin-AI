import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="BioTwin AI - Your Digital Health Twin", layout="wide", page_icon="🧬")

st.title("BioTwin AI")
st.subheader("Your Real-Time Digital Health Twin • Simulating Your Future Body")


# ====================== AI ENGINE ======================
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'bmi': np.random.normal(25, 5, n).clip(15, 40),
        'sleep_hours': np.random.normal(7, 1.5, n).clip(3, 12),
        'daily_steps': np.random.normal(7000, 3000, n).clip(1000, 20000),
        'diet_score': np.random.randint(1, 11, n),
        'stress_level': np.random.randint(1, 11, n),
        'smoking': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    data['risk_score'] = (
            (data['bmi'] > 30) * 0.4 +
            (data['sleep_hours'] < 6) * 0.3 +
            (data['daily_steps'] < 5000) * 0.25 +
            (data['stress_level'] > 7) * 0.2 +
            data['smoking'] * 0.25 +
            np.random.normal(0, 0.1, n)
    ).clip(0, 1)
    data['risk_label'] = (data['risk_score'] > 0.5).astype(int)

    X = data.drop(['risk_score', 'risk_label'], axis=1)
    y = data['risk_label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns


model, feature_cols = train_model()

# ====================== SIDEBAR INPUTS ======================
st.sidebar.header("Build Your Digital Twin")

age = st.sidebar.slider("Age", 18, 80, 32)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])
weight = st.sidebar.number_input("Weight (kg)", 40, 200, 75)
height = st.sidebar.number_input("Height (cm)", 140, 220, 170)
sleep = st.sidebar.slider("Avg Sleep (hours/day)", 3.0, 12.0, 6.5, 0.1)
steps = st.sidebar.slider("Daily Steps", 1000, 20000, 6500, 100)
diet = st.sidebar.slider("Diet Quality (1-10)", 1, 10, 6)
stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 7)
smoking = st.sidebar.checkbox("Smoker")

st.sidebar.subheader("Describe Your Lifestyle (optional)")
user_text = st.sidebar.text_area("e.g. I smoke occasionally, sleep poorly, love junk food", height=80)

bmi = round(weight / ((height / 100) ** 2), 1)

# Basic NLP auto-adjust
if user_text:
    lower = user_text.lower()
    if any(word in lower for word in ["smoke", "cigarette", "tobacco"]):
        smoking = True
    if any(word in lower for word in ["junk", "fast food", "unhealthy", "fried"]):
        diet = max(1, diet - 2)
    if "sleep" in lower and any(word in lower for word in ["bad", "poor", "little", "less", "insomnia"]):
        sleep = max(3.0, sleep - 1.5)

# ====================== CURRENT TWIN DASHBOARD ======================
col1, col2, col3, col4 = st.columns(4)
col1.metric("BMI", f"{bmi}", "Healthy" if bmi < 25 else "Elevated")
col2.metric("Sleep", f"{sleep} hrs", "Optimal" if sleep >= 7 else "Risky")
col3.metric("Steps", f"{steps:,}", "Active" if steps >= 7500 else "Sedentary")
col4.metric("Current Risk", "Moderate")

st.subheader("🔬 Your Current Digital Twin")
input_data = pd.DataFrame([[age, bmi, sleep, steps, diet, stress, int(smoking)]],
                          columns=feature_cols)
current_risk_prob = model.predict_proba(input_data)[0][1] * 100

# NEW: Risk Category
if current_risk_prob < 30:
    risk_label = "🟢 Low"
elif current_risk_prob < 60:
    risk_label = "🟡 Moderate"
else:
    risk_label = "🔴 High"

st.write(f"### Risk Level: {risk_label}")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=current_risk_prob,
    title={'text': "Simulated 5-Year Metabolic Risk"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "darkblue"},
           'steps': [{'range': [0, 30], 'color': "green"},
                     {'range': [30, 60], 'color': "yellow"},
                     {'range': [60, 100], 'color': "red"}]}))
st.plotly_chart(fig, use_container_width=True)

# ====================== ORGAN IMPACT ======================
st.subheader("Organ-Level Impact Analysis")

organ_scores = {
    "❤️ Heart": 0,
    "🫁 Lungs": 0,
    "🧠 Brain": 0,
    "⚡ Metabolism": 0
}

if smoking:
    organ_scores["❤️ Heart"] += 2
    organ_scores["🫁 Lungs"] += 3
if sleep < 6:
    organ_scores["🧠 Brain"] += 3
    organ_scores["❤️ Heart"] += 1
if diet < 5:
    organ_scores["⚡ Metabolism"] += 3
if steps < 5000:
    organ_scores["⚡ Metabolism"] += 2
    organ_scores["❤️ Heart"] += 1
if stress > 7:
    organ_scores["🧠 Brain"] += 2
    organ_scores["❤️ Heart"] += 1

for organ, score in organ_scores.items():
    level = "🟢 Low" if score <= 1 else "🟡 Moderate" if score <= 3 else "🔴 High"
    st.write(f"**{organ}**: {level} impact")

# NEW: Highlight worst organ
worst = max(organ_scores, key=organ_scores.get)
st.warning(f"⚠️ Most affected system right now: **{worst}**")

# ====================== DIGITAL TWIN INTERPRETATION ======================
st.subheader("Digital Twin Interpretation")

insights = []
if smoking:
    insights.append("🚬 Smoking is accelerating lung and heart damage — projected 35% higher risk in 3 years")
if sleep < 6:
    insights.append("⏰ Chronic short sleep is harming brain recovery and increasing metabolic stress")
if steps < 5000:
    insights.append("🏃‍♂️ Low daily activity is the biggest hidden driver of future metabolic disorder")
if stress > 7:
    insights.append("😟 High stress is silently elevating cardiovascular risk even if other numbers look okay")

for insight in insights:
    st.write("•", insight)
if not insights:
    st.success("✅ Your current habits are well-balanced — keep it up!")

# ====================== FUTURE SIMULATION ======================
st.subheader("⏳ Simulate Your Future Body (1–5 Years)")
scenario = st.radio("Choose scenario",
                    ["Current Lifestyle Continues", "Improved Habits (+1h sleep, +2000 steps, -2 stress, better diet)"],
                    horizontal=True)

years = list(range(1, 6))
risks_current = []
risks_improved = []

for y in years:
    proj_sleep = sleep - y * 0.2
    proj_steps = steps - y * 300
    proj_stress = stress + y * 0.3
    proj_diet = max(1, diet - y * 0.3)

    proj_df_current = pd.DataFrame([[age + y * 5, bmi + y * 0.8, proj_sleep, proj_steps,
                                     proj_diet, proj_stress, int(smoking)]], columns=feature_cols)
    risks_current.append(model.predict_proba(proj_df_current)[0][1] * 100)

    proj_df_imp = pd.DataFrame([[age + y * 5, bmi - y * 0.4, sleep + 1, steps + 2000,
                                 diet + 2, stress - 2, 0]], columns=feature_cols)
    risks_improved.append(model.predict_proba(proj_df_imp)[0][1] * 100)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=years, y=risks_current, name="Current Lifestyle", line=dict(color="red", width=4)))
fig2.add_trace(go.Scatter(x=years, y=risks_improved, name="Improved Lifestyle", line=dict(color="green", width=4)))
fig2.update_layout(title="Your Digital Twin Future Trajectory (1–5 Years)",
                   xaxis_title="Years Ahead",
                   yaxis_title="Risk of Metabolic Disorder (%)",
                   template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

st.success("Improving sleep + exercise dramatically lowers your future risk!")

# NEW: Disclaimer
st.caption("⚠️ This is a simulation tool for awareness, not medical advice.")

st.caption("BioTwin AI • Built for Codecure AI Hackathon 2026 @ IIT BHU by Soham Chaudhary and Akshita Panchal")