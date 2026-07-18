# ============================================================
# AI-Powered Healthcare Diagnosis Assistant
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from datetime import datetime
from google import genai

# ============================================================
# Configure Streamlit Page
# ============================================================

st.set_page_config(
    page_title="AI-Powered Healthcare Diagnosis Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Session State for Prediction History
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0    

# ============================================================
# Load Required Files
# ============================================================

MODEL_PATH = "Saved_Models/logistic_regression.pkl"
ENCODER_PATH = "Saved_Models/label_encoder.pkl"
FEATURE_PATH = "Processed_Data/feature_names.pkl"

required_files = {
    "Model": MODEL_PATH,
    "Label Encoder": ENCODER_PATH,
    "Feature Names": FEATURE_PATH
}

for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):

        st.error(f"❌ {file_name} not found!")

        st.stop()

# ============================================================
# Load Model Files
# ============================================================

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)

feature_names = joblib.load(FEATURE_PATH)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# PDF Report Generator
# ============================================================

def generate_pdf(symptoms, disease):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(
    Paragraph(
        f"<b>Date & Time:</b> {datetime.now().strftime('%d-%m-%Y %I:%M %p')}",
        styles["BodyText"]
    )
)

    story.append(Paragraph("<br/>", styles["BodyText"]))

    story.append(Paragraph("<b>AI-Powered Healthcare Diagnosis Assistant</b>", styles["Title"]))

    story.append(Paragraph("<br/><b>Selected Symptoms</b>", styles["Heading2"]))

    for symptom in symptoms:
            story.append(Paragraph(f"• {symptom}", styles["BodyText"]))

    story.append(Paragraph("<br/><b>Predicted Disease</b>", styles["Heading2"]))

    story.append(Paragraph(disease, styles["BodyText"]))

    story.append(Paragraph("<br/><b>General Health Advice</b>", styles["Heading2"]))

    advice = [
        "Drink plenty of clean water.",
        "Take proper rest.",
        "Eat nutritious food.",
        "Consult a qualified healthcare professional."
    ]

    for item in advice:
        story.append(Paragraph(f"• {item}", styles["BodyText"]))

    story.append(Paragraph("<br/><b>Disclaimer</b>", styles["Heading2"]))

    story.append(
        Paragraph(
            "This report is generated using a Machine Learning model for educational purposes only.",
            styles["BodyText"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


def get_ai_health_advice(disease):

    prompt = f"""
The predicted disease is {disease}.

Give:

1. Disease Description
2. Possible Causes
3. Recommended Diet
4. Precautions
5. When to consult a doctor

Keep it simple.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as e:
        return f"❌ AI Error: {e}"


# ============================================================
# Main Title
# ============================================================

st.title("🩺 AI-Powered Healthcare Diagnosis Assistant")

st.success("✅ Logistic Regression Model Loaded Successfully")

st.success(f"✅ {len(feature_names)} Symptoms Loaded Successfully")

# ============================================================
# Sidebar Navigation
# ============================================================

st.sidebar.title("🩺 Navigation")

st.sidebar.markdown("---")

st.sidebar.subheader("📜 Prediction History")

page = st.sidebar.radio(
    "Select a Page",
    [
        "🏠 Home",
        "🩺 Disease Prediction",
        "ℹ️ About Project"
    ]
)


if len(st.session_state.prediction_history) == 0:

    st.sidebar.info("No predictions yet.")

else:

    for item in reversed(st.session_state.prediction_history[-5:]):

        st.sidebar.success(item["Disease"])

        st.sidebar.markdown("---")

st.sidebar.subheader("📊 Project Statistics")

st.sidebar.metric("🦠 Diseases", len(label_encoder.classes_))

st.sidebar.metric("🤒 Symptoms", len(feature_names))

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model Information")

st.sidebar.write("**Model:** Logistic Regression")

st.sidebar.write("**Version:** 1.0")

st.sidebar.write("**Accuracy:** 86.00%")

# ============================================================
# Home Page
# ============================================================

if page == "🏠 Home":

    st.header("🏠 Welcome")

    st.markdown("""
## AI-Powered Healthcare Diagnosis Assistant

Welcome to the **AI-Powered Healthcare Diagnosis Assistant**.

This application predicts possible diseases using a trained
**Logistic Regression Machine Learning Model** based on the
symptoms selected by the user.

---

### ✨ Features

✅ Predict diseases using AI

✅ Select from **377 symptoms**

✅ Fast and accurate prediction

✅ Easy-to-use interface

✅ Disease Prediction

✅ Interactive Symptom Selection

✅ Beautiful Prediction Result

✅ General Health Advice

✅ Medical Disclaimer

✅ Professional User Interface
---

### 🚀 How to Use

1. Open **Disease Prediction** from the sidebar.
2. Select one or more symptoms.
3. Click **Predict Disease**.
4. View the predicted disease.

---
""")
    
    st.markdown("""
## 🌟 Key Features

✅ AI-based Disease Prediction

✅ Searchable Symptom Selection

✅ Instant Prediction Results

✅ Prediction History

✅ Downloadable PDF Report

✅ General Health Advice

✅ Easy-to-use Interface
""")
    

    col1, col2 = st.columns(2)

    with col1:
     st.info("""
### 🧠 AI Prediction

Predicts diseases using a trained Machine Learning model.
""")

    with col2:
     st.success("""
### 📄 PDF Report

Download your prediction report instantly.
""")

    col3, col4 = st.columns(2)

    with col3:
     st.warning("""
### 📜 History

View all previous predictions during your session.
""")

    with col4:
     st.error("""
### ⚠ Medical Disclaimer

Educational purposes only. Always consult a doctor.
""")

# ============================================================
# About Project Page
# ============================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About Project")

    st.markdown("""
## Project Information

### 🩺 Project Name

**AI-Powered Healthcare Diagnosis Assistant**

---

### 💻 Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Logistic Regression
- Pandas
- NumPy
- Joblib

---

### 🎯 Objective

The objective of this project is to predict diseases
using Machine Learning based on symptoms selected
by the user.

---

### ⚠️ Disclaimer

This application is developed only for educational
and research purposes.

It should **NOT** be considered a substitute for
professional medical advice.

Always consult a qualified healthcare professional.
""")
    
# ============================================================
# Disease Prediction Page
# ============================================================

elif page == "🩺 Disease Prediction":

    st.header("🩺 Disease Prediction")

    st.write(
        "Select one or more symptoms and click the Predict Disease button."
    )

    st.markdown("---")

    # ========================================================
    # Symptom Selection
    # ========================================================

    selected_symptoms = st.multiselect(
        label="🔍 Search and Select Symptoms",
        options=sorted(feature_names),
        placeholder="Type here to search symptoms..."
    )

    st.markdown("")

    # ========================================================
    # Display Selected Symptoms
    # ========================================================

    if len(selected_symptoms) > 0:

        st.success("✅ Selected Symptoms")

        symptom_html = ""

        for symptom in selected_symptoms:

            symptom_html += f"""
            <span style="
                display:inline-block;
                background:#0E7490;
                color:white;
                padding:8px 16px;
                margin:5px;
                border-radius:20px;
                font-size:15px;
                font-weight:bold;
            ">
                {symptom.title()}
            </span>
            """

        st.markdown(
            symptom_html,
            unsafe_allow_html=True
        )

    else:

        st.info("Please select one or more symptoms.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # Predict Button
    # ========================================================

    predict_button = st.button(
        "🔍 Predict Disease",
        use_container_width=True
    )


    # ========================================================
    # Prediction Logic
    # ========================================================

    if predict_button:

        # Check if user selected symptoms
        if len(selected_symptoms) == 0:

            st.warning("⚠️ Please select at least one symptom.")

        else:

            # Create input vector
            input_vector = np.zeros(len(feature_names))

            # Mark selected symptoms as 1
            for symptom in selected_symptoms:

                if symptom in feature_names:

                    index = feature_names.index(symptom)

                    input_vector[index] = 1

            # Convert into DataFrame
            input_df = pd.DataFrame(
                [input_vector],
                columns=feature_names
            )

        with st.spinner("🧠 AI is analyzing your symptoms..."):

            # Predict disease
            prediction = model.predict(input_df)

            predicted_disease = label_encoder.inverse_transform(
                prediction
            )[0]

            # ====================================================
            # Prediction Probability
            # ====================================================

           # probabilities = model.predict_proba(input_df)[0]

           # confidence = np.max(probabilities) * 100

       
    
           # top_3_indices = np.argsort(probabilities)[::-1][:3]

           # top_3_diseases = label_encoder.inverse_transform(top_3_indices)

           # top_3_scores = probabilities[top_3_indices] * 100

            st.markdown("---")

            # ====================================================
            # Prediction Result
            # ====================================================

            st.markdown(
                f"""
                <div style="
                    background:#E8F5E9;
                    padding:25px;
                    border-radius:15px;
                    border-left:8px solid #2E7D32;
                    margin-top:10px;
                ">

                <h2 style="color:#2E7D32;">
                🩺 Predicted Disease
                </h2>

                <h3 style="color:#000;">
                {predicted_disease}
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success("✅ Prediction Completed Successfully!")

            # ============================================================
            # Save Prediction History
            # ============================================================

            st.session_state.prediction_history.append({
            "Disease": predicted_disease,
            "Symptoms": ", ".join(selected_symptoms)
            })

            st.session_state.prediction_count += 1

            st.balloons()    

           # st.metric(
           #     label="🎯 Confidence Score",
           #     value=f"{confidence:.2f}%"
           # )

           # st.markdown("### 🏆 Top 3 Predicted Diseases")

           # for disease, score in zip(top_3_diseases, top_3_scores):

           #     st.progress(float(score / 100))

           #     st.write(
           #         f"**{disease}** — {score:.2f}%"
           #     )


            st.info("""
### 📋 General Health Advice

✔ Drink plenty of clean water.

✔ Take proper rest.

✔ Eat nutritious food.

✔ Avoid self-medication.

✔ Monitor your symptoms regularly.

✔ Consult a qualified healthcare professional.
""")

            st.warning("""
### ⚠ Medical Disclaimer

This prediction is generated using a Machine Learning model.

It is intended only for educational purposes.

Always consult a qualified doctor before taking any medicine.
""")

            pdf = generate_pdf(selected_symptoms, predicted_disease)

            st.download_button(
                label="📄 Download Prediction Report",
                data=pdf,
                file_name="Healthcare_Report.pdf",
                mime="application/pdf",
                use_container_width=True
)

            st.markdown("---")

            st.subheader("🤖 AI Health Assistant")

            with st.spinner("Generating AI Health Advice..."):

             advice = get_ai_health_advice(predicted_disease)

            st.markdown(advice)