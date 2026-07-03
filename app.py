"""
app.py

This is the main file for our Streamlit Web Application.
In Phase 4, we build an advanced, portfolio-grade healthcare dashboard:
1. Performs predictions via the PredictionAgent (ML model).
2. Computes real-time SHAP local explainability explanations via the ExplanationAgent.
3. Translates metrics to empathetic summaries via the GeminiReasoningAgent.
4. Generates lifestyle guidance via the RecommendationAgent.
5. Provides interactive Q&A directly related to patient diagnostic outcomes via the QandAAgent.
6. Generates downloadable PDF reports via the ReportAgent.

To run this application, open your terminal and run:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Multi-Agent modules and report generator
from gemini_agent import (
    PredictionAgent,
    ExplanationAgent,
    GeminiReasoningAgent,
    RecommendationAgent,
    QandAAgent
)
from report_generator import ReportAgent

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Explainable Diabetes Assistant",
    page_icon="🏥",
    layout="wide"
)

# ----------------------------------------------------
# INSTANTIATE AGENTS (Cached to prevent reload delay)
# ----------------------------------------------------
@st.cache_resource
def get_agents():
    try:
        pred_agent = PredictionAgent()
        explain_agent = ExplanationAgent()
        reasoning_agent = GeminiReasoningAgent()
        reco_agent = RecommendationAgent()
        qa_agent = QandAAgent()
        report_agent = ReportAgent()
        return pred_agent, explain_agent, reasoning_agent, reco_agent, qa_agent, report_agent
    except Exception as e:
        st.error(f"Error loading agent components: {str(e)}")
        return None, None, None, None, None, None

pred_agent, explain_agent, reasoning_agent, reco_agent, qa_agent, report_agent = get_agents()

# ----------------------------------------------------
# APPLICATION TITLE & LAYOUT
# ----------------------------------------------------
st.title("🏥 Diabetes Prediction & AI Health Assistant")
st.write(
    "A modular, multi-agent AI platform combining Machine Learning diagnostics "
    "with generative reasoning for patient-friendly healthcare summaries."
)

# Check if the Gemini API Key is configured in the environment
gemini_key = os.getenv("GEMINI_API_KEY")
is_api_key_configured = gemini_key and gemini_key != "your_gemini_api_key_here"

if not is_api_key_configured:
    st.warning(
        "💡 **Gemini AI Integration is inactive**: A valid `GEMINI_API_KEY` was not found in your `.env` file. "
        "The machine learning predictions and local SHAP explanations will work, but the "
        "personalized AI summaries, recommendations, and chatbot require an active API key."
    )

st.markdown("---")

if pred_agent is None:
    st.error("⚠️ Model payload not found. Please train the model first by running `python train_model.py` in your terminal.")
else:
    # ----------------------------------------------------
    # GRID LAYOUT: INPUTS (LEFT) vs OUTPUTS (RIGHT)
    # ----------------------------------------------------
    left_col, right_col = st.columns([2, 3])
    
    with left_col:
        st.header("Patient Clinical Metrics")
        st.write("Enter values below to run predictions:")
        
        # Clinical parameters input widgets
        pregnancies = st.slider("Pregnancies", 0, 20, 1, help="Number of pregnancies")
        glucose = st.slider("Glucose Level (mg/dL)", 0, 300, 115, help="2-hour oral glucose tolerance test level")
        blood_pressure = st.slider("Diastolic Blood Pressure (mmHg)", 0, 150, 70, help="Diastolic pressure")
        skin_thickness = st.slider("Triceps Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")
        insulin = st.slider("2-Hour Insulin (mu U/ml)", 0, 900, 80, help="2-hour serum insulin level")
        bmi = st.slider("Body Mass Index (BMI)", 0.0, 70.0, 25.4, step=0.1, help="Weight in kg / (height in m)^2")
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47, step=0.01, help="Genetic family history index score")
        age = st.slider("Age (Years)", 21, 120, 33, help="Age of patient")
        
        # Package inputs into dictionary
        patient_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }
        
        predict_clicked = st.button("Analyze Patient Data", type="primary", use_container_width=True)

    with right_col:
        # Check if the inputs have changed or predict is clicked to persist state
        if predict_clicked or "last_prediction" in st.session_state:
            
            # Run prediction and fetch AI responses if it is a new run
            if predict_clicked or "last_prediction" not in st.session_state:
                with st.spinner("Analyzing patient metrics using ML model and AI reasoning agents..."):
                    # 1. Prediction Agent (ML execution)
                    prediction = pred_agent.run_prediction(patient_data)
                    
                    # 2. Explanation Agent (Feature influence parsing)
                    explanation = explain_agent.explain_prediction(patient_data, prediction)
                    
                    # 3. Gemini Reasoning & Recommendation Agents (Generative AI)
                    ai_explanation = reasoning_agent.get_explanation(patient_data, prediction, explanation)
                    ai_recommendations = reco_agent.get_recommendations(patient_data, prediction)
                    
                    # Save states in session state to persist tabs clicking and chatbot context
                    st.session_state.last_patient_data = patient_data
                    st.session_state.last_prediction = prediction
                    st.session_state.last_explanation = explanation
                    st.session_state.last_ai_explanation = ai_explanation
                    st.session_state.last_ai_recommendations = ai_recommendations
                    
                    # Reset chat history for the new prediction
                    st.session_state.chat_history = []
            
            # Retrieve cached outputs
            patient_data = st.session_state.last_patient_data
            prediction = st.session_state.last_prediction
            explanation = st.session_state.last_explanation
            ai_explanation = st.session_state.last_ai_explanation
            ai_recommendations = st.session_state.last_ai_recommendations
            
            # --------------------------------------------
            # RENDER TABS
            # --------------------------------------------
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Dashboard", 
                "🔬 SHAP Explanation", 
                "🩺 AI Medical Summary", 
                "🥗 Lifestyle Plan",
                "💬 Chat with Assistant"
            ])
            
            with tab1:
                st.subheader("Model Diagnostic Outcome")
                
                # Risk coloring setup
                prob_percent = prediction["probability"] * 100
                risk_level = prediction["risk_level"]
                if risk_level == "Low Risk":
                    risk_color = "green"
                    risk_bg = "#e6f4ea"
                elif risk_level == "Medium Risk":
                    risk_color = "#f2994a"
                    risk_bg = "#fff8e6"
                else:
                    risk_color = "#eb5757"
                    risk_bg = "#fce8e6"
                    
                diag_text = "Diabetic (High Likelihood)" if prediction["prediction"] == 1 else "Non-Diabetic (Low Likelihood)"
                
                # Styled Risk Card display
                st.markdown(
                    f"""
                    <div style="background-color: {risk_bg}; padding: 25px; border-radius: 10px; border-left: 10px solid {risk_color}; margin-bottom: 25px;">
                        <h2 style="margin-top: 0; color: {risk_color}; font-size: 28px;">{risk_level}</h2>
                        <p style="margin-bottom: 8px; font-size: 18px;"><b>Classification:</b> {diag_text}</p>
                        <p style="margin-bottom: 0; font-size: 18px;"><b>Prediction Probability:</b> {prob_percent:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Top Risk factors listing
                st.subheader("Primary Diagnostic Risk Factors")
                st.write("These values contributed most to the model's prediction score:")
                for factor in explanation["top_risk_factors"]:
                    st.markdown(
                        f"- **{factor['feature']}**: {factor['value']} "
                        f"(elevated relative to normal baseline, adding to prediction risk)"
                    )
                
                # PDF Generation Action
                st.markdown("---")
                st.subheader("Download Assessment Report")
                st.write("Generate a professionally formatted PDF containing ML outputs, patient values, and lifestyle plan:")
                
                if st.button("Generate & Download PDF Report", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        pdf_path = report_agent.generate_pdf_report(
                            patient_data, prediction, ai_explanation, ai_recommendations
                        )
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            label="📥 Download Patient Report (PDF)",
                            data=pdf_bytes,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
            with tab2:
                st.subheader("Local Explainability: SHAP Contribution")
                st.write(
                    "SHAP (SHapley Additive exPlanations) values decompose the model's prediction into contributions "
                    "from individual features. This shows how much each metric pushed the score towards or away from diabetes."
                )
                
                # Visualizing the local SHAP values
                shap_values = explanation["shap_values"]
                shap_df = pd.DataFrame({
                    "Feature": list(shap_values.keys()),
                    "SHAP Value": list(shap_values.values())
                }).sort_values(by="SHAP Value", ascending=True)
                
                # Colors: Red for positive contribution (increases risk), Green for negative (reduces risk)
                colors = ['#eb5757' if val >= 0 else '#27ae60' for val in shap_df['SHAP Value']]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    x="SHAP Value", 
                    y="Feature", 
                    data=shap_df, 
                    ax=ax, 
                    palette=colors
                )
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                ax.set_title("Local Feature Contribution (SHAP Explanation)")
                ax.set_xlabel("Contribution to Prediction (SHAP Value)")
                ax.set_ylabel("Medical Feature")
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info(
                    "🔴 **Red bars** indicate values that increased the patient's predicted diabetes probability.\n"
                    "🟢 **Green bars** indicate values that decreased/protected against diabetes probability."
                )
                
            with tab3:
                st.subheader("Empathic Medical Summary")
                st.write("Generated by the **Gemini Reasoning Agent**:")
                st.markdown(ai_explanation)
                
            with tab4:
                st.subheader("Lifestyle Action Plan")
                st.write("Generated by the **Recommendation Agent**:")
                st.markdown(ai_recommendations)
                
            with tab5:
                st.subheader("Chat with Health Assistant")
                st.write("Ask any questions regarding your prediction results, metrics, or the lifestyle guidelines:")
                
                # Initialize chat history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                # Display chat messages
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                
                # Accept chat input
                if user_question := st.chat_input("Ask a question about your prediction..."):
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(user_question)
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    
                    # Run Q&A Agent
                    with st.spinner("Generating answer..."):
                        answer = qa_agent.answer_question(
                            user_question, 
                            patient_data, 
                            prediction, 
                            ai_explanation
                        )
                    
                    # Display assistant message
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
        else:
            st.info("👈 Enter patient measurements and click 'Analyze Patient Data' on the left to run predictions.")
