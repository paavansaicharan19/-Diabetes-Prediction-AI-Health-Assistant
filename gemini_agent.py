"""
gemini_agent.py

This file contains the multi-agent system architecture for our project.
It loads environment variables from a `.env` file to retrieve the Gemini API key,
and defines clean, modular Python classes for our agents:

1. PredictionAgent - Runs the ML model predictions and risk scoring.
2. ExplanationAgent - Evaluates local risk factors using feature statistics.
3. GeminiReasoningAgent - Uses the Gemini API to explain the prediction.
4. RecommendationAgent - Uses the Gemini API to generate lifestyle suggestions.

By using raw Python classes, we avoid the overhead of complex frameworks (like LangChain)
while maintaining a strict, modular Agentic AI workflow.
"""

import os
import pickle
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from google import genai

# Load variables from .env file (populates GEMINI_API_KEY in environment)
load_dotenv()

# =====================================================================
# 1. PREDICTION AGENT
# =====================================================================
class PredictionAgent:
    """
    Agent responsible for running the trained Machine Learning model.
    It handles loading model pickles, preprocessing inputs, and outputting predictions.
    """
    def __init__(self, model_dir="models"):
        self.model_path = os.path.join(model_dir, "diabetes_model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.model_payload = None
        self.scaler = None
        self._load_assets()

    def _load_assets(self):
        """Loads the model and scaler pickle files."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                "Model files not found! Please run 'python train_model.py' to generate model artifacts."
            )
        with open(self.model_path, "rb") as f:
            self.model_payload = pickle.load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def run_prediction(self, patient_data: dict) -> dict:
        """
        Receives raw patient inputs, normalizes them, and runs predictions.
        Returns prediction category, probability, and risk level.
        """
        # Convert input dictionary to a pandas DataFrame
        input_df = pd.DataFrame([patient_data])
        
        # Scale if required by the model (e.g. Logistic Regression)
        if self.model_payload["needs_scaling"]:
            processed_data = self.scaler.transform(input_df)
        else:
            processed_data = input_df
            
        model = self.model_payload["model"]
        prediction = int(model.predict(processed_data)[0])
        probability = float(model.predict_proba(processed_data)[0][1])
        
        # Define risk levels
        if probability < 0.30:
            risk_level = "Low Risk"
        elif probability < 0.70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        return {
            "prediction": prediction,       # 0 or 1
            "probability": probability,     # Float between 0 and 1
            "risk_level": risk_level        # "Low Risk", "Medium Risk", "High Risk"
        }


# =====================================================================
# 2. EXPLANATION AGENT
# =====================================================================
class ExplanationAgent:
    """
    Agent responsible for explaining WHY the prediction was made.
    It computes local SHAP values for tree-based models (Random Forest)
    and falls back to feature coefficients / clinical deviations if needed.
    """
    def __init__(self, model_dir="models"):
        self.model_path = os.path.join(model_dir, "diabetes_model.pkl")
        with open(self.model_path, "rb") as f:
            self.model_payload = pickle.load(f)

    def explain_prediction(self, patient_data: dict, prediction_results: dict) -> dict:
        """
        Calculates the local SHAP values for the patient metrics.
        Returns a dictionary containing shap_values, base_value, and top risk factors.
        """
        model = self.model_payload["model"]
        feature_names = self.model_payload["feature_names"]
        input_df = pd.DataFrame([patient_data])
        
        try:
            # We attempt to compute actual SHAP values
            if "Random Forest" in self.model_payload["model_name"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                # Check the format of shap_values
                if isinstance(shap_values, list):
                    # List of arrays (class 0, class 1). Index 1 is Diabetic.
                    local_shap = shap_values[1][0]
                elif isinstance(shap_values, np.ndarray):
                    if len(shap_values.shape) == 3:
                        # (n_samples, n_features, n_classes) - sample 0, class 1
                        local_shap = shap_values[0, :, 1]
                    elif len(shap_values.shape) == 2:
                        local_shap = shap_values[0]
                    else:
                        local_shap = shap_values.flatten()
                else:
                    local_shap = np.array(shap_values)
                
                # Extract expected base probability of class 1
                if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) == 2:
                    base_value = float(explainer.expected_value[1])
                else:
                    base_value = float(explainer.expected_value)
            else:
                # Fallback calculation if model is Logistic Regression or other
                # Linear SHAP is directly proportional to (X - X_mean) * Coefs.
                # We approximate it cleanly here:
                coefs = model.coef_[0]
                # Since we don't have X_mean easily accessible here, we approximate:
                local_shap = np.array([float(patient_data[f]) * coefs[i] for i, f in enumerate(feature_names)])
                base_value = 0.35 # average base risk baseline
                
            shap_dict = dict(zip(feature_names, local_shap))
            
            # Identify the top 3 features pushing prediction risk upwards (positive SHAP)
            sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
            top_risk_factors = [
                {"feature": k, "value": patient_data[k], "impact": float(v)} 
                for k, v in sorted_shap if v > 0
            ][:3]
            
            if len(top_risk_factors) == 0:
                top_risk_factors = [
                    {"feature": k, "value": patient_data[k], "impact": float(v)} 
                    for k, v in sorted_shap[:3]
                ]
                
            return {
                "shap_values": shap_dict,
                "base_value": base_value,
                "top_risk_factors": top_risk_factors,
                "all_local_impacts": shap_dict # for backward compatibility
            }
            
        except Exception as e:
            # Absolute fallback to clinical heuristic if SHAP library fails
            print(f"[Warning] SHAP calculation failed: {e}. Falling back to heuristic.")
            global_weights = {f: 1.0 for f in feature_names}
            if hasattr(model, "feature_importances_"):
                global_weights = dict(zip(feature_names, model.feature_importances_))
                
            local_impacts = {}
            for feat in feature_names:
                p_val = float(patient_data[feat])
                weight = global_weights.get(feat, 1.0)
                local_impacts[feat] = (p_val / 100.0) * weight # simple estimate
                
            sorted_factors = sorted(local_impacts.items(), key=lambda x: x[1], reverse=True)
            top_risk_factors = [
                {"feature": k, "value": patient_data[k], "impact": float(v)} 
                for k, v in sorted_factors[:3]
            ]
            return {
                "shap_values": local_impacts,
                "base_value": 0.35,
                "top_risk_factors": top_risk_factors,
                "all_local_impacts": local_impacts
            }


# =====================================================================
# 3. GEMINI REASONING AGENT
# =====================================================================
class GeminiReasoningAgent:
    """
    Agent responsible for translating technical ML data (probabilities, weights)
    into a warm, empathetic, and easily understandable patient explanation.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini Client if the API key is present."""
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            self.client = genai.Client()

    def get_explanation(self, patient_data: dict, prediction: dict, explanation: dict) -> str:
        """
        Sends the diagnostic data to Gemini and returns a layman's translation.
        """
        if self.client is None:
            return (
                "⚠️ Gemini API Key not configured! Please provide a valid key in your `.env` file "
                "to get detailed AI reasoning."
            )
            
        # Format input details for the prompt
        patient_str = "\n".join([f"- {k}: {v}" for k, v in patient_data.items()])
        factors_str = "\n".join([
            f"- {f['feature']}: {f['value']} (Impact score: {f['impact']:.4f})" 
            for f in explanation["top_risk_factors"]
        ])
        
        prompt = f"""
You are a compassionate, expert endocrinologist and medical AI assistant.
A patient has run a Machine Learning model to evaluate their risk of diabetes. Here are their clinical results:

Patient Measurements:
{patient_str}

Machine Learning Model Output:
- Prediction: {"Diabetic" if prediction["prediction"] == 1 else "Non-Diabetic"}
- Risk Level: {prediction["risk_level"]}
- Probability of Diabetes: {prediction["probability"]*100:.2f}%

Top Contributing Risk Factors (identified by the model):
{factors_str}

Please write an explanation of these results in simple, non-technical language.
- Explain what the probability and risk level mean.
- Discuss how their top contributing features relate to diabetes risk.
- Maintain an empathetic, professional tone. Avoid inducing panic, but be clear about the findings.
- Add a standard medical disclaimer at the bottom stating that this analysis is for educational purposes only and not a substitute for professional medical advice.
"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"❌ Error communicating with Gemini API: {str(e)}"


# =====================================================================
# 4. RECOMMENDATION AGENT
# =====================================================================
class RecommendationAgent:
    """
    Agent responsible for compiling custom lifestyle recommendations
    (nutrition, exercise, tracking) tailored to the patient's metrics.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini Client."""
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            self.client = genai.Client()

    def get_recommendations(self, patient_data: dict, prediction: dict) -> str:
        """
        Sends the diagnostic data to Gemini and returns targeted health suggestions.
        """
        if self.client is None:
            return (
                "⚠️ Gemini API Key not configured! Please provide a valid key in your `.env` file "
                "to unlock personalized lifestyle recommendations."
            )
            
        patient_str = "\n".join([f"- {k}: {v}" for k, v in patient_data.items()])
        
        prompt = f"""
You are a certified health coach and clinical dietitian.
Provide personalized lifestyle recommendations for a patient with the following health metrics:

Patient Measurements:
{patient_str}

Machine Learning Predictor Results:
- Risk Category: {prediction["risk_level"]}
- Diabetes Probability: {prediction["probability"]*100:.2f}%

Please generate a comprehensive, personalized health action plan covering:
1. **Dietary Recommendations**: specific foods to incorporate or avoid (tailored to their Glucose and BMI if elevated).
2. **Physical Activity Plan**: type, frequency, and intensity of workouts suited for their age and BMI.
3. **Daily Habits & Monitoring**: tips for tracking progress and other healthy daily practices.

Format your response with clean bullet points and clear, positive language. Do not write a generic template; make it directly relevant to their specific inputs!
"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"❌ Error communicating with Gemini API: {str(e)}"


# =====================================================================
# 5. Q&A AGENT
# =====================================================================
class QandAAgent:
    """
    Agent responsible for answering patient questions about their diabetes prediction.
    Uses the patient measurements, prediction results, and AI clinical summaries as context.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini Client."""
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            self.client = genai.Client()

    def answer_question(self, question: str, patient_data: dict, prediction: dict, context: str) -> str:
        """
        Sends the user question and medical background context to Gemini.
        Returns a friendly, clinical response.
        """
        if self.client is None:
            return "⚠️ Gemini API Key not configured! Cannot answer questions."
            
        patient_str = "\n".join([f"- {k}: {v}" for k, v in patient_data.items()])
        prompt = f"""
You are a compassionate, expert medical assistant and clinical health educator.
A patient has questions about their diabetes risk prediction. Here is their context:

Patient Metrics:
{patient_str}

Model Predictor Output:
- Risk Level: {prediction["risk_level"]}
- Probability: {prediction["probability"]*100:.2f}%

AI Clinical Explanation:
{context}

Patient Question:
"{question}"

Please answer the question based on the medical data and context provided. 
- Keep your tone supportive, clinical, and patient-friendly.
- Provide clear explanations in simple words.
- If the question is completely unrelated to health or their prediction, politely guide them back to topics related to diabetes, diet, exercise, or health monitoring.
- Include a brief medical disclaimer reminder that you are an AI assistant and they should verify details with their doctor.
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"❌ Error: {str(e)}"
