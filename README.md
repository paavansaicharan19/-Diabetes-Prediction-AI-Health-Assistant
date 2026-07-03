# Agentic AI Powered Explainable Diabetes Prediction & Personalized Health Assistant

This is a portfolio-ready, beginner-friendly clinical assistant application. It uses a **Random Forest** model trained on the Pima Indians Diabetes Dataset to evaluate patient risk, explains predictions locally using **SHAP (SHapley Additive exPlanations)**, integrates **Gemini AI** to write layman-friendly diagnostic analyses and recommendations, and utilizes a **Report Agent** to generate downloadable PDF health summaries.

---

## 🚀 Key Features

* **Phased Data Pipeline**: Automated clean-up of physiological anomalies (placeholder zeros replaced by training medians) and feature normalization.
* **Explainable AI (SHAP)**: Individual patient predictions are explained using real-time local contribution bar charts (red for pushing towards diabetic risk, green for protecting/reducing risk).
* **Multi-Agent Architecture**:
  1. **Prediction Agent**: Evaluates the model statistics to produce risk classification (Low, Medium, High).
  2. **Explanation Agent**: Performs local SHAP value calculations.
  3. **Gemini Reasoning Agent**: Translates complex risk ratios into warm, empathetic layperson descriptions.
  4. **Recommendation Agent**: Formulates a detailed, personalized action plan covering diet and physical workouts.
  5. **Q&A Agent**: A conversational chatbot interface to answer patient questions regarding their metrics.
  6. **Report Agent**: Creates and exports professional PDF reports containing metrics, results, and recommendations.

---

## 📁 Folder Structure

```
Diabetes_Prediction/
│
├── dataset/
│   └── diabetes.csv          # Source dataset
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis & visual checkups
├── models/
│   ├── diabetes_model.pkl    # Random Forest classifier
│   ├── scaler.pkl            # Normalized input standardizer
│   └── imputation_values.pkl # Training dataset medians for placeholder zeros
├── app.py                    # Streamlit Dashboard and Q&A Chat UI
├── train_model.py            # Clean-up, Model comparison & saving script
├── gemini_agent.py           # Core Multi-Agent backend (GenAI & ML wrappers)
├── report_generator.py       # PDF generator logic (Report Agent)
├── requirements.txt          # Python package dependency list
├── .env                      # Local key configurations (ignored by git)
└── README.md                 # Project summary and running instructions
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.8+ (preferably **Python 3.12**) is installed on your computer.

### 2. Install Dependencies
Open your command terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 3. API Key Setup
Create a `.env` file in the root of the project directory and configure your Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*Note: You can get your free Gemini API key from [Google AI Studio](https://aistudio.google.com/).*

---

## 💻 Running the Project

### Step 1: Model Training
To train and export the Machine Learning model, run the training script:
```bash
python train_model.py
```
This compares **Logistic Regression** vs **Random Forest** and saves the best model inside the `models/` folder.

### Step 2: Launch Streamlit Web Application
Run the web application server:
```bash
streamlit run app.py
```
This opens the browser dashboard where you can enter patient metrics, view predictions, check SHAP charts, chat with the AI assistant, and download PDF reports.
