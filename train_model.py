"""
train_model.py

This script trains Machine Learning models to predict diabetes based on patient medical metrics.
It handles data loading, automatic schema inspection, preprocessing (imputing invalid zeros to prevent bias),
feature scaling, model training, evaluation, and saving the best model using pickle.

Target Audience: Beginners in Machine Learning.
Detailed explanations are provided in the comments.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    # Force UTF-8 output encoding to prevent Windows UnicodeEncodeErrors
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("="*60)
    print("PHASE 1: Starting Machine Learning Training Pipeline")
    print("="*60)

    # ----------------------------------------------------
    # 1. LOAD THE DATASET
    # ----------------------------------------------------
    data_path = os.path.join("dataset", "diabetes.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at: {data_path}")
        print("Please run the folder setup commands first.")
        return
        
    print(f"[*] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[✓] Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------------------------------
    # 2. AUTOMATIC COLUMN ANALYSIS
    # ----------------------------------------------------
    print("\n[*] Running Automatic Column Analysis:")
    for col in df.columns:
        print(f"    - Column: {col:<25} | Type: {str(df[col].dtype):<10} | Unique Values: {df[col].nunique()}")

    # ----------------------------------------------------
    # 3. PREPROCESSING: SPLITTING DATA FIRST (To Avoid Data Leakage)
    # ----------------------------------------------------
    # In professional Machine Learning, we split the data into Train and Test sets BEFORE
    # calculating medians or scaling. This ensures that the test set remains completely
    # unseen, mimicking real-world clinical usage.
    
    X = df.drop(columns=["Outcome"])  # Features (medical inputs)
    y = df["Outcome"]                 # Target variable (0 = Non-diabetic, 1 = Diabetic)

    # Split into 80% Training data and 20% Testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\n[✓] Data split completed:")
    print(f"    - Training set size: {X_train.shape[0]} rows")
    print(f"    - Testing set size:  {X_test.shape[0]} rows")

    # ----------------------------------------------------
    # 4. PREPROCESSING: IMPUTING PHYSIOLOGICAL ZEROS
    # ----------------------------------------------------
    # Medical features like Glucose, BloodPressure, SkinThickness, Insulin, and BMI
    # cannot physically be zero. A zero value represents missing data.
    # We will compute the median of these features using ONLY the training set,
    # and use those computed values to fill zeros in both train and test sets.
    
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    print("\n[*] Checking for invalid zero values in features...")
    
    # Calculate medians from the training set only
    imputation_values = {}
    for col in zero_columns:
        # Calculate training median for non-zero values
        non_zero_train_values = X_train[X_train[col] != 0][col]
        median_val = non_zero_train_values.median()
        imputation_values[col] = median_val
        
        # Count zeros in train and test
        train_zeros = (X_train[col] == 0).sum()
        test_zeros = (X_test[col] == 0).sum()
        print(f"    - '{col}': found {train_zeros} zeros in train, {test_zeros} in test. Imputing with median: {median_val:.2f}")
        
        # Replace 0s with median
        X_train[col] = X_train[col].replace(0, median_val)
        X_test[col] = X_test[col].replace(0, median_val)

    # Save imputation values directory for backup/app usage if needed
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "imputation_values.pkl"), "wb") as f:
        pickle.dump(imputation_values, f)
    print("[✓] Saved training median imputation values to models/imputation_values.pkl")

    # ----------------------------------------------------
    # 5. FEATURE SCALING
    # ----------------------------------------------------
    # Logistic Regression requires scaled inputs because it relies on gradient descent.
    # We fit the scaler on the training set and transform both training and test sets.
    
    print("\n[*] Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    # Fit on training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test data using the fitted scaler (unseen data normalization)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler so we can normalize user inputs in our Streamlit web app
    with open(os.path.join("models", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("[✓] Saved trained scaler to models/scaler.pkl")

    # ----------------------------------------------------
    # 6. MODEL TRAINING
    # ----------------------------------------------------
    print("\n[*] Training Machine Learning Models...")
    
    # Model A: Logistic Regression (Highly interpretable baseline)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    print("    - [✓] Logistic Regression trained.")
    
    # Model B: Random Forest Classifier (Robust non-linear ensemble)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6)
    rf_model.fit(X_train, y_train)  # Tree-based models do not strictly need scaled data, but we train on raw imputed data
    print("    - [✓] Random Forest trained.")

    # ----------------------------------------------------
    # 7. MODEL EVALUATION & COMPARISON
    # ----------------------------------------------------
    print("\n[*] Evaluating models on unseen test data...")
    
    # Predict using Logistic Regression (requires scaled test data)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    
    # Predict using Random Forest (uses raw/unscaled imputed test data)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    
    print("-" * 50)
    print(f"{'Metric':<25} | {'Logistic Regression':<20} | {'Random Forest':<15}")
    print("-" * 50)
    print(f"{'Accuracy (Overall)':<25} | {f'{lr_acc:.4f}':<20} | {f'{rf_acc:.4f}':<15}")
    print(f"{'Precision (Positive class)':<25} | {f'{lr_prec:.4f}':<20} | {f'{rf_prec:.4f}':<15}")
    print(f"{'Recall (Sensitivity)':<25} | {f'{lr_rec:.4f}':<20} | {f'{rf_rec:.4f}':<15}")
    print(f"{'F1-Score':<25} | {f'{lr_f1:.4f}':<20} | {f'{rf_f1:.4f}':<15}")
    print("-" * 50)

    # ----------------------------------------------------
    # 8. SELECT BEST MODEL AND SAVE
    # ----------------------------------------------------
    # We compare accuracy, but in clinical tasks, F1-Score or Recall is often preferred.
    # For general usability, we select the model with the highest Accuracy.
    
    best_model_name = ""
    best_model = None
    
    if rf_acc >= lr_acc:
        best_model_name = "Random Forest"
        best_model = rf_model
        best_preds = rf_preds
        needs_scaling = False
    else:
        best_model_name = "Logistic Regression"
        best_model = lr_model
        best_preds = lr_preds
        needs_scaling = True

    print(f"\n[✓] Best model selected: {best_model_name} (Accuracy: {max(lr_acc, rf_acc):.4f})")
    
    # Save the chosen model
    model_export_path = os.path.join("models", "diabetes_model.pkl")
    
    # We will save a dictionary containing metadata, which helps the Streamlit app
    # automatically know whether it needs to scale the input data or not!
    model_payload = {
        "model_name": best_model_name,
        "model": best_model,
        "needs_scaling": needs_scaling,
        "feature_names": list(X.columns)
    }
    
    with open(model_export_path, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"[✓] Saved best model payload to: {model_export_path}")

    # ----------------------------------------------------
    # 9. DETAILED CONFUSION MATRIX FOR BEST MODEL
    # ----------------------------------------------------
    print(f"\n[*] Detailed confusion matrix for {best_model_name}:")
    cm = confusion_matrix(y_test, best_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"    - True Negatives (Healthy predicted Healthy): {tn}")
    print(f"    - False Positives (Healthy predicted Diabetic): {fp}")
    print(f"    - False Negatives (Diabetic predicted Healthy): {fn}")
    print(f"    - True Positives (Diabetic predicted Diabetic): {tp}")
    print(f"    - Confusion Matrix layout:\n      [[{tn} {fp}]\n       [{fn} {tp}]]")

    print("\n" + "="*60)
    print("PHASE 1 COMPLETE: Model training, comparison and save finished successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
