# ==========================================
# INSURANCE FRAUD DETECTION 
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("insurance_claims.csv")

df.replace("?", np.nan, inplace=True)
df.drop(columns=["policy_number", "_c39"], errors="ignore", inplace=True)

# Date processing
df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
df["incident_date"] = pd.to_datetime(df["incident_date"])

df["policy_year"] = df["policy_bind_date"].dt.year
df["incident_year"] = df["incident_date"].dt.year

df.drop(columns=["policy_bind_date", "incident_date"], inplace=True)

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ------------------------------------------
# 2. MODEL COMPARISON 
# ------------------------------------------
X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train_smote)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {"Accuracy": acc, "F1 Score": f1}

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results).T
results_df.to_excel("model_comparison.xlsx")
print("\nModel comparison saved as model_comparison.xlsx")

# ==========================================
# 3. FINAL DEPLOYMENT MODEL (8 FEATURES ONLY)
# ==========================================
print("\n===== Creating Final Deployment Model (8 Features) =====")

feature_columns = [
    "months_as_customer",
    "age",
    "policy_annual_premium",
    "umbrella_limit",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim"
]

X_final = df[feature_columns]
y_final = df["fraud_reported"]

# Handle imbalance
smote_final = SMOTE(random_state=42)
X_final_smote, y_final_smote = smote_final.fit_resample(X_final, y_final)

# Scale
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final_smote)

# Train final model (Best for deployment: RandomForest)
final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
final_model.fit(X_final_scaled, y_final_smote)

# ------------------------------------------
# 4. SAVE DEPLOYMENT PIPELINE
# ------------------------------------------
joblib.dump({
    "model": final_model,
    "scaler": scaler_final,
    "feature_columns": feature_columns  # ✅ unified name
}, "fraud_pipeline.pkl")

print("\nFinal deployment model saved as fraud_pipeline.pkl ✅")
print("Ready for Flask Dashboard 🚀")