import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
file_path = "data/creditcard.csv"

if not os.path.exists(file_path):
    print("ERROR: Dataset not found!")
    exit()

print("Dataset found successfully!")

df = pd.read_csv(file_path)

print("\nDataset shape:", df.shape)

# Detect target column
if 'Class' in df.columns:
    target = 'Class'
elif 'is_fraud' in df.columns:
    target = 'is_fraud'
else:
    raise Exception("No target column found!")

print("\nClass distribution:")
print(df[target].value_counts())

# -------------------------------
# 🧹 PREPROCESSING
# -------------------------------

# Drop ID column
if 'transaction_id' in df.columns:
    df = df.drop(columns=['transaction_id'])

# Separate features & target
X = df.drop(columns=[target])
y = df[target]

# 🔥 HANDLE CATEGORICAL DATA (IMPORTANT FIX)
X = pd.get_dummies(X, drop_first=True)

print("\nAfter encoding shape:", X.shape)

# -------------------------------
# 🔀 TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# ⚙️ SCALING
# -------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# ⚖️ SMOTE
# -------------------------------

print("\nApplying SMOTE...")
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train))

# -------------------------------
# 🤖 MODEL TRAINING
# -------------------------------

print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 🔍 PREDICTION
# -------------------------------

# -------------------------------
# 🔍 PROBABILITY PREDICTION
# -------------------------------

y_prob = model.predict_proba(X_test)[:, 1]

# Try custom threshold (tune this)
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)

print(f"\nUsing Threshold: {threshold}")

# -------------------------------
# 📈 EVALUATION
# -------------------------------

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

print("\n✅ Model training complete!")
import joblib

joblib.dump({
    "model": model,
    "scaler": scaler,
    "threshold": threshold,
    "columns": X.columns.tolist()
}, "models/fraud_model.pkl")

print("\n✅ Model saved successfully!")