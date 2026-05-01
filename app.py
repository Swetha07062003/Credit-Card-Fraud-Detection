from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load saved model
bundle = joblib.load("models/fraud_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]
columns = bundle["columns"]

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # Encode categorical variables
        df = pd.get_dummies(df)

        # Match training columns
        for col in columns:
            if col not in df:
                df[col] = 0

        df = df[columns]

        # Scale input
        df = scaler.transform(df)

        # Predict probability
        prob = model.predict_proba(df)[0][1]

        # Apply threshold
        prediction = int(prob >= threshold)

        return {
            "fraud_probability": float(prob),
            "prediction": prediction,
            "decision": "FRAUD" if prediction == 1 else "SAFE"
        }

    except Exception as e:
        return {"error": str(e)}