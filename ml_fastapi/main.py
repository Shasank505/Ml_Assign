from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load PCA and trained model
pca = joblib.load("pca.pkl")
model = joblib.load("sales_prediction_model.pkl")

app = FastAPI(title="Sales Price Prediction API", description="Predict Sales Price based on input features.")

# Define input schema
class PredictionInput(BaseModel):
    features: list[float]  # Expect a list of numerical features

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input to NumPy array
        input_array = np.array(input_data.features).reshape(1, -1)

        # Check if input matches expected feature count (before PCA transformation)
        expected_features = pca.n_features_in_ if hasattr(pca, 'n_features_in_') else pca.components_.shape[1]
        if input_array.shape[1] != expected_features:
            raise HTTPException(status_code=400, detail=f"Expected {expected_features} features, but got {input_array.shape[1]}")

        # Apply PCA transformation (Keep top 7 components)
        processed_input = pca.transform(input_array)[:, :7]

        # Predict sales
        prediction = model.predict(processed_input)

        return {"predicted_sales": float(prediction[0])}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"ValueError: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
def home():
    return {"message": "Welcome to the Sales Price Prediction API!"}
