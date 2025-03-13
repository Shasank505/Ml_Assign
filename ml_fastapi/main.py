from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional

# Initialize FastAPI
app = FastAPI(
    title="Sales Prediction API",
    description="Predict sales using trained ML models.",
    version="1.2"
)

# Function to load models with error handling
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

# Load ML models
models = {
    "SVM": load_model("models/svm_model.pkl"),
    "Random Forest": load_model("models/random_forest_model.pkl"),
    "KNN": load_model("models/knn_model.pkl"),
    "Gradient Boosting": load_model("models/gradient_boosting_model.pkl"),
}

# Feature names (16 required)
REQUIRED_COLUMNS = {
    "Ship Mode": "Ship_Mode_LE",
    "Segment": "Segment_LE",
    "City": "City_LE",
    "State": "State_LE",
    "Region": "Region_LE",
    "Category": "Category_LE",
    "Sub-Category": "Sub_Category_LE",
    "Avg Category Sales": "Avg_Category_Sales",
    "Avg Sub-Category Sales": "Avg_SubCategory_Sales",
    "Avg State Sales": "Avg_State_Sales",
    "Discount": "Discount",
    "Quantity": "Quantity",
    "Profit": "Profit",
    "Shipping Cost": "Shipping_Cost",
    "Order Processing Time": "Order_Processing_Time",
    "Customer Rating": "Customer_Rating"
}

# Default values for missing fields (fill with 0 or mean values)
DEFAULT_VALUES = {
    "Discount": 0.1,  # Example default value
    "Quantity": 1,
    "Profit": 50.0,
    "Shipping Cost": 10.0,
    "Order Processing Time": 2.0,
    "Customer Rating": 4.0
}

# Define schema with friendly field names
class PredictionInput(BaseModel):
    data: Dict[str, Optional[float]] = Field(
        ..., 
        example={
            "Ship Mode": 2,
            "Segment": 0,
            "City": 194,
            "State": 15,
            "Region": 2,
            "Category": 0,
            "Sub-Category": 4,
            "Avg Category Sales": 227.08,
            "Avg Sub-Category Sales": 314.41,
            "Avg State Sales": 164.89,
            "Discount": 0.15,
            "Quantity": 3,
            "Profit": 120.50,
            "Shipping Cost": 12.0,
            "Order Processing Time": 1.5,
            "Customer Rating": 4.5
        }
    )

    @validator("data")
    def check_missing_fields(cls, value):
        """Ensures all required fields are present, fills missing with defaults"""
        missing_cols = [col for col in REQUIRED_COLUMNS.keys() if col not in value]
        if missing_cols:
            print(f"⚠️ Warning: Missing columns {missing_cols}. Filling with default values.")
            for col in missing_cols:
                value[col] = DEFAULT_VALUES.get(col, 0)  # Fill with default value or 0
        return value

    def get_feature_vector(self):
        """Returns ordered feature vector for prediction"""
        return [self.data[col] for col in REQUIRED_COLUMNS.keys()]

# Unified prediction function with error handling
def make_prediction(model_name: str, data: PredictionInput):
    """Handles prediction logic for any model"""
    model = models.get(model_name)
    if model is None:
        raise HTTPException(status_code=500, detail=f"{model_name} model is not available.")
    
    try:
        # Get correctly ordered feature vector
        input_vector = np.array(data.get_feature_vector()).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_vector)

        return {"model": model_name, "prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")

# API Endpoints
@app.post("/predict/svm")
def predict_svm(data: PredictionInput):
    return make_prediction("SVM", data)

@app.post("/predict/random_forest")
def predict_rf(data: PredictionInput):
    return make_prediction("Random Forest", data)

@app.post("/predict/knn")
def predict_knn(data: PredictionInput):
    return make_prediction("KNN", data)

@app.post("/predict/gradient_boosting")
def predict_gb(data: PredictionInput):
    return make_prediction("Gradient Boosting", data)
