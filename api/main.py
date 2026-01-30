from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sys

# Add the parent directory to sys.path to resolve any local module imports if needed, 
# though we are just loading pickles.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="NBC Nexus Analytics API", description="API for RUL Prediction and Dealer Segmentation models")

# Paths to models
# Models are now located directly in the api folder
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

RUL_MODEL_PATH = os.path.join(MODEL_DIR, "rul_model.pkl")
DEALER_MODEL_PATH = os.path.join(MODEL_DIR, "dealer_segmentation.pkl")

# Load Models
try:
    rul_model = joblib.load(RUL_MODEL_PATH)
    print(f"RUL Model loaded from {RUL_MODEL_PATH}")
except Exception as e:
    print(f"Failed to load RUL model: {e}")
    rul_model = None

try:
    dealer_bundle = joblib.load(DEALER_MODEL_PATH)
    dealer_model = dealer_bundle['model']
    dealer_scaler = dealer_bundle['scaler']
    dealer_labels = dealer_bundle['label_map']
    print(f"Dealer Segmentation Model loaded from {DEALER_MODEL_PATH}")
except Exception as e:
    print(f"Failed to load Dealer model: {e}")
    dealer_bundle = None

# Pydantic Models for Validation
class RULInput(BaseModel):
    Operating_Hours: float
    RPM: float
    Temperature_C: float
    Vibration_mm_s: float
    Lubrication_Level_Pct: float
    Load_Factor: float

class DealerInput(BaseModel):
    Inventory_Level: int
    Service_Responsiveness_Score: float
    Turnaround_Time_Hrs: float
    Customer_Satisfaction_Index: float

@app.get("/")
def home():
    return {"message": "Welcome to NBC Nexus Analytics API. Use /docs for documentation."}

@app.post("/predict/rul")
def predict_rul(data: RULInput):
    if rul_model is None:
        raise HTTPException(status_code=503, detail="RUL Model is not available")
    
    # Prepare data for prediction
    # model expects a DataFrame or 2D array with specific feature order
    features = [
        data.Operating_Hours, 
        data.RPM, 
        data.Temperature_C, 
        data.Vibration_mm_s, 
        data.Lubrication_Level_Pct, 
        data.Load_Factor
    ]
    
    # Predict
    try:
        prediction = rul_model.predict([features])[0]
        return {"predicted_rul_days": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/dealer-segment")
def predict_dealer_segment(data: DealerInput):
    if dealer_bundle is None:
        raise HTTPException(status_code=503, detail="Dealer Segmentation Model is not available")
    
    # Prepare data
    features = [
        data.Inventory_Level,
        data.Service_Responsiveness_Score,
        data.Turnaround_Time_Hrs,
        data.Customer_Satisfaction_Index
    ]
    
    try:
        # Scale data
        scaled_features = dealer_scaler.transform([features])
        
        # Predict cluster
        cluster = dealer_model.predict(scaled_features)[0]
        
        # Get label
        label = dealer_labels.get(cluster, "Unknown")
        
        return {
            "cluster_id": int(cluster),
            "segment_label": label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
