from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import uvicorn
import logging
from pathlib import Path
from typing import Optional
from src.logger import get_logger

# Configure logging
logger = get_logger(__name__)

app = FastAPI(
    title="Hotel Reservation Prediction API",
    description="API for predicting hotel reservation cancellations",
    version="1.0.0"
)

# Global variable for model
loaded_model = None
MODEL_OUTPUT_PATH = "artifacts\models\lgbm_model.pkl" 

class PredictionRequest(BaseModel):
    lead_time: int = Field(..., ge=0, le=365, description="Lead time in days (0-365)")
    no_of_special_request: int = Field(..., ge=0, le=10, description="Number of special requests (0-10)")
    avg_price_per_room: float = Field(..., gt=0, le=1000, description="Average price per room (positive value, max 1000)")
    arrival_month: int = Field(..., ge=1, le=12, description="Arrival month (1-12)")
    arrival_date: int = Field(..., ge=1, le=31, description="Arrival date (1-31)")
    market_segment_type: int = Field(..., ge=0, description="Market segment type (encoded)")
    no_of_week_nights: int = Field(..., ge=0, le=30, description="Number of week nights (0-30)")
    no_of_weekend_nights: int = Field(..., ge=0, le=10, description="Number of weekend nights (0-10)")
    type_of_meal_plan: int = Field(..., ge=0, description="Type of meal plan (encoded)")
    room_type_reserved: int = Field(..., ge=0, description="Room type reserved (encoded)")
    
    @validator('arrival_date')
    def validate_date(cls, v, values):
        if 'arrival_month' in values:
            month = values['arrival_month']
            # Basic validation for days in month
            if month in [2] and v > 29:
                raise ValueError('February cannot have more than 29 days')
            elif month in [4, 6, 9, 11] and v > 30:
                raise ValueError('This month cannot have more than 30 days')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    prediction_text: str
    confidence: Optional[float] = None
    model_info: dict

def load_model():
    """Load the ML model with proper error handling"""
    global loaded_model
    try:
        if not Path(MODEL_OUTPUT_PATH).exists():
            logger.error(f"Model file not found at: {MODEL_OUTPUT_PATH}")
            return False
            
        loaded_model = joblib.load(MODEL_OUTPUT_PATH)
        
        # Validate model has required methods
        if not hasattr(loaded_model, 'predict'):
            logger.error("Loaded model doesn't have predict method")
            loaded_model = None
            return False
            
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        loaded_model = None
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.warning("Starting API without model - predictions will fail")

@app.get("/")
def root():
    return {
        "message": "Hotel Reservation Prediction API is running! 🏨",
        "model_status": "loaded" if loaded_model else "not loaded",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        if loaded_model is None:
            raise HTTPException(
                status_code=503, 
                detail="Model not available. Please check server logs."
            )
        
        # Create feature array - ensure this matches your training feature order
        features = np.array([[
            request.lead_time,
            request.no_of_special_request,
            request.avg_price_per_room,
            request.arrival_month,
            request.arrival_date,
            request.market_segment_type,
            request.no_of_week_nights,
            request.no_of_weekend_nights,
            request.type_of_meal_plan,
            request.room_type_reserved
        ]])
        
        # Make prediction
        prediction = loaded_model.predict(features)[0]
        
        # Get prediction text
        prediction_mapping = {
            0: "The customer is likely to cancel the reservation",
            1: "The customer is likely to keep the reservation"
        }
        
        prediction_text = prediction_mapping.get(
            prediction, 
            f"Unexpected prediction value: {prediction}"
        )
        
        # Get confidence score if available
        confidence = None
        if hasattr(loaded_model, 'predict_proba'):
            try:
                proba = loaded_model.predict_proba(features)[0]
                confidence = float(max(proba))
            except Exception as e:
                logger.warning(f"Could not get prediction probability: {e}")
        
        # Model info
        model_info = {
            "model_type": type(loaded_model).__name__,
            "features_used": 10,
            "prediction_classes": [0, 1]
        }
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_text=prediction_text,
            confidence=confidence,
            model_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal prediction error: {str(e)}"
        )

@app.get("/health")
def health_check():
    model_status = "healthy" if loaded_model is not None else "unhealthy"
    
    return {
        "status": "running",
        "model_status": model_status,
        "model_loaded": loaded_model is not None,
        "model_type": type(loaded_model).__name__ if loaded_model else None
    }

@app.post("/reload-model")
def reload_model():
    """Endpoint to reload the model without restarting the server"""
    success = load_model()
    return {
        "success": success,
        "message": "Model reloaded successfully" if success else "Failed to reload model",
        "model_loaded": loaded_model is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )