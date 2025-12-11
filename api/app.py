"""
FastAPI Prediction Service for Delivery ETA Prediction System

This module provides a REST API for making ETA predictions:
- POST /predict: Predict delivery ETA given location and time
- GET /health: Health check endpoint

Usage:
    uvicorn api.app:app --reload --port 8000
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np

# Import project modules
from src.features.feature_engineering import prepare_prediction_features, get_feature_names
from src.utils.helpers import (
    validate_prediction_input,
    calculate_prediction_interval_simple,
    format_eta,
    format_distance,
    haversine_distance,
    DEFAULT_COURIER_SPEED,
    DEFAULT_COURIER_RELIABILITY,
    DEFAULT_COURIER_AVG_ETA,
    DEFAULT_COURIER_DELIVERIES,
    DEFAULT_EVENT_COUNT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Delivery ETA Prediction API",
    description="""
    A machine learning API for predicting delivery ETA (Estimated Time of Arrival).
    
    This API uses a trained XGBoost model on the LaDe last-mile delivery dataset
    to predict how long a delivery will take based on:
    - Pickup and delivery locations
    - Time of day and day of week
    - Courier behavior metrics
    - Route complexity
    
    **Key Features:**
    - Real-time ETA predictions
    - Confidence intervals for predictions
    - Input validation
    - Health monitoring
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# GLOBAL MODEL LOADING
# =============================================================================

# Global model storage
model_package: Optional[Dict[str, Any]] = None
model_rmse: float = 0.5  # Default RMSE for confidence intervals


def load_model():
    """Load the trained model on startup."""
    global model_package, model_rmse
    
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "best_model.pkl"
    )
    
    if os.path.exists(model_path):
        try:
            model_package = joblib.load(model_path)
            logger.info(f"Model loaded successfully: {model_package['model_name']}")
            
            # Get RMSE from model metrics
            if 'metrics' in model_package and 'rmse' in model_package['metrics']:
                model_rmse = model_package['metrics']['rmse']
                logger.info(f"Model RMSE: {model_rmse:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    else:
        logger.warning(f"Model file not found at: {model_path}")
        logger.warning("API will return mock predictions until model is trained")
        return False


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting Delivery ETA Prediction API...")
    load_model()
    logger.info("API startup complete")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for ETA prediction."""
    
    pickup_time: str = Field(
        ...,
        description="Pickup timestamp in ISO format (e.g., '2024-01-15T10:30:00')",
        example="2024-01-15T10:30:00"
    )
    pickup_lat: float = Field(
        ...,
        description="Pickup location latitude",
        ge=-90, le=90,
        example=31.2304
    )
    pickup_lng: float = Field(
        ...,
        description="Pickup location longitude",
        ge=-180, le=180,
        example=121.4737
    )
    drop_lat: float = Field(
        ...,
        description="Delivery location latitude",
        ge=-90, le=90,
        example=31.2397
    )
    drop_lng: float = Field(
        ...,
        description="Delivery location longitude",
        ge=-180, le=180,
        example=121.4996
    )
    courier_id: Optional[str] = Field(
        default="unknown",
        description="Courier identifier (optional)",
        example="courier_001"
    )
    event_count: Optional[int] = Field(
        default=DEFAULT_EVENT_COUNT,
        description="Expected number of route events",
        ge=0, le=50,
        example=5
    )
    
    # Optional courier behavior parameters
    courier_avg_speed: Optional[float] = Field(
        default=DEFAULT_COURIER_SPEED,
        description="Courier's average speed in km/h",
        ge=0,
        example=10.0
    )
    courier_reliability: Optional[float] = Field(
        default=DEFAULT_COURIER_RELIABILITY,
        description="Courier reliability score (0-1)",
        ge=0, le=1,
        example=0.75
    )

    @validator('pickup_time')
    def validate_pickup_time(cls, v):
        """Validate that pickup_time is a valid datetime string."""
        try:
            pd.to_datetime(v)
            return v
        except Exception:
            raise ValueError('Invalid datetime format. Use ISO format: YYYY-MM-DDTHH:MM:SS')


class PredictionResponse(BaseModel):
    """Response model for ETA prediction."""
    
    predicted_eta_hours: float = Field(
        ...,
        description="Predicted delivery time in hours"
    )
    predicted_eta_formatted: str = Field(
        ...,
        description="Predicted delivery time in human-readable format"
    )
    lower_bound_hours: float = Field(
        ...,
        description="Lower bound of confidence interval (hours)"
    )
    upper_bound_hours: float = Field(
        ...,
        description="Upper bound of confidence interval (hours)"
    )
    confidence_level: float = Field(
        default=0.87,
        description="Confidence level for the interval"
    )
    distance_km: float = Field(
        ...,
        description="Straight-line distance between pickup and delivery (km)"
    )
    distance_formatted: str = Field(
        ...,
        description="Distance in human-readable format"
    )
    model_used: str = Field(
        ...,
        description="Name of the model used for prediction"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    timestamp: str = Field(..., description="Current server timestamp")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    detail: str = Field(..., description="Error message")
    errors: Optional[list] = Field(None, description="List of validation errors")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Delivery ETA Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check API health status"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API and model loading state.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_package is not None,
        model_name=model_package['model_name'] if model_package else None,
        timestamp=datetime.now().isoformat()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    },
    tags=["Prediction"],
    summary="Predict delivery ETA"
)
async def predict_eta(request: PredictionRequest):
    """
    Predict delivery ETA based on pickup/delivery locations and time.
    
    This endpoint accepts location coordinates and pickup time,
    then returns a predicted ETA with confidence intervals.
    
    **Input Parameters:**
    - pickup_time: When the package will be picked up
    - pickup_lat/lng: Pickup location coordinates
    - drop_lat/lng: Delivery location coordinates
    - courier_id: Optional courier identifier
    - event_count: Expected number of route events
    
    **Returns:**
    - Predicted ETA in hours and formatted string
    - Confidence interval (lower and upper bounds)
    - Distance between locations
    - Model information
    """
    logger.info(f"Received prediction request: pickup=({request.pickup_lat}, {request.pickup_lng}), "
                f"drop=({request.drop_lat}, {request.drop_lng}), time={request.pickup_time}")
    
    try:
        # Validate inputs
        valid, errors = validate_prediction_input(
            request.pickup_lat, request.pickup_lng,
            request.drop_lat, request.drop_lng,
            request.pickup_time
        )
        
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "Invalid input", "errors": errors}
            )
        
        # Calculate distance
        distance_km = haversine_distance(
            request.pickup_lat, request.pickup_lng,
            request.drop_lat, request.drop_lng
        )
        
        # Prepare features for prediction
        features_df = prepare_prediction_features(
            pickup_lat=request.pickup_lat,
            pickup_lng=request.pickup_lng,
            drop_lat=request.drop_lat,
            drop_lng=request.drop_lng,
            pickup_time=request.pickup_time,
            courier_avg_speed=request.courier_avg_speed,
            courier_reliability=request.courier_reliability,
            courier_avg_eta=DEFAULT_COURIER_AVG_ETA,
            courier_total_deliveries=DEFAULT_COURIER_DELIVERIES,
            event_count=request.event_count
        )
        
        # Make prediction
        if model_package is not None:
            # Use trained model
            model = model_package['model']
            
            # Ensure features are in correct order
            feature_names = model_package.get('feature_names', get_feature_names())
            features_df = features_df[feature_names]
            
            prediction = float(model.predict(features_df)[0])
            model_name = model_package['model_name']
            
            logger.info(f"Model prediction: {prediction:.4f} hours")
        else:
            # Fallback: simple distance-based estimation
            # Assume average speed of 15 km/h (accounting for traffic, stops, etc.)
            prediction = distance_km / 15.0 + 0.25  # Add 15 min base time
            model_name = "distance_estimation (no model loaded)"
            
            logger.warning("Using fallback estimation - model not loaded")
        
        # Ensure non-negative prediction
        prediction = max(0.1, prediction)
        
        # Calculate confidence interval
        lower, upper = calculate_prediction_interval_simple(prediction, model_rmse)
        
        # Create response
        response = PredictionResponse(
            predicted_eta_hours=round(prediction, 4),
            predicted_eta_formatted=format_eta(prediction),
            lower_bound_hours=round(lower, 4),
            upper_bound_hours=round(upper, 4),
            confidence_level=0.87,
            distance_km=round(distance_km, 4),
            distance_formatted=format_distance(distance_km),
            model_used=model_name
        )
        
        logger.info(f"Prediction response: ETA={prediction:.2f}h ({format_eta(prediction)})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/reload-model", tags=["Admin"], summary="Reload the prediction model")
async def reload_model():
    """
    Reload the prediction model from disk.
    
    Use this endpoint after training a new model to load it
    without restarting the API server.
    """
    success = load_model()
    
    if success:
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_name": model_package['model_name'] if model_package else None
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
