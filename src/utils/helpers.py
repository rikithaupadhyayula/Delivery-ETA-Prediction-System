"""
Utility Functions Module for Delivery ETA Prediction System

This module provides common helper functions used across the project:
- Logging configuration
- Data validation
- Confidence interval calculation
- General utilities
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict, Any
from datetime import datetime
import os
import json


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path to write logs.
        log_format: Custom log format string.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('delivery_eta')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# HAVERSINE DISTANCE (Standalone Version)
# =============================================================================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate the Haversine distance between two points on Earth.
    
    The Haversine formula determines the great-circle distance between
    two points on a sphere given their longitudes and latitudes.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees).
        lat2, lon2: Latitude and longitude of second point (degrees).
    
    Returns:
        float: Distance in kilometers.
    
    Example:
        >>> haversine_distance(31.2304, 121.4737, 31.2397, 121.4996)
        2.891...
    """
    R = 6371.0  # Earth's radius in kilometers
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


# =============================================================================
# CONFIDENCE INTERVAL CALCULATION
# =============================================================================

def calculate_prediction_interval(
    prediction: float,
    model_rmse: float,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate prediction interval for a single prediction.
    
    Uses the model's RMSE to estimate prediction bounds.
    For a normal distribution:
    - 68% confidence: ±1 RMSE
    - 95% confidence: ±1.96 RMSE
    - 99% confidence: ±2.576 RMSE
    
    Args:
        prediction: Point prediction value.
        model_rmse: Model's RMSE from validation.
        confidence_level: Desired confidence level (default 0.95).
    
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    from scipy import stats
    
    # Get z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate bounds
    margin = z_score * model_rmse
    lower = max(0, prediction - margin)  # ETA can't be negative
    upper = prediction + margin
    
    return lower, upper


def calculate_prediction_interval_simple(
    prediction: float,
    model_rmse: float,
    multiplier: float = 1.5
) -> Tuple[float, float]:
    """
    Simple prediction interval calculation without scipy.
    
    Uses a simple multiplier on RMSE for confidence bounds.
    
    Args:
        prediction: Point prediction value.
        model_rmse: Model's RMSE from validation.
        multiplier: Multiplier for RMSE (default 1.5 ≈ 87% confidence).
    
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    margin = multiplier * model_rmse
    lower = max(0, prediction - margin)
    upper = prediction + margin
    
    return lower, upper


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_coordinates(
    lat: float,
    lng: float
) -> Tuple[bool, str]:
    """
    Validate geographic coordinates.
    
    Args:
        lat: Latitude value.
        lng: Longitude value.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        return False, "Coordinates must be numeric"
    
    if lat < -90 or lat > 90:
        return False, f"Latitude must be between -90 and 90, got {lat}"
    
    if lng < -180 or lng > 180:
        return False, f"Longitude must be between -180 and 180, got {lng}"
    
    return True, ""


def validate_timestamp(
    timestamp: Union[str, datetime]
) -> Tuple[bool, str, Optional[datetime]]:
    """
    Validate and parse a timestamp.
    
    Args:
        timestamp: Timestamp string or datetime object.
    
    Returns:
        Tuple of (is_valid, error_message, parsed_datetime).
    """
    if isinstance(timestamp, datetime):
        return True, "", timestamp
    
    if isinstance(timestamp, str):
        try:
            parsed = pd.to_datetime(timestamp)
            return True, "", parsed.to_pydatetime()
        except Exception as e:
            return False, f"Invalid timestamp format: {e}", None
    
    return False, "Timestamp must be string or datetime", None


def validate_prediction_input(
    pickup_lat: float,
    pickup_lng: float,
    drop_lat: float,
    drop_lng: float,
    pickup_time: Union[str, datetime]
) -> Tuple[bool, List[str]]:
    """
    Validate all inputs for a prediction request.
    
    Args:
        pickup_lat, pickup_lng: Pickup coordinates.
        drop_lat, drop_lng: Delivery coordinates.
        pickup_time: Pickup timestamp.
    
    Returns:
        Tuple of (all_valid, list_of_errors).
    """
    errors = []
    
    # Validate pickup coordinates
    valid, msg = validate_coordinates(pickup_lat, pickup_lng)
    if not valid:
        errors.append(f"Pickup location: {msg}")
    
    # Validate drop coordinates
    valid, msg = validate_coordinates(drop_lat, drop_lng)
    if not valid:
        errors.append(f"Drop location: {msg}")
    
    # Validate timestamp
    valid, msg, _ = validate_timestamp(pickup_time)
    if not valid:
        errors.append(f"Pickup time: {msg}")
    
    return len(errors) == 0, errors


# =============================================================================
# DATA FORMATTING
# =============================================================================

def format_eta(hours: float) -> str:
    """
    Format ETA hours into human-readable string.
    
    Args:
        hours: ETA in decimal hours.
    
    Returns:
        str: Formatted string like "1h 30m" or "45m".
    """
    if hours < 0:
        return "Invalid"
    
    total_minutes = int(hours * 60)
    h = total_minutes // 60
    m = total_minutes % 60
    
    if h > 0:
        return f"{h}h {m}m"
    else:
        return f"{m}m"


def format_distance(km: float) -> str:
    """
    Format distance in km to human-readable string.
    
    Args:
        km: Distance in kilometers.
    
    Returns:
        str: Formatted string like "2.5 km" or "800 m".
    """
    if km < 1:
        return f"{int(km * 1000)} m"
    else:
        return f"{km:.1f} km"


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def numpy_to_python(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert.
    
    Returns:
        Python native type.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    return obj


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file with numpy type handling.
    
    Args:
        data: Dictionary to save.
        filepath: Path to save file.
        indent: JSON indentation level.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(numpy_to_python(data), f, indent=indent)


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file.
    
    Returns:
        Dict: Loaded data.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default model parameters
DEFAULT_MODEL_RMSE = 0.5  # Default RMSE in hours for confidence intervals

# Feature engineering defaults
DEFAULT_COURIER_SPEED = 10.0  # km/h
DEFAULT_COURIER_RELIABILITY = 0.5
DEFAULT_COURIER_AVG_ETA = 1.5  # hours
DEFAULT_COURIER_DELIVERIES = 100
DEFAULT_EVENT_COUNT = 3

# Distance categorization thresholds (km)
DISTANCE_SHORT_THRESHOLD = 2.0
DISTANCE_LONG_THRESHOLD = 5.0

# Time period definitions
TIME_PERIODS = {
    'morning': (6, 12),
    'afternoon': (12, 17),
    'evening': (17, 21),
    'night': (21, 6)
}

# Rush hour definitions (7-9 AM, 5-7 PM)
RUSH_HOURS = [7, 8, 9, 17, 18, 19]


# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        str: Absolute path to project root.
    """
    current = os.path.dirname(os.path.abspath(__file__))
    # Go up from src/utils to project root
    return os.path.dirname(os.path.dirname(current))


def get_model_path(model_name: str = "best_model.pkl") -> str:
    """
    Get the full path to a model file.
    
    Args:
        model_name: Name of the model file.
    
    Returns:
        str: Full path to the model file.
    """
    return os.path.join(get_project_root(), "models", model_name)


def get_data_path(filename: str = "processed_data.parquet") -> str:
    """
    Get the full path to a data file.
    
    Args:
        filename: Name of the data file.
    
    Returns:
        str: Full path to the data file.
    """
    return os.path.join(get_project_root(), "dataset", filename)


# Entry point for testing
if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test Haversine
    dist = haversine_distance(31.2304, 121.4737, 31.2397, 121.4996)
    print(f"\nHaversine Distance: {dist:.2f} km")
    print(f"Formatted: {format_distance(dist)}")
    
    # Test ETA formatting
    print(f"\nETA 1.5 hours: {format_eta(1.5)}")
    print(f"ETA 0.25 hours: {format_eta(0.25)}")
    
    # Test confidence intervals
    lower, upper = calculate_prediction_interval_simple(1.5, 0.3)
    print(f"\nPrediction: 1.5 hours")
    print(f"Confidence Interval: [{lower:.2f}, {upper:.2f}] hours")
    
    # Test validation
    valid, errors = validate_prediction_input(
        31.2304, 121.4737,
        31.2397, 121.4996,
        "2024-01-15T10:30:00"
    )
    print(f"\nValidation passed: {valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test paths
    print(f"\nProject root: {get_project_root()}")
    print(f"Model path: {get_model_path()}")
