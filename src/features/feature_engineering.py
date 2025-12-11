"""
Feature Engineering Module for Delivery ETA Prediction System

This module creates features for the ETA prediction model including:
- Geospatial features (distance, direction)
- Temporal features (hour, day, month, weekend)
- Courier behavior features (speed, reliability)
- Event-based features (delays, counts)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# GEOSPATIAL FEATURES
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
        lat1, lon1: Latitude and longitude of first point (in degrees).
        lat2, lon2: Latitude and longitude of second point (in degrees).
    
    Returns:
        float: Distance in kilometers.
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def calculate_haversine_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Haversine distance for entire DataFrame using vectorized operations.
    
    Args:
        df: DataFrame with lat/lon columns for pickup and delivery.
    
    Returns:
        pd.Series: Distance in kilometers for each record.
    """
    # Get coordinate columns
    lat1 = df['accept_gps_lat'].values
    lon1 = df['accept_gps_lng'].values
    lat2 = df['delivery_gps_lat'].values
    lon2 = df['delivery_gps_lng'].values
    
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    # Haversine formula (vectorized)
    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return pd.Series(R * c, index=df.index)


def categorize_distance(distance_km: pd.Series) -> pd.Series:
    """
    Categorize delivery distance into short, medium, or long.
    
    Categories:
    - short: < 2 km
    - medium: 2-5 km
    - long: > 5 km
    
    Args:
        distance_km: Series of distances in kilometers.
    
    Returns:
        pd.Series: Categorical distance labels.
    """
    conditions = [
        distance_km < 2,
        (distance_km >= 2) & (distance_km < 5),
        distance_km >= 5
    ]
    choices = ['short', 'medium', 'long']
    
    return pd.Series(
        np.select(conditions, choices, default='medium'),
        index=distance_km.index
    )


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all geospatial features to the DataFrame.
    
    Features added:
    - haversine_distance_km: Straight-line distance
    - distance_category: short/medium/long classification
    - lat_diff: Latitude difference
    - lng_diff: Longitude difference
    
    Args:
        df: DataFrame with GPS coordinates.
    
    Returns:
        pd.DataFrame: DataFrame with geospatial features added.
    """
    logger.info("Adding geospatial features...")
    df = df.copy()
    
    # Calculate Haversine distance
    df['haversine_distance_km'] = calculate_haversine_vectorized(df)
    
    # Categorize distance
    df['distance_category'] = categorize_distance(df['haversine_distance_km'])
    
    # Add coordinate differences
    df['lat_diff'] = df['delivery_gps_lat'] - df['accept_gps_lat']
    df['lng_diff'] = df['delivery_gps_lng'] - df['accept_gps_lng']
    
    logger.info(f"Distance stats: mean={df['haversine_distance_km'].mean():.2f}km, "
                f"median={df['haversine_distance_km'].median():.2f}km")
    
    return df


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the accept_time (pickup time).
    
    Features added:
    - hour_of_day: Hour (0-23)
    - day_of_week: Day (0=Monday, 6=Sunday)
    - month: Month (1-12)
    - is_weekend: Boolean (Saturday or Sunday)
    - is_rush_hour: Boolean (7-9 AM or 5-7 PM)
    - time_period: morning/afternoon/evening/night
    
    Args:
        df: DataFrame with accept_time column.
    
    Returns:
        pd.DataFrame: DataFrame with temporal features added.
    """
    logger.info("Adding temporal features...")
    df = df.copy()
    
    # Ensure datetime type
    if df['accept_time'].dtype != 'datetime64[ns]':
        df['accept_time'] = pd.to_datetime(df['accept_time'])
    
    # Extract basic temporal features
    df['hour_of_day'] = df['accept_time'].dt.hour
    df['day_of_week'] = df['accept_time'].dt.dayofweek
    df['month'] = df['accept_time'].dt.month
    df['day_of_month'] = df['accept_time'].dt.day
    
    # Weekend flag (Saturday=5, Sunday=6)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Rush hour flag (7-9 AM or 5-7 PM)
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Time period categorization
    conditions = [
        (df['hour_of_day'] >= 6) & (df['hour_of_day'] < 12),   # Morning
        (df['hour_of_day'] >= 12) & (df['hour_of_day'] < 17),  # Afternoon
        (df['hour_of_day'] >= 17) & (df['hour_of_day'] < 21),  # Evening
    ]
    choices = ['morning', 'afternoon', 'evening']
    df['time_period'] = np.select(conditions, choices, default='night')
    
    logger.info(f"Temporal features added. Weekend orders: {df['is_weekend'].sum()}, "
                f"Rush hour orders: {df['is_rush_hour'].sum()}")
    
    return df


# =============================================================================
# COURIER BEHAVIOR FEATURES
# =============================================================================

def calculate_courier_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate courier-level statistics for behavior features.
    
    Statistics calculated per courier:
    - Total deliveries
    - Average delivery speed
    - Average delivery time
    - Reliability score
    
    Args:
        df: DataFrame with courier_id and delivery information.
    
    Returns:
        pd.DataFrame: Courier statistics DataFrame.
    """
    logger.info("Calculating courier statistics...")
    
    # Group by courier
    courier_stats = df.groupby('courier_id').agg({
        'eta_hours': ['count', 'mean', 'std'],
        'haversine_distance_km': ['mean', 'sum']
    }).reset_index()
    
    # Flatten column names
    courier_stats.columns = [
        'courier_id',
        'courier_total_deliveries',
        'courier_avg_eta',
        'courier_eta_std',
        'courier_avg_distance',
        'courier_total_distance'
    ]
    
    # Calculate average speed (km/h) = total_distance / total_time
    total_time = df.groupby('courier_id')['eta_hours'].sum().reset_index()
    total_time.columns = ['courier_id', 'courier_total_time']
    
    courier_stats = courier_stats.merge(total_time, on='courier_id')
    courier_stats['courier_avg_speed'] = (
        courier_stats['courier_total_distance'] / 
        courier_stats['courier_total_time']
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate reliability score (normalized deliveries)
    max_deliveries = courier_stats['courier_total_deliveries'].max()
    courier_stats['courier_reliability'] = (
        courier_stats['courier_total_deliveries'] / max_deliveries
    )
    
    # Fill NaN values
    courier_stats = courier_stats.fillna(0)
    
    return courier_stats


def add_courier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add courier behavior features to the DataFrame.
    
    Features added:
    - courier_avg_speed: Historical average speed (km/h)
    - courier_reliability: Reliability score (0-1)
    - courier_avg_eta: Historical average delivery time
    - courier_total_deliveries: Total completed deliveries
    
    Args:
        df: DataFrame with courier_id column.
    
    Returns:
        pd.DataFrame: DataFrame with courier features added.
    """
    logger.info("Adding courier behavior features...")
    df = df.copy()
    
    # Calculate courier statistics
    courier_stats = calculate_courier_stats(df)
    
    # Select features to merge
    features_to_add = [
        'courier_id',
        'courier_avg_speed',
        'courier_reliability',
        'courier_avg_eta',
        'courier_total_deliveries'
    ]
    
    # Merge with main DataFrame
    df = df.merge(
        courier_stats[features_to_add],
        on='courier_id',
        how='left'
    )
    
    # Fill any missing values with median
    for col in features_to_add[1:]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    logger.info(f"Courier features added. Unique couriers: {df['courier_id'].nunique()}")
    
    return df


# =============================================================================
# EVENT-BASED FEATURES
# =============================================================================

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add event-based features (derived from available features without target leakage).
    
    Features added:
    - event_count: Estimated number of route events based on distance
    - pickup_delay_hours: Estimated delay based on hour of day
    - delivery_complexity: Complexity score based on distance and temporal features
    
    Args:
        df: DataFrame with delivery information.
    
    Returns:
        pd.DataFrame: DataFrame with event features added.
    """
    logger.info("Adding event-based features...")
    df = df.copy()
    
    # Estimate event count based on distance only (no target leakage)
    # More distance = more likely route events
    df['event_count'] = np.clip(
        (df['haversine_distance_km'] / 1.5).astype(int) + 1,
        1, 20
    )
    
    # Estimate pickup delay based on hour of day (busy hours = more delay)
    # Peak hours (11-13, 17-19) have higher delays
    peak_hours = df['hour_of_day'].isin([11, 12, 13, 17, 18, 19])
    df['pickup_delay_hours'] = np.where(peak_hours, 0.3, 0.1)
    
    # Delivery complexity score (0-1) based on distance and temporal features
    # NO target variable used here
    distance_norm = (df['haversine_distance_km'] - df['haversine_distance_km'].min()) / \
                    (df['haversine_distance_km'].max() - df['haversine_distance_km'].min() + 1e-6)
    
    df['delivery_complexity'] = (
        distance_norm * 0.6 + 
        df['is_weekend'] * 0.2 + 
        df['is_rush_hour'] * 0.2
    )
    
    logger.info(f"Event features added. Mean event count: {df['event_count'].mean():.1f}")
    
    return df


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def create_feature_matrix(
    df: pd.DataFrame,
    target_col: str = 'eta_hours',
    include_target: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Create the complete feature matrix for model training.
    
    This is the main function that orchestrates all feature engineering.
    It applies all feature transformations and returns a clean feature matrix.
    
    Args:
        df: Raw/preprocessed DataFrame from load_and_clean.
        target_col: Name of the target variable column.
        include_target: Whether to return the target variable.
    
    Returns:
        Tuple of (X, y) where:
        - X: DataFrame with all engineered features
        - y: Series with target values (or None if include_target=False)
    """
    logger.info("=" * 50)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 50)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Step 1: Add geospatial features
    df = add_geospatial_features(df)
    
    # Step 2: Add temporal features
    df = add_temporal_features(df)
    
    # Step 3: Add courier behavior features
    if 'courier_id' in df.columns:
        df = add_courier_features(df)
    
    # Step 4: Add event-based features
    df = add_event_features(df)
    
    # Define numeric features for the model
    numeric_features = [
        # Geospatial
        'haversine_distance_km',
        'lat_diff',
        'lng_diff',
        
        # Temporal
        'hour_of_day',
        'day_of_week',
        'month',
        'day_of_month',
        'is_weekend',
        'is_rush_hour',
        
        # Courier behavior
        'courier_avg_speed',
        'courier_reliability',
        'courier_avg_eta',
        'courier_total_deliveries',
        
        # Event-based
        'event_count',
        'pickup_delay_hours',
        'delivery_complexity',
    ]
    
    # Keep only features that exist in the DataFrame
    available_features = [f for f in numeric_features if f in df.columns]
    
    # Create feature matrix
    X = df[available_features].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Get target variable
    y = df[target_col].copy() if include_target and target_col in df.columns else None
    
    logger.info("=" * 50)
    logger.info(f"Feature engineering complete!")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    if y is not None:
        logger.info(f"Target shape: {y.shape}")
    logger.info("=" * 50)
    
    return X, y


def get_feature_names() -> List[str]:
    """
    Get the list of feature names used in the model.
    
    Returns:
        List[str]: List of feature names.
    """
    return [
        'haversine_distance_km',
        'lat_diff',
        'lng_diff',
        'hour_of_day',
        'day_of_week',
        'month',
        'day_of_month',
        'is_weekend',
        'is_rush_hour',
        'courier_avg_speed',
        'courier_reliability',
        'courier_avg_eta',
        'courier_total_deliveries',
        'event_count',
        'pickup_delay_hours',
        'delivery_complexity',
    ]


def prepare_prediction_features(
    pickup_lat: float,
    pickup_lng: float,
    drop_lat: float,
    drop_lng: float,
    pickup_time: datetime,
    courier_avg_speed: float = 10.0,
    courier_reliability: float = 0.5,
    courier_avg_eta: float = 1.5,
    courier_total_deliveries: int = 100,
    event_count: int = 3
) -> pd.DataFrame:
    """
    Prepare features for a single prediction request.
    
    This function is used by the API to transform request data
    into the feature format expected by the model.
    
    Args:
        pickup_lat, pickup_lng: Pickup location coordinates.
        drop_lat, drop_lng: Delivery location coordinates.
        pickup_time: Pickup timestamp.
        courier_avg_speed: Courier's average speed (default=10 km/h).
        courier_reliability: Courier reliability score (default=0.5).
        courier_avg_eta: Courier's average ETA (default=1.5 hours).
        courier_total_deliveries: Courier's total deliveries (default=100).
        event_count: Expected number of events (default=3).
    
    Returns:
        pd.DataFrame: Single-row DataFrame with all features.
    """
    # Calculate geospatial features
    distance_km = haversine_distance(pickup_lat, pickup_lng, drop_lat, drop_lng)
    lat_diff = drop_lat - pickup_lat
    lng_diff = drop_lng - pickup_lng
    
    # Extract temporal features
    if isinstance(pickup_time, str):
        pickup_time = pd.to_datetime(pickup_time)
    
    hour_of_day = pickup_time.hour
    day_of_week = pickup_time.weekday()
    month = pickup_time.month
    day_of_month = pickup_time.day
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if hour_of_day in [7, 8, 9, 17, 18, 19] else 0
    
    # Estimate derived features
    pickup_delay_hours = 0.1  # Default small delay
    
    # Estimate complexity
    distance_norm = min(distance_km / 10, 1.0)  # Normalize to ~10km max
    delivery_complexity = distance_norm * 0.5 + is_weekend * 0.1 + is_rush_hour * 0.1
    
    # Create feature dictionary
    features = {
        'haversine_distance_km': distance_km,
        'lat_diff': lat_diff,
        'lng_diff': lng_diff,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'month': month,
        'day_of_month': day_of_month,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'courier_avg_speed': courier_avg_speed,
        'courier_reliability': courier_reliability,
        'courier_avg_eta': courier_avg_eta,
        'courier_total_deliveries': courier_total_deliveries,
        'event_count': event_count,
        'pickup_delay_hours': pickup_delay_hours,
        'delivery_complexity': delivery_complexity,
    }
    
    return pd.DataFrame([features])


# Entry point for direct execution
if __name__ == "__main__":
    # Example usage - requires data from load_and_clean module
    import sys
    sys.path.append('..')
    
    from data.load_and_clean import load_and_preprocess
    
    print("Loading data...")
    df = load_and_preprocess(city="shanghai")
    
    print("\nCreating feature matrix...")
    X, y = create_feature_matrix(df)
    
    print("\nFeature Matrix:")
    print(X.head())
    print(f"\nShape: {X.shape}")
    print(f"\nFeature Statistics:")
    print(X.describe())
    
    # Test single prediction preparation
    print("\n" + "=" * 50)
    print("Testing single prediction feature preparation...")
    test_features = prepare_prediction_features(
        pickup_lat=31.2304,
        pickup_lng=121.4737,
        drop_lat=31.2397,
        drop_lng=121.4996,
        pickup_time=datetime.now()
    )
    print(test_features)
