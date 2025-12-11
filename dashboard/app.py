"""
Streamlit Dashboard for Delivery ETA Prediction System

This dashboard provides:
1. ETA Predictor UI - Interactive form for making predictions
2. Model Insights - Feature importance and error analysis
3. Data Explorer - Visualize the LaDe dataset

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
import requests
import json

# API URL configuration - set this in Streamlit Cloud secrets for production
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import project modules
try:
    from src.utils.helpers import (
        haversine_distance, format_eta, format_distance,
        DEFAULT_COURIER_SPEED, DEFAULT_COURIER_RELIABILITY
    )
    from src.features.feature_engineering import prepare_prediction_features
    import joblib
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Delivery ETA Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-result {
        font-size: 3rem;
        font-weight: bold;
        color: #28a745;
        text-align: center;
    }
    .confidence-interval {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained model."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "best_model.pkl"
    )
    
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"Error loading model: {e}")
            return None
    return None


@st.cache_data
def load_sample_data():
    """Load sample data for visualization."""
    # Try to load from processed data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset", "processed_data.parquet"
    )
    
    if os.path.exists(data_path):
        return pd.read_parquet(data_path)
    
    # Generate sample data if no processed data exists
    np.random.seed(42)
    n_samples = 1000
    
    # Shanghai coordinates
    base_lat, base_lng = 31.2304, 121.4737
    
    sample_data = pd.DataFrame({
        'accept_gps_lat': base_lat + np.random.uniform(-0.1, 0.1, n_samples),
        'accept_gps_lng': base_lng + np.random.uniform(-0.1, 0.1, n_samples),
        'delivery_gps_lat': base_lat + np.random.uniform(-0.1, 0.1, n_samples),
        'delivery_gps_lng': base_lng + np.random.uniform(-0.1, 0.1, n_samples),
        'eta_hours': np.random.exponential(1.5, n_samples),
        'haversine_distance_km': np.random.exponential(3, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'courier_avg_speed': np.random.uniform(5, 20, n_samples),
        'courier_reliability': np.random.uniform(0.3, 1.0, n_samples),
    })
    
    return sample_data


def make_prediction(pickup_lat, pickup_lng, drop_lat, drop_lng, 
                   pickup_time, courier_speed, courier_reliability, event_count):
    """Make a prediction using the loaded model or API."""
    model_package = load_model()
    
    if model_package is not None and MODULES_AVAILABLE:
        # Use loaded model directly
        features_df = prepare_prediction_features(
            pickup_lat=pickup_lat,
            pickup_lng=pickup_lng,
            drop_lat=drop_lat,
            drop_lng=drop_lng,
            pickup_time=pickup_time,
            courier_avg_speed=courier_speed,
            courier_reliability=courier_reliability,
            event_count=event_count
        )
        
        model = model_package['model']
        feature_names = model_package.get('feature_names', list(features_df.columns))
        features_df = features_df[feature_names]
        
        prediction = float(model.predict(features_df)[0])
        rmse = model_package.get('metrics', {}).get('rmse', 0.5)
        
        return {
            'prediction': max(0.1, prediction),
            'lower': max(0, prediction - 1.5 * rmse),
            'upper': prediction + 1.5 * rmse,
            'model_name': model_package['model_name']
        }
    else:
        # Try calling the deployed API
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "pickup_time": pickup_time.isoformat(),
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "drop_lat": drop_lat,
                    "drop_lng": drop_lng,
                    "courier_id": "dashboard_user",
                    "event_count": event_count
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    'prediction': data.get('predicted_eta_hours', 1.0),
                    'lower': data.get('lower_bound_hours', 0.5),
                    'upper': data.get('upper_bound_hours', 1.5),
                    'model_name': f"API ({data.get('model_used', 'xgboost')})"
                }
        except Exception as e:
            st.warning(f"API call failed: {e}")
        
        # Final fallback: distance-based estimation
        distance = haversine_distance(pickup_lat, pickup_lng, drop_lat, drop_lng) if MODULES_AVAILABLE else 5.0
        prediction = distance / 15.0 + 0.25
        
        return {
            'prediction': prediction,
            'lower': max(0, prediction - 0.5),
            'upper': prediction + 0.5,
            'model_name': 'Distance Estimation (Fallback)'
        }


# =============================================================================
# PAGE: ETA PREDICTOR
# =============================================================================

def render_predictor_page():
    """Render the ETA Predictor page."""
    st.markdown('<h1 class="main-header">🚚 Delivery ETA Predictor</h1>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Pickup Location")
        pickup_lat = st.number_input(
            "Latitude", 
            value=31.2304, 
            format="%.6f",
            key="pickup_lat",
            help="Pickup location latitude"
        )
        pickup_lng = st.number_input(
            "Longitude", 
            value=121.4737, 
            format="%.6f",
            key="pickup_lng",
            help="Pickup location longitude"
        )
    
    with col2:
        st.subheader("📦 Delivery Location")
        drop_lat = st.number_input(
            "Latitude", 
            value=31.2397, 
            format="%.6f",
            key="drop_lat",
            help="Delivery location latitude"
        )
        drop_lng = st.number_input(
            "Longitude", 
            value=121.4996, 
            format="%.6f",
            key="drop_lng",
            help="Delivery location longitude"
        )
    
    # Pickup time
    st.subheader("⏰ Pickup Time")
    col3, col4 = st.columns(2)
    with col3:
        pickup_date = st.date_input("Date", value=datetime.now().date())
    with col4:
        pickup_time_input = st.time_input("Time", value=datetime.now().time())
    
    pickup_datetime = datetime.combine(pickup_date, pickup_time_input)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col5, col6, col7 = st.columns(3)
        with col5:
            courier_speed = st.slider(
                "Courier Avg Speed (km/h)", 
                min_value=5.0, 
                max_value=30.0, 
                value=10.0,
                step=0.5
            )
        with col6:
            courier_reliability = st.slider(
                "Courier Reliability", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                step=0.05
            )
        with col7:
            event_count = st.slider(
                "Expected Route Events", 
                min_value=1, 
                max_value=20, 
                value=3
            )
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔮 Predict ETA", type="primary", use_container_width=True):
        with st.spinner("Calculating ETA..."):
            result = make_prediction(
                pickup_lat, pickup_lng,
                drop_lat, drop_lng,
                pickup_datetime,
                courier_speed,
                courier_reliability,
                event_count
            )
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Main prediction
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric(
                label="Predicted ETA",
                value=f"{result['prediction']:.1f} hours",
                delta=f"~{int(result['prediction'] * 60)} minutes"
            )
        
        with col_res2:
            distance = haversine_distance(pickup_lat, pickup_lng, drop_lat, drop_lng) if MODULES_AVAILABLE else 5.0
            st.metric(
                label="Distance",
                value=f"{distance:.2f} km"
            )
        
        with col_res3:
            st.metric(
                label="Confidence Range",
                value=f"{result['lower']:.1f} - {result['upper']:.1f} h"
            )
        
        # Visualization
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Bar(
            x=['Lower Bound', 'Predicted ETA', 'Upper Bound'],
            y=[result['lower'], result['prediction'], result['upper']],
            marker_color=['#17becf', '#1f77b4', '#17becf'],
            text=[f"{result['lower']:.2f}h", f"{result['prediction']:.2f}h", f"{result['upper']:.2f}h"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="ETA Prediction with Confidence Interval",
            yaxis_title="Time (hours)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model info
        st.info(f"**Model Used:** {result['model_name']}")


# =============================================================================
# PAGE: MODEL INSIGHTS
# =============================================================================

def render_insights_page():
    """Render the Model Insights page."""
    st.markdown('<h1 class="main-header">📈 Model Insights</h1>', unsafe_allow_html=True)
    
    model_package = load_model()
    
    if model_package is None:
        st.warning("⚠️ No trained model found. Please run the training pipeline first.")
        st.code("python src/models/train_model.py", language="bash")
        return
    
    # Model info
    st.subheader("🤖 Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", model_package['model_name'])
    with col2:
        metrics = model_package.get('metrics', {})
        st.metric("Test RMSE", f"{metrics.get('rmse', 'N/A'):.4f}" if metrics.get('rmse') else "N/A")
    with col3:
        st.metric("Test MAE", f"{metrics.get('mae', 'N/A'):.4f}" if metrics.get('mae') else "N/A")
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("🎯 Feature Importance")
    
    model = model_package['model']
    feature_names = model_package.get('feature_names', [])
    
    if hasattr(model, 'feature_importances_') and feature_names:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top factors
        st.subheader("🔝 Top Factors Affecting ETA")
        top_features = importance_df.tail(5).iloc[::-1]
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            st.write(f"{i}. **{row['Feature']}** - Importance: {row['Importance']:.4f}")
    
    elif hasattr(model, 'coef_') and feature_names:
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': np.abs(model.coef_)
        }).sort_values('Coefficient', ascending=True)
        
        fig = px.bar(
            coef_df,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Feature Coefficients (Absolute)',
            color='Coefficient',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")
    
    # Error Distribution (simulated)
    st.markdown("---")
    st.subheader("📊 Error Distribution")
    
    # Simulate error distribution
    np.random.seed(42)
    rmse = metrics.get('rmse', 0.5)
    errors = np.random.normal(0, rmse, 1000)
    
    fig = px.histogram(
        errors,
        nbins=50,
        title='Prediction Error Distribution',
        labels={'value': 'Error (hours)', 'count': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: DATA EXPLORER
# =============================================================================

def render_explorer_page():
    """Render the Data Explorer page."""
    st.markdown('<h1 class="main-header">🔍 Data Explorer</h1>', unsafe_allow_html=True)
    
    df = load_sample_data()
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        if 'eta_hours' in df.columns:
            st.metric("Avg ETA", f"{df['eta_hours'].mean():.2f}h")
    with col4:
        if 'haversine_distance_km' in df.columns:
            st.metric("Avg Distance", f"{df['haversine_distance_km'].mean():.2f}km")
    
    st.markdown("---")
    
    # Sample data
    st.subheader("📝 Sample Data")
    st.dataframe(df.head(100), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["ETA Distribution", "Time Patterns", "Location Map"])
    
    with viz_tab1:
        if 'eta_hours' in df.columns:
            fig = px.histogram(
                df[df['eta_hours'] < 10],  # Filter outliers
                x='eta_hours',
                nbins=50,
                title='Distribution of Delivery ETA',
                labels={'eta_hours': 'ETA (hours)'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'haversine_distance_km' in df.columns:
            fig = px.histogram(
                df[df['haversine_distance_km'] < 20],  # Filter outliers
                x='haversine_distance_km',
                nbins=50,
                title='Distribution of Delivery Distance',
                labels={'haversine_distance_km': 'Distance (km)'},
                color_discrete_sequence=['#2ca02c']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        if 'hour_of_day' in df.columns:
            hourly_avg = df.groupby('hour_of_day')['eta_hours'].mean().reset_index()
            fig = px.line(
                hourly_avg,
                x='hour_of_day',
                y='eta_hours',
                title='Average ETA by Hour of Day',
                labels={'hour_of_day': 'Hour', 'eta_hours': 'Avg ETA (hours)'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'day_of_week' in df.columns:
            daily_avg = df.groupby('day_of_week')['eta_hours'].mean().reset_index()
            daily_avg['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig = px.bar(
                daily_avg,
                x='day_name',
                y='eta_hours',
                title='Average ETA by Day of Week',
                labels={'day_name': 'Day', 'eta_hours': 'Avg ETA (hours)'},
                color='eta_hours',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab3:
        st.write("📍 Pickup and Delivery Locations (Sample)")
        
        # Create a sample of locations for the map
        sample_df = df.head(100).copy()
        
        if 'accept_gps_lat' in sample_df.columns and 'accept_gps_lng' in sample_df.columns:
            # Pickup locations
            fig = px.scatter_mapbox(
                sample_df,
                lat='accept_gps_lat',
                lon='accept_gps_lng',
                color='eta_hours' if 'eta_hours' in sample_df.columns else None,
                size='haversine_distance_km' if 'haversine_distance_km' in sample_df.columns else None,
                color_continuous_scale='Viridis',
                zoom=10,
                height=500,
                title='Pickup Locations'
            )
            fig.update_layout(mapbox_style='open-street-map')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Location data not available in the current dataset.")
    
    # Correlation analysis
    st.markdown("---")
    st.subheader("🔗 Feature Correlations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar
    st.sidebar.title("🚚 Delivery ETA")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["🔮 ETA Predictor", "📈 Model Insights", "🔍 Data Explorer"],
        index=0
    )
    
    # Model status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Status")
    
    model_package = load_model()
    if model_package:
        st.sidebar.success(f"✅ Model Loaded: {model_package['model_name']}")
        if 'metrics' in model_package:
            st.sidebar.write(f"RMSE: {model_package['metrics'].get('rmse', 'N/A'):.4f}")
    else:
        st.sidebar.warning("⚠️ No model loaded")
    
    # Render selected page
    if page == "🔮 ETA Predictor":
        render_predictor_page()
    elif page == "📈 Model Insights":
        render_insights_page()
    elif page == "🔍 Data Explorer":
        render_explorer_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center; color: #888;">
            <small>Delivery ETA Prediction System<br>
            Built with Streamlit + XGBoost</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
