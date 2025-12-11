# 🚚 Delivery ETA Prediction System

A complete end-to-end machine learning project that predicts delivery ETA (Estimated Time of Arrival) in hours using the **LaDe** (Last-mile Delivery) dataset from HuggingFace.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Feature Engineering](#-feature-engineering)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Dashboard](#-dashboard)
- [Contributing](#-contributing)

## 🎯 Overview

This project demonstrates a production-style ML pipeline for predicting how long a last-mile delivery will take. It includes:

- **Data Loading**: Direct integration with HuggingFace datasets
- **Feature Engineering**: Geospatial, temporal, courier behavior, and event-based features
- **Model Training**: Multiple model comparison (Linear Regression, Random Forest, XGBoost)
- **REST API**: FastAPI-based prediction service
- **Dashboard**: Interactive Streamlit visualization and prediction interface

## 📊 Dataset

### LaDe - Last-mile Delivery Dataset

The project uses the **Cainiao-AI/LaDe** dataset from HuggingFace, one of the largest publicly available last-mile delivery datasets.

| Attribute | Value |
|-----------|-------|
| **Source** | [HuggingFace - Cainiao-AI/LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) |
| **Size** | 10,677k packages |
| **Couriers** | 21k couriers |
| **Duration** | 6 months of real-world data |
| **Cities** | Shanghai, Hangzhou, Chongqing, Jilin, Yantai |

### Key Columns

| Column | Description |
|--------|-------------|
| `courier_id` | Unique courier identifier |
| `accept_time` | Package pickup/accept timestamp |
| `delivery_time` | Package delivery completion timestamp |
| `accept_gps_lat/lng` | Pickup location coordinates |
| `delivery_gps_lat/lng` | Delivery location coordinates |
| `order_id` | Unique order identifier |
| `city` | City code (sh, hz, cq, jl, yt) |

## 📁 Project Structure

```
delivery-eta/
│
├── api/
│   ├── __init__.py
│   └── app.py                 # FastAPI prediction service
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                 # Streamlit dashboard
│
├── dataset/
│   └── .gitkeep               # Placeholder for processed data
│
├── models/
│   └── best_model.pkl         # Trained model (after training)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_and_clean.py  # Data loading & preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Feature creation
│   ├── models/
│   │   ├── __init__.py
│   │   └── train_model.py     # Model training pipeline
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Utility functions
│
├── requirements.txt           # Project dependencies
├── README.md                  # This file
└── .gitignore
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd delivery-eta
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Train the Model

Run the training pipeline to download data, engineer features, and train models:

```bash
python src/models/train_model.py
```

This will:
- Load the LaDe dataset from HuggingFace
- Preprocess and clean the data
- Create features (geospatial, temporal, behavioral)
- Train multiple models (Linear Regression, Random Forest, XGBoost)
- Save the best model to `models/best_model.pkl`

Expected output:
```
============================================================
TRAINING SUMMARY
============================================================
Best Model: xgboost
Test MAE: 0.XXXX hours
Test RMSE: 0.XXXX hours
Test R²: 0.XXXX

Top 5 Important Features:
  1. haversine_distance_km: 0.XXXX
  2. hour_of_day: 0.XXXX
  ...
```

### 2. Run the FastAPI Server

Start the prediction API:

```bash
cd delivery-eta
uvicorn api.app:app --reload --port 8000
```

The API will be available at:
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health
- **Predict**: http://localhost:8000/predict (POST)

### 3. Run the Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

Access the dashboard at http://localhost:8501

## 🔧 Feature Engineering

### Geospatial Features

| Feature | Description |
|---------|-------------|
| `haversine_distance_km` | Great-circle distance between pickup and delivery |
| `distance_category` | Categorical: short (<2km), medium (2-5km), long (>5km) |
| `lat_diff` | Latitude difference |
| `lng_diff` | Longitude difference |

### Temporal Features

| Feature | Description |
|---------|-------------|
| `hour_of_day` | Hour (0-23) |
| `day_of_week` | Day (0=Monday, 6=Sunday) |
| `month` | Month (1-12) |
| `is_weekend` | Boolean flag for weekend |
| `is_rush_hour` | Boolean flag for peak hours (7-9 AM, 5-7 PM) |

### Courier Behavior Features

| Feature | Description |
|---------|-------------|
| `courier_avg_speed` | Historical average speed (km/h) |
| `courier_reliability` | Reliability score (0-1) |
| `courier_avg_eta` | Historical average delivery time |
| `courier_total_deliveries` | Total completed deliveries |

### Event-Based Features

| Feature | Description |
|---------|-------------|
| `event_count` | Estimated route events |
| `pickup_delay_hours` | Delay at pickup |
| `delivery_complexity` | Complexity score (0-1) |

## 📈 Model Performance

### Model Comparison

| Model | MAE (hours) | RMSE (hours) | R² Score | Training Time |
|-------|-------------|--------------|----------|---------------|
| Linear Regression | ~0.XX | ~0.XX | ~0.XX | ~Xs |
| Ridge Regression | ~0.XX | ~0.XX | ~0.XX | ~Xs |
| Random Forest | ~0.XX | ~0.XX | ~0.XX | ~Xs |
| **XGBoost** ⭐ | ~0.XX | ~0.XX | ~0.XX | ~Xs |

*Note: Actual values depend on the training run.*

### Best Model: XGBoost

XGBoost is selected as the primary model due to:
- Best balance of accuracy and training speed
- Robust handling of mixed feature types
- Built-in feature importance
- Excellent generalization

## 🌐 API Documentation

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "xgboost",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### `POST /predict`

Predict delivery ETA.

**Request:**
```json
{
  "pickup_time": "2024-01-15T10:30:00",
  "pickup_lat": 31.2304,
  "pickup_lng": 121.4737,
  "drop_lat": 31.2397,
  "drop_lng": 121.4996,
  "courier_id": "courier_001",
  "event_count": 5
}
```

**Response:**
```json
{
  "predicted_eta_hours": 1.25,
  "predicted_eta_formatted": "1h 15m",
  "lower_bound_hours": 0.75,
  "upper_bound_hours": 1.75,
  "confidence_level": 0.87,
  "distance_km": 2.89,
  "distance_formatted": "2.9 km",
  "model_used": "xgboost"
}
```

### Example Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "pickup_time": "2024-01-15T10:30:00",
        "pickup_lat": 31.2304,
        "pickup_lng": 121.4737,
        "drop_lat": 31.2397,
        "drop_lng": 121.4996,
        "courier_id": "courier_001",
        "event_count": 5
    }
)

print(response.json())
```

## 📱 Dashboard

The Streamlit dashboard provides three main pages:

### 1. ETA Predictor
- Interactive form for entering prediction parameters
- Real-time ETA predictions with confidence intervals
- Visual representation of results

### 2. Model Insights
- Feature importance visualization
- Error distribution analysis
- Top factors affecting ETA

### 3. Data Explorer
- Dataset overview and statistics
- ETA and distance distributions
- Time pattern analysis
- Location maps
- Feature correlations

## 🔄 Quick Start Commands

### Using Run Scripts (Recommended)

**Windows:**
```cmd
run.bat setup      # First time: create venv and install dependencies
run.bat train      # Train the ML model
run.bat api        # Start FastAPI server
run.bat dashboard  # Start Streamlit dashboard
run.bat all        # Train + start both servers
```

**Linux/macOS:**
```bash
bash run.sh setup      # First time: create venv and install dependencies
bash run.sh train      # Train the ML model
bash run.sh api        # Start FastAPI server
bash run.sh dashboard  # Start Streamlit dashboard
bash run.sh all        # Train + start both servers
```

### Manual Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/models/train_model.py

# Start API server
uvicorn api.app:app --reload --port 8000

# Start dashboard (in another terminal)
streamlit run dashboard/app.py
```

## 📝 License

This project is for educational and research purposes. The LaDe dataset is provided by Cainiao AI and is subject to their terms of use.

## 🙏 Acknowledgments

- **Cainiao AI** for providing the LaDe dataset
- **HuggingFace** for dataset hosting
- **Scikit-learn**, **XGBoost**, **FastAPI**, and **Streamlit** communities

---
