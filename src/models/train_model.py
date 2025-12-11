"""
Model Training Module for Delivery ETA Prediction System

This module handles:
- Loading processed data
- Training multiple ML models (Linear Regression, Random Forest, XGBoost)
- Evaluating models using MAE and RMSE
- Selecting and saving the best model
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_models() -> Dict[str, Any]:
    """
    Get dictionary of models to train.
    
    Returns:
        Dict[str, Any]: Dictionary mapping model names to model instances.
    """
    models = {
        'linear_regression': LinearRegression(),
        'ridge_regression': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }
    return models


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate model predictions using multiple metrics.
    
    Metrics calculated:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² Score
    - MAPE (Mean Absolute Percentage Error)
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        model_name: Name of the model for logging.
    
    Returns:
        Dict[str, float]: Dictionary of metric names and values.
    """
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (avoiding division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    logger.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    
    return metrics


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_single_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model"
) -> Tuple[Any, float]:
    """
    Train a single model and return it with training time.
    
    Args:
        model: Scikit-learn compatible model instance.
        X_train: Training features.
        y_train: Training target.
        model_name: Name for logging.
    
    Returns:
        Tuple of (trained_model, training_time_seconds).
    """
    logger.info(f"Training {model_name}...")
    start_time = datetime.now()
    
    model.fit(X_train, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"{model_name} trained in {training_time:.2f} seconds")
    
    return model, training_time


def train_and_evaluate_all(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Dict]:
    """
    Train and evaluate all models.
    
    Args:
        X_train, X_test: Training and test features.
        y_train, y_test: Training and test targets.
    
    Returns:
        Dict containing results for each model.
    """
    logger.info("=" * 60)
    logger.info("Starting model training and evaluation")
    logger.info("=" * 60)
    
    models = get_models()
    results = {}
    
    for name, model in models.items():
        logger.info("-" * 40)
        
        # Train model
        trained_model, train_time = train_single_model(model, X_train, y_train, name)
        
        # Make predictions
        y_train_pred = trained_model.predict(X_train)
        y_test_pred = trained_model.predict(X_test)
        
        # Evaluate
        train_metrics = evaluate_model(y_train.values, y_train_pred, f"{name} (train)")
        test_metrics = evaluate_model(y_test.values, y_test_pred, f"{name} (test)")
        
        # Store results
        results[name] = {
            'model': trained_model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'training_time': train_time,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            }
        }
    
    return results


def select_best_model(results: Dict[str, Dict]) -> Tuple[str, Any]:
    """
    Select the best model based on test RMSE.
    
    Args:
        results: Dictionary of training results.
    
    Returns:
        Tuple of (best_model_name, best_model_instance).
    """
    logger.info("=" * 60)
    logger.info("Selecting best model based on test RMSE")
    logger.info("=" * 60)
    
    best_name = None
    best_rmse = float('inf')
    best_model = None
    
    # Create comparison table
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Time (s)':>12}")
    print("=" * 80)
    
    for name, result in results.items():
        test_metrics = result['test_metrics']
        train_time = result['training_time']
        
        print(f"{name:<25} {test_metrics['mae']:>10.4f} {test_metrics['rmse']:>10.4f} "
              f"{test_metrics['r2']:>10.4f} {train_time:>12.2f}")
        
        if test_metrics['rmse'] < best_rmse:
            best_rmse = test_metrics['rmse']
            best_name = name
            best_model = result['model']
    
    print("=" * 80)
    print(f"\n🏆 Best Model: {best_name} (RMSE: {best_rmse:.4f})")
    
    logger.info(f"Best model selected: {best_name} with RMSE: {best_rmse:.4f}")
    
    return best_name, best_model


def get_feature_importance(
    model: Any,
    feature_names: list,
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Extract feature importance from the model.
    
    Args:
        model: Trained model.
        feature_names: List of feature names.
        model_name: Name of the model.
    
    Returns:
        pd.DataFrame: Feature importance DataFrame.
    """
    importance = None
    
    # Try different methods to get feature importance
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest, XGBoost)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_)
    
    if importance is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nFeature Importance ({model_name}):")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    return pd.DataFrame()


# =============================================================================
# MODEL SAVING/LOADING
# =============================================================================

def save_model(
    model: Any,
    model_name: str,
    feature_names: list,
    metrics: Dict,
    save_dir: str = "models"
) -> str:
    """
    Save the trained model with metadata.
    
    Args:
        model: Trained model instance.
        model_name: Name of the model.
        feature_names: List of feature names.
        metrics: Model evaluation metrics.
        save_dir: Directory to save the model.
    
    Returns:
        str: Path to the saved model file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model package with metadata
    model_package = {
        'model': model,
        'model_name': model_name,
        'feature_names': feature_names,
        'metrics': metrics,
        'created_at': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    # Save as best_model.pkl
    save_path = os.path.join(save_dir, 'best_model.pkl')
    joblib.dump(model_package, save_path)
    
    logger.info(f"Model saved to: {save_path}")
    
    return save_path


def load_model(model_path: str = "models/best_model.pkl") -> Dict:
    """
    Load a saved model with its metadata.
    
    Args:
        model_path: Path to the model file.
    
    Returns:
        Dict: Model package with model and metadata.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model_package = joblib.load(model_path)
    logger.info(f"Model loaded from: {model_path}")
    logger.info(f"Model: {model_package['model_name']}, Created: {model_package['created_at']}")
    
    return model_package


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_training_pipeline(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model_flag: bool = True,
    model_dir: str = "models",
    sample_size: Optional[int] = 50000
) -> Dict:
    """
    Run the complete training pipeline.
    
    This is the main entry point for training models:
    1. Load processed data
    2. Create feature matrix
    3. Split into train/test
    4. Train all models
    5. Evaluate and select best
    6. Save best model
    
    Args:
        data_path: Path to processed data (optional, will load from HuggingFace if not provided).
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.
        save_model_flag: Whether to save the best model.
        model_dir: Directory to save models.
        sample_size: Maximum samples to use for training (for faster runs). Default 50000.
    
    Returns:
        Dict: Training results and best model information.
    """
    logger.info("=" * 60)
    logger.info("DELIVERY ETA PREDICTION - MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Import data and feature modules
    from src.data.load_and_clean import load_and_preprocess
    from src.features.feature_engineering import create_feature_matrix
    
    # Step 1: Load and preprocess data
    logger.info("\n[Step 1] Loading and preprocessing data...")
    
    if data_path and os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded data from: {data_path}")
        # Sample if needed
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            logger.info(f"Sampled {sample_size} records for training")
    else:
        df = load_and_preprocess(city="shanghai", sample_size=sample_size)
    
    logger.info(f"Data loaded: {len(df)} records")
    
    # Step 2: Create feature matrix
    logger.info("\n[Step 2] Creating feature matrix...")
    X, y = create_feature_matrix(df)
    feature_names = list(X.columns)
    logger.info(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Step 3: Split data
    logger.info(f"\n[Step 3] Splitting data ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Step 4: Train and evaluate all models
    logger.info("\n[Step 4] Training and evaluating models...")
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    # Step 5: Select best model
    logger.info("\n[Step 5] Selecting best model...")
    best_name, best_model = select_best_model(results)
    
    # Get feature importance
    importance_df = get_feature_importance(best_model, feature_names, best_name)
    
    # Step 6: Save best model
    if save_model_flag:
        logger.info(f"\n[Step 6] Saving best model to {model_dir}/...")
        save_path = save_model(
            model=best_model,
            model_name=best_name,
            feature_names=feature_names,
            metrics=results[best_name]['test_metrics'],
            save_dir=model_dir
        )
    
    # Prepare final results
    final_results = {
        'best_model_name': best_name,
        'best_model': best_model,
        'feature_names': feature_names,
        'test_metrics': results[best_name]['test_metrics'],
        'all_results': results,
        'feature_importance': importance_df,
        'data_stats': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(feature_names)
        }
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 60)
    
    return final_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run training pipeline
    results = run_training_pipeline(
        test_size=0.2,
        random_state=42,
        save_model_flag=True,
        model_dir="models"
    )
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Test MAE: {results['test_metrics']['mae']:.4f} hours")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.4f} hours")
    print(f"Test R²: {results['test_metrics']['r2']:.4f}")
    print(f"\nTop 5 Important Features:")
    if not results['feature_importance'].empty:
        for i, (_, row) in enumerate(results['feature_importance'].head(5).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
