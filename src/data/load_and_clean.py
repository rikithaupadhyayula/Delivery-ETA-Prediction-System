"""
Data Loading and Cleaning Module for Delivery ETA Prediction System

This module handles loading the LaDe (Last-mile Delivery) dataset from HuggingFace,
cleaning the data, and preparing it for feature engineering.

Dataset: Cainiao-AI/LaDe (LaDe-D for delivery scenario)
Source: https://huggingface.co/datasets/Cainiao-AI/LaDe
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from datetime import datetime
from typing import Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_lade_dataset(
    city: str = "shanghai",
    cache_dir: Optional[str] = None,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load the LaDe-D (delivery) dataset from HuggingFace.
    
    The dataset contains delivery records with courier information,
    pickup/delivery locations, and timestamps.
    
    Args:
        city: City to load data for. Options: 'shanghai', 'hangzhou', 
              'chongqing', 'jilin', 'yantai'. Default is 'shanghai'.
        cache_dir: Optional directory to cache the downloaded data.
        sample_size: Optional limit on number of records to load (for faster testing).
    
    Returns:
        pd.DataFrame: Raw delivery data from the specified city.
    """
    logger.info(f"Loading LaDe-D dataset for city: {city}")
    
    # Map city names to dataset split names
    city_to_split = {
        'shanghai': 'delivery_sh',
        'hangzhou': 'delivery_hz',
        'chongqing': 'delivery_cq',
        'jilin': 'delivery_jl',
        'yantai': 'delivery_yt'
    }
    
    try:
        # Load the LaDe-D (delivery) dataset
        dataset = load_dataset(
            "Cainiao-AI/LaDe-D",
            cache_dir=cache_dir
        )
        
        # Get the split name for the requested city
        split_name = city_to_split.get(city.lower(), f'delivery_{city.lower()[:2]}')
        
        # Check if split exists
        if split_name not in dataset:
            available_splits = list(dataset.keys())
            logger.warning(f"Split '{split_name}' not found. Available: {available_splits}")
            # Use first available split
            split_name = available_splits[0]
            logger.info(f"Using split: {split_name}")
        
        # Convert to pandas
        df = dataset[split_name].to_pandas()
        
        # Log available columns  
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Limit sample size if specified
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample_size} records from {len(dataset[split_name])} total")
        
        logger.info(f"Loaded {len(df)} records from {city} (split: {split_name})")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and convert timestamp columns to datetime objects.
    
    The LaDe dataset uses format like '06-04 11:05:00' (MM-DD HH:MM:SS without year).
    This function standardizes them to pandas datetime.
    
    Args:
        df: DataFrame with raw timestamp columns.
    
    Returns:
        pd.DataFrame: DataFrame with parsed datetime columns.
    """
    logger.info("Parsing timestamp columns...")
    df = df.copy()
    
    # Timestamp columns in the LaDe dataset
    time_columns = ['accept_time', 'delivery_time', 'accept_gps_time', 'delivery_gps_time']
    
    for col in time_columns:
        if col in df.columns:
            try:
                # Check if already datetime
                if df[col].dtype == 'datetime64[ns]':
                    continue
                
                # Try to parse as Unix timestamp (seconds) - fastest method
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
                else:
                    # LaDe format is like '06-04 11:05:00' (MM-DD HH:MM:SS without year)
                    # Add a year prefix to make it parseable
                    sample = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else None
                    if sample and len(sample) <= 14:  # Format: MM-DD HH:MM:SS
                        # Add year 2023 as prefix
                        df[col] = '2023-' + df[col].astype(str)
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    else:
                        # Try standard formats
                        df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
                    
                logger.info(f"Parsed column: {col}")
            except Exception as e:
                logger.warning(f"Could not parse column {col}: {e}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the delivery data by removing invalid records.
    
    Cleaning steps:
    1. Remove records with missing critical timestamps
    2. Remove records with missing GPS coordinates
    3. Filter out unreasonable delivery durations
    4. Remove duplicate records
    
    Args:
        df: Raw DataFrame with delivery records.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info(f"Starting data cleaning. Initial records: {len(df)}")
    initial_count = len(df)
    df = df.copy()
    
    # Step 1: Remove records with missing timestamps
    critical_time_cols = ['accept_time', 'delivery_time']
    for col in critical_time_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
    logger.info(f"After removing missing timestamps: {len(df)} records")
    
    # Step 2: Remove records with missing GPS coordinates
    gps_cols = ['accept_gps_lat', 'accept_gps_lng', 'delivery_gps_lat', 'delivery_gps_lng']
    existing_gps_cols = [col for col in gps_cols if col in df.columns]
    if existing_gps_cols:
        df = df.dropna(subset=existing_gps_cols)
    logger.info(f"After removing missing GPS: {len(df)} records")
    
    # Step 3: Compute delivery duration and filter unreasonable values
    if 'accept_time' in df.columns and 'delivery_time' in df.columns:
        # Ensure datetime types
        if df['accept_time'].dtype != 'datetime64[ns]':
            df['accept_time'] = pd.to_datetime(df['accept_time'], errors='coerce')
        if df['delivery_time'].dtype != 'datetime64[ns]':
            df['delivery_time'] = pd.to_datetime(df['delivery_time'], errors='coerce')
        
        # Calculate duration in hours
        df['duration_hours'] = (
            df['delivery_time'] - df['accept_time']
        ).dt.total_seconds() / 3600
        
        # Filter: duration should be between 1 minute (0.0167 hours) and 24 hours
        min_duration = 1 / 60  # 1 minute in hours
        max_duration = 24  # 24 hours
        
        df = df[
            (df['duration_hours'] >= min_duration) & 
            (df['duration_hours'] <= max_duration)
        ]
        logger.info(f"After filtering durations ({min_duration:.4f}h - {max_duration}h): {len(df)} records")
    
    # Step 4: Remove duplicates
    if 'order_id' in df.columns:
        df = df.drop_duplicates(subset=['order_id'], keep='first')
    else:
        df = df.drop_duplicates()
    logger.info(f"After removing duplicates: {len(df)} records")
    
    # Log cleaning summary
    removed = initial_count - len(df)
    logger.info(f"Cleaning complete. Removed {removed} records ({removed/initial_count*100:.2f}%)")
    
    return df


def compute_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the target variable: ETA in hours.
    
    The target is calculated as:
    eta_hours = (delivery_time - accept_time).total_seconds() / 3600
    
    Args:
        df: DataFrame with accept_time and delivery_time columns.
    
    Returns:
        pd.DataFrame: DataFrame with eta_hours column added.
    """
    logger.info("Computing target variable (eta_hours)...")
    df = df.copy()
    
    if 'duration_hours' in df.columns:
        # Already computed during cleaning
        df['eta_hours'] = df['duration_hours']
    elif 'accept_time' in df.columns and 'delivery_time' in df.columns:
        # Compute from timestamps
        df['eta_hours'] = (
            df['delivery_time'] - df['accept_time']
        ).dt.total_seconds() / 3600
    else:
        raise ValueError("Missing required columns: accept_time and delivery_time")
    
    logger.info(f"Target variable stats: mean={df['eta_hours'].mean():.2f}h, "
                f"median={df['eta_hours'].median():.2f}h, "
                f"std={df['eta_hours'].std():.2f}h")
    
    return df


def load_and_preprocess(
    city: str = "shanghai",
    cache_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Main function to load, clean, and preprocess the LaDe dataset.
    
    This is the primary entry point for the data loading module.
    It performs all preprocessing steps in sequence:
    1. Load data from HuggingFace
    2. Parse timestamps
    3. Clean invalid records
    4. Compute target variable
    
    Args:
        city: City to load data for. Default is 'shanghai'.
        cache_dir: Optional directory to cache downloaded data.
        save_path: Optional path to save the processed data.
        sample_size: Optional limit on records (for faster testing).
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for feature engineering.
    """
    logger.info("=" * 50)
    logger.info("Starting data loading and preprocessing pipeline")
    logger.info("=" * 50)
    
    # Step 1: Load data
    df = load_lade_dataset(city=city, cache_dir=cache_dir, sample_size=sample_size)
    
    # Step 2: Parse timestamps
    df = parse_timestamps(df)
    
    # Step 3: Clean data
    df = clean_data(df)
    
    # Step 4: Compute target variable
    df = compute_target_variable(df)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info(f"Saved processed data to: {save_path}")
    
    logger.info("=" * 50)
    logger.info(f"Pipeline complete. Final dataset: {len(df)} records, {len(df.columns)} columns")
    logger.info("=" * 50)
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for reporting.
    
    Args:
        df: Processed DataFrame.
    
    Returns:
        dict: Summary statistics and information.
    """
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
    }
    
    # Add ETA statistics if available
    if 'eta_hours' in df.columns:
        summary['eta_stats'] = {
            'mean': df['eta_hours'].mean(),
            'median': df['eta_hours'].median(),
            'std': df['eta_hours'].std(),
            'min': df['eta_hours'].min(),
            'max': df['eta_hours'].max(),
            'q25': df['eta_hours'].quantile(0.25),
            'q75': df['eta_hours'].quantile(0.75),
        }
    
    # Add courier statistics if available
    if 'courier_id' in df.columns:
        summary['courier_stats'] = {
            'unique_couriers': df['courier_id'].nunique(),
            'deliveries_per_courier': df.groupby('courier_id').size().describe().to_dict()
        }
    
    return summary


# Entry point for direct execution
if __name__ == "__main__":
    # Example usage
    print("Loading and preprocessing LaDe dataset...")
    
    # Process data and save
    df = load_and_preprocess(
        city="shanghai",
        save_path="dataset/processed_data.parquet"
    )
    
    # Print summary
    summary = get_data_summary(df)
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total Records: {summary['total_records']}")
    print(f"Total Columns: {summary['total_columns']}")
    
    if 'eta_stats' in summary:
        print("\nETA Statistics (hours):")
        for key, value in summary['eta_stats'].items():
            print(f"  {key}: {value:.4f}")
    
    print("\nSample of processed data:")
    print(df.head())
