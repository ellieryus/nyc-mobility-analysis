"""
Data processing and cleaning script for NYC TLC trip data.

This script:
1. Loads raw trip data
2. Cleans and validates data
3. Handles missing values and outliers
4. Standardizes column names across datasets
5. Saves processed data

Usage:
    python src/data/process_data.py
    python src/data/process_data.py --input data/raw --output data/processed
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TripDataProcessor:
    """Process NYC TLC trip data."""
    
    # Column mapping to standardize names
    COLUMN_MAPPING = {
        'tpep_pickup_datetime': 'pickup_datetime',
        'tpep_dropoff_datetime': 'dropoff_datetime',
        'lpep_pickup_datetime': 'pickup_datetime',
        'lpep_dropoff_datetime': 'dropoff_datetime',
        'pickup_datetime': 'pickup_datetime',
        'dropOff_datetime': 'dropoff_datetime',
        'PULocationID': 'pickup_location_id',
        'DOLocationID': 'dropoff_location_id',
        'trip_distance': 'trip_distance',
        'fare_amount': 'fare_amount',
        'total_amount': 'total_amount',
        'passenger_count': 'passenger_count',
        'payment_type': 'payment_type',
    }
    
    def __init__(self):
        """Initialize processor."""
        pass
    
    def clean_data(self, df: pd.DataFrame, vehicle_type: str) -> pd.DataFrame:
        """
        Clean and validate trip data.
        
        Args:
            df: Raw dataframe
            vehicle_type: Type of vehicle (yellow, green, fhv, fhvhv)
        
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Cleaning {vehicle_type} data. Initial shape: {df.shape}")
        
        # Rename columns to standard names
        df = df.rename(columns=self.COLUMN_MAPPING)
        
        # Ensure datetime columns are proper datetime
        datetime_cols = ['pickup_datetime', 'dropoff_datetime']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Remove rows with null pickup/dropoff times
        initial_rows = len(df)
        df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])
        logger.info(f"Removed {initial_rows - len(df)} rows with null datetime")
        
        # Calculate trip duration
        if 'pickup_datetime' in df.columns and 'dropoff_datetime' in df.columns:
            df['trip_duration_seconds'] = (
                df['dropoff_datetime'] - df['pickup_datetime']
            ).dt.total_seconds()
        
        # Remove invalid trips
        if 'trip_duration_seconds' in df.columns:
            initial_rows = len(df)
            df = df[
                (df['trip_duration_seconds'] > 0) &  # Positive duration
                (df['trip_duration_seconds'] < 86400)  # Less than 24 hours
            ]
            logger.info(f"Removed {initial_rows - len(df)} rows with invalid duration")
        
        # Clean distance
        if 'trip_distance' in df.columns:
            initial_rows = len(df)
            df = df[
                (df['trip_distance'] >= 0) &
                (df['trip_distance'] <= 100)  # Reasonable max distance
            ]
            logger.info(f"Removed {initial_rows - len(df)} rows with invalid distance")
        
        # Clean fare amount
        if 'fare_amount' in df.columns:
            initial_rows = len(df)
            df = df[
                (df['fare_amount'] >= 0) &
                (df['fare_amount'] <= 1000)  # Reasonable max fare
            ]
            logger.info(f"Removed {initial_rows - len(df)} rows with invalid fare")
        
        # Clean passenger count
        if 'passenger_count' in df.columns:
            initial_rows = len(df)
            df = df[
                (df['passenger_count'] >= 0) &
                (df['passenger_count'] <= 8)  # Reasonable max passengers
            ]
            logger.info(f"Removed {initial_rows - len(df)} rows with invalid passenger count")
        
        # Add vehicle type column
        df['vehicle_type'] = vehicle_type
        
        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from datetime columns.
        
        Args:
            df: Dataframe with datetime columns
        
        Returns:
            Dataframe with additional temporal features
        """
        if 'pickup_datetime' in df.columns:
            df['pickup_year'] = df['pickup_datetime'].dt.year
            df['pickup_month'] = df['pickup_datetime'].dt.month
            df['pickup_day'] = df['pickup_datetime'].dt.day
            df['pickup_hour'] = df['pickup_datetime'].dt.hour
            df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
            df['pickup_weekday'] = df['pickup_datetime'].dt.weekday < 5
            
            # Classify time of day
            df['time_of_day'] = pd.cut(
                df['pickup_hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
        
        return df
    
    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        vehicle_type: str
    ) -> bool:
        """
        Process a single trip data file.
        
        Args:
            input_path: Path to input file
            output_path: Path to save processed file
            vehicle_type: Type of vehicle
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing {input_path.name}")
            
            # Read data
            df = pd.read_parquet(input_path)
            
            # Clean data
            df = self.clean_data(df, vehicle_type)
            
            # Extract temporal features
            df = self.extract_temporal_features(df)
            
            # Save processed data
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Process NYC TLC trip data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw',
        help='Input directory containing raw data (default: data/raw)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed data (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = TripDataProcessor()
    
    # Get all parquet files
    files = list(input_dir.glob('*.parquet'))
    logger.info(f"Found {len(files)} files to process")
    
    successful = 0
    failed = 0
    
    for file_path in tqdm(files, desc="Processing files"):
        # Determine vehicle type from filename
        if 'yellow' in file_path.name:
            vehicle_type = 'yellow'
        elif 'green' in file_path.name:
            vehicle_type = 'green'
        elif 'fhvhv' in file_path.name:
            vehicle_type = 'fhvhv'
        elif 'fhv' in file_path.name:
            vehicle_type = 'fhv'
        else:
            logger.warning(f"Unknown vehicle type for {file_path.name}, skipping")
            continue
        
        output_path = output_dir / file_path.name
        
        if processor.process_file(file_path, output_path, vehicle_type):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")


if __name__ == "__main__":
    main()
