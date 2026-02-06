"""
Script to download NYC TLC trip record data.

This script downloads yellow taxi, green taxi, and FHV trip data
from the NYC TLC website for specified date ranges.

Usage:
    python src/data/download_data.py --start-year 2009 --end-year 2024
    python src/data/download_data.py --start-year 2020 --end-year 2020 --vehicle-type yellow
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base URL for NYC TLC data
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# Vehicle types available
VEHICLE_TYPES = {
    'yellow': 'yellow_tripdata',
    'green': 'green_tripdata',
    'fhv': 'fhv_tripdata',
    'fhvhv': 'fhvhv_tripdata'  # High Volume FHV (Uber, Lyft)
}


def generate_urls(
    start_year: int,
    end_year: int,
    vehicle_types: List[str]
) -> List[tuple]:
    """
    Generate URLs for downloading trip data.
    
    Args:
        start_year: Starting year for data download
        end_year: Ending year for data download
        vehicle_types: List of vehicle types to download
    
    Returns:
        List of tuples (url, filename, vehicle_type)
    """
    urls = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Skip future months
            current_date = datetime.now()
            if year == current_date.year and month > current_date.month:
                continue
                
            for vtype in vehicle_types:
                if vtype not in VEHICLE_TYPES:
                    logger.warning(f"Unknown vehicle type: {vtype}, skipping")
                    continue
                
                filename = f"{VEHICLE_TYPES[vtype]}_{year}-{month:02d}.parquet"
                url = f"{BASE_URL}/{filename}"
                urls.append((url, filename, vtype))
    
    return urls


def download_file(url: str, output_path: Path) -> bool:
    """
    Download a file from URL to output path.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        
        if response.status_code == 404:
            logger.warning(f"File not found: {url}")
            return False
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Successfully downloaded: {output_path.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download NYC TLC trip record data'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2020,
        help='Start year for data download (default: 2020)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=datetime.now().year,
        help='End year for data download (default: current year)'
    )
    parser.add_argument(
        '--vehicle-type',
        type=str,
        nargs='+',
        default=['yellow', 'green', 'fhvhv'],
        choices=['yellow', 'green', 'fhv', 'fhvhv', 'all'],
        help='Vehicle types to download (default: yellow green fhvhv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded files (default: data/raw)'
    )
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.vehicle_type:
        vehicle_types = list(VEHICLE_TYPES.keys())
    else:
        vehicle_types = args.vehicle_type
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting download from {args.start_year} to {args.end_year}")
    logger.info(f"Vehicle types: {', '.join(vehicle_types)}")
    
    # Generate URLs
    urls = generate_urls(args.start_year, args.end_year, vehicle_types)
    logger.info(f"Total files to download: {len(urls)}")
    
    # Download files
    successful = 0
    failed = 0
    skipped = 0
    
    for url, filename, vtype in urls:
        output_path = output_dir / filename
        
        # Skip if file already exists
        if output_path.exists():
            logger.info(f"File already exists, skipping: {filename}")
            skipped += 1
            continue
        
        if download_file(url, output_path):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Download complete!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")

if __name__ == "__main__":
    main()
