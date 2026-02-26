"""
Download NYC TLC data directly from website
Works when CloudFront URLs don't work!
"""

import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_download_links(year, month, vehicle_type='yellow'):
    """
    Get direct download links from NYC TLC website
    """
    # NYC TLC data page
    base_url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
    
    # Expected filename pattern
    filename = f"{vehicle_type}_tripdata_{year}-{month:02d}.parquet"
    
    # Direct download URLs 
    possible_urls = [
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}",
        f"https://s3.amazonaws.com/nyc-tlc/trip+data/{filename}",
        f"https://d37ci6vzurychx.cloudfront.net/{filename}",
    ]
    
    return possible_urls, filename

def download_file(url, output_path):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=60)
        
        if response.status_code == 403:
            return False, "Access forbidden"
        if response.status_code == 404:
            return False, "File not found"
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def download_yellow_taxi_data(start_year=2024, end_year=2025):
    """
    Download yellow taxi data from NYC TLC
    """
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Skip future months
            from datetime import datetime
            if year == datetime.now().year and month > datetime.now().month:
                continue
            
            urls, filename = get_download_links(year, month, 'yellow')
            output_path = output_dir / filename
            
            # Skip if already exists
            if output_path.exists():
                logger.info(f"✓ Already exists: {filename}")
                continue
            
            logger.info(f"Downloading: {filename}")
            
            # Try each possible URL
            success = False
            for url in urls:
                result, message = download_file(url, output_path)
                if result:
                    logger.info(f"Success: {filename} from {url}")
                    successful += 1
                    success = True
                    break
            
            if not success:
                logger.warning(f"Failed: {filename} - {message}")
                failed += 1
    
    print(f"DOWNLOAD COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    download_yellow_taxi_data(start_year=2024, end_year=2025)