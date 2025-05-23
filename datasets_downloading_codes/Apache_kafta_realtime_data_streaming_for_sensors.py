import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, base_path="E:\\Agrospere\\models\\datas\\RNN_realtime data"):
        """Initialize the downloader with the target directory path."""
        self.base_path = Path(base_path)
        
    def create_directory(self):
        """Create the directory structure if it doesn't exist."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory structure created at: {self.base_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return False
            
    def download_file(self, url, filename):
        """Download a file from the given URL and save it to the specified path."""
        try:
            filepath = self.base_path / filename
            
            # Check if file already exists
            if filepath.exists():
                logger.info(f"File {filename} already exists. Skipping download.")
                return filepath
                
            logger.info(f"Downloading {filename} from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Save the downloaded file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Successfully downloaded {filename} to {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file: {e}")
            return None
            
    def extract_zip(self, zip_path):
        """Extract a zip file to the base directory."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_path)
            logger.info(f"Successfully extracted {zip_path} to {self.base_path}")
            return True
        except Exception as e:
            logger.error(f"Error extracting zip file: {e}")
            return False
            
    def download_sample_dataset(self):
        """Download a sample time series dataset for RNN training."""
        # Example: Download a sample time series dataset from GitHub
        # Replace this URL with your actual dataset source
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        filename = "airline-passengers.csv"
        
        return self.download_file(url, filename)
        
    def download_custom_dataset(self, url, filename):
        """Download a custom dataset from the specified URL."""
        return self.download_file(url, filename)

def main():
    """Main function to execute the dataset download process."""
    # Initialize the downloader
    downloader = DatasetDownloader()
    
    # Create the directory structure
    if not downloader.create_directory():
        logger.error("Failed to create directory structure. Exiting.")
        return
    
    # Download the sample dataset
    sample_dataset_path = downloader.download_sample_dataset()
    if sample_dataset_path:
        # Load and display dataset info
        try:
            df = pd.read_csv(sample_dataset_path)
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"First few rows:\n{df.head()}")
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
    
    # Example: Download a custom dataset (uncomment and modify as needed)
    # custom_dataset_url = "YOUR_CUSTOM_DATASET_URL"
    # custom_filename = "custom_dataset.csv"
    # downloader.download_custom_dataset(custom_dataset_url, custom_filename)
    
    logger.info("Dataset download process completed.")

if __name__ == "__main__":
    main()
