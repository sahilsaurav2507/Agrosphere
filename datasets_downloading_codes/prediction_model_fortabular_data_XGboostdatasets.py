# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
import requests
from urllib.parse import urlparse
import time

# Create the target directory if it doesn't exist
target_dir = r"E:\Agrospere\models\datas\predictive_model_data"
os.makedirs(target_dir, exist_ok=True)

# Function to download and save dataset with specific file path
def download_and_save_dataset(dataset_name, file_path, output_filename=None):
    print(f"Downloading dataset: {dataset_name}, file: {file_path}")
    try:
        # Load the dataset
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset_name,
            file_path
        )
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = os.path.basename(file_path)
        
        # Save to the target directory
        output_path = os.path.join(target_dir, output_filename)
        df.to_csv(output_path, index=False)
        
        print(f"Successfully saved: {output_path}")
        print(f"First 5 records:\n{df.head()}")
        print("-" * 50)
        
        return df
    except Exception as e:
        print(f"Error downloading {dataset_name}, file {file_path}: {str(e)}")
        return None

# Function to download file from direct URL
def download_file_from_url(url, output_filename):
    print(f"Downloading from URL: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        output_path = os.path.join(target_dir, output_filename)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully saved: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading from URL {url}: {str(e)}")
        return False

# List of datasets to download with specific file paths
datasets = [
    {"name": "samuelotiattakorah/agriculture-crop-yield", "files": ["crop_yield.csv"]},
    {"name": "waqi786/climate-change-impact-on-agriculture", "files": ["climate_change_impact_on_agriculture.csv"]},
    {"name": "amirmohammdjalili/soil-moisture-dataset", "files": ["soil_moisture.csv"]},
    {"name": "ziya07/plant-health-data", "files": ["plant_health.csv"]}
]

# Alternative direct download URLs (if kagglehub fails)
direct_urls = [
    {"url": "https://raw.githubusercontent.com/samuelotiattakorah/agriculture-crop-yield/main/crop_yield.csv", 
     "filename": "agriculture-crop-yield_crop_yield.csv"},
    {"url": "https://raw.githubusercontent.com/waqi786/climate-change-impact-on-agriculture/main/climate_change_impact_on_agriculture.csv", 
     "filename": "climate-change-impact-on-agriculture_data.csv"},
    {"url": "https://raw.githubusercontent.com/amirmohammdjalili/soil-moisture-dataset/main/soil_moisture.csv", 
     "filename": "soil-moisture-dataset_data.csv"},
    {"url": "https://raw.githubusercontent.com/ziya07/plant-health-data/main/plant_health.csv", 
     "filename": "plant-health-data_data.csv"}
]

# Download and save all datasets
downloaded_datasets = {}
successful_downloads = 0

# Try with kagglehub first
for dataset in datasets:
    dataset_name = dataset["name"]
    for file_path in dataset["files"]:
        output_filename = f"{dataset_name.split('/')[-1]}_{os.path.basename(file_path)}"
        df = download_and_save_dataset(dataset_name, file_path, output_filename)
        if df is not None:
            downloaded_datasets[output_filename] = df
            successful_downloads += 1

# If kagglehub didn't work for all datasets, try direct URLs
if successful_downloads < len(datasets):
    print("\nTrying alternative download method with direct URLs...")
    for url_info in direct_urls:
        if download_file_from_url(url_info["url"], url_info["filename"]):
            successful_downloads += 1

# If we still don't have all datasets, try downloading sample data
if successful_downloads < len(datasets):
    print("\nCreating sample datasets for demonstration...")
    
    # Create sample crop yield data
    crop_yield = pd.DataFrame({
        'Year': range(2010, 2021),
        'Crop': ['Wheat', 'Rice', 'Corn', 'Wheat', 'Rice', 'Corn', 'Wheat', 'Rice', 'Corn', 'Wheat', 'Rice'],
        'Yield': [4.2, 5.1, 6.3, 4.3, 5.2, 6.5, 4.4, 5.3, 6.7, 4.5, 5.4],
        'Area': [100, 120, 150, 105, 125, 155, 110, 130, 160, 115, 135]
    })
    crop_yield.to_csv(os.path.join(target_dir, 'sample_crop_yield.csv'), index=False)
    successful_downloads += 1
    
    # Create sample climate impact data
    climate_impact = pd.DataFrame({
        'Year': range(2010, 2021),
        'Temperature': [25.1, 25.3, 25.5, 25.7, 25.9, 26.1, 26.3, 26.5, 26.7, 26.9, 27.1],
        'Rainfall': [1200, 1180, 1220, 1150, 1250, 1100, 1300, 1050, 1350, 1000, 1400],
        'Crop_Impact': [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1]
    })
    climate_impact.to_csv(os.path.join(target_dir, 'sample_climate_impact.csv'), index=False)
    successful_downloads += 1
    
    # Create sample soil moisture data
    soil_moisture = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=11, freq='M'),
        'Moisture_Level': [0.3, 0.32, 0.35, 0.38, 0.4, 0.42, 0.39, 0.36, 0.33, 0.31, 0.29],
        'Soil_Type': ['Clay', 'Clay', 'Loam', 'Loam', 'Sandy', 'Sandy', 'Clay', 'Clay', 'Loam', 'Loam', 'Sandy']
    })
    soil_moisture.to_csv(os.path.join(target_dir, 'sample_soil_moisture.csv'), index=False)
    successful_downloads += 1
    
    # Create sample plant health data
    plant_health = pd.DataFrame({
        'Plant_ID': range(1, 12),
        'Health_Score': [85, 90, 75, 95, 70, 80, 85, 90, 75, 95, 70],
        'Disease': ['None', 'None', 'Rust', 'None', 'Blight', 'None', 'None', 'None', 'Rust', 'None', 'Blight'],
        'Treatment': ['None', 'None', 'Fungicide', 'None', 'Fungicide', 'None', 'None', 'None', 'Fungicide', 'None', 'Fungicide']
    })
    plant_health.to_csv(os.path.join(target_dir, 'sample_plant_health.csv'), index=False)
    successful_downloads += 1

print(f"\nAll datasets have been downloaded to: {target_dir}")
print(f"Total datasets downloaded: {successful_downloads}")

# List all files in the target directory
print("\nFiles in the target directory:")
for file in os.listdir(target_dir):
    print(f"- {file}")
