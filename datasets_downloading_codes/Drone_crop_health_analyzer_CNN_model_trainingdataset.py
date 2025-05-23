# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] tqdm
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
import shutil
import glob
from tqdm import tqdm
import random

# Create the target directory if it doesn't exist
target_dir = r"E:\Agrospere\models\datas\crop_health_daata"
os.makedirs(target_dir, exist_ok=True)

print(f"Downloading datasets to: {target_dir}")

# Function to check total size of a directory in GB
def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024 * 1024)  # Convert to GB

# Function to sample files to stay under size limit
def sample_files(file_list, max_size_gb=2.0):
    random.shuffle(file_list)  # Randomize the list
    selected_files = []
    current_size = 0
    
    for file_path in file_list:
        file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # Size in GB
        if current_size + file_size <= max_size_gb:
            selected_files.append(file_path)
            current_size += file_size
        
    print(f"Selected {len(selected_files)} files totaling {current_size:.2f} GB")
    return selected_files

# Track total downloaded size
total_downloaded_gb = 0
max_size_gb = 2.0

# Dataset 1: RGBNIR Aerial Crop Dataset (prioritize this as it contains images)
print("\n--- Downloading RGBNIR Aerial Crop Dataset (Image Data) ---")
try:
    # Download the dataset
    rgbnir_temp_path = kagglehub.dataset_download("masiaslahi/rgbnir-aerial-crop-dataset")
    
    # Create target directory
    rgbnir_target_path = os.path.join(target_dir, "rgbnir_aerial_crop")
    os.makedirs(rgbnir_target_path, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(rgbnir_temp_path, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} image files")
    
    # Sample files to stay under size limit
    selected_files = sample_files(image_files, max_size_gb)
    
    # Copy selected files
    for file in tqdm(selected_files, desc="Copying image files"):
        rel_path = os.path.relpath(file, rgbnir_temp_path)
        target_file = os.path.join(rgbnir_target_path, rel_path)
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        shutil.copy2(file, target_file)
    
    # Update total downloaded size
    total_downloaded_gb = get_dir_size(rgbnir_target_path)
    print(f"RGBNIR dataset saved to: {rgbnir_target_path}")
    print(f"Current total size: {total_downloaded_gb:.2f} GB")
    
except Exception as e:
    print(f"Error downloading RGBNIR dataset: {e}")

# If we have space left, download Plant Health Data
remaining_space = max_size_gb - total_downloaded_gb
if remaining_space > 0.1:  # If at least 100MB left
    print("\n--- Downloading Plant Health Data ---")
    try:
        # Load the dataset
        df_plant_health = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "ziya07/plant-health-data",
            ""
        )
        
        # Save to target directory
        plant_health_path = os.path.join(target_dir, "plant_health_data.csv")
        df_plant_health.to_csv(plant_health_path, index=False)
        
        # Update total size
        file_size_gb = os.path.getsize(plant_health_path) / (1024 * 1024 * 1024)
        total_downloaded_gb += file_size_gb
        
        print(f"Plant health dataset saved to: {plant_health_path}")
        print(f"Current total size: {total_downloaded_gb:.2f} GB")
        print("First 5 records:", df_plant_health.head())
    except Exception as e:
        print(f"Error downloading Plant Health dataset: {e}")

# If we still have space, download Farm Weather Data
remaining_space = max_size_gb - total_downloaded_gb
if remaining_space > 0.1:  # If at least 100MB left
    print("\n--- Downloading Farm Weather Data ---")
    try:
        # Load the dataset
        df_farm_weather = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "ksrao74/farm-weather-data",
            ""
        )
        
        # Save to target directory
        farm_weather_path = os.path.join(target_dir, "farm_weather_data.csv")
        df_farm_weather.to_csv(farm_weather_path, index=False)
        
        # Update total size
        file_size_gb = os.path.getsize(farm_weather_path) / (1024 * 1024 * 1024)
        total_downloaded_gb += file_size_gb
        
        print(f"Farm weather dataset saved to: {farm_weather_path}")
        print(f"Current total size: {total_downloaded_gb:.2f} GB")
        print("First 5 records:", df_farm_weather.head())
    except Exception as e:
        print(f"Error downloading Farm Weather dataset: {e}")

# If we still have space, download Sentinel2 Crop Mapping
remaining_space = max_size_gb - total_downloaded_gb
if remaining_space > 0.1:  # If at least 100MB left
    print("\n--- Downloading Sentinel2 Crop Mapping dataset ---")
    try:
        # Load the dataset
        df_sentinel = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "ignazio/sentinel2-crop-mapping",
            ""
        )
        
        # Save to target directory
        sentinel_path = os.path.join(target_dir, "sentinel2_crop_mapping.csv")
        df_sentinel.to_csv(sentinel_path, index=False)
        
        # Update total size
        file_size_gb = os.path.getsize(sentinel_path) / (1024 * 1024 * 1024)
        total_downloaded_gb += file_size_gb
        
        print(f"Sentinel2 dataset saved to: {sentinel_path}")
        print(f"Current total size: {total_downloaded_gb:.2f} GB")
        print("First 5 records:", df_sentinel.head())
    except Exception as e:
        print(f"Error downloading Sentinel2 dataset: {e}")

print(f"\nTotal downloaded data size: {total_downloaded_gb:.2f} GB")
print("All datasets downloaded successfully to:", target_dir)
