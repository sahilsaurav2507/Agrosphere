import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd

# Create the target directory if it doesn't exist
target_dir = r"E:\Agrospere\models\datas\crop_recom_data"
os.makedirs(target_dir, exist_ok=True)

# Function to download and save dataset
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
        print(f"First 5 records:\n{df.head()}\n")
        
        return df
    except Exception as e:
        print(f"Error downloading {dataset_name}, file {file_path}: {str(e)}")
        return None

# List of datasets to download with their specific file paths
datasets = [
    {"name": "atharvaingle/crop-recommendation-dataset", "file_path": "Crop_recommendation.csv", "filename": "crop_recommendation.csv"},
    {"name": "bhadramohit/agriculture-and-farming-dataset", "file_path": "agriculture_farming_dataset.csv", "filename": "agriculture_farming.csv"},
    {"name": "ziya07/plant-health-data", "file_path": "plant_health_data.csv", "filename": "plant_health.csv"},
    {"name": "amirmohammdjalili/soil-moisture-dataset", "file_path": "soil_moisture.csv", "filename": "soil_moisture.csv"}
]

# Download all datasets
for dataset in datasets:
    download_and_save_dataset(
        dataset["name"], 
        dataset["file_path"], 
        dataset["filename"]
    )

print(f"All datasets have been downloaded to {target_dir}")
