import os
import json
import kagglehub
import pandas as pd
import kaggle

# Define the target directory
target_dir = r"E:\Agrospere\models\datas\substanability_tracker_data"

# Create the directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Set up Kaggle credentials
def setup_kaggle_credentials():
    # Create Kaggle credentials directory if it doesn't exist
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Create credentials file
    credentials = {
        "username": "sahil2507",
        "key": "b202d0d2e68e94ba6790cd6f9542698b"
    }
    
    # Write credentials to file
    credentials_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(credentials_path, "w") as f:
        json.dump(credentials, f)
    
    # Set appropriate permissions on Unix systems
    try:
        os.chmod(credentials_path, 0o600)
    except:
        pass  # Skip if on Windows
    
    print(f"Kaggle credentials set up at {credentials_path}")

# Set up credentials before downloading
setup_kaggle_credentials()

# List of datasets to download
datasets = [
    {
        "dataset_id": "waqi786/climate-change-impact-on-agriculture",
        "output_name": "climate_change_impact.csv"
    },
    {
        "dataset_id": "amirmohammdjalili/soil-moisture-dataset",
        "output_name": "soil_moisture.csv"
    },
    {
        "dataset_id": "bhadramohit/agriculture-and-farming-dataset",
        "output_name": "agriculture_farming.csv"
    }
]

# Download and save each dataset using Kaggle API
for dataset in datasets:
    try:
        print(f"\nDownloading {dataset['dataset_id']}...")
        dataset_path = os.path.join(target_dir, dataset['dataset_id'].replace('/', '_'))
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            dataset["dataset_id"],
            path=dataset_path,
            unzip=True
        )
        
        print(f"Dataset downloaded to {dataset_path}")
        
        # Find CSV files in the downloaded dataset
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        
        if csv_files:
            # Use the first CSV file found
            source_file = os.path.join(dataset_path, csv_files[0])
            target_file = os.path.join(target_dir, dataset["output_name"])
            
            # Read and save the file with the desired name
            df = pd.read_csv(source_file)
            df.to_csv(target_file, index=False)
            
            print(f"Successfully saved to {target_file}")
            print(f"First 5 records:")
            print(df.head())
        else:
            print(f"No CSV files found in the downloaded dataset")
        
    except Exception as e:
        print(f"Error downloading {dataset['dataset_id']}: {str(e)}")
    
    print("-" * 50)

print("\nAll downloads completed!")
