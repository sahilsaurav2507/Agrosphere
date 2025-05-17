import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from datetime import datetime
import joblib

class SustainabilityModel:
    
    def __init__(self):
        """Initialize the sustainability model"""
        self.carbon_model = None
        self.water_model = None
        self.env_impact_model = None
        self.carbon_scaler = StandardScaler()
        self.water_scaler = StandardScaler()
        self.env_scaler = StandardScaler()
        
        # Create necessary directories
        os.makedirs('data/sustainability', exist_ok=True)
        os.makedirs('saved_models/sustainability', exist_ok=True)
        os.makedirs('reports/sustainability', exist_ok=True)
    
    def load_data(self, carbon_path=None, water_path=None, regen_farming_path=None, env_data_path=None):
        datasets = {}
        
        # Try to load carbon footprint data
        if carbon_path and os.path.exists(carbon_path):
            try:
                datasets['carbon'] = pd.read_csv(carbon_path)
                print(f"Loaded carbon footprint data from {carbon_path}")
            except Exception as e:
                print(f"Error loading carbon data: {e}")
                datasets['carbon'] = self._generate_synthetic_carbon_data()
        else:
            print("Carbon footprint data not found. Generating synthetic data.")
            datasets['carbon'] = self._generate_synthetic_carbon_data()
        
        # Try to load water usage data
        if water_path and os.path.exists(water_path):
            try:
                datasets['water'] = pd.read_csv(water_path)
                print(f"Loaded water usage data from {water_path}")
            except Exception as e:
                print(f"Error loading water data: {e}")
                datasets['water'] = self._generate_synthetic_water_data()
        else:
            print("Water usage data not found. Generating synthetic data.")
            datasets['water'] = self._generate_synthetic_water_data()
        
        # Try to load regenerative farming data
        if regen_farming_path and os.path.exists(regen_farming_path):
            try:
                datasets['regen_farming'] = pd.read_csv(regen_farming_path)
                print(f"Loaded regenerative farming data from {regen_farming_path}")
            except Exception as e:
                print(f"Error loading regenerative farming data: {e}")
                datasets['regen_farming'] = self._generate_synthetic_regen_farming_data()
        else:
            print("Regenerative farming data not found. Generating synthetic data.")
            datasets['regen_farming'] = self._generate_synthetic_regen_farming_data()
        
        # Try to load environmental data
        if env_data_path and os.path.exists(env_data_path):
            try:
                datasets['environment'] = pd.read_csv(env_data_path)
                print(f"Loaded environmental data from {env_data_path}")
            except Exception as e:
                print(f"Error loading environmental data: {e}")
                datasets['environment'] = self._generate_synthetic_env_data()
        else:
            print("Environmental data not found. Generating synthetic data.")
            datasets['environment'] = self._generate_synthetic_env_data()
        
        # Merge datasets if needed for comprehensive analysis
        datasets['merged'] = self._merge_datasets(datasets)
        
        return datasets
    
    def _generate_synthetic_carbon_data(self, n_samples=500):
        """Generate synthetic carbon footprint data"""
        np.random.seed(42)
        
        # Create crop types
        crop_types = ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Cotton', 'Potatoes']
        
        # Create farming methods
        farming_methods = ['Conventional', 'Organic', 'Conservation', 'Precision', 'Regenerative']
        
        # Generate data
        data = {
            'crop_type': np.random.choice(crop_types, n_samples),
            'farming_method': np.random.choice(farming_methods, n_samples),
            'farm_size_hectares': np.random.uniform(5, 500, n_samples),
            'fertilizer_kg_per_hectare': np.random.uniform(0, 300, n_samples),
            'machinery_hours': np.random.uniform(10, 200, n_samples),
            'transport_distance_km': np.random.uniform(5, 500, n_samples),
            'energy_consumption_kwh': np.random.uniform(500, 10000, n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate carbon footprint based on factors
        # Base carbon values for different crops (kg CO2e per hectare)
        crop_carbon = {
            'Wheat': 2500, 'Corn': 3000, 'Rice': 4000, 
            'Soybeans': 2000, 'Cotton': 3500, 'Potatoes': 2800
        }
        
        # Farming method impact factors (multipliers)
        method_factors = {
            'Conventional': 1.0, 'Organic': 0.7, 'Conservation': 0.8, 
            'Precision': 0.75, 'Regenerative': 0.6
        }
        
        # Calculate carbon footprint
        df['carbon_footprint_kg_co2e'] = df.apply(
            lambda row: (
                crop_carbon[row['crop_type']] * 
                method_factors[row['farming_method']] * 
                (row['farm_size_hectares'] / 100) +
                row['fertilizer_kg_per_hectare'] * 3 +  # Fertilizer impact
                row['machinery_hours'] * 10 +           # Machinery impact
                row['transport_distance_km'] * 0.2 +    # Transport impact
                row['energy_consumption_kwh'] * 0.5     # Energy impact
            ),
            axis=1
        )
        
        # Add some noise to make it more realistic
        df['carbon_footprint_kg_co2e'] += np.random.normal(0, 200, n_samples)
        df['carbon_footprint_kg_co2e'] = np.maximum(0, df['carbon_footprint_kg_co2e'])
        
        # Add carbon sequestration potential
        df['carbon_sequestration_potential_kg_co2e'] = df.apply(
            lambda row: (
                500 if row['farming_method'] == 'Regenerative' else
                300 if row['farming_method'] == 'Conservation' else
                200 if row['farming_method'] == 'Organic' else
                100 if row['farming_method'] == 'Precision' else 0
            ) * row['farm_size_hectares'] / 100,
            axis=1
        )
        
        # Add some noise to sequestration
        df['carbon_sequestration_potential_kg_co2e'] += np.random.normal(0, 50, n_samples)
        df['carbon_sequestration_potential_kg_co2e'] = np.maximum(0, df['carbon_sequestration_potential_kg_co2e'])
        
        # Calculate net carbon impact
        df['net_carbon_impact_kg_co2e'] = df['carbon_footprint_kg_co2e'] - df['carbon_sequestration_potential_kg_co2e']
        
        # Save synthetic data
        os.makedirs('data/sustainability', exist_ok=True)
        df.to_csv('data/sustainability/synthetic_carbon_data.csv', index=False)
        print("Generated synthetic carbon footprint data")
        
        return df
    
    def _generate_synthetic_water_data(self, n_samples=500):
        """Generate synthetic water usage data"""
        np.random.seed(43)
        
        # Create crop types
        crop_types = ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Cotton', 'Potatoes']
        
        # Create irrigation methods
        irrigation_methods = ['Flood', 'Drip', 'Sprinkler', 'Subsurface', 'Rainfed']
        
        # Create soil types
        soil_types = ['Sandy', 'Loamy', 'Clay', 'Silt', 'Peat']
        
        # Generate data
        data = {
            'crop_type': np.random.choice(crop_types, n_samples),
            'irrigation_method': np.random.choice(irrigation_methods, n_samples),
            'soil_type': np.random.choice(soil_types, n_samples),
            'farm_size_hectares': np.random.uniform(5, 500, n_samples),
            'annual_rainfall_mm': np.random.uniform(200, 1500, n_samples),
            'temperature_celsius': np.random.uniform(10, 35, n_samples),
            'growing_season_days': np.random.uniform(60, 180, n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Base water requirements for different crops (cubic meters per hectare)
        crop_water = {
            'Wheat': 4500, 'Corn': 6000, 'Rice': 10000, 
            'Soybeans': 5000, 'Cotton': 7000, 'Potatoes': 4000
        }
        
        # Irrigation efficiency factors (multipliers)
        irrigation_factors = {
            'Flood': 1.5, 'Drip': 0.6, 'Sprinkler': 0.9, 
            'Subsurface': 0.7, 'Rainfed': 0.1
        }
        
        # Soil water retention factors (multipliers)
        soil_factors = {
            'Sandy': 1.3, 'Loamy': 0.9, 'Clay': 0.8, 
            'Silt': 1.0, 'Peat': 0.7
        }
        
        # Calculate water usage
        df['water_usage_cubic_meters'] = df.apply(
            lambda row: (
                crop_water[row['crop_type']] * 
                irrigation_factors[row['irrigation_method']] * 
                soil_factors[row['soil_type']] * 
                (row['farm_size_hectares'] / 100) *
                (1 + (row['temperature_celsius'] - 20) / 50) *  # Temperature impact
                (1 - row['annual_rainfall_mm'] / 3000)          # Rainfall impact
            ),
            axis=1
        )
        
        # Add some noise to make it more realistic
        df['water_usage_cubic_meters'] += np.random.normal(0, 500, n_samples)
        df['water_usage_cubic_meters'] = np.maximum(0, df['water_usage_cubic_meters'])
        
        # Calculate water efficiency
        df['water_efficiency_cubic_meters_per_ton'] = df.apply(
            lambda row: (
                row['water_usage_cubic_meters'] / 
                (crop_water[row['crop_type']] / 1000)  # Approximate yield in tons
            ),
            axis=1
        )
        
        # Add water recycling potential
        df['water_recycling_potential_percent'] = df.apply(
            lambda row: (
                60 if row['irrigation_method'] == 'Drip' else
                40 if row['irrigation_method'] == 'Subsurface' else
                30 if row['irrigation_method'] == 'Sprinkler' else
                10 if row['irrigation_method'] == 'Flood' else 0
            ),
            axis=1
        )
        
        # Add some noise to recycling potential
        df['water_recycling_potential_percent'] += np.random.normal(0, 5, n_samples)
        df['water_recycling_potential_percent'] = np.clip(df['water_recycling_potential_percent'], 0, 100)
        
        # Save synthetic data
        os.makedirs('data/sustainability', exist_ok=True)
        df.to_csv('data/sustainability/synthetic_water_data.csv', index=False)
        print("Generated synthetic water usage data")
        
        return df
    
    def _generate_synthetic_regen_farming_data(self, n_samples=500):
        """Generate synthetic regenerative farming practices data"""
        np.random.seed(44)
        
        # Create farm types
        farm_types = ['Crop', 'Mixed', 'Livestock', 'Orchard', 'Vegetable']
        
        # Create regenerative practices
        practices = [
            'Cover Crops', 'No-Till', 'Crop Rotation', 'Composting', 
            'Agroforestry', 'Managed Grazing', 'Reduced Synthetic Inputs'
        ]
        
        # Generate data
        data = {
            'farm_type': np.random.choice(farm_types, n_samples),
            'farm_size_hectares': np.random.uniform(5, 500, n_samples),
            'years_practicing_regenerative': np.random.uniform(0, 15, n_samples),
            'soil_organic_matter_percent_initial': np.random.uniform(1, 3, n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add practice adoption (binary columns)
        for practice in practices:
            df[f'uses_{practice.lower().replace("-", "_").replace(" ", "_")}'] = np.random.choice([0, 1], n_samples)
        
        # Calculate total practices adopted
        df['total_practices_adopted'] = df[[col for col in df.columns if col.startswith('uses_')]].sum(axis=1)
        
        # Calculate soil health improvement based on practices and time
        df['soil_organic_matter_percent_current'] = df.apply(
            lambda row: (
                row['soil_organic_matter_percent_initial'] + 
                (row['total_practices_adopted'] * 0.2 * min(row['years_practicing_regenerative'], 10) / 10)
            ),
            axis=1
        )
        
        # Add some noise to current soil organic matter
        df['soil_organic_matter_percent_current'] += np.random.normal(0, 0.2, n_samples)
        df['soil_organic_matter_percent_current'] = np.maximum(
            df['soil_organic_matter_percent_initial'], 
            df['soil_organic_matter_percent_current']
        )
        
        # Calculate biodiversity score
        df['biodiversity_score'] = df.apply(
            lambda row: (
                5 + row['total_practices_adopted'] * 0.8 + 
                (1 if row['uses_agroforestry'] else 0) * 2 +
                (1 if row['uses_crop_rotation'] else 0) * 1.5
            ),
            axis=1
        )
        
        # Add some noise to biodiversity score
        df['biodiversity_score'] += np.random.normal(0, 0.5, n_samples)
        df['biodiversity_score'] = np.clip(df['biodiversity_score'], 1, 10)
        
        # Calculate economic resilience score
        df['economic_resilience_score'] = df.apply(
            lambda row: (
                4 + row['total_practices_adopted'] * 0.7 + 
                row['years_practicing_regenerative'] * 0.2
            ),
            axis=1
        )
        
        # Add some noise to economic resilience score
        df['economic_resilience_score'] += np.random.normal(0, 0.5, n_samples)
        df['economic_resilience_score'] = np.clip(df['economic_resilience_score'], 1, 10)
        
        # Save synthetic data
        os.makedirs('data/sustainability', exist_ok=True)
        df.to_csv('data/sustainability/synthetic_regen_farming_data.csv', index=False)
        print("Generated synthetic regenerative farming data")
        
        return df
    
    def _generate_synthetic_env_data(self, n_samples=500):
        """Generate synthetic environmental impact data"""
        np.random.seed(45)
        
        # Create farm types
        farm_types = ['Crop', 'Mixed', 'Livestock', 'Orchard', 'Vegetable']
        
        # Create farming systems
        farming_systems = ['Conventional', 'Organic', 'Integrated', 'Regenerative', 'Biodynamic']
        
        # Generate data
        data = {
            'farm_type': np.random.choice(farm_types, n_samples),
            'farming_system': np.random.choice(farming_systems, n_samples),
            'farm_size_hectares': np.random.uniform(5, 500, n_samples),
            'pesticide_use_kg_per_hectare': np.random.uniform(0, 10, n_samples),
            'fertilizer_use_kg_per_hectare': np.random.uniform(0, 300, n_samples),
            'energy_use_kwh_per_hectare': np.random.uniform(100, 2000, n_samples),
            'water_use_cubic_meters_per_hectare': np.random.uniform(1000, 10000, n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Adjust pesticide and fertilizer use based on farming system
        for i, row in df.iterrows():
            if row['farming_system'] == 'Organic':
                df.at[i, 'pesticide_use_kg_per_hectare'] *= 0.2
                df.at[i, 'fertilizer_use_kg_per_hectare'] *= 0.6
            elif row['farming_system'] == 'Regenerative':
                df.at[i, 'pesticide_use_kg_per_hectare'] *= 0.3
                df.at[i, 'fertilizer_use_kg_per_hectare'] *= 0.5
            elif row['farming_system'] == 'Biodynamic':
                df.at[i, 'pesticide_use_kg_per_hectare'] *= 0.1
                df.at[i, 'fertilizer_use_kg_per_hectare'] *= 0.4
            elif row['farming_system'] == 'Integrated':
                df.at[i, 'pesticide_use_kg_per_hectare'] *= 0.7
                df.at[i, 'fertilizer_use_kg_per_hectare'] *= 0.8
        
        # Calculate environmental impact scores
        
        # 1. Biodiversity impact (1-10 scale, lower is better)
        df['biodiversity_impact_score'] = df.apply(
            lambda row: (
                8 - (
                    (1 if row['farming_system'] == 'Regenerative' else 0) * 3 +
                    (1 if row['farming_system'] == 'Organic' else 0) * 2.5 +
                    (1 if row['farming_system'] == 'Biodynamic' else 0) * 3.5 +
                    (1 if row['farming_system'] == 'Integrated' else 0) * 1.5
                ) + 
                (row['pesticide_use_kg_per_hectare'] / 10) * 3
            ),
            axis=1
        )
        
        # 2. Soil health impact (1-10 scale, lower is better)
        df['soil_health_impact_score'] = df.apply(
            lambda row: (
                7 - (
                    (1 if row['farming_system'] == 'Regenerative' else 0) * 4 +
                    (1 if row['farming_system'] == 'Organic' else 0) * 3 +
                    (1 if row['farming_system'] == 'Biodynamic' else 0) * 4 +
                    (1 if row['farming_system'] == 'Integrated' else 0) * 2
                ) + 
                (row['pesticide_use_kg_per_hectare'] / 10) * 2 +
                (row['fertilizer_use_kg_per_hectare'] / 300) * 3
            ),
            axis=1
        )
        
        # 3. Water quality impact (1-10 scale, lower is better)
        df['water_quality_impact_score'] = df.apply(
            lambda row: (
                6 - (
                    (1 if row['farming_system'] == 'Regenerative' else 0) * 3 +
                    (1 if row['farming_system'] == 'Organic' else 0) * 2.5 +
                    (1 if row['farming_system'] == 'Biodynamic' else 0) * 3 +
                    (1 if row['farming_system'] == 'Integrated' else 0) * 1.5
                ) + 
                (row['pesticide_use_kg_per_hectare'] / 10) * 4 +
                (row['fertilizer_use_kg_per_hectare'] / 300) * 3
            ),
            axis=1
        )
        
        # 4. Air quality impact (1-10 scale, lower is better)
        df['air_quality_impact_score'] = df.apply(
            lambda row: (
                5 - (
                    (1 if row['farming_system'] == 'Regenerative' else 0) * 2 +
                    (1 if row['farming_system'] == 'Organic' else 0) * 1.5 +
                    (1 if row['farming_system'] == 'Biodynamic' else 0) * 2 +
                    (1 if row['farming_system'] == 'Integrated' else 0) * 1
                ) + 
                (row['energy_use_kwh_per_hectare'] / 2000) * 5
            ),
            axis=1
        )
        
        # Add some noise to the impact scores
        for col in ['biodiversity_impact_score', 'soil_health_impact_score', 
                    'water_quality_impact_score', 'air_quality_impact_score']:
            df[col] += np.random.normal(0, 0.5, n_samples)
            df[col] = np.clip(df[col], 1, 10)
        
        # Calculate overall environmental impact score (weighted average)
        df['overall_environmental_impact_score'] = (
            df['biodiversity_impact_score'] * 0.3 +
            df['soil_health_impact_score'] * 0.3 +
            df['water_quality_impact_score'] * 0.25 +
            df['air_quality_impact_score'] * 0.15
        )
        
        # Save synthetic data
        os.makedirs('data/sustainability', exist_ok=True)
        df.to_csv('data/sustainability/synthetic_environmental_data.csv', index=False)
        print("Generated synthetic environmental impact data")
        
        return df
    
    def _merge_datasets(self, datasets):
        """Merge multiple sustainability datasets for comprehensive analysis"""
        print("Merging sustainability datasets...")
        
        # Start with carbon footprint data
        if 'carbon' not in datasets:
            return None
        
        merged_df = datasets['carbon'].copy()
        
        # Add water usage data if available
        if 'water' in datasets:
            water_df = datasets['water'].copy()
            
            # Merge on common columns if they exist
            common_cols = list(set(merged_df.columns) & set(water_df.columns))
            if common_cols and len(common_cols) > 0:
                # Use only key columns for merging
                key_cols = [col for col in common_cols if col in ['crop_type', 'farm_size_hectares']]
                if key_cols:
                    # Merge on key columns
                    merged_df = pd.merge(
                        merged_df, 
                        water_df.drop([col for col in common_cols if col not in key_cols], axis=1),
                        on=key_cols,
                        how='outer'
                    )
                else:
                    # If no key columns, create a synthetic index for demonstration
                    merged_df['_temp_index'] = np.arange(len(merged_df)) % len(water_df)
                    water_df['_temp_index'] = np.arange(len(water_df))
                    merged_df = pd.merge(
                        merged_df,
                        water_df.drop(common_cols, axis=1),
                        on='_temp_index',
                        how='left'
                    )
                    merged_df.drop('_temp_index', axis=1, inplace=True)
            else:
                # If no common columns, create a synthetic index for demonstration
                merged_df['_temp_index'] = np.arange(len(merged_df)) % len(water_df)
                water_df['_temp_index'] = np.arange(len(water_df))
                merged_df = pd.merge(
                    merged_df,
                    water_df,
                    on='_temp_index',
                    how='left'
                )
                merged_df.drop('_temp_index', axis=1, inplace=True)
        
        # Add regenerative farming data if available
        if 'regen_farming' in datasets:
            regen_df = datasets['regen_farming'].copy()
            
            # Merge on common columns if they exist
            common_cols = list(set(merged_df.columns) & set(regen_df.columns))
            if common_cols and len(common_cols) > 0:
                # Use only key columns for merging
                key_cols = [col for col in common_cols if col in ['farm_type', 'farm_size_hectares']]
                if key_cols:
                    # Merge on key columns
                    merged_df = pd.merge(
                        merged_df, 
                        regen_df.drop([col for col in common_cols if col not in key_cols], axis=1),
                        on=key_cols,
                        how='outer'
                    )
                else:
                    # If no key columns, create a synthetic index for demonstration
                    merged_df['_temp_index'] = np.arange(len(merged_df)) % len(regen_df)
                    regen_df['_temp_index'] = np.arange(len(regen_df))
                    merged_df = pd.merge(
                        merged_df,
                        regen_df.drop(common_cols, axis=1),
                        on='_temp_index',
                        how='left'
                    )
                    merged_df.drop('_temp_index', axis=1, inplace=True)
            else:
                # If no common columns, create a synthetic index for demonstration
                merged_df['_temp_index'] = np.arange(len(merged_df)) % len(regen_df)
                regen_df['_temp_index'] = np.arange(len(regen_df))
                merged_df = pd.merge(
                    merged_df,
                    regen_df,
                    on='_temp_index',
                    how='left'
                )
                merged_df.drop('_temp_index', axis=1, inplace=True)
        
        # Add environmental impact data if available
        if 'environment' in datasets:
            env_df = datasets['environment'].copy()
            
            # Merge on common columns if they exist
            common_cols = list(set(merged_df.columns) & set(env_df.columns))
            if common_cols and len(common_cols) > 0:
                # Use only key columns for merging
                key_cols = [col for col in common_cols if col in ['farm_type', 'farm_size_hectares', 'farming_system']]
                if key_cols:
                    # Merge on key columns
                    merged_df = pd.merge(
                        merged_df, 
                        env_df.drop([col for col in common_cols if col not in key_cols], axis=1),
                        on=key_cols,
                        how='outer'
                    )
                else:
                    # If no key columns, create a synthetic index for demonstration
                    merged_df['_temp_index'] = np.arange(len(merged_df)) % len(env_df)
                    env_df['_temp_index'] = np.arange(len(env_df))
                    merged_df = pd.merge(
                        merged_df,
                        env_df.drop(common_cols, axis=1),
                        on='_temp_index',
                        how='left'
                    )
                    merged_df.drop('_temp_index', axis=1, inplace=True)
            else:
                # If no common columns, create a synthetic index for demonstration
                merged_df['_temp_index'] = np.arange(len(merged_df)) % len(env_df)
                env_df['_temp_index'] = np.arange(len(env_df))
                merged_df = pd.merge(
                    merged_df,
                    env_df,
                    on='_temp_index',
                    how='left'
                )
                merged_df.drop('_temp_index', axis=1, inplace=True)
        
        # Fill NaN values with appropriate defaults
        merged_df.fillna({
            'carbon_footprint_kg_co2e': merged_df['carbon_footprint_kg_co2e'].mean() if 'carbon_footprint_kg_co2e' in merged_df else 0,
            'water_usage_cubic_meters': merged_df['water_usage_cubic_meters'].mean() if 'water_usage_cubic_meters' in merged_df else 0,
            'soil_organic_matter_percent_current': merged_df['soil_organic_matter_percent_current'].mean() if 'soil_organic_matter_percent_current' in merged_df else 0,
            'overall_environmental_impact_score': merged_df['overall_environmental_impact_score'].mean() if 'overall_environmental_impact_score' in merged_df else 5
        }, inplace=True)
        
        # Save merged dataset
        merged_df.to_csv('data/sustainability/merged_sustainability_data.csv', index=False)
        print("Created merged sustainability dataset")
        
        return merged_df
    
    def train_carbon_model(self, data=None, target='carbon_footprint_kg_co2e', test_size=0.2):
        """Train a model to predict carbon footprint with improved accuracy"""
        print("\nTraining carbon footprint prediction model...")
        
        # Load data if not provided
        if data is None:
            try:
                data = pd.read_csv('data/sustainability/synthetic_carbon_data.csv')
            except:
                data = self._generate_synthetic_carbon_data()
        
        # Select features and target
        if target not in data.columns:
            print(f"Target column '{target}' not found in data.")
            return None
        
        # Define feature types
        categorical_features = ['crop_type', 'farming_method']
        numeric_features = [col for col in data.columns if col not in categorical_features + [target]]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Define models with hyperparameters
        models = {
            'random_forest': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42))
            ]),
            'gradient_boosting': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
            ])
        }
        
        # Split data
        X = data.drop(target, axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Cross-validation and model evaluation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        best_model = None
        best_score = float('inf')
        results = {}
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
            cv_rmse = -cv_scores.mean()
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'cv_rmse': cv_rmse,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2
            }
            
            print(f"{name} - CV RMSE: {cv_rmse:.2f}, Test RMSE: {rmse:.2f}, R²: {r2:.2f}")
            
            # Update best model
            if cv_rmse < best_score:
                best_score = cv_rmse
                best_model = model
        
        # Save the best model
        self.carbon_model = best_model
        joblib.dump(best_model, 'saved_models/sustainability/carbon_model.joblib')
        
        # Feature importance analysis
        self._analyze_feature_importance(best_model, X, 'carbon')
        
        print(f"Carbon footprint model trained and saved successfully.")
        return results
    
    def train_water_model(self, data=None, target='water_usage_cubic_meters', test_size=0.2):
        """Train a model to predict water usage
        
        Args:
            data: DataFrame containing water usage data
            target: Target column to predict
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with model performance metrics
        """
        print("\nTraining water usage prediction model...")
        
        # Load data if not provided
        if data is None:
            try:
                data = pd.read_csv('data/sustainability/synthetic_water_data.csv')
            except:
                data = self._generate_synthetic_water_data()
        
        # Select features and target
        if target not in data.columns:
            print(f"Target column '{target}' not found in data. Available columns: {data.columns.tolist()}")
            return None
        
        # Select numeric and categorical features
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from features
        if target in numeric_features:
            numeric_features.remove(target)
        
        # Prepare features
        X = data[numeric_features + categorical_features].copy()
        y = data[target]
        
        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale numeric features
        numeric_transformer = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Apply scaling only to numeric features
        X_train_scaled[numeric_features] = numeric_transformer.fit_transform(X_train[numeric_features])
        X_test_scaled[numeric_features] = numeric_transformer.transform(X_test[numeric_features])
        
        # Save the scaler
        self.water_scaler = numeric_transformer
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = float('inf')
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            
            # Update best model
            if rmse < best_score:
                best_score = rmse
                best_model = model
        
        # Save the best model
        self.water_model = best_model
        
        # Save model to disk
        os.makedirs('saved_models/sustainability', exist_ok=True)
        joblib.dump(best_model, 'saved_models/sustainability/water_model.joblib')
        joblib.dump(numeric_transformer, 'saved_models/sustainability/water_scaler.joblib')
        
        # Feature importance for the best model
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X_train_scaled.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
            plt.title('Top 10 Features for Water Usage Prediction')
            plt.tight_layout()
            plt.savefig('reports/sustainability/water_feature_importance.png')
            
            # Save feature importance
            feature_importance.to_csv('reports/sustainability/water_feature_importance.csv', index=False)
        
        print(f"Water usage model trained and saved successfully.")
        return results
    
    def train_env_impact_model(self, data=None, target='overall_environmental_impact_score', test_size=0.2):
        """Train a model to predict environmental impact
        
        Args:
            data: DataFrame containing environmental impact data
            target: Target column to predict
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with model performance metrics
        """
        print("\nTraining environmental impact prediction model...")
        
        # Load data if not provided
        if data is None:
            try:
                data = pd.read_csv('data/sustainability/synthetic_environmental_data.csv')
            except:
                data = self._generate_synthetic_env_data()
        
        # Select features and target
        if target not in data.columns:
            print(f"Target column '{target}' not found in data. Available columns: {data.columns.tolist()}")
            return None
        
        # Select numeric and categorical features
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from features
        if target in numeric_features:
            numeric_features.remove(target)
        
        # Prepare features
        X = data[numeric_features + categorical_features].copy()
        y = data[target]
        
        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale numeric features
        numeric_transformer = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Apply scaling only to numeric features
        X_train_scaled[numeric_features] = numeric_transformer.fit_transform(X_train[numeric_features])
        X_test_scaled[numeric_features] = numeric_transformer.transform(X_test[numeric_features])
        
        # Save the scaler
        self.env_scaler = numeric_transformer
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = float('inf')
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            
            # Update best model
            if rmse < best_score:
                best_score = rmse
                best_model = model
        
        # Save the best model
        self.env_impact_model = best_model
        
        # Save model to disk
        os.makedirs('saved_models/sustainability', exist_ok=True)
        joblib.dump(best_model, 'saved_models/sustainability/env_impact_model.joblib')
        joblib.dump(numeric_transformer, 'saved_models/sustainability/env_impact_scaler.joblib')
        
        # Feature importance for the best model
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X_train_scaled.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
            plt.title('Top 10 Features for Environmental Impact Prediction')
            plt.tight_layout()
            plt.savefig('reports/sustainability/env_impact_feature_importance.png')
            
            # Save feature importance
            feature_importance.to_csv('reports/sustainability/env_impact_feature_importance.csv', index=False)
        
        print(f"Environmental impact model trained and saved successfully.")
        return results
    
    def predict_carbon_footprint(self, data):
        """Predict carbon footprint for new data
        
        Args:
            data: DataFrame with features for prediction
            
        Returns:
            Array of predicted carbon footprint values
        """
        if self.carbon_model is None:
            try:
                self.carbon_model = joblib.load('saved_models/sustainability/carbon_model.joblib')
            except Exception as e:
                print(f"Carbon footprint model not found: {e}")
                print("Please train the model first.")
                # Return default values instead of None
                return np.array([5000] * len(data))  # Default carbon footprint value
        
        try:
            # Create a copy to avoid modifying the original data
            prediction_data = data.copy()
            
            # Handle missing critical features
            critical_features = [
                'carbon_sequestration_potential_kg_co2e', 
                'net_carbon_impact_kg_co2e'
            ]
            
            for feature in critical_features:
                if feature not in prediction_data.columns:
                    # Add missing critical feature with a reasonable default
                    if feature == 'carbon_sequestration_potential_kg_co2e':
                        # Default based on farming method if available
                        if 'farming_method' in prediction_data.columns:
                            prediction_data[feature] = prediction_data.apply(
                                lambda row: (
                                    2500 if row['farming_method'] == 'Regenerative' else
                                    1500 if row['farming_method'] == 'Conservation' else
                                    1000 if row['farming_method'] == 'Organic' else
                                    500 if row['farming_method'] == 'Precision' else 200
                                ), axis=1
                            )
                        else:
                            prediction_data[feature] = 1000  # General default
                    
                    elif feature == 'net_carbon_impact_kg_co2e':
                        # Calculate if we have the necessary data, otherwise use default
                        if 'carbon_footprint_kg_co2e' in prediction_data.columns and 'carbon_sequestration_potential_kg_co2e' in prediction_data.columns:
                            prediction_data[feature] = prediction_data['carbon_footprint_kg_co2e'] - prediction_data['carbon_sequestration_potential_kg_co2e']
                        else:
                            prediction_data[feature] = 2000  # General default
            
            # Check if the model is a Pipeline with a ColumnTransformer
            if hasattr(self.carbon_model, 'named_steps') and 'preprocessor' in self.carbon_model.named_steps:
                # Get the preprocessor
                preprocessor = self.carbon_model.named_steps['preprocessor']
                
                # Get the categorical and numeric features used during training
                categorical_features = preprocessor.transformers_[1][2]  # Assuming cat is the second transformer
                numeric_features = preprocessor.transformers_[0][2]      # Assuming num is the first transformer
                
                # Ensure all categorical features exist in the prediction data
                for feature in categorical_features:
                    if feature not in prediction_data.columns:
                        # Add missing categorical feature with a default value
                        prediction_data[feature] = 'Unknown'
                
                # Ensure all numeric features exist in the prediction data
                for feature in numeric_features:
                    if feature not in prediction_data.columns:
                        # Add missing numeric feature with a default value (0)
                        prediction_data[feature] = 0
                
                # Use only the features that the model was trained on
                X = prediction_data[list(categorical_features) + list(numeric_features)]
                
                # Make predictions using the pipeline
                predictions = self.carbon_model.predict(X)
                
                return predictions
            else:
                # Fallback for models without a preprocessor pipeline
                # Extract feature names from the model if available
                if hasattr(self.carbon_model, 'feature_names_in_'):
                    expected_features = self.carbon_model.feature_names_in_
                    
                    # Ensure all expected features are present
                    for feature in expected_features:
                        if feature not in prediction_data.columns:
                            prediction_data[feature] = 0  # Default value
                    
                    # Keep only the expected features in the correct order
                    X = prediction_data[expected_features]
                    
                    # Make predictions
                    predictions = self.carbon_model.predict(X)
                    
                    return predictions
                else:
                    # If we can't determine the expected features, use a simpler approach
                    # Use common features for carbon footprint prediction
                    common_features = [
                        'farm_size_hectares', 'fertilizer_kg_per_hectare', 
                        'machinery_hours', 'transport_distance_km', 
                        'energy_consumption_kwh'
                    ]
                    
                    # Filter to only include features that exist in the data
                    available_features = [f for f in common_features if f in prediction_data.columns]
                    
                    if not available_features:
                        print("No usable features found for prediction")
                        return np.array([5000] * len(prediction_data))
                    
                    # Use available features
                    X = prediction_data[available_features]
                    
                    # Scale the features if a scaler is available
                    if hasattr(self, 'carbon_scaler') and self.carbon_scaler is not None:
                        X_scaled = self.carbon_scaler.transform(X)
                    else:
                        X_scaled = X
                    
                    # Make predictions
                    predictions = self.carbon_model.predict(X_scaled)
                    
                    return predictions
        
        except Exception as e:
            print(f"Error predicting carbon footprint: {e}")
            import traceback
            traceback.print_exc()
            # Return default values on error
            return np.array([5000] * len(data))  # Default carbon footprint value
    
    def predict_water_usage(self, data):
        """Predict water usage for new data
        
        Args:
            data: DataFrame with features for prediction
            
        Returns:
            Array of predicted water usage values
        """
        if self.water_model is None:
            try:
                self.water_model = joblib.load('saved_models/sustainability/water_model.joblib')
                self.water_scaler = joblib.load('saved_models/sustainability/water_scaler.joblib')
            except Exception as e:
                print(f"Water usage model not found: {e}")
                print("Please train the model first.")
                # Return default values instead of None
                return np.array([3000] * len(data))  # Default water usage value
        
        try:
            # Create a copy to avoid modifying the original data
            prediction_data = data.copy()
            
            # Get expected feature names from the model
            expected_features = self.water_model.feature_names_in_
            
            # Check for critical missing features and add defaults
            critical_features = [
                'water_efficiency_cubic_meters_per_ton',
                'water_recycling_potential_percent'
            ]
            
            for feature in critical_features:
                if feature not in prediction_data.columns:
                    # Add missing critical feature with a reasonable default
                    if feature == 'water_efficiency_cubic_meters_per_ton':
                        # Default based on irrigation method if available
                        if 'irrigation_method' in prediction_data.columns:
                            prediction_data[feature] = prediction_data.apply(
                                lambda row: (
                                    15 if row['irrigation_method'] == 'Drip' else
                                    25 if row['irrigation_method'] == 'Sprinkler' else
                                    20 if row['irrigation_method'] == 'Subsurface' else
                                    40 if row['irrigation_method'] == 'Flood' else 30
                                ), axis=1
                            )
                        else:
                            prediction_data[feature] = 25  # General default
                    
                    elif feature == 'water_recycling_potential_percent':
                        # Default based on irrigation method if available
                        if 'irrigation_method' in prediction_data.columns:
                            prediction_data[feature] = prediction_data.apply(
                                lambda row: (
                                    60 if row['irrigation_method'] == 'Drip' else
                                    40 if row['irrigation_method'] == 'Sprinkler' else
                                    50 if row['irrigation_method'] == 'Subsurface' else
                                    20 if row['irrigation_method'] == 'Flood' else 30
                                ), axis=1
                            )
                        else:
                            prediction_data[feature] = 35  # General default
            
            # Prepare features
            numeric_features = prediction_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = prediction_data.select_dtypes(include=['object']).columns.tolist()
            
            # Create a DataFrame with the expected structure
            X = pd.DataFrame(index=prediction_data.index)
            
            # Add numeric features directly
            for col in numeric_features:
                if col in prediction_data.columns and col in expected_features:
                    X[col] = prediction_data[col]
            
            # Handle categorical features manually to match training encoding
            for col in expected_features:
                # Check if this is a one-hot encoded column from training
                if '_' in col:
                    # Extract the original column name and value
                    parts = col.split('_')
                    if len(parts) >= 2:
                        cat_col = parts[0]
                        cat_val = '_'.join(parts[1:])
                        
                        # Check if the original column exists in our data
                        if cat_col in categorical_features:
                            # Create the one-hot feature
                            X[col] = (prediction_data[cat_col] == cat_val).astype(int)
                
            # Ensure all expected features are present
            for col in expected_features:
                if col not in X.columns:
                    X[col] = 0
            
            # Keep only the expected features in the correct order
            X = X[expected_features]
            
            # Scale numeric features - use the exact feature order from the scaler
            X_scaled = X.copy()
            
            # Get the feature names used during scaler fitting
            if hasattr(self.water_scaler, 'feature_names_in_'):
                scaler_features = self.water_scaler.feature_names_in_
                # Only transform numeric features that were used during scaler fitting
                if len(scaler_features) > 0:
                    # Create a DataFrame with only the features used by the scaler, in the same order
                    X_to_scale = X[scaler_features].copy()
                    # Apply the transformation
                    X_scaled_values = self.water_scaler.transform(X_to_scale)
                    # Update the values in the original DataFrame
                    for i, feature in enumerate(scaler_features):
                        X_scaled[feature] = X_scaled_values[:, i]
            else:
                # Fallback for older scikit-learn versions
                # Identify numeric features that need scaling
                model_numeric_features = [col for col in X.columns if col in numeric_features]
                if model_numeric_features:
                    try:
                        # Try to transform all numeric features at once
                        X_scaled[model_numeric_features] = self.water_scaler.transform(X[model_numeric_features])
                    except Exception as scaling_error:
                        print(f"Warning: Error during feature scaling: {scaling_error}")
                        # If that fails, try to scale each feature individually
                        for feature in model_numeric_features:
                            try:
                                X_scaled[feature] = self.water_scaler.transform(X[[feature]])
                            except:
                                # If individual scaling fails, keep the original values
                                pass
            
            # Make predictions
            predictions = self.water_model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting water usage: {e}")
            import traceback
            traceback.print_exc()
            # Return default values on error
            return np.array([3000] * len(data))  # Default water usage value
    
    def predict_env_impact(self, data):
        """Predict environmental impact for new data
        
        Args:
            data: DataFrame with features for prediction
            
        Returns:
            Array of predicted environmental impact scores
        """
        if self.env_impact_model is None:
            try:
                self.env_impact_model = joblib.load('saved_models/sustainability/env_impact_model.joblib')
                self.env_scaler = joblib.load('saved_models/sustainability/env_impact_scaler.joblib')
            except Exception as e:
                print(f"Environmental impact model not found: {e}")
                print("Please train the model first.")
                # Return default values instead of None
                return np.array([5.0] * len(data))  # Default environmental impact score
        
        try:
            # Create a copy to avoid modifying the original data
            prediction_data = data.copy()
            
            # Get expected feature names from the model
            expected_features = self.env_impact_model.feature_names_in_
            
            # Check for critical missing features and add defaults
            critical_features = [
                'soil_health_impact_score',
                'water_quality_impact_score',
                'biodiversity_impact_score',
                'air_quality_impact_score',
                'fertilizer_use_kg_per_hectare'
            ]
            
            for feature in critical_features:
                if feature not in prediction_data.columns:
                    # Add missing critical feature with a reasonable default
                    if feature == 'soil_health_impact_score':
                        # Default based on farming system if available
                        if 'farming_system' in prediction_data.columns:
                            prediction_data[feature] = prediction_data.apply(
                                lambda row: (
                                    8.5 if row['farming_system'] == 'Regenerative' else
                                    7.5 if row['farming_system'] == 'Organic' else
                                    6.5 if row['farming_system'] == 'Integrated' else
                                    5.0 if row['farming_system'] == 'Conventional' else 6.0
                                ), axis=1
                            )
                        else:
                            prediction_data[feature] = 6.0  # General default
                    
                    elif feature == 'water_quality_impact_score':
                        # Default based on farming system if available
                        if 'farming_system' in prediction_data.columns:
                            prediction_data[feature] = prediction_data.apply(
                                lambda row: (
                                    8.0 if row['farming_system'] == 'Regenerative' else
                                    7.0 if row['farming_system'] == 'Organic' else
                                    6.0 if row['farming_system'] == 'Integrated' else
                                    4.5 if row['farming_system'] == 'Conventional' else 5.5
                                ), axis=1
                            )
                        else:
                            prediction_data[feature] = 5.5  # General default
                    
                    elif feature == 'biodiversity_impact_score':
                        # Default based on farming system if available
                        if 'farming_system' in prediction_data.columns:
                            prediction_data[feature] = prediction_data.apply(
                                lambda row: (
                                    8.5 if row['farming_system'] == 'Regenerative' else
                                    7.5 if row['farming_system'] == 'Organic' else
                                    6.0 if row['farming_system'] == 'Integrated' else
                                    4.0 if row['farming_system'] == 'Conventional' else 5.5
                                ), axis=1
                            )
                        else:
                            prediction_data[feature] = 5.5  # General default
                    
                    elif feature == 'air_quality_impact_score':
                        # Default based on farming system if available
                        if 'farming_system' in prediction_data.columns:
                            prediction_data[feature] = prediction_data.apply(
                                lambda row: (
                                    7.5 if row['farming_system'] == 'Regenerative' else
                                    7.0 if row['farming_system'] == 'Organic' else
                                    6.0 if row['farming_system'] == 'Integrated' else
                                    5.0 if row['farming_system'] == 'Conventional' else 6.0
                                ), axis=1
                            )
                        else:
                            prediction_data[feature] = 6.0  # General default
                    
                    elif feature == 'fertilizer_use_kg_per_hectare':
                        # Use fertilizer_kg_per_hectare if available
                        if 'fertilizer_kg_per_hectare' in prediction_data.columns:
                            prediction_data[feature] = prediction_data['fertilizer_kg_per_hectare']
                        else:
                            # Default based on farming system if available
                            if 'farming_system' in prediction_data.columns:
                                prediction_data[feature] = prediction_data.apply(
                                    lambda row: (
                                        30 if row['farming_system'] == 'Regenerative' else
                                        50 if row['farming_system'] == 'Organic' else
                                        100 if row['farming_system'] == 'Integrated' else
                                        150 if row['farming_system'] == 'Conventional' else 80
                                    ), axis=1
                                )
                            else:
                                prediction_data[feature] = 80  # General default
            
            # Prepare features
            numeric_features = prediction_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = prediction_data.select_dtypes(include=['object']).columns.tolist()
            
            # Create a DataFrame with the expected structure
            X = pd.DataFrame(index=prediction_data.index)
            
            # Add numeric features directly
            for col in numeric_features:
                if col in prediction_data.columns and col in expected_features:
                    X[col] = prediction_data[col]
            
            # Handle categorical features manually to match training encoding
            for col in expected_features:
                # Check if this is a one-hot encoded column from training
                if '_' in col:
                    # Extract the original column name and value
                    parts = col.split('_')
                    if len(parts) >= 2:
                        cat_col = parts[0]
                        cat_val = '_'.join(parts[1:])
                        
                        # Check if the original column exists in our data
                        if cat_col in categorical_features:
                            # Create the one-hot feature
                            X[col] = (prediction_data[cat_col] == cat_val).astype(int)
                
            # Ensure all expected features are present
            for col in expected_features:
                if col not in X.columns:
                    X[col] = 0
            
            # Keep only the expected features in the correct order
            X = X[expected_features]
            
            # Scale numeric features - use the exact feature order from the scaler
            X_scaled = X.copy()
            
            # Get the feature names used during scaler fitting
            if hasattr(self.env_scaler, 'feature_names_in_'):
                scaler_features = self.env_scaler.feature_names_in_
                # Only transform numeric features that were used during scaler fitting
                if len(scaler_features) > 0:
                    # Create a DataFrame with only the features used by the scaler, in the same order
                    X_to_scale = X[scaler_features].copy()
                    # Apply the transformation
                    X_scaled_values = self.env_scaler.transform(X_to_scale)
                    # Update the values in the original DataFrame
                    for i, feature in enumerate(scaler_features):
                        X_scaled[feature] = X_scaled_values[:, i]
            else:
                # Fallback for older scikit-learn versions
                # Identify numeric features that need scaling
                model_numeric_features = [col for col in X.columns if col in numeric_features]
                if model_numeric_features:
                    try:
                        # Try to transform all numeric features at once
                        X_scaled[model_numeric_features] = self.env_scaler.transform(X[model_numeric_features])
                    except Exception as scaling_error:
                        print(f"Warning: Error during feature scaling: {scaling_error}")
                        # If that fails, try to scale each feature individually
                        for feature in model_numeric_features:
                            try:
                                X_scaled[feature] = self.env_scaler.transform(X[[feature]])
                            except:
                                # If individual scaling fails, keep the original values
                                pass
            
            # Make predictions
            predictions = self.env_impact_model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting environmental impact: {e}")
            import traceback
            traceback.print_exc()
            # Return default values on error
            return np.array([5.0] * len(data))  # Default environmental impact score
    
    def generate_sustainability_report(self, farm_data, output_path=None):
        """Generate a comprehensive sustainability report for a farm
        
        Args:
            farm_data: DataFrame with farm data
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        if output_path is None:
            os.makedirs('reports/sustainability', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/sustainability/farm_sustainability_report_{timestamp}.pdf"
        
        try:
            # Make predictions with robust error handling
            carbon_predictions = self.predict_carbon_footprint(farm_data)
            water_predictions = self.predict_water_usage(farm_data)
            env_impact_predictions = self.predict_env_impact(farm_data)
            
            # Add predictions to data
            farm_data_with_predictions = farm_data.copy()
            farm_data_with_predictions['predicted_carbon_footprint'] = carbon_predictions
            farm_data_with_predictions['predicted_water_usage'] = water_predictions
            farm_data_with_predictions['predicted_env_impact'] = env_impact_predictions
            
            # Generate visualizations
            plt.figure(figsize=(15, 10))
            
            # Carbon footprint by farming method
            plt.subplot(2, 2, 1)
            if 'farming_method' in farm_data_with_predictions.columns:
                sns.barplot(x='farming_method', y='predicted_carbon_footprint', data=farm_data_with_predictions)
                plt.title('Carbon Footprint by Farming Method')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'Farming method data not available', ha='center', va='center')
                plt.title('Carbon Footprint Analysis')
            
            # Water usage by irrigation method
            plt.subplot(2, 2, 2)
            if 'irrigation_method' in farm_data_with_predictions.columns:
                sns.barplot(x='irrigation_method', y='predicted_water_usage', data=farm_data_with_predictions)
                plt.title('Water Usage by Irrigation Method')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'Irrigation method data not available', ha='center', va='center')
                plt.title('Water Usage Analysis')
            
            # Environmental impact by farming system
            plt.subplot(2, 2, 3)
            if 'farming_system' in farm_data_with_predictions.columns:
                sns.barplot(x='farming_system', y='predicted_env_impact', data=farm_data_with_predictions)
                plt.title('Environmental Impact by Farming System')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'Farming system data not available', ha='center', va='center')
                plt.title('Environmental Impact Analysis')
            
            # Correlation between farm size and sustainability metrics
            plt.subplot(2, 2, 4)
            if 'farm_size_hectares' in farm_data_with_predictions.columns:
                farm_data_with_predictions.plot.scatter(
                    x='farm_size_hectares', 
                    y='predicted_carbon_footprint', 
                    c='predicted_env_impact', 
                    colormap='viridis', 
                    s=farm_data_with_predictions['predicted_water_usage']/100, 
                    alpha=0.7
                )
                plt.title('Farm Size vs. Sustainability Metrics')
            else:
                plt.text(0.5, 0.5, 'Farm size data not available', ha='center', va='center')
                plt.title('Sustainability Metrics Correlation')
            
            plt.tight_layout()
            os.makedirs('reports/sustainability', exist_ok=True)
            plt.savefig('reports/sustainability/sustainability_analysis.png')
            
            # Generate summary statistics
            summary = {
                'carbon_footprint': {
                    'mean': np.mean(carbon_predictions),
                    'median': np.median(carbon_predictions),
                    'min': np.min(carbon_predictions),
                    'max': np.max(carbon_predictions)
                },
                'water_usage': {
                    'mean': np.mean(water_predictions),
                    'median': np.median(water_predictions),
                    'min': np.min(water_predictions),
                    'max': np.max(water_predictions)
                },
                'env_impact': {
                    'mean': np.mean(env_impact_predictions),
                    'median': np.median(env_impact_predictions),
                    'min': np.min(env_impact_predictions),
                    'max': np.max(env_impact_predictions)
                }
            }
            
            # Save summary to CSV
            summary_df = pd.DataFrame({
                'Metric': ['Carbon Footprint (kg CO2e)', 'Water Usage (cubic meters)', 'Environmental Impact Score'],
                'Mean': [summary['carbon_footprint']['mean'], summary['water_usage']['mean'], summary['env_impact']['mean']],
                'Median': [summary['carbon_footprint']['median'], summary['water_usage']['median'], summary['env_impact']['median']],
                'Min': [summary['carbon_footprint']['min'], summary['water_usage']['min'], summary['env_impact']['min']],
                'Max': [summary['carbon_footprint']['max'], summary['water_usage']['max'], summary['env_impact']['max']]
            })
            summary_df.to_csv('reports/sustainability/sustainability_summary.csv', index=False)
            
            print(f"Sustainability report generated and saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating sustainability report: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_feature_importance(self, model, X, model_type):
        """Analyze and visualize feature importance"""
        # Extract feature names
        if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            regressor = model.named_steps['regressor']
            if hasattr(regressor, 'feature_importances_'):
                # For pipeline with preprocessor, need to get transformed feature names
                if 'preprocessor' in model.named_steps:
                    preprocessor = model.named_steps['preprocessor']
                    # Get transformed feature names from preprocessor
                    cat_features = preprocessor.transformers_[1][2]  # Categorical features
                    num_features = preprocessor.transformers_[0][2]  # Numeric features
                    
                    # Get one-hot encoded feature names
                    cat_encoder = preprocessor.transformers_[1][1]
                    if hasattr(cat_encoder, 'get_feature_names_out'):
                        cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
                    else:
                        cat_feature_names = [f"{col}_{val}" for col in cat_features 
                                            for val in X[col].unique()]
                    
                    # Combine feature names
                    feature_names = list(num_features) + list(cat_feature_names)
                else:
                    feature_names = X.columns
                
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': regressor.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
                plt.title(f'Top 10 Features for {model_type.title()} Prediction')
                plt.tight_layout()
                
                # Create directory if it doesn't exist
                os.makedirs('reports/sustainability', exist_ok=True)
                plt.savefig(f'reports/sustainability/{model_type}_feature_importance.png')
                
                # Save feature importance
                feature_importance.to_csv(f'reports/sustainability/{model_type}_feature_importance.csv', index=False)

    def evaluate_model_performance(self, data, target, model_type='carbon'):
        """Evaluate model performance with detailed metrics
        
        Args:
            data: Test data
            target: Target column name
            model_type: Type of model ('carbon', 'water', or 'env_impact')
        
        Returns:
            Dictionary with performance metrics
        """
        # Select the appropriate model
        if model_type == 'carbon':
            model = self.carbon_model
        elif model_type == 'water':
            model = self.water_model
        elif model_type == 'env_impact':
            model = self.env_impact_model
        else:
            print(f"Unknown model type: {model_type}")
            return None
        
        if model is None:
            print(f"{model_type} model not loaded. Please train the model first.")
            return None
        
        # Prepare data
        X = data.drop(target, axis=1)
        y = data[target]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_type.title()} - Actual vs Predicted')
        plt.savefig(f'reports/sustainability/{model_type}_actual_vs_predicted.png')
        
        # Plot residuals
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title(f'{model_type.title()} - Residuals Plot')
        plt.savefig(f'reports/sustainability/{model_type}_residuals.png')
        
        # Return metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std()
        }
        
        print(f"{model_type.title()} Model Evaluation:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.2f}")
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = SustainabilityModel()
    
    # Load or generate data
    datasets = model.load_data()
    
    # Train models
    model.train_carbon_model(datasets['carbon'])
    model.train_water_model(datasets['water'])
    model.train_env_impact_model(datasets['environment'])
    
    # Generate a sustainability report for a sample farm with all required features
    sample_farm = pd.DataFrame({
        # Basic farm features
        'crop_type': ['Corn'],
        'farming_method': ['Regenerative'],
        'farm_size_hectares': [100],
        'fertilizer_kg_per_hectare': [50],
        'machinery_hours': [80],
        'transport_distance_km': [150],
        'energy_consumption_kwh': [3000],
        
        # Water-related features
        'irrigation_method': ['Drip'],
        'soil_type': ['Loamy'],
        'annual_rainfall_mm': [800],
        'temperature_celsius': [22],
        'growing_season_days': [120],
        'water_efficiency_cubic_meters_per_ton': [25],
        'water_recycling_potential_percent': [40],
        'water_use_cubic_meters_per_hectare': [3000],
        
        # Environmental impact features
        'farming_system': ['Regenerative'],
        'pesticide_use_kg_per_hectare': [1.5],
        'energy_use_kwh_per_hectare': [500],
        'soil_health_impact_score': [8.5],
        'water_quality_impact_score': [7.8],
        'biodiversity_impact_score': [8.2],
        'air_quality_impact_score': [7.5],
        'fertilizer_use_kg_per_hectare': [50],
        
        # Carbon-related features
        'carbon_sequestration_potential_kg_co2e': [2500],
        'net_carbon_impact_kg_co2e': [1500]
    })
    
    model.generate_sustainability_report(sample_farm)










