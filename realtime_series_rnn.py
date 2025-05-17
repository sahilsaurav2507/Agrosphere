import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os

class RealtimeSeriesRNN:
    def __init__(self, sequence_length=30, n_features=5, output_size=1, scaling_method='minmax', use_lr_schedule=False):
        """Initialize RNN model with configuration parameters"""
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.output_size = output_size
        self.scaling_method = scaling_method
        self.use_lr_schedule = use_lr_schedule
        self.history = None
        self.model_dir = 'saved_models'
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        
        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_scaler()
        self.model = self._build_model()
        
    def _initialize_scaler(self):
        """Initialize data scaler based on specified method"""
        scalers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        self.scaler = scalers.get(self.scaling_method, MinMaxScaler())
        
    def _build_model(self):
        """Build and compile the RNN model"""
        # Configure optimizer with gradient clipping
        optimizer = Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        # Define model architecture
        model = Sequential([
            # Input layer
            LSTM(128, return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.005),
                 recurrent_regularizer=tf.keras.regularizers.l2(0.005),
                 input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layers
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(self.output_size)
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model

    def preprocess_data(self, data, target_column=None):
        """Enhanced data preprocessing with validation"""
        # Input validation
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be pandas DataFrame or numpy array")
            
        # Handle missing values
        if isinstance(data, pd.DataFrame):
            data = data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Extract features and target
        X, y = self._extract_features_target(data, target_column)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        return self._create_sequences(X_scaled, y)

    def _extract_features_target(self, data, target_column):
        if isinstance(data, pd.DataFrame):
            if target_column:
                y = data[target_column].values
                X = data.drop(columns=[target_column]).values
            else:
                X = data.values
                y = None
        else:
            X = data
            y = None
        return X, y

    def _create_sequences(self, X_scaled, y=None):
        seq_length = min(self.sequence_length, X_scaled.shape[0] - 1)
        X_sequences = []
        y_targets = []
        
        for i in range(len(X_scaled) - seq_length):
            sequence = X_scaled[i:i+seq_length]
            X_sequences.append(sequence)
            if y is not None:
                y_targets.append(y[i+seq_length])
        
        X_sequences = np.array(X_sequences)
        return (X_sequences, np.array(y_targets)) if y is not None else X_sequences

    def train(self, data, target_column, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model with the provided data"""
        try:
            # Preprocess and split data
            X_sequences, y_targets = self.preprocess_data(data, target_column)
            X_train, X_val, y_train, y_val = train_test_split(
                X_sequences, y_targets, 
                test_size=validation_split, 
                shuffle=False
            )
            
            # Augment training data
            X_train_aug, y_train_aug = self._augment_data(X_train, y_train)
            
            # Train model
            self.history = self.model.fit(
                X_train_aug, y_train_aug,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=self._get_callbacks(),
                verbose=1
            )
            
            return self.history
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

    def _augment_data(self, X, y):
        """Enhanced data augmentation"""
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            # Original sequence
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # Add noise
            noise = np.random.normal(0, 0.01, X[i].shape)
            augmented_X.append(X[i] + noise)
            augmented_y.append(y[i])
            
            # Scale variation
            scale_factor = np.random.uniform(0.95, 1.05)
            augmented_X.append(X[i] * scale_factor)
            augmented_y.append(y[i])
        
        return np.array(augmented_X), np.array(augmented_y)

    def _get_callbacks(self):
        """Configure training callbacks"""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.keras')
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-6
            )
        ]

    def _plot_training_history(self, history):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def predict(self, data):
        """Generate predictions for input data"""
        try:
            X_sequences = self.preprocess_data(data)
            return self.model.predict(X_sequences)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise
    
    def predict_next_n_steps(self, initial_sequence, n_steps=10):
        """Predict next n steps using recursive forecasting"""
        if not isinstance(initial_sequence, np.ndarray):
            initial_sequence = np.array(initial_sequence)
            
        if initial_sequence.shape[0] < self.sequence_length:
            raise ValueError(f"Initial sequence must have at least {self.sequence_length} time steps")
        
        # Scale the initial sequence
        scaled_sequence = self.scaler.transform(initial_sequence)
        
        # Initialize the forecast array
        forecasts = []
        
        # Get the most recent sequence
        current_sequence = scaled_sequence[-self.sequence_length:].reshape(1, self.sequence_length, self.n_features)
        
        # Predict n steps ahead
        for _ in range(n_steps):
            # Predict the next step
            next_step = self.model.predict(current_sequence)[0]
            forecasts.append(next_step)
            
            # Update the sequence for the next prediction
            # Create a new row with the predicted value and zeros for other features
            new_row = np.zeros((1, self.n_features))
            new_row[0, 0] = next_step  # Assuming the target is the first feature
            
            # Remove the oldest time step and add the new prediction
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                        [new_row], 
                                        axis=1)
        
        return np.array(forecasts)
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model
            self.model.save(filepath)
            
            # Save scaler configuration
            scaler_path = f"{os.path.splitext(filepath)[0]}_scaler.pkl"
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            print(f"Model saved to {filepath}")
            print(f"Scaler saved to {scaler_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """Load model and scaler from disk"""
        self.model = tf.keras.models.load_model(filepath)
        scaler_path = f"{os.path.splitext(filepath)[0]}_scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        print(f"Model loaded from {filepath}")

    def evaluate(self, test_data, target_column):
        """Enhanced evaluation with correct return values"""
        try:
            # Preprocess test data
            X_test, y_test = self.preprocess_data(test_data, target_column)
            
            # Get model metrics
            metrics = self.model.evaluate(X_test, y_test, verbose=1)
            loss, mae, mse = metrics  # Unpack the three metrics
            
            # Generate predictions
            predictions = self.model.predict(X_test)
            
            # Calculate additional metrics
            rmse = np.sqrt(mse)
            
            # Import r2_score if not already imported
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, predictions)
            
            # Plot results
            self._plot_evaluation_metrics(y_test, predictions.flatten(), loss, mae, rmse, r2)
            
            # Return exactly three values as expected
            return loss, mae, predictions.flatten()
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            # Return empty arrays instead of None
            return 0.0, 0.0, np.array([])

    def _plot_evaluation_metrics(self, y_true, y_pred, loss, mae, rmse, r2):
        """Plot evaluation results with enhanced visualization"""
        plt.figure(figsize=(15, 10))
        
        # Time series plot
        plt.subplot(2, 1, 1)
        plt.plot(y_true, 'b-', label='Actual', linewidth=2)
        plt.plot(y_pred, 'r--', label='Predicted', linewidth=2)
        plt.title(f'Prediction Results\nLoss: {loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        plt.legend()
        plt.grid(True)
        
        # Correlation plot
        plt.subplot(2, 1, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.title(f'Prediction Correlation (R² = {r2:.4f})')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png')
        plt.close()





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import zipfile
import io
from sklearn.preprocessing import MinMaxScaler

# Create directories if they don't exist
os.makedirs('saved_models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('saved_models/checkpoints', exist_ok=True)

def download_datasets():
    """Download agricultural datasets from Kaggle or direct sources"""
    print("Downloading agricultural datasets...")
    
    try:
        # Try using Kaggle API
        import kaggle
        
        # Define datasets to download
        kaggle_datasets = [
            ('anshtanwar/current-daily-price-of-various-commodities-india', 'data/commodities'),
            ('saeedahmadi/agricultural-commodities-dataset', 'data/commodities_backup'),
            ('vipullrathod/daily-min-temperatures', 'data/weather/temperatures'),
            ('sumanthvrao/daily-climate-time-series-data', 'data/weather/climate'),
            ('patelris/yield-prediction', 'data/farm_yield'),
            ('manish1809/mtpproject', 'data/soil/improved'),
            ('cdminix/us-drought-meteorological-data', 'data/soil/drought'),
            ('atharvaingle/crop-recommendation-dataset', 'data/practices')
        ]
        
        # Download each dataset
        for dataset, path in kaggle_datasets:
            print(f"Downloading {dataset}...")
            kaggle.api.dataset_download_files(dataset, path=path, unzip=True)
            
        print("All datasets downloaded successfully using Kaggle API.")
        
    except Exception as e:
        print(f"Error using Kaggle API: {e}")
        print("Downloading datasets directly using requests instead...")
        
        # Direct download URLs for the datasets
        datasets = {
            'commodities': 'https://raw.githubusercontent.com/datasets/agriculture/master/data/crop-production.csv',
            'weather': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',
            'farm_yield': 'https://raw.githubusercontent.com/rishabh-patel/farm-yield-prediction/main/yield_df.csv',
            'practices': 'https://raw.githubusercontent.com/datasets/fertilizer/master/data/Fertilizer.csv'
        }
        
        # Download each dataset
        for dataset_name, url in datasets.items():
            try:
                print(f"Downloading {dataset_name} dataset...")
                os.makedirs(f'data/{dataset_name}', exist_ok=True)
                
                response = requests.get(url)
                response.raise_for_status()
                
                with open(f'data/{dataset_name}/{dataset_name}.csv', 'wb') as f:
                    f.write(response.content)
                
                print(f"{dataset_name} dataset downloaded successfully.")
                
            except Exception as e:
                print(f"Error downloading {dataset_name} dataset: {e}")
        
        print("\nIf you want to use the Kaggle API in the future:")
        print("1. Install the Kaggle package: pip install kaggle")
        print("2. Go to https://www.kaggle.com/account")
        print("3. Click 'Create New API Token' to download kaggle.json")
        print("4. Place this file in ~/.kaggle/ or C:\\Users\\<username>\\.kaggle\\")

def load_and_prepare_data():
    """Load and prepare agricultural datasets for time series analysis"""
    print("\nLoading and preparing agricultural time series data...")
    
    # Create required directories
    for dir_path in ['data/commodities', 'data/weather', 'data/farm_yield', 'data/soil']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Try to load the farm yield dataset first (integrated data)
    farm_yield_paths = [
        'data/farm_yield/yield_df.csv',
        'data/farm_yield/yield_prediction.csv',
        'data/yield_df.csv',
    ]
    
    # Try loading farm yield dataset
    for path in farm_yield_paths:
        if os.path.exists(path):
            try:
                farm_yield_df = pd.read_csv(path)
                print(f"Loaded integrated farm yield data from {path}")
                
                # Check if dataset has expected structure
                if 'Year' in farm_yield_df.columns and 'hg/ha_yield' in farm_yield_df.columns:
                    print("Using integrated farm yield dataset")
                    
                    # Sort by year
                    farm_yield_df = farm_yield_df.sort_values('Year')
                    
                    # Create dummy variables for categorical columns
                    cat_columns = [col for col in farm_yield_df.columns 
                                  if farm_yield_df[col].dtype == 'object' and col not in ['Year']]
                    
                    if cat_columns:
                        farm_yield_df = pd.get_dummies(farm_yield_df, columns=cat_columns)
                    
                    # Create the final dataset
                    feature_cols = [col for col in farm_yield_df.columns 
                                   if col not in ['Year', 'hg/ha_yield'] 
                                   and not pd.api.types.is_string_dtype(farm_yield_df[col])]
                    
                    time_series_data = farm_yield_df[feature_cols + ['hg/ha_yield']].copy()
                    years = farm_yield_df['Year'].values
                    
                    print(f"Prepared time series data: {time_series_data.shape} with {len(feature_cols)} features")
                    return time_series_data, years
            except Exception as e:
                print(f"Error processing {path}: {e}")
    
    print("Integrated dataset not found or invalid, merging separate datasets...")
    
    # Load crop data
    crop_df = load_dataset([
        'data/commodities/Daily_Price_of_Various_Commodities.csv',
        'data/commodities/daily_price.csv',
        'data/commodities_backup/yield_df.csv',
        'data/commodities/crop-production.csv'
    ], "crop")
    
    # Load weather data
    weather_df = load_dataset([
        'data/weather/temperatures/daily-min-temperatures.csv',
        'data/weather/climate/DailyDelhiClimateTrain.csv',
        'data/weather/daily-min-temperatures.csv'
    ], "weather")
    
    # Load soil data
    soil_df = load_dataset([
        'data/soil/improved/soil_data.csv',
        'data/soil/drought/drought.csv',
        'data/soil/soil-carbon.csv'
    ], "soil")
    
    # Process and merge datasets
    crop_data = process_crop_data(crop_df)
    yearly_weather = process_weather_data(weather_df)
    yearly_soil = process_soil_data(soil_df)
    
    # Merge datasets
    merged_data = merge_datasets(crop_data, yearly_weather, yearly_soil)
    
    # Create final dataset
    feature_cols = [col for col in merged_data.columns 
                   if col not in ['Year', 'hg/ha_yield', 'Area', 'Item'] 
                   and not pd.api.types.is_string_dtype(merged_data[col])]
    
    time_series_data = merged_data[feature_cols + ['hg/ha_yield']].copy()
    years = merged_data['Year'].values if 'Year' in merged_data.columns else np.arange(len(merged_data))
    
    print(f"Prepared time series data: {time_series_data.shape} with {len(feature_cols)} features")
    return time_series_data, years

def load_dataset(paths, dataset_type):
    """Load a dataset from multiple possible paths"""
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"Loaded {dataset_type} data from {path}")
                return df
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    print(f"No {dataset_type} data found. Please run download_datasets() first.")
    return pd.DataFrame()  # Return empty DataFrame if no data found

def process_crop_data(crop_df):
    """Process crop yield data"""
    if crop_df.empty:
        return pd.DataFrame({'Year': [], 'hg/ha_yield': []})
    
    # Process based on dataset structure
    if 'Date' in crop_df.columns and 'Price' in crop_df.columns:
        # Indian commodities dataset
        crop_df['Date'] = pd.to_datetime(crop_df['Date'])
        crop_df['Year'] = crop_df['Date'].dt.year
        
        if 'Commodity' in crop_df.columns:
            main_commodity = crop_df['Commodity'].value_counts().index[0]
            crop_data = crop_df[crop_df['Commodity'] == main_commodity]
            crop_data = crop_data.groupby('Year').agg({'Price': 'mean'}).reset_index()
            crop_data.rename(columns={'Price': 'hg/ha_yield'}, inplace=True)
        else:
            crop_data = crop_df.groupby('Year').agg({'Price': 'mean'}).reset_index()
            crop_data.rename(columns={'Price': 'hg/ha_yield'}, inplace=True)
    else:
        # Try to find year and yield columns
        year_cols = [col for col in crop_df.columns if 'year' in col.lower()]
        yield_cols = [col for col in crop_df.columns if 'yield' in col.lower() or 'production' in col.lower()]
        
        if year_cols and yield_cols:
            crop_data = crop_df[[year_cols[0], yield_cols[0]]].copy()
            crop_data.columns = ['Year', 'hg/ha_yield']
        else:
            # Create synthetic data if needed
            crop_data = pd.DataFrame({
                'Year': np.arange(2000, 2020),
                'hg/ha_yield': np.random.normal(5000, 500, 20)
            })
    
    return crop_data

def process_weather_data(weather_df):
    """Process weather data"""
    if weather_df.empty:
        return pd.DataFrame({'year': []})
    
    # Convert date column to datetime
    date_cols = [col for col in weather_df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if date_cols:
        date_col = date_cols[0]
        weather_df[date_col] = pd.to_datetime(weather_df[date_col], errors='coerce')
        weather_df['year'] = weather_df[date_col].dt.year
        
        # Aggregate numeric columns by year
        num_cols = weather_df.select_dtypes(include=['number']).columns
        agg_cols = [col for col in num_cols if col != 'year']
        
        yearly_weather = weather_df.groupby('year').agg({
            col: 'mean' for col in agg_cols
        }).reset_index()
    else:
        # Create synthetic data if needed
        yearly_weather = pd.DataFrame({
            'year': np.arange(2000, 2020),
            'temperature': np.random.normal(25, 5, 20),
            'humidity': np.random.normal(60, 10, 20)
        })
    
    return yearly_weather

def process_soil_data(soil_df):
    """Process soil data"""
    if soil_df.empty:
        return pd.DataFrame({'year': []})
    
    # Look for date columns
    date_cols = [col for col in soil_df.columns if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower()]
    
    if date_cols:
        date_col = date_cols[0]
        if 'year' not in date_col.lower():
            soil_df[date_col] = pd.to_datetime(soil_df[date_col], errors='coerce')
            soil_df['year'] = soil_df[date_col].dt.year
        else:
            soil_df['year'] = soil_df[date_col]
        
        # Aggregate numeric columns by year
        num_cols = soil_df.select_dtypes(include=['number']).columns
        soil_features = [col for col in num_cols if col != 'year' and 'id' not in col.lower()]
        
        yearly_soil = soil_df.groupby('year').agg({
            col: 'mean' for col in soil_features
        }).reset_index()
    else:
        # Create synthetic data if needed
        yearly_soil = pd.DataFrame({
            'year': np.arange(2000, 2020),
            'moisture': np.random.normal(30, 5, 20),
            'ph': np.random.normal(6.5, 0.5, 20)
        })
    
    return yearly_soil

def merge_datasets(crop_data, weather_data, soil_data):
    """Merge crop, weather and soil datasets"""
    merged_data = crop_data.copy()
    
    # Merge with weather data if available
    if not weather_data.empty:
        common_years = set(weather_data['year']).intersection(set(merged_data['Year']))
        if common_years:
            merged_data = pd.merge(
                merged_data, 
                weather_data,
                left_on='Year',
                right_on='year',
                how='left'
            ).drop(columns=['year'])
        else:
            print("Warning: No overlapping years between crop and weather data")
    
    # Merge with soil data if available
    if not soil_data.empty:
        common_years = set(soil_data['year']).intersection(set(merged_data['Year']))
        if common_years:
            merged_data = pd.merge(
                merged_data, 
                soil_data,
                left_on='Year',
                right_on='year',
                how='left'
            ).drop(columns=['year'])
        else:
            print("Warning: No overlapping years between crop and soil data")
    
    # Fill missing values
    merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
    
    return merged_data

def main():
    # Check if datasets exist, if not download them
    if not all(os.path.exists(f'data/{dir}') for dir in ['commodities', 'weather', 'farm_yield']):
        download_datasets()
    
    # Load and prepare the data
    data, years = load_and_prepare_data()
    
    # Plot the data
    plt.figure(figsize=(15, 10))
    plot_cols = [col for col in data.columns if col != 'hg/ha_yield' 
                and not col.startswith('Area_') and not col.startswith('Item_')]
    
    for i, column in enumerate(plot_cols[:5]):  # Plot up to 5 features
        plt.subplot(3, 2, i+1)
        plt.plot(years, data[column])
        plt.title(column)
    
    # Plot the target variable
    plt.subplot(3, 2, 6)
    plt.plot(years, data['hg/ha_yield'], 'r-')
    plt.title('Crop Yield (hg/ha)')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png')
    
    # Split into train and test with 90:10 ratio
    train_size = int(len(data) * 0.9)
    train_data = data[:train_size]
    test_data = data[train_size-5:]  # Include overlap for sequence
    
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # Initialize and train the model
    print("\nInitializing RNN model...")
    model = RealtimeSeriesRNN(
        sequence_length=5,  # Shorter sequence due to limited data points
        n_features=len(data.columns) - 1,  # Excluding the target column
        output_size=1,
        scaling_method='standard',  # Use standard scaling
        use_lr_schedule=False  # Use fixed learning rate with ReduceLROnPlateau
    )
    
    print("\nTraining the model...")
    try:
        history = model.train(
            data=train_data,
            target_column='hg/ha_yield',
            epochs=100,
            batch_size=4,  # Smaller batch size for smaller dataset
            validation_split=0.2
        )
        
        # Evaluate on test data
        print("\nEvaluating the model on test data...")
        loss, mae, predictions = model.evaluate(test_data, 'hg/ha_yield')

        # Save the model
        model.save_model('saved_models/crop_yield_rnn.keras')

        # Demonstrate forecasting
        print("\nDemonstrating multi-step forecasting...")
        # Use the last available sequence for forecasting
        last_sequence = test_data.iloc[-10:].drop(columns=['hg/ha_yield'])
        forecast = model.predict_next_n_steps(last_sequence.values, n_steps=5)

        # Plot forecast
        plt.figure(figsize=(12, 6))
        # Plot actual values
        plt.plot(years[-len(test_data):], test_data['hg/ha_yield'].values, label='Actual')

        # Plot predicted values (align with actual data points)
        if len(predictions) > 0:  # Check if predictions is not empty
            pred_years = years[-len(predictions):]
            plt.plot(pred_years, predictions, label='Predicted')

            # Plot forecast (future years)
            forecast_years = np.arange(years[-1] + 1, years[-1] + 6)
            plt.plot(forecast_years, forecast, label='Forecast', linestyle='--')

            plt.axvline(x=years[-1], color='r', linestyle='-')
            plt.title('Crop Yield Prediction and Forecast')
            plt.xlabel('Year')
            plt.ylabel('Yield (hg/ha)')
            plt.legend()
            plt.savefig('forecast_results.png')
            plt.show()
        else:
            print("Warning: No predictions available for plotting")

        print("\nTraining and evaluation complete. Model saved to 'saved_models/crop_yield_rnn.keras'")
        
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()








