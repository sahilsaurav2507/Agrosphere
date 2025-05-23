import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import threading
import time
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flag to determine if Kafka is available
KAFKA_AVAILABLE = False

# Try to import Kafka libraries, but don't fail if they're not available
try:
    from confluent_kafka import Consumer, Producer
    KAFKA_AVAILABLE = True
except ImportError:
    logger.warning("confluent_kafka package not available. Running in offline mode.")

# Constants for file extensions based on TF version
# TF 2.13+ uses .keras, older versions use .h5
MODEL_FILE_EXT = '.keras' if tf.__version__ >= '2.13' else '.h5'

class RealtimeDataAnalyzer:
    def __init__(self, kafka_bootstrap_servers='localhost:9092', 
                 kafka_topic='agro_iot_sensors', 
                 data_path='E:\\Agrospere\\models\\datas\\RNN_realtime data',
                 model_save_path='E:\\Agrospere\\models\\saved_models',
                 offline_mode=not KAFKA_AVAILABLE):
        """
        Initialize the RNN model trainer for real-time agricultural data analysis.
        
        Args:
            kafka_bootstrap_servers: Kafka server address
            kafka_topic: Topic to consume IoT sensor data from
            data_path: Path to historical data for initial training
            model_save_path: Path to save trained models
            offline_mode: If True, don't try to connect to Kafka
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.offline_mode = offline_mode
        
        # Ensure directories exist
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Initialize data containers
        self.historical_data = None
        self.streaming_data_buffer = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_model_trained = False
        self.sequence_length = 30  # Number of time steps to look back
        
        # Initialize lock for thread-safe operations
        self.data_lock = threading.Lock()
        
        # Load historical data if available
        self._load_historical_data()
        
        # If no historical data and in offline mode, create synthetic data
        if self.historical_data is None and self.offline_mode:
            logger.info("No historical data found. Creating synthetic data for testing.")
            self._create_synthetic_data()
    
    def _load_historical_data(self):
        """Load historical data from the specified path for initial model training."""
        try:
            # Assuming CSV format with columns: timestamp, temperature, humidity, rainfall, soil_moisture, etc.
            data_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            
            if not data_files:
                logger.warning(f"No CSV files found in {self.data_path}. Will wait for streaming data or create synthetic data.")
                return
            
            # Load and concatenate all data files
            dfs = []
            for file in data_files:
                file_path = os.path.join(self.data_path, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            self.historical_data = pd.concat(dfs, ignore_index=True)
            
            # Ensure timestamp is in datetime format and sort by it
            if 'timestamp' in self.historical_data.columns:
                self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
                self.historical_data.sort_values('timestamp', inplace=True)
            
            logger.info(f"Loaded historical data with {len(self.historical_data)} records")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.historical_data = None
    
    def _create_synthetic_data(self):
        """Create synthetic data for testing when no historical data is available."""
        # Create a date range for the past year
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Initialize DataFrame
        data = {
            'timestamp': dates,
            'temperature': [],
            'humidity': [],
            'rainfall': [],
            'soil_moisture': [],
            'solar_radiation': [],
            'wind_speed': [],
            'yield': []
        }
        
        # Generate synthetic data with seasonal patterns
        for date in dates:
            day_of_year = date.dayofyear
            hour_of_day = date.hour
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            daily_factor = np.sin(2 * np.pi * hour_of_day / 24)
            
            # Generate sensor readings with patterns
            temperature = 25 + 10 * season_factor + 5 * daily_factor + np.random.normal(0, 2)
            humidity = 60 + 20 * season_factor - 10 * daily_factor + np.random.normal(0, 5)
            rainfall = max(0, 5 * season_factor - 2 * daily_factor + np.random.exponential(1))
            soil_moisture = 30 + 15 * season_factor + np.random.normal(0, 3)
            solar_radiation = max(0, 800 + 400 * season_factor + 300 * daily_factor + np.random.normal(0, 50))
            wind_speed = 5 + 2 * season_factor + 3 * daily_factor + np.random.exponential(2)
            
            # Calculate synthetic yield
            base_yield = 100
            temp_effect = -0.5 * (temperature - 25)**2 / 100
            moisture_effect = -0.5 * (soil_moisture - 35)**2 / 100
            rain_effect = 0.1 * rainfall if rainfall < 10 else (1 - 0.02 * (rainfall - 10))
            
            yield_estimate = base_yield * (1 + temp_effect + moisture_effect + rain_effect) / 3
            
            # Add to data dictionary
            data['temperature'].append(temperature)
            data['humidity'].append(humidity)
            data['rainfall'].append(rainfall)
            data['soil_moisture'].append(soil_moisture)
            data['solar_radiation'].append(solar_radiation)
            data['wind_speed'].append(wind_speed)
            data['yield'].append(yield_estimate)
        
        # Create DataFrame
        self.historical_data = pd.DataFrame(data)
        
        # Save synthetic data
        synthetic_data_path = os.path.join(self.data_path, 'synthetic_data.csv')
        self.historical_data.to_csv(synthetic_data_path, index=False)
        logger.info(f"Created synthetic dataset with {len(self.historical_data)} records and saved to {synthetic_data_path}")
    
    def preprocess_data(self, data, target_column='yield'):
        """
        Preprocess data for RNN model training.
        
        Args:
            data: DataFrame containing the data
            target_column: Column name for the target variable to predict
            
        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        # Drop non-numeric columns except timestamp
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Check if target column exists, if not, try to create it or use another column
        if target_column not in numeric_data.columns:
            logger.warning(f"Target column '{target_column}' not found in data")
            
            # If we have temperature and soil_moisture, we can create a synthetic yield
            if 'temperature' in numeric_data.columns and 'soil_moisture' in numeric_data.columns:
                logger.info("Creating synthetic 'yield' column from existing data")
                
                # Create a synthetic yield based on temperature and soil_moisture
                temp = numeric_data['temperature']
                moisture = numeric_data['soil_moisture'] if 'soil_moisture' in numeric_data.columns else numeric_data['humidity']
                
                # Simple formula: yield is optimal at 25°C and 35% soil moisture
                temp_effect = -0.5 * (temp - 25)**2 / 100
                moisture_effect = -0.5 * (moisture - 35)**2 / 100
                
                base_yield = 100
                numeric_data[target_column] = base_yield * (1 + temp_effect + moisture_effect) / 2
            else:
                # If we can't create yield, use the first column as target
                logger.warning(f"Using '{numeric_data.columns[0]}' as target instead")
                target_column = numeric_data.columns[0]
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(numeric_data)
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        
        # Create sequences for time series prediction
        X, y = [], []
        for i in range(len(scaled_df) - self.sequence_length):
            X.append(scaled_df.drop(columns=[target_column]).iloc[i:i+self.sequence_length].values)
            y.append(scaled_df[target_column].iloc[i+self.sequence_length])
        
        X, y = np.array(X), np.array(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """
        Build and compile the RNN model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1))  # Output layer
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def get_model_filepath(self, filename_base):
        """
        Generate a model filepath with the correct extension based on TF version.
        
        Args:
            filename_base: Base filename without extension
            
        Returns:
            Full filepath with appropriate extension
        """
        return os.path.join(self.model_save_path, f"{filename_base}{MODEL_FILE_EXT}")
    
    def train_model(self, data=None, target_column='yield', epochs=100, batch_size=32):
        """
        Train the RNN model on the provided data.
        
        Args:
            data: DataFrame to use for training (uses historical_data if None)
            target_column: Column to predict
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if data is None:
            if self.historical_data is None or len(self.historical_data) < self.sequence_length + 1:
                logger.warning("Insufficient data for training. Need more data points.")
                return None
            data = self.historical_data
        
        # Check data columns
        logger.info(f"Available columns in data: {data.columns.tolist()}")
        
        # Preprocess the data
        try:
            X_train, X_test, y_train, y_test = self.preprocess_data(data, target_column)
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None
        
        # Build the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            self.get_model_filepath('best_model'),  # Using helper method for correct extension
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model training completed. MSE: {mse}, R²: {r2}")
        
        # Save the final model
        self.model.save(self.get_model_filepath('final_model'))
        
        # Plot training history
        self._plot_training_history(history)
        
        # Plot predictions vs actual
        self._plot_predictions(y_test, y_pred)
        
        self.is_model_trained = True
        return history
    
    def _plot_training_history(self, history):
        """Plot and save the training history."""
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(self.model_save_path, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training history plot saved to {plot_path}")
    
    def _plot_predictions(self, y_true, y_pred):
        """Plot and save actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(self.model_save_path, 'prediction_scatter.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Prediction scatter plot saved to {plot_path}")
    
    def start_kafka_consumer(self):
        """Start a separate thread to consume data from Kafka."""
        if self.offline_mode:
            logger.info("Running in offline mode. Kafka consumer not started.")
            return
        
        if not KAFKA_AVAILABLE:
            logger.error("Kafka libraries not available. Cannot start consumer.")
            self.offline_mode = True
            return
        
        # Start Kafka consumer in a separate thread
        consumer_thread = threading.Thread(target=self._consume_kafka_data)
        consumer_thread.daemon = True  # Thread will exit when main program exits
        consumer_thread.start()
        logger.info("Kafka consumer thread started")
    
    def _consume_kafka_data(self):
        """Consume data from Kafka and add it to the streaming buffer."""
        if not KAFKA_AVAILABLE:
            logger.error("Kafka libraries not available. Cannot consume data.")
            return
            
        try:
            # Configure the Kafka consumer
            consumer_conf = {
                'bootstrap.servers': self.kafka_bootstrap_servers,
                'group.id': 'agro_analyzer_group',
                'auto.offset.reset': 'latest',
                'session.timeout.ms': 6000
            }
            
            consumer = Consumer(consumer_conf)
            consumer.subscribe([self.kafka_topic])
            
            logger.info(f"Kafka consumer connected and subscribed to {self.kafka_topic}")
            
            while True:
                # Poll for messages
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                try:
                    # Process the message
                    data = json.loads(msg.value().decode('utf-8'))
                    
                    # Convert to DataFrame row
                    df_row = pd.DataFrame([data])
                    
                    # Add to buffer with thread safety
                    with self.data_lock:
                        self.streaming_data_buffer.append(df_row)
                        
                        # If buffer gets large enough, update the model
                        if len(self.streaming_data_buffer) >= 100:  # Arbitrary threshold
                            self._process_streaming_buffer()
                    
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
        
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
        
        finally:
            # Close down consumer to commit final offsets
            try:
                consumer.close()
            except:
                pass
    
    def _process_streaming_buffer(self):
        """Process the accumulated streaming data and update the model if needed."""
        with self.data_lock:
            if not self.streaming_data_buffer:
                return
            
            # Concatenate all buffered data
            buffer_df = pd.concat(self.streaming_data_buffer, ignore_index=True)
            
            # Clear the buffer
            self.streaming_data_buffer = []
            
            # Update historical data
            if self.historical_data is not None:
                self.historical_data = pd.concat([self.historical_data, buffer_df], ignore_index=True)
            else:
                self.historical_data = buffer_df
            
            # Save the updated dataset
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.historical_data.to_csv(os.path.join(self.data_path, f'updated_dataset_{timestamp}.csv'), index=False)
            
            # Retrain model if we have enough data and either:
            # 1. The model hasn't been trained yet, or
            # 2. We have accumulated a significant amount of new data
            if (len(self.historical_data) > self.sequence_length + 100 and 
                (not self.is_model_trained or len(buffer_df) > 500)):
                logger.info("Retraining model with updated data...")
                self.train_model()
    
    def predict(self, input_data):
        """
        Make predictions using the trained model.
        
        Args:
            input_data: DataFrame with the same features as training data
            
        Returns:
            Predicted values
        """
        if not self.is_model_trained or self.model is None:
            logger.error("Model not trained yet. Cannot make predictions.")
            return None
        
        try:
            # Preprocess input data
            numeric_data = input_data.select_dtypes(include=[np.number])
            scaled_data = self.scaler.transform(numeric_data)
            
            # Create sequences
            X = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                X.append(scaled_data[i:i+self.sequence_length])
            
            X = np.array(X)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Inverse transform to get original scale
            # We need to create a dummy array with the same shape as our original data
            dummy = np.zeros((len(predictions), numeric_data.shape[1]))
            # Put predictions in the target column position
            target_idx = list(numeric_data.columns).index('yield')
            dummy[:, target_idx] = predictions.flatten()
            
            # Inverse transform
            unscaled_predictions = self.scaler.inverse_transform(dummy)[:, target_idx]
            
            return unscaled_predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

    def run(self):
        """Main method to run the analyzer."""
        # Train initial model if historical data is available
        if self.historical_data is not None and len(self.historical_data) > self.sequence_length + 1:
            logger.info("Training initial model with historical data...")
            self.train_model()
        
        # Start consuming from Kafka if not in offline mode
        if not self.offline_mode:
            self.start_kafka_consumer()
        
        # Keep the main thread running
        try:
            while True:
                # In offline mode, simulate new data periodically
                if self.offline_mode and self.is_model_trained:
                    logger.info("Running in offline mode. Simulating new data...")
                    self._simulate_new_data()
                
                time.sleep(60)  # Check every minute if we need to do anything
                
                # Periodically save the model and data
                if self.is_model_trained:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    self.model.save(self.get_model_filepath(f'model_checkpoint_{timestamp}'))
                    logger.info(f"Saved model checkpoint at {timestamp}")
        
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        
        finally:
            # Save final model and data before exiting
            if self.is_model_trained and self.model is not None:
                self.model.save(self.get_model_filepath('final_model'))
                logger.info("Final model saved")
            
            if self.historical_data is not None:
                self.historical_data.to_csv(os.path.join(self.data_path, 'final_dataset.csv'), index=False)
                logger.info("Final dataset saved")
    
    def _simulate_new_data(self):
        """In offline mode, simulate new data points to test model updating."""
        # Create a few new synthetic data points
        new_data = []
        current_time = pd.Timestamp.now()
        
        for i in range(10):  # Generate 10 new points
            timestamp = current_time + pd.Timedelta(hours=i)
            day_of_year = timestamp.dayofyear
            hour_of_day = timestamp.hour
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            daily_factor = np.sin(2 * np.pi * hour_of_day / 24)
            
            # Generate sensor readings with patterns
            temperature = 25 + 10 * season_factor + 5 * daily_factor + np.random.normal(0, 2)
            humidity = 60 + 20 * season_factor - 10 * daily_factor + np.random.normal(0, 5)
            rainfall = max(0, 5 * season_factor - 2 * daily_factor + np.random.exponential(1))
            soil_moisture = 30 + 15 * season_factor + np.random.normal(0, 3)
            solar_radiation = max(0, 800 + 400 * season_factor + 300 * daily_factor + np.random.normal(0, 50))
            wind_speed = 5 + 2 * season_factor + 3 * daily_factor + np.random.exponential(2)
            
            # Calculate synthetic yield
            base_yield = 100
            temp_effect = -0.5 * (temperature - 25)**2 / 100
            moisture_effect = -0.5 * (soil_moisture - 35)**2 / 100
            rain_effect = 0.1 * rainfall if rainfall < 10 else (1 - 0.02 * (rainfall - 10))
            
            yield_estimate = base_yield * (1 + temp_effect + moisture_effect + rain_effect) / 3
            
            # Create data point
            data = {
                'timestamp': timestamp,
                'temperature': temperature,
                'humidity': humidity,
                'rainfall': rainfall,
                'soil_moisture': soil_moisture,
                'solar_radiation': solar_radiation,
                'wind_speed': wind_speed,
                'yield': yield_estimate
            }
            
            new_data.append(data)
        
        # Convert to DataFrame
        new_df = pd.DataFrame(new_data)
        
        # Add to buffer with thread safety
        with self.data_lock:
            self.streaming_data_buffer.append(new_df)
            self._process_streaming_buffer()
        
        logger.info(f"Added {len(new_data)} simulated data points in offline mode")


class DataSimulator:
    """
    Simulates IoT sensor data and sends it to Kafka for testing purposes.
    This is useful for development when real IoT sensors are not available.
    """
    def __init__(self, kafka_bootstrap_servers='localhost:9092', kafka_topic='agro_iot_sensors', 
                 offline_mode=not KAFKA_AVAILABLE):
        """
        Initialize the data simulator.
        
        Args:
            kafka_bootstrap_servers: Kafka server address
            kafka_topic: Topic to produce IoT sensor data to
            offline_mode: If True, save data to CSV instead of sending to Kafka
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.offline_mode = offline_mode
        self.data_path = 'E:\\Agrospere\\models\\datas\\RNN_realtime data'
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        
        # Initialize Kafka producer if not in offline mode
        self.producer = None
        if not self.offline_mode and KAFKA_AVAILABLE:
            try:
                producer_conf = {
                    'bootstrap.servers': kafka_bootstrap_servers,
                    'message.timeout.ms': 10000
                }
                self.producer = Producer(producer_conf)
                logger.info(f"Kafka producer initialized for {kafka_bootstrap_servers}")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka producer: {e}")
                logger.info("Switching to offline mode")
                self.offline_mode = True
    
    def generate_sample_data(self):
        """Generate a single sample of simulated sensor data."""
        # Current timestamp
        timestamp = pd.Timestamp.now().isoformat()
        
        # Simulate seasonal patterns
        day_of_year = pd.Timestamp.now().dayofyear
        hour_of_day = pd.Timestamp.now().hour
        season_factor = np.sin(2 * np.pi * day_of_year / 365)
        daily_factor = np.sin(2 * np.pi * hour_of_day / 24)
        
        # Generate sensor readings with some randomness and seasonal effects
        temperature = 25 + 10 * season_factor + 5 * daily_factor + np.random.normal(0, 2)  # °C
        humidity = 60 + 20 * season_factor - 10 * daily_factor + np.random.normal(0, 5)     # %
        rainfall = max(0, 5 * season_factor - 2 * daily_factor + np.random.exponential(1)) # mm
        soil_moisture = 30 + 15 * season_factor + np.random.normal(0, 3) # %
        solar_radiation = max(0, 800 + 400 * season_factor + 300 * daily_factor + np.random.normal(0, 50)) # W/m²
        wind_speed = 5 + 2 * season_factor + 3 * daily_factor + np.random.exponential(2)  # m/s
        
        # Calculate a simulated yield estimate based on the above factors
        # This is a simplified model for demonstration
        base_yield = 100  # base yield in arbitrary units
        temp_effect = -0.5 * (temperature - 25)**2 / 100  # optimal temp around 25°C
        moisture_effect = -0.5 * (soil_moisture - 35)**2 / 100  # optimal moisture around 35%
        rain_effect = 0.1 * rainfall if rainfall < 10 else (1 - 0.02 * (rainfall - 10))
        
        # Combine effects (simplified)
        yield_estimate = base_yield * (1 + temp_effect + moisture_effect + rain_effect) / 3
        
        # Create data point
        data = {
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'soil_moisture': soil_moisture,
            'solar_radiation': solar_radiation,
            'wind_speed': wind_speed,
            'yield': yield_estimate
        }
        
        return data
    
    def delivery_report(self, err, msg):
        """Callback for message delivery reports."""
        if err is not None:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def run_simulation(self, interval_seconds=5.0, duration_minutes=None):
        """
        Run the data simulation, sending data to Kafka or saving to CSV.
        
        Args:
            interval_seconds: Time interval between data points
            duration_minutes: How long to run the simulation (None = indefinitely)
        """
        if self.offline_mode:
            logger.info(f"Starting data simulation in offline mode. Saving to {self.data_path}")
        else:
            logger.info(f"Starting data simulation. Sending to Kafka topic {self.kafka_topic}")
        
        start_time = time.time()
        count = 0
        simulated_data = []  # For offline mode
        
        try:
            while True:
                # Generate a data point
                data = self.generate_sample_data()
                count += 1
                
                # In offline mode, collect data for CSV
                if self.offline_mode:
                    simulated_data.append(data)
                # In Kafka mode, send to topic
                elif KAFKA_AVAILABLE and self.producer:
                    try:
                        # Convert data to JSON and send
                        json_data = json.dumps(data).encode('utf-8')
                        self.producer.produce(self.kafka_topic, json_data)
                        self.producer.poll(0)  # Trigger any callbacks
                    except Exception as e:
                        logger.error(f"Error sending to Kafka: {e}")
                
                # Log progress periodically
                if count % 20 == 0:
                    if self.offline_mode:
                        logger.info(f"Generated {count} simulated data points (offline mode)")
                    else:
                        logger.info(f"Sent {count} simulated data points to Kafka")
                    
                    # In offline mode, periodically save data to CSV
                    if self.offline_mode and len(simulated_data) >= 100:
                        df = pd.DataFrame(simulated_data)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        csv_path = os.path.join(self.data_path, f'simulated_data_{timestamp}.csv')
                        df.to_csv(csv_path, index=False)
                        logger.info(f"Saved {len(simulated_data)} data points to {csv_path}")
                        simulated_data = []  # Clear after saving
                
                # Check if we've reached the duration limit
                if duration_minutes is not None:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        logger.info(f"Simulation completed after {duration_minutes} minutes")
                        break
                
                # Wait for the next interval
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user")
        
        finally:
            # Save any remaining data in offline mode
            if self.offline_mode and simulated_data:
                df = pd.DataFrame(simulated_data)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(self.data_path, f'simulated_data_{timestamp}.csv')
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved final {len(simulated_data)} data points to {csv_path}")
            
            # Flush any remaining messages in Kafka mode
            if not self.offline_mode and KAFKA_AVAILABLE and self.producer:
                try:
                    self.producer.flush(timeout=5)
                except:
                    pass
                
            logger.info(f"Simulation ended. Generated {count} data points.")


def main():
    """Main function to run the RNN model training on real-time data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time data analyzer with RNN model for agricultural data')
    parser.add_argument('--kafka-server', type=str, default='localhost:9092', help='Kafka bootstrap server')
    parser.add_argument('--kafka-topic', type=str, default='agro_iot_sensors', help='Kafka topic for IoT sensor data')
    parser.add_argument('--data-path', type=str, default='E:\\Agrospere\\models\\datas\\RNN_realtime data', 
                        help='Path to historical data')
    parser.add_argument('--model-path', type=str, default='E:\\Agrospere\\models\\saved_models', 
                        help='Path to save trained models')
    parser.add_argument('--simulate', action='store_true', help='Run data simulator instead of analyzer')
    parser.add_argument('--sim-interval', type=float, default=5.0, help='Simulation interval in seconds')
    parser.add_argument('--sim-duration', type=float, default=None, help='Simulation duration in minutes')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode (no Kafka)')
    
    args = parser.parse_args()
    
    # Check if Kafka is available and warn if not
    if not KAFKA_AVAILABLE and not args.offline:
        logger.warning("confluent_kafka package not available. Forcing offline mode.")
        offline_mode = True
    else:
        offline_mode = args.offline
    
    if args.simulate:
        # Run the data simulator
        simulator = DataSimulator(
            kafka_bootstrap_servers=args.kafka_server,
            kafka_topic=args.kafka_topic,
            offline_mode=offline_mode
        )
        simulator.run_simulation(
            interval_seconds=args.sim_interval,
            duration_minutes=args.sim_duration
        )
    else:
        # Run the data analyzer
        analyzer = RealtimeDataAnalyzer(
            kafka_bootstrap_servers=args.kafka_server,
            kafka_topic=args.kafka_topic,
            data_path=args.data_path,
            model_save_path=args.model_path,
            offline_mode=offline_mode
        )
        analyzer.run()


if __name__ == "__main__":
    main()
