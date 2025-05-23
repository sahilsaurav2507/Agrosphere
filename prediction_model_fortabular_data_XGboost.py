import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Define the data directory
DATA_DIR = r'E:\Agrospere\models\datas\predictive_model_data'

def load_data(data_dir):
    """Load all datasets from the specified directory"""
    all_files = os.listdir(data_dir)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    # Load all CSV files into a dictionary of dataframes
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df_name = os.path.splitext(file)[0]
        try:
            dataframes[df_name] = pd.read_csv(file_path)
            print(f"Loaded {df_name} with shape {dataframes[df_name].shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return dataframes

def explore_data(dataframes):
    """Perform basic exploratory data analysis on the dataframes"""
    for name, df in dataframes.items():
        print(f"\n--- Exploring {name} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"Sample data:\n{df.head()}")
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        print(f"Numeric columns: {numeric_cols.tolist()}")
        print(f"Categorical columns: {categorical_cols.tolist()}")
        
        # Basic statistics for numeric columns
        if len(numeric_cols) > 0:
            print(f"Numeric statistics:\n{df[numeric_cols].describe()}")
        
        # Value counts for categorical columns (limited to first 5 for brevity)
        for col in categorical_cols[:5]:
            print(f"\nValue counts for {col}:\n{df[col].value_counts().head()}")

def preprocess_data(dataframes):
    """
    Preprocess the data for modeling.
    This function will:
    1. Combine dataframes if needed
    2. Handle missing values
    3. Encode categorical variables
    4. Scale numerical features
    """
    # For this example, we'll assume we're working with a single dataframe
    # If you have multiple dataframes, you'll need to decide how to combine them
    
    # For demonstration, let's use the first dataframe
    if not dataframes:
        raise ValueError("No dataframes to process")
    
    # Choose the first dataframe for processing
    df_name = list(dataframes.keys())[0]
    df = dataframes[df_name]
    
    print(f"Preprocessing dataframe: {df_name} with shape {df.shape}")
    
    # Identify target variable (this will depend on your specific dataset)
    # For this example, let's assume the last column is the target
    target_col = df.columns[-1]
    print(f"Using {target_col} as the target variable")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Check if target is categorical or numerical
    if y.dtype == 'object' or y.dtype.name == 'category':
        print("Target is categorical - performing classification")
        le = LabelEncoder()
        y = le.fit_transform(y)
        task_type = 'classification'
    else:
        print("Target is numerical - performing regression")
        task_type = 'regression'
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, task_type, target_col

def feature_engineering(X_train, X_test, preprocessor):
    """
    Perform feature engineering on the data
    """
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed training features shape: {X_train_processed.shape}")
    print(f"Processed test features shape: {X_test_processed.shape}")
    
    # Create a simplified list of feature names
    # This is a workaround since we can't easily get the exact feature names after transformation
    feature_names = []
    
    # Get numeric feature names
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
    
    # Add placeholder names for categorical features
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'cat':
            for col in cols:
                # Add placeholder names for each categorical column
                # We don't know exactly how many categories were created
                feature_names.append(f"{col}_encoded")
    
    print(f"Created {len(feature_names)} feature name placeholders")
    
    # If feature_names length doesn't match the actual features, use generic names
    if len(feature_names) != X_train_processed.shape[1]:
        print(f"Warning: Feature names count ({len(feature_names)}) doesn't match actual features ({X_train_processed.shape[1]})")
        print("Using generic feature names instead")
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
    
    return X_train_processed, X_test_processed, feature_names

def train_model(X_train, y_train, task_type):
    """
    Train an XGBoost model on the processed data
    """
    eval_metric = 'logloss' if task_type == 'classification' else 'rmse'
    
    if task_type == 'classification':
        # For classification
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic' if len(np.unique(y_train)) == 2 else 'multi:softprob',
            eval_metric=eval_metric,  # Add eval_metric here
            random_state=42
        )
    else:
        # For regression
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            eval_metric=eval_metric,  # Add eval_metric here
            random_state=42
        )
    
    # Train the model - removed eval_metric from here
    model.fit(
        X_train, 
        y_train,
        verbose=True
    )
    
    # Cross-validation
    cv_score = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=5,
        scoring='accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
    )
    
    if task_type == 'classification':
        print(f"Cross-validation accuracy: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
    else:
        print(f"Cross-validation RMSE: {np.sqrt(-cv_score).mean():.4f} ± {np.sqrt(-cv_score).std():.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, task_type, feature_names=None):
    """
    Evaluate the trained model on the test set
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate based on task type
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
    else:  # regression
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.savefig('actual_vs_predicted.png')
        plt.close()
    
    # Feature importance
    if feature_names is not None and len(feature_names) == len(model.feature_importances_):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance.head(20))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    return y_pred

def save_model(model, preprocessor, target_col, task_type):
    """
    Save the trained model and preprocessor
    """
    import joblib
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_filename = f"models/xgboost_{task_type}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    
    # Save the preprocessor
    preprocessor_filename = f"models/preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"Preprocessor saved to {preprocessor_filename}")
    
    # Save model metadata
    metadata = {
        'target_column': target_col,
        'task_type': task_type,
        'model_params': model.get_params(),
        'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_filename = f"models/model_metadata.json"
    import json
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {metadata_filename}")

def main():
    """Main function to run the entire pipeline"""
    print("Starting the XGBoost prediction model pipeline...")
    
    # Load data
    print("\n--- Loading Data ---")
    dataframes = load_data(DATA_DIR)
    
    # Explore data
    print("\n--- Exploring Data ---")
    explore_data(dataframes)
    
    # Preprocess data
    print("\n--- Preprocessing Data ---")
    X_train, X_test, y_train, y_test, preprocessor, task_type, target_col = preprocess_data(dataframes)
    
    # Feature engineering
    print("\n--- Feature Engineering ---")
    X_train_processed, X_test_processed, feature_names = feature_engineering(X_train, X_test, preprocessor)
    
    # Train model
    print("\n--- Training Model ---")
    model = train_model(X_train_processed, y_train, task_type)
    
    # Evaluate model
    print("\n--- Evaluating Model ---")
    y_pred = evaluate_model(model, X_test_processed, y_test, task_type, feature_names)
    
    # Save model
    print("\n--- Saving Model ---")
    save_model(model, preprocessor, target_col, task_type)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
