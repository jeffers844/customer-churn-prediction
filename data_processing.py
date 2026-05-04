"""
Data Processing Module for Customer Churn Prediction
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnDataProcessor:
    """
    Process customer data for churn prediction
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, download=False):
        """
        Load the Telco Customer Churn dataset
        
        Args:
            download: If True, downloads data from Kaggle
        
        Returns:
            DataFrame with customer data
        """
        if download:
            logger.info("Downloading dataset from Kaggle...")
            # In production, use: kaggle datasets download -d blastchar/telco-customer-churn
            # For this example, we'll create synthetic data
            return self._generate_synthetic_data()
        
        if self.data_path and Path(self.data_path).exists():
            logger.info(f"Loading data from {self.data_path}")
            return pd.read_csv(self.data_path)
        else:
            logger.info("Generating synthetic data...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic customer churn data for demonstration
        """
        np.random.seed(42)
        
        # Generate features
        data = {
            'customerID': [f'CUST{i:06d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.exponential(20, n_samples).astype(int),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples),
            'MonthlyCharges': np.random.uniform(18, 120, n_samples),
            'TotalCharges': np.zeros(n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate TotalCharges based on tenure and MonthlyCharges
        df['TotalCharges'] = df['tenure'] * df['MonthlyCharges'] + np.random.normal(0, 100, n_samples)
        df['TotalCharges'] = df['TotalCharges'].clip(lower=0)
        
        # Generate churn based on realistic patterns
        churn_prob = (
            0.1 +  # base rate
            0.4 * (df['Contract'] == 'Month-to-month').astype(int) +
            0.3 * (df['tenure'] < 6).astype(int) +
            0.2 * (df['PaymentMethod'] == 'Electronic check').astype(int) +
            0.15 * (df['TechSupport'] == 'No').astype(int) -
            0.2 * (df['Contract'] == 'Two year').astype(int)
        ).clip(0, 1)
        
        df['Churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
        df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
        
        logger.info(f"Generated {n_samples} synthetic customer records")
        logger.info(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
        
        return df
    
    def clean_data(self, df):
        """
        Clean the dataset
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Remove customerID (not a feature)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Handle missing values
        df = df.replace(' ', np.nan)
        
        # Convert TotalCharges to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing values
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def engineer_features(self, df):
        """
        Create advanced features for better prediction
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        # Tenure groups
        df['tenure_group'] = pd.cut(df['tenure'], 
                                    bins=[0, 12, 24, 48, 72],
                                    labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
        
        # Average monthly charges
        df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero
        
        # Price increase indicator
        df['ChargeIncrease'] = (df['MonthlyCharges'] > df['AvgMonthlyCharges']).astype(int)
        
        # Service count (how many services customer has)
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        
        df['ServiceCount'] = 0
        for col in service_cols:
            if col in df.columns:
                df['ServiceCount'] += (df[col] == 'Yes').astype(int)
        
        # Customer value score (RFM-like)
        df['CustomerValue'] = (
            df['tenure'] * 0.3 +  # Recency
            df['MonthlyCharges'] * 0.4 +  # Monetary
            df['ServiceCount'] * 10 * 0.3  # Frequency/usage
        )
        
        # Contract risk score
        contract_risk = {
            'Month-to-month': 3,
            'One year': 2,
            'Two year': 1
        }
        df['ContractRisk'] = df['Contract'].map(contract_risk)
        
        # Payment risk score
        payment_risk = {
            'Electronic check': 3,
            'Mailed check': 2,
            'Bank transfer (automatic)': 1,
            'Credit card (automatic)': 1
        }
        df['PaymentRisk'] = df['PaymentMethod'].map(payment_risk)
        
        logger.info(f"Features engineered. New shape: {df.shape}")
        return df
    
    def encode_features(self, df, fit=True):
        """
        Encode categorical variables
        
        Args:
            df: DataFrame with features
            fit: If True, fit encoders. If False, use existing encoders
            
        Returns:
            Encoded DataFrame
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Binary categorical columns
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                      'PaperlessBilling', 'Churn']
        
        for col in binary_cols:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        # One-hot encode multi-class categoricals
        categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract', 
                          'PaymentMethod', 'tenure_group']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
        
        logger.info(f"Features encoded. Final shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_features(self, X, fit=True):
        """
        Scale numerical features
        
        Args:
            X: Feature DataFrame
            fit: If True, fit scaler. If False, use existing scaler
            
        Returns:
            Scaled features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def prepare_data(self, df=None, test_size=0.2, random_state=42):
        """
        Complete data preparation pipeline
        
        Args:
            df: Input DataFrame (if None, loads data)
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if df is None:
            df = self.load_data()
        
        # Clean
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode
        df_encoded = self.encode_features(df, fit=True)
        
        # Separate features and target
        if 'Churn' in df_encoded.columns:
            y = df_encoded['Churn']
            X = df_encoded.drop('Churn', axis=1)
        else:
            raise ValueError("Target column 'Churn' not found")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train = self.scale_features(X_train, fit=True)
        X_test = self.scale_features(X_test, fit=False)
        
        logger.info(f"Data preparation complete!")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Churn rate in train: {y_train.mean():.2%}")
        logger.info(f"Churn rate in test: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processors(self, path='data/processors/'):
        """Save scalers and encoders"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')
        joblib.dump(self.feature_names, f'{path}/feature_names.pkl')
        
        logger.info(f"Processors saved to {path}")
    
    def load_processors(self, path='data/processors/'):
        """Load saved scalers and encoders"""
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.label_encoders = joblib.load(f'{path}/label_encoders.pkl')
        self.feature_names = joblib.load(f'{path}/feature_names.pkl')
        
        logger.info(f"Processors loaded from {path}")


if __name__ == "__main__":
    # Example usage
    processor = ChurnDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_data()
    
    # Save for later use
    processor.save_processors()
    
    # Save processed data
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("✓ Data processing complete!")
    print(f"✓ Files saved to data/processed/")
