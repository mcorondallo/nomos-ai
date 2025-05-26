import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os

class ArrearsPredictor:
    """
    A machine learning model to predict the likelihood of arrears payment
    based on historical payment data and various features.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.features = None
        self.feature_importance = None
        self.trained = False
        
        # Define expected feature categories
        self.numeric_features = [
            'days_overdue', 'previous_late_payments', 'average_days_late',
            'lease_count', 'lease_amount', 'property_size_sqm'
        ]
        
        self.categorical_features = [
            'payment_method', 'country', 'region'
        ]
        
    def train(self, data):
        """
        Train the model using historical payment data
        
        Args:
            data (DataFrame): Historical payment data with features and payment status
        """
        # For dummy implementation, ensure data has the target variable
        if 'was_paid' not in data.columns:
            # Add a dummy target for demonstration
            data['was_paid'] = np.random.binomial(1, 0.7, size=len(data))
            
        # Store the feature names
        self.features = self.numeric_features + self.categorical_features
        
        # Prepare X and y
        X = data[self.features]
        y = data['was_paid'].astype(float)  # Ensure target is numeric
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Create and train the model pipeline
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        # Split the data for training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit the preprocessor
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Train the model
        model.fit(X_train_processed, y_train)
        
        # Store the trained model
        self.model = model
        self.trained = True
        
        # Calculate feature importance (for numeric features only, as categorical are one-hot encoded)
        self.feature_importance = model.feature_importances_
        
        # Evaluate on test set
        X_test_processed = self.preprocessor.transform(X_test)
        test_score = model.score(X_test_processed, y_test)
        
        print(f"Model trained successfully. Test R² score: {test_score:.4f}")
        
        return test_score
    
    def predict(self, data, include_confidence=False, risk_thresholds=None):
        """
        Make predictions on new data
        
        Args:
            data (DataFrame): New data to predict on
            include_confidence (bool): Whether to include confidence intervals
            risk_thresholds (dict): Custom thresholds for risk categories
            
        Returns:
            DataFrame: Original data with payment probability predictions
        """
        if not self.trained:
            # If model isn't trained, train on dummy data
            print("Model not trained yet, training on dummy data first...")
            from utils import load_dummy_data
            self.train(load_dummy_data())
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in data.columns:
                if feature in self.numeric_features:
                    # Fill with mean value for numeric
                    data[feature] = 0
                else:
                    # Fill with most common value for categorical
                    data[feature] = "unknown"
        
        # Preprocess the data
        X = data[self.features]
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Ensure predictions are within [0,1] range
        predictions = np.clip(predictions, 0, 1)
        
        # Add predictions to the original data
        results = data.copy()
        results['payment_probability'] = predictions
        
        # Add risk categories with custom thresholds if provided
        if risk_thresholds is not None:
            # Handle both dictionary and tuple formats
            if isinstance(risk_thresholds, dict):
                low_threshold = risk_thresholds.get('low', 0.7)
                medium_threshold = risk_thresholds.get('medium', 0.3)
            elif isinstance(risk_thresholds, (list, tuple)) and len(risk_thresholds) >= 2:
                # Assume format is (medium_threshold, low_threshold)
                medium_threshold = risk_thresholds[0]
                low_threshold = risk_thresholds[1]
            else:
                # Default values
                medium_threshold = 0.3
                low_threshold = 0.7
            
            bins = [0, medium_threshold, low_threshold, 1]
        else:
            bins = [0, 0.3, 0.7, 1]
            
        results['risk_category'] = pd.cut(
            results['payment_probability'],
            bins=bins,
            labels=['High Risk', 'Medium Risk', 'Low Risk']
        )
        
        # Add confidence intervals if requested
        if include_confidence:
            # For demonstration purposes, create simple confidence intervals
            # In a real model, these would be calculated using proper statistical methods
            confidence_margin = 0.1  # 10% margin
            
            # Adjust margins for extreme values
            results['lower_bound'] = np.maximum(0, results['payment_probability'] - confidence_margin)
            results['upper_bound'] = np.minimum(1, results['payment_probability'] + confidence_margin)
            
            # Add confidence range
            results['confidence_range'] = results['upper_bound'] - results['lower_bound']
        
        return results
    
    def save_model(self, path="model.pkl"):
        """Save the trained model to a file"""
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'preprocessor': self.preprocessor,
                'features': self.features,
                'feature_importance': self.feature_importance
            }, f)
    
    def load_model(self, path="model.pkl"):
        """Load a trained model from a file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.features = model_data['features']
        self.feature_importance = model_data['feature_importance']
        self.trained = True
