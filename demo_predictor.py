"""
Demo Predictor - Backup prediction function for DKD Risk Platform
================================================================
This module provides a rule-based prediction system when the main model
cannot be loaded, allowing the app to function for demonstration purposes.
"""

import numpy as np
import pandas as pd

class DemoPredictor:
    """
    Rule-based predictor that mimics the ML model behavior for demo purposes.
    Based on clinical guidelines and the documented feature importance.
    """
    
    def __init__(self):
        self.feature_weights = {
            'baseline_eGFR': -0.25,      # Strong negative predictor
            'baseline_ACR': 0.15,        # Strong positive predictor  
            'baseline_HbA1c': 0.12,      # Positive predictor
            'DR_grade': 0.08,            # Positive predictor
            'NSAID_cum90d': 0.06,        # Positive predictor
            'Family_Hx_CKD': 0.05,       # Positive predictor
            'IMD_quintile': 0.04,        # Positive predictor
            'baseline_BMI': 0.03,        # Weak positive predictor
            'Age': 0.02,                 # Weak positive predictor
            'BB_flag': 0.02,             # Weak positive predictor
            'CCB_flag': 0.02,            # Weak positive predictor
            'Diuretic_flag': 0.02,       # Weak positive predictor
            'MRA_use': -0.03,            # Protective factor
            'baseline_SBP': 0.01,        # Very weak predictor
            'baseline_DBP': 0.01,        # Very weak predictor
            'Gender': 0.01,              # Minimal effect
            'Nationality': 0.005         # Minimal effect
        }
        
        self.baseline_risk = 0.095  # 9.5% baseline prevalence
        
    def predict_proba(self, X):
        """
        Predict probabilities using rule-based approach.
        
        Args:
            X: DataFrame with patient features
            
        Returns:
            Array of shape (n_samples, 2) with [negative_prob, positive_prob]
        """
        
        predictions = []
        
        for _, row in X.iterrows():
            # Start with baseline risk (logit scale)
            logit_risk = np.log(self.baseline_risk / (1 - self.baseline_risk))
            
            # Apply feature contributions
            for feature, weight in self.feature_weights.items():
                if feature in row:
                    value = row[feature]
                    
                    if feature == 'baseline_eGFR':
                        # eGFR: normalized around 90, stronger effect for low values
                        normalized_value = (90 - value) / 20  # Higher score for lower eGFR
                        contribution = weight * normalized_value
                        
                    elif feature == 'baseline_ACR':
                        # ACR: log transformation, higher is worse
                        log_acr = np.log(max(value, 1))  # Avoid log(0)
                        normalized_value = log_acr / 5  # Normalize
                        contribution = weight * normalized_value
                        
                    elif feature == 'baseline_HbA1c':
                        # HbA1c: normalized around 7%, higher is worse
                        normalized_value = (value - 7) / 2
                        contribution = weight * normalized_value
                        
                    elif feature == 'baseline_BMI':
                        # BMI: J-shaped relationship, worst at extremes
                        if value < 18.5 or value > 35:
                            normalized_value = 2
                        elif 25 <= value <= 30:
                            normalized_value = 0.5
                        else:
                            normalized_value = 1
                        contribution = weight * normalized_value
                        
                    elif feature == 'Age':
                        # Age: linear relationship, normalized around 60
                        normalized_value = (value - 60) / 15
                        contribution = weight * normalized_value
                        
                    elif feature == 'DR_grade':
                        # DR grade: ordinal 0-4, exponential effect
                        normalized_value = value * (1.5 ** value)  # Exponential growth
                        contribution = weight * normalized_value
                        
                    elif feature == 'IMD_quintile':
                        # Deprivation: linear 1-5, higher is worse
                        normalized_value = (value - 1) / 2  # 0-2 scale
                        contribution = weight * normalized_value
                        
                    elif feature in ['baseline_SBP', 'baseline_DBP']:
                        # Blood pressure: normalized around 130/80
                        if feature == 'baseline_SBP':
                            normalized_value = (value - 130) / 20
                        else:
                            normalized_value = (value - 80) / 15
                        contribution = weight * normalized_value
                        
                    else:
                        # Binary features: direct multiplication
                        contribution = weight * value
                    
                    logit_risk += contribution
            
            # Convert back to probability
            probability = 1 / (1 + np.exp(-logit_risk))
            
            # Add some realistic noise
            noise = np.random.normal(0, 0.02)  # 2% standard deviation
            probability = np.clip(probability + noise, 0.01, 0.99)
            
            predictions.append([1 - probability, probability])
        
        return np.array(predictions)
    
    def predict(self, X):
        """
        Predict binary outcomes.
        
        Args:
            X: DataFrame with patient features
            
        Returns:
            Array of binary predictions
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

def get_demo_predictor():
    """
    Factory function to create a demo predictor instance.
    
    Returns:
        DemoPredictor instance
    """
    return DemoPredictor()

# Test the demo predictor
if __name__ == "__main__":
    # Create test data
    test_data = pd.DataFrame({
        'baseline_eGFR': [45, 90, 120],  # Low, normal, high
        'baseline_ACR': [50, 10, 5],     # High, moderate, low
        'baseline_HbA1c': [9.5, 7.2, 6.8],  # Poor, fair, good control
        'DR_grade': [3, 1, 0],           # Severe, mild, none
        'NSAID_cum90d': [1, 0, 0],       # Yes, no, no
        'Family_Hx_CKD': [1, 1, 0],     # Yes, yes, no
        'IMD_quintile': [5, 3, 1],       # Most, moderate, least deprived
        'baseline_BMI': [32, 28, 24],    # Obese, overweight, normal
        'Age': [65, 55, 45],             # Older, middle, younger
        'BB_flag': [1, 1, 0],            # Yes, yes, no
        'CCB_flag': [1, 0, 0],           # Yes, no, no
        'Diuretic_flag': [1, 1, 0],      # Yes, yes, no
        'MRA_use': [0, 1, 0],            # No, yes, no
        'baseline_SBP': [160, 135, 120], # High, borderline, normal
        'baseline_DBP': [95, 85, 75],    # High, borderline, normal
        'Gender': [1, 0, 1],             # Male, female, male
        'Nationality': [1, 1, 1]         # Saudi, saudi, saudi
    })
    
    # Test predictor
    predictor = get_demo_predictor()
    probabilities = predictor.predict_proba(test_data)
    
    print("Demo Predictor Test Results:")
    print("Input scenarios: High risk, Moderate risk, Low risk")
    print(f"Predicted probabilities: {probabilities[:, 1] * 100:.1f}%")
    print("Expected: High > Moderate > Low risk")
