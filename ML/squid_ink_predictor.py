"""
Squid Ink Price Predictor

This module contains the SquidInkPredictor class which is used to predict
future squid ink prices based on a trained model.
"""
import json
from typing import List, Dict, Any
import math

class SquidInkPredictor:
    """Machine learning predictor for squid ink patterns"""
    def __init__(self, model_type='ar'):
        self.model_type = model_type
        
        # AR model parameters (to be replaced with your trained values)
        if model_type == 'ar':
            # These are placeholder values - replace with your trained values
            self.coefficients = [-0.01591316, 0.07101847, 0.12484902, 0.82004215]
            self.intercept = 0.00047196049877129553
            
            # Additional parameters from pattern detection
            self.cycle_period = 0  # Will be populated if a cycle is detected
            self.cycle_strength = 0  # Correlation strength
        
        # Linear model parameters (if you choose this model type)
        elif model_type == 'linear':
            # These would be replaced with your trained feature coefficients
            self.feature_columns = ['return_1', 'return_5', 'volume_imbalance', 'spread_pct']
            self.coefficients = [0.1, 0.05, 0.2, -0.1]  # Example values
            self.intercept = 0.0002
    
    def predict_next_price(self, prices: List[float]) -> float:
        """
        Predict the next price using the AR model
        
        Args:
            prices: List of recent prices, with the most recent price last
                   Should have at least as many prices as the model has coefficients
        
        Returns:
            Predicted next price
        """
        if self.model_type == 'ar':
            # Get the last n prices matching coefficient count
            n = len(self.coefficients)
            if len(prices) < n:
                return prices[-1]  # Not enough data, return last known price
            
            recent_prices = prices[-n:]
            
            # Apply AR model: intercept + sum(coef_i * price_i)
            prediction = self.intercept
            for i, coef in enumerate(self.coefficients):
                prediction += coef * recent_prices[i]
            
            return prediction
        
        # For other model types, we would need the actual feature values
        # This is just a placeholder
        return prices[-1]
    
    def predict_direction(self, prices: List[float]) -> int:
        """
        Predict if the price will go up or down
        
        Returns:
            1 if price predicted to go up, -1 if down, 0 if unchanged
        """
        if len(prices) < len(self.coefficients) + 1:
            return 0  # Not enough data
            
        current_price = prices[-1]
        predicted_price = self.predict_next_price(prices)
        
        if predicted_price > current_price:
            return 1
        elif predicted_price < current_price:
            return -1
        else:
            return 0
    
    def get_confidence(self, prices: List[float]) -> float:
        """
        Calculate confidence in the prediction
        Higher when recent predictions have been accurate
        
        Returns:
            Confidence score between 0 and 1
        """
        # This is a simple placeholder implementation
        # A real implementation would track prediction accuracy
        return 0.7
    
    @classmethod
    def from_json(cls, filename: str) -> 'SquidInkPredictor':
        """
        Load model from a JSON file created by the training pipeline
        
        Returns:
            Initialized SquidInkPredictor with loaded parameters
        """
        import json
        
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        model = cls(model_type=model_data.get('model_type', 'ar'))
        model.coefficients = model_data.get('coefficients', [])
        model.intercept = model_data.get('intercept', 0)
        
        if 'feature_info' in model_data:
            if model.model_type == 'linear':
                model.feature_columns = model_data['feature_info']
            elif model.model_type == 'ar':
                # Just informational for AR models
                pass
        
        return model