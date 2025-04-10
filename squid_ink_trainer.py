import pandas as pd
import numpy as np
import math
import statistics
from typing import List, Dict, Tuple, Optional
import json
import os

class SquidInkModelTrainer:
    """
    A comprehensive pipeline for training predictive models for Squid Ink trading
    using historical price and trade data.
    """
    
    def __init__(self):
        self.trades_data = None
        self.prices_data = None
        self.squid_data = None
        self.features = None
        self.labels = None
        self.models = {}
        self.best_model = None
        self.features_df = None
        
    def load_data(self, trades_file: str, prices_file: str):
        """Load and parse the trades and prices data"""
        print(f"Loading data files: {trades_file}, {prices_file}")
        
        try:
            # Load trades data
            trades_df = pd.read_csv(trades_file, sep=';')
            self.trades_data = trades_df
            print(f"Loaded {len(trades_df)} trades")
            
            # Load prices data
            prices_df = pd.read_csv(prices_file, sep=';')
            self.prices_data = prices_df
            print(f"Loaded {len(prices_df)} price points")
            
            # Extract squid ink specific data
            if 'product' in prices_df.columns:
                self.squid_data = prices_df[prices_df['product'] == 'SQUID_INK'].copy()
                print(f"Found {len(self.squid_data)} Squid Ink price points")
            else:
                print("Warning: 'product' column not found in prices data")
        except Exception as e:
            print(f"Error loading data: {e}")
        
        return self
    
    def load_multiple_days(self, data_dir: str, days: List[int] = [0, -1, -2]):
        """Load and combine data from multiple days"""
        print(f"Loading data from {len(days)} days: {days}")
        
        all_prices = []
        all_trades = []
        
        for day in days:
            day_str = f"day_{day}" if day <= 0 else f"day_{day}"
            prices_file = os.path.join(data_dir, f"prices_round_1_{day_str}.csv")
            trades_file = os.path.join(data_dir, f"trades_round_1_{day_str}.csv")
            
            if os.path.exists(prices_file) and os.path.exists(trades_file):
                print(f"Processing day {day}...")
                
                # Load prices
                try:
                    prices_df = pd.read_csv(prices_file, sep=';')
                    prices_df['day'] = day  # Tag with day for reference
                    all_prices.append(prices_df)
                    print(f"  - Loaded {len(prices_df)} price points")
                except Exception as e:
                    print(f"  - Error loading prices for day {day}: {e}")
                
                # Load trades
                try:
                    trades_df = pd.read_csv(trades_file, sep=';')
                    trades_df['day'] = day  # Tag with day for reference
                    all_trades.append(trades_df)
                    print(f"  - Loaded {len(trades_df)} trades")
                except Exception as e:
                    print(f"  - Error loading trades for day {day}: {e}")
            else:
                print(f"Files for day {day} not found, skipping.")
        
        # Combine all data
        if all_prices:
            self.prices_data = pd.concat(all_prices, ignore_index=True)
            print(f"Combined prices data: {len(self.prices_data)} rows")
            
            # Extract squid ink data
            if 'product' in self.prices_data.columns:
                self.squid_data = self.prices_data[self.prices_data['product'] == 'SQUID_INK'].copy()
                print(f"Combined Squid Ink data: {len(self.squid_data)} rows")
        
        if all_trades:
            self.trades_data = pd.concat(all_trades, ignore_index=True)
            print(f"Combined trades data: {len(self.trades_data)} rows")
        
        return self
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        if self.squid_data is None or len(self.squid_data) == 0:
            print("No data loaded. Call load_data() or load_multiple_days() first.")
            return self
            
        print("Preprocessing data...")
        
        # Ensure numeric data types
        numeric_columns = [
            'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3',
            'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3',
            'mid_price'
        ]
        
        for col in numeric_columns:
            if col in self.squid_data.columns:
                self.squid_data[col] = pd.to_numeric(self.squid_data[col], errors='coerce')
        
        # Sort by day and timestamp
        sort_cols = []
        if 'day' in self.squid_data.columns:
            sort_cols.append('day')
        if 'timestamp' in self.squid_data.columns:
            sort_cols.append('timestamp')
            
        if sort_cols:
            self.squid_data = self.squid_data.sort_values(sort_cols)
        
        # Handle missing values
        self.squid_data = self.squid_data.dropna(subset=['mid_price'])
        
        print(f"After preprocessing: {len(self.squid_data)} valid data points")
        return self
    
    def detect_cycles(self) -> 'SquidInkModelTrainer':
        """Detect cyclic patterns in the price data using autocorrelation"""
        if self.squid_data is None or len(self.squid_data) == 0:
            print("No preprocessed data available.")
            return self
            
        print("Detecting cyclic patterns...")
        
        # Get the mid price series
        prices = self.squid_data['mid_price'].values
        
        # Normalize prices
        mean_price = np.mean(prices)
        normalized = prices - mean_price
        
        # Calculate autocorrelation for different lags
        max_lag = min(100, len(normalized) // 4)
        autocorr = {}
        
        for lag in range(1, max_lag):
            if lag >= len(normalized):
                continue
                
            # Calculate autocorrelation for this lag
            corr = np.corrcoef(normalized[:-lag], normalized[lag:])[0, 1]
            autocorr[lag] = corr
        
        # Find significant cycles
        significant_cycles = []
        threshold = 0.2  # Correlation threshold for significance
        
        for lag, corr in sorted(autocorr.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(corr) > threshold:
                significant_cycles.append((lag, corr))
                if len(significant_cycles) >= 5:  # Limit to top 5
                    break
        
        # Find the dominant cycle
        dominant_cycle = max(autocorr.items(), key=lambda x: abs(x[1])) if autocorr else (0, 0)
        
        # Store results in the class instead of returning them
        self.cycle_results = {
            "dominant_cycle": dominant_cycle,
            "significant_cycles": significant_cycles,
            "autocorrelations": autocorr
        }
        
        # Set the cycle information for use in prediction
        self.cycle_period = dominant_cycle[0]
        self.cycle_strength = dominant_cycle[1]
        
        print(f"Dominant cycle detected at lag {dominant_cycle[0]} with correlation {dominant_cycle[1]:.4f}")
        
        # Return self for method chaining
        return self
    
    def extract_features(self, lookback_periods: List[int] = [1, 2, 3, 4, 5, 10, 20]):
        """Extract features from the price data"""
        if self.squid_data is None or len(self.squid_data) == 0:
            print("No preprocessed data available.")
            return self
            
        print("Extracting features...")
        
        # Make a copy to avoid modifying the original
        df = self.squid_data.copy()
        
        # Print initial data shape
        initial_rows = len(df)
        print(f"Initial data shape: {df.shape}")
        
        # Check for NaN values in mid_price
        nan_midprice = df['mid_price'].isna().sum()
        print(f"NaN values in mid_price: {nan_midprice}")
        
        # Price changes (returns) at different lookback periods
        for period in lookback_periods:
            if period < len(df):
                df[f'return_{period}'] = df['mid_price'].pct_change(period)
                nan_count = df[f'return_{period}'].isna().sum()
                print(f"NaN values in return_{period}: {nan_count}")
        
        # Volume features - check column existence first
        volume_cols = ['bid_volume_1', 'ask_volume_1', 'bid_volume_2', 'ask_volume_2', 'bid_volume_3', 'ask_volume_3']
        existing_vol_cols = [col for col in volume_cols if col in df.columns]
        print(f"Existing volume columns: {existing_vol_cols}")
        
        # If we have at least bid_volume_1 and ask_volume_1, compute volume features
        if 'bid_volume_1' in df.columns and 'ask_volume_1' in df.columns:
            # Fill missing values with 0 for volume columns
            for col in ['bid_volume_2', 'bid_volume_3', 'ask_volume_2', 'ask_volume_3']:
                if col in df.columns:
                    df[col].fillna(0, inplace=True)
                else:
                    df[col] = 0
            
            # Total volumes
            df['total_bid_volume'] = df['bid_volume_1'] + df['bid_volume_2'] + df['bid_volume_3']
            df['total_ask_volume'] = df['ask_volume_1'] + df['ask_volume_2'] + df['ask_volume_3']
            
            # Volume imbalance - handle division by zero
            total_volume = df['total_bid_volume'] + df['total_ask_volume']
            df['volume_imbalance'] = np.where(
                total_volume > 0,
                (df['total_bid_volume'] - df['total_ask_volume']) / total_volume,
                0
            )
        
        # Price spreads
        if all(col in df.columns for col in ['bid_price_1', 'ask_price_1']):
            df['spread'] = df['ask_price_1'] - df['bid_price_1']
            df['spread_pct'] = df['spread'] / df['mid_price'].replace(0, np.nan)  # Avoid division by zero
        
        # Add lagged price features
        max_lag = max(lookback_periods) if lookback_periods else 5
        for i in range(1, max_lag + 1):
            df[f'price_lag_{i}'] = df['mid_price'].shift(i)
            nan_count = df[f'price_lag_{i}'].isna().sum()
            print(f"NaN values in price_lag_{i}: {nan_count}")
        
        # Feature engineering based on detected patterns
        if hasattr(self, 'cycle_period') and self.cycle_period > 0:
            # Add cyclical features if a strong cycle was detected
            dominant_cycle = self.cycle_period
            df[f'cycle_price'] = df['mid_price'].shift(dominant_cycle)
            
            # Calculate cycle return safely
            if f'cycle_price' in df.columns:
                cycle_denom = df[f'cycle_price'].replace(0, np.nan)  # Avoid division by zero
                df[f'cycle_return'] = (df['mid_price'] / cycle_denom - 1)
        
        # Check data shape before dropping NaNs
        before_dropna = len(df)
        print(f"Data shape before dropping NaNs: {df.shape}")
        
        # Identify essential columns that should not have NaNs
        essential_cols = ['mid_price']
        # Add some key feature columns but not all to avoid dropping too many rows
        essential_cols.extend([f'price_lag_{i}' for i in range(1, min(5, max_lag + 1))])
        # Only drop rows with NaNs in essential columns
        df.dropna(subset=essential_cols, inplace=True)
        
        # For remaining features, fill NaNs with appropriate values
        # Returns and other calculated values can be filled with 0
        calculated_cols = [col for col in df.columns if col.startswith('return_') or 
                        col.startswith('volume_') or col.startswith('spread')]
        df[calculated_cols] = df[calculated_cols].fillna(0)
        
        # Check data shape after handling NaNs
        after_handling_nans = len(df)
        print(f"Data shape after handling NaNs: {df.shape}")
        print(f"Rows removed: {initial_rows - after_handling_nans}")
        
        # Save the processed dataframe
        self.features_df = df
        print(f"Extracted {len(self.features_df)} data points with {len(self.features_df.columns)} features")
        
        # Initialize X_train and y_train with empty arrays
        # This ensures they exist even if prepare_training_data() fails
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_test = np.array([])
        self.y_test = np.array([])
        
        return self
    
    def prepare_training_data(self, target_horizon: int = 1, test_size: float = 0.2):
        """Prepare data for training by creating features and labels"""
        if self.features_df is None:
            print("No feature data available. Run extract_features() first.")
            # Initialize with empty arrays to prevent AttributeError
            self.X_train = np.array([])
            self.y_train = np.array([])
            self.X_test = np.array([])
            self.y_test = np.array([])
            self.feature_columns = []
            return self
            
        if len(self.features_df) == 0:
            print("Warning: features_df is empty. No data to prepare for training.")
            # Initialize with empty arrays to prevent AttributeError
            self.X_train = np.array([])
            self.y_train = np.array([])
            self.X_test = np.array([])
            self.y_test = np.array([])
            self.feature_columns = []
            return self
                
        print(f"Preparing training data with target horizon of {target_horizon}...")
        
        df = self.features_df.copy()
        
        # Create target variable: future price change
        df['target'] = df['mid_price'].shift(-target_horizon) / df['mid_price'] - 1
        
        # Check for NaN in target
        nan_target = df['target'].isna().sum()
        print(f"NaN values in target: {nan_target}")
        
        # Remove rows where target is NaN
        df_with_target = df.dropna(subset=['target'])
        print(f"Rows after removing NaN targets: {len(df_with_target)}")
        
        if len(df_with_target) == 0:
            print("Warning: No valid data points with target. Cannot prepare training data.")
            # Initialize with empty arrays to prevent AttributeError
            self.X_train = np.array([])
            self.y_train = np.array([])
            self.X_test = np.array([])
            self.y_test = np.array([])
            self.feature_columns = []
            return self
        
        # Split into train and test
        train_size = int(len(df_with_target) * (1 - test_size))
        train_df = df_with_target.iloc[:train_size]
        test_df = df_with_target.iloc[train_size:]
        
        # Define feature columns to use - exclude certain columns
        exclude_cols = ['target', 'day', 'timestamp', 'product', 'mid_price', 'profit_and_loss']
        feature_cols = [col for col in df_with_target.columns 
                    if col.startswith(('return_', 'price_lag_', 'volume_', 'spread')) 
                    and col not in exclude_cols]
        
        # Add cycle features if they exist
        if 'cycle_price' in df_with_target.columns:
            feature_cols.append('cycle_price')
        if 'cycle_return' in df_with_target.columns:
            feature_cols.append('cycle_return')
        
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Store training and testing data
        self.X_train = train_df[feature_cols].values
        self.y_train = train_df['target'].values
        self.X_test = test_df[feature_cols].values
        self.y_test = test_df['target'].values
        self.feature_columns = feature_cols
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Testing set: {len(self.X_test)} samples")
        
        return self
    
    def train_linear_regression(self):
        """Train a linear regression model using NumPy"""
        if self.X_train is None or self.y_train is None:
            print("No training data available. Run prepare_training_data() first.")
            return self
            
        if len(self.X_train) == 0 or len(self.y_train) == 0:
            print("Training data is empty. Cannot train linear regression model.")
            # Create empty model to avoid errors later
            self.models['linear'] = {
                'intercept': 0.0,
                'coefficients': np.array([]),
                'feature_columns': [],
                'mse': float('inf')
            }
            return self
                
        print("Training linear regression model...")
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        
        # Add constant term for intercept
        X_train_with_const = np.column_stack([np.ones(len(self.X_train)), self.X_train])
        
        # Solve using normal equations: (X^T X)^-1 X^T y
        try:
            # Use least squares to fit the model
            coeffs = np.linalg.lstsq(X_train_with_const, self.y_train, rcond=None)[0]
            
            # Store the model
            self.models['linear'] = {
                'intercept': coeffs[0],
                'coefficients': coeffs[1:],
                'feature_columns': self.feature_columns
            }
            
            # Evaluate on test data
            if len(self.X_test) > 0:
                X_test_with_const = np.column_stack([np.ones(len(self.X_test)), self.X_test])
                y_pred = X_test_with_const.dot(coeffs)
                
                # Calculate mean squared error
                mse = np.mean((y_pred - self.y_test) ** 2)
                print(f"Linear regression MSE on test data: {mse:.8f}")
                
                # Store evaluation metrics
                self.models['linear']['mse'] = mse
                
                # Set as best model if it's the first or better than previous
                if self.best_model is None or mse < self.best_model.get('mse', float('inf')):
                    self.best_model = self.models['linear']
                    print("Linear regression is currently the best model")
            else:
                # No test data available
                print("No test data available for evaluation")
                self.models['linear']['mse'] = float('inf')
                    
        except np.linalg.LinAlgError as e:
            print(f"Error fitting linear regression model: {e}")
            print("Check for multicollinearity or insufficient data.")
            # Create empty model to avoid errors later
            self.models['linear'] = {
                'intercept': 0.0,
                'coefficients': np.array([]),
                'feature_columns': [],
                'mse': float('inf')
            }
        except Exception as e:
            print(f"Unexpected error in linear regression: {e}")
            # Create empty model to avoid errors later
            self.models['linear'] = {
                'intercept': 0.0,
                'coefficients': np.array([]),
                'feature_columns': [],
                'mse': float('inf')
            }
        
        return self
    
    def train_ar_model(self, lags: List[int] = None):
        """Train an autoregressive model on price data"""
        if self.squid_data is None or len(self.squid_data) == 0:
            print("No preprocessed data available.")
            # Create empty model to avoid errors later
            self.models['ar'] = {
                'intercept': 0.0,
                'coefficients': np.array([]),
                'lags': lags or [],
                'mse': float('inf'),
                'normalized_mse': float('inf'),
                'direction_accuracy': 0.0
            }
            return self
                
        # Use only price lag features
        prices = self.squid_data['mid_price'].values
        print(f"Training AR model on {len(prices)} price points")
        
        # Default to using 4 lags if not specified
        if lags is None:
            lags = [1, 2, 3, 4]
                
        print(f"Training AR({max(lags)}) model with lags: {lags}")
        
        # Ensure we have enough data
        if len(prices) <= max(lags) + 1:
            print(f"Insufficient data: need at least {max(lags) + 2} points, have {len(prices)}")
            # Create empty model to avoid errors later
            self.models['ar'] = {
                'intercept': 0.0,
                'coefficients': np.array([]),
                'lags': lags,
                'mse': float('inf'),
                'normalized_mse': float('inf'),
                'direction_accuracy': 0.0
            }
            return self
        
        # Prepare data matrix
        data = []
        targets = []
        max_lag = max(lags)
        
        for i in range(max_lag, len(prices) - 1):
            # Create feature vector with specified lags
            features = [prices[i - lag] for lag in lags]
            target = prices[i + 1]  # Next price
            
            data.append(features)
            targets.append(target)
        
        print(f"Prepared {len(data)} samples for AR model training")
        
        if len(data) == 0:
            print("No valid samples for AR model training")
            # Create empty model to avoid errors later
            self.models['ar'] = {
                'intercept': 0.0,
                'coefficients': np.array([]),
                'lags': lags,
                'mse': float('inf'),
                'normalized_mse': float('inf'),
                'direction_accuracy': 0.0
            }
            return self
        
        X = np.array(data)
        y = np.array(targets)
        
        # Split into train and test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"AR model - Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Add constant term for intercept
        X_train_with_const = np.column_stack([np.ones(len(X_train)), X_train])
        
        try:
            # Fit model
            coeffs = np.linalg.lstsq(X_train_with_const, y_train, rcond=None)[0]
            
            # Store model
            self.models['ar'] = {
                'intercept': coeffs[0],
                'coefficients': coeffs[1:],
                'lags': lags
            }
            
            # Evaluate
            X_test_with_const = np.column_stack([np.ones(len(X_test)), X_test])
            y_pred = X_test_with_const.dot(coeffs)
            
            mse = np.mean((y_pred - y_test) ** 2)
            
            # Calculate normalized MSE (as percentage of price)
            price_scale = np.mean(y_test) if len(y_test) > 0 else 1
            normalized_mse = mse / (price_scale ** 2) if price_scale != 0 else float('inf')
            
            print(f"AR model MSE: {mse:.2f}, Normalized MSE: {normalized_mse:.8f}")
            
            # Calculate directional accuracy (up/down prediction)
            last_prices = np.array([features[-1] for features in data[train_size:]])
            if len(last_prices) > 0:
                actual_direction = (y_test > last_prices).astype(int)
                predicted_direction = (y_pred > last_prices).astype(int)
                direction_accuracy = np.mean(actual_direction == predicted_direction)
                print(f"Directional accuracy: {direction_accuracy:.4f}")
            else:
                direction_accuracy = 0.0
                print("No test samples for directional accuracy calculation")
            
            # Store evaluation metrics
            self.models['ar']['mse'] = mse
            self.models['ar']['normalized_mse'] = normalized_mse
            self.models['ar']['direction_accuracy'] = direction_accuracy
            
            # Check if this is the best model
            if self.best_model is None or normalized_mse < self.best_model.get('normalized_mse', float('inf')):
                self.best_model = self.models['ar']
                print("AR model is currently the best model")
                
        except np.linalg.LinAlgError as e:
            print(f"Error fitting AR model: {e}")
            # Create empty model to avoid errors later
            self.models['ar'] = {
                'intercept': 0.0,
                'coefficients': np.array([]),
                'lags': lags,
                'mse': float('inf'),
                'normalized_mse': float('inf'),
                'direction_accuracy': 0.0
            }
        except Exception as e:
            print(f"Unexpected error in AR model training: {e}")
            # Create empty model to avoid errors later
            self.models['ar'] = {
                'intercept': 0.0,
                'coefficients': np.array([]),
                'lags': lags,
                'mse': float('inf'),
                'normalized_mse': float('inf'),
                'direction_accuracy': 0.0
            }
        
        return self
    
    def optimize_ar_lags(self, max_lag: int = 10):
        """Find the optimal set of lags for the AR model"""
        if self.squid_data is None or len(self.squid_data) == 0:
            print("No preprocessed data available.")
            return self
            
        print(f"Optimizing AR model lags up to {max_lag}...")
        
        best_lags = []
        best_mse = float('inf')
        
        # Try different combinations of lags
        # Start with just lag 1
        for i in range(1, max_lag + 1):
            lags = list(range(1, i + 1))  # [1], [1,2], [1,2,3], etc.
            
            # Train model with these lags
            self.train_ar_model(lags)
            
            # Check if it's the best so far
            if 'ar' in self.models and self.models['ar'].get('normalized_mse', float('inf')) < best_mse:
                best_mse = self.models['ar']['normalized_mse']
                best_lags = lags
                print(f"New best lags: {best_lags} with normalized MSE: {best_mse:.8f}")
        
        # Final training with best lags
        if best_lags:
            print(f"Final training with optimal lags: {best_lags}")
            self.train_ar_model(best_lags)
        
        return self
    
    def save_model(self, filename: str = 'squid_ink_model.json'):
        """Save the best model parameters to a JSON file"""
        if not self.models:
            print("No trained models available to save.")
            return self
        
        # Find best model using normalized MSE
        best_model = None
        best_model_name = 'unknown'
        best_normalized_mse = float('inf')
        
        for name, model in self.models.items():
            if 'normalized_mse' in model and model['normalized_mse'] < best_normalized_mse:
                best_normalized_mse = model['normalized_mse']
                best_model = model
                best_model_name = name
        
        if best_model is None:
            print("Could not determine best model. No model saved.")
            return self
                
        print(f"Saving best model ({best_model_name}) to {filename}...")
        
        # Create a simplified version of the model for the trading algorithm
        model_data = {
            'model_type': best_model_name,
            'intercept': float(best_model['intercept']),
            'coefficients': [float(c) for c in best_model['coefficients']],
            'feature_info': best_model.get('feature_columns', best_model.get('lags', [])),
            'metrics': {
                'mse': float(best_model.get('mse', 0)),
                'normalized_mse': float(best_model.get('normalized_mse', 0)),
                'direction_accuracy': float(best_model.get('direction_accuracy', 0))
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filename}")
        
        # Output model for easy copy-paste
        print("\nModel parameters for direct use in trading algorithm:")
        if best_model_name == 'ar':
            print(f"self.coefficients = {[float(c) for c in best_model['coefficients']]}")
            print(f"self.intercept = {float(best_model['intercept'])}")
        
        return self
    
    def print_summary(self):
        """Print a summary of trained models and their performance"""
        if not self.models:
            print("No models have been trained yet.")
            return self
                
        print("\n===== MODEL TRAINING SUMMARY =====")
        
        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper()} MODEL ---")
            
            if 'mse' in model:
                print(f"MSE: {model['mse']:.8f}")
            
            if 'normalized_mse' in model:
                print(f"Normalized MSE: {model['normalized_mse']:.8f}")
                    
            if 'direction_accuracy' in model:
                print(f"Direction Accuracy: {model['direction_accuracy']:.4f}")
            
            print("Parameters:")
            print(f"  Intercept: {model['intercept']:.8f}")
            
            if 'coefficients' in model and len(model['coefficients']) > 0:
                print("  Coefficients:")
                
                if model_name == 'linear' and 'feature_columns' in model:
                    # Print with feature names
                    for feature, coef in zip(model['feature_columns'], model['coefficients']):
                        print(f"    {feature}: {coef:.8f}")
                elif model_name == 'ar' and 'lags' in model:
                    # Print with lag numbers
                    for lag, coef in zip(model['lags'], model['coefficients']):
                        print(f"    Lag {lag}: {coef:.8f}")
                else:
                    # Generic printing
                    for i, coef in enumerate(model['coefficients']):
                        print(f"    Coef_{i+1}: {coef:.8f}")
        
        if self.best_model:
            # Find best model using model identity and normalized MSE
            best_model_name = 'unknown'
            best_normalized_mse = float('inf')
            
            for name, model in self.models.items():
                if 'normalized_mse' in model and model['normalized_mse'] < best_normalized_mse:
                    best_normalized_mse = model['normalized_mse']
                    best_model_name = name
            
            print(f"\nBEST MODEL: {best_model_name.upper()}")
        
        print("\n===================================")
        
        return self