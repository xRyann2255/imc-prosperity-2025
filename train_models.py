#!/usr/bin/env python
"""
Squid Ink Trading Model Trainer

This script trains a machine learning model for predicting squid ink prices
using historical price and trade data.
"""
import os
import sys
from squid_ink_trainer import SquidInkModelTrainer

def main():
    # Get the training data directory path
    data_dir = "training_data"
    if not os.path.exists(data_dir):
        print(f"Error: Training data directory '{data_dir}' not found!")
        print("Please ensure the directory exists and contains the CSV files.")
        return 1

    print("=" * 60)
    print("SQUID INK TRADING MODEL TRAINER")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    
    # Initialize trainer
    trainer = SquidInkModelTrainer()
    
    # Load data from all available days
    trainer.load_multiple_days(data_dir, days=[0, -1, -2])
    
    # Run the complete training pipeline
    (trainer
        .preprocess_data()
        .detect_cycles()  # Analyze for cyclic patterns
        .extract_features()
        .prepare_training_data(target_horizon=1)  # Predict 1 step ahead
        .train_linear_regression()  # Try a linear model
        .train_ar_model([1, 2, 3, 4])  # Try an AR(4) model
        .optimize_ar_lags(max_lag=10)  # Find optimal AR structure
        .print_summary()
        .save_model("squid_ink_model.json"))
    
    print("\nTraining complete! Model saved to squid_ink_model.json")
    print("You can now copy the coefficient values into your trading algorithm")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())