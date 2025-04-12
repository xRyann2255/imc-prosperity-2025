import optuna
import subprocess
import re
import os
import numpy as np
from datetime import datetime

# Function to evaluate a set of parameters
def objective(trial):
    # Define the parameter search space with ranges that include the defaults
    squid_ma_window = trial.suggest_int('squid_ma_window', 5, 50)
    squid_entry_threshold = trial.suggest_float('squid_entry_threshold', 1.0, 3.0)
    squid_exit_threshold = trial.suggest_float('squid_exit_threshold', 0.1, 1.0)
    squid_stop_loss_threshold = trial.suggest_float('squid_stop_loss_threshold', 2.0, 6.0)
    squid_price_stop_loss_pct = trial.suggest_float('squid_price_stop_loss_pct', 0.001, 0.01, log=True)
    squid_trend_window = trial.suggest_int('squid_trend_window', 10, 100)
    squid_trend_threshold = trial.suggest_float('squid_trend_threshold', 0.001, 0.05)
    squid_trade_cooldown = trial.suggest_int('squid_trade_cooldown', 1, 10)
    squid_volatility_lookback = trial.suggest_int('squid_volatility_lookback', 10, 100)
    
    # Create a modified version of the algorithm by editing the code file
    try:
        with open('ryan_r2_v2.py', 'r') as file:
            code = file.read()
        
        # Replace parameters using regex substitutions
        code = re.sub(r'self\.squid_ma_window\s*=\s*\d+', f'self.squid_ma_window = {squid_ma_window}', code)
        code = re.sub(r'self\.squid_entry_threshold\s*=\s*\d+(?:\.\d+)?', f'self.squid_entry_threshold = {squid_entry_threshold}', code)
        code = re.sub(r'self\.squid_exit_threshold\s*=\s*\d+(?:\.\d+)?', f'self.squid_exit_threshold = {squid_exit_threshold}', code)
        code = re.sub(r'self\.squid_stop_loss_threshold\s*=\s*\d+(?:\.\d+)?', f'self.squid_stop_loss_threshold = {squid_stop_loss_threshold}', code)
        code = re.sub(r'self\.squid_price_stop_loss_pct\s*=\s*\d+(?:\.\d+)?', f'self.squid_price_stop_loss_pct = {squid_price_stop_loss_pct}', code)
        code = re.sub(r'self\.squid_trend_window\s*=\s*\d+', f'self.squid_trend_window = {squid_trend_window}', code)
        code = re.sub(r'self\.squid_trend_threshold\s*=\s*\d+(?:\.\d+)?', f'self.squid_trend_threshold = {squid_trend_threshold}', code)
        code = re.sub(r'self\.squid_trade_cooldown\s*=\s*\d+', f'self.squid_trade_cooldown = {squid_trade_cooldown}', code)
        code = re.sub(r'self\.squid_volatility_lookback\s*=\s*\d+', f'self.squid_volatility_lookback = {squid_volatility_lookback}', code)
        
        # Optionally, if the code also contains JSON-styled settings for trade cooldown, update that as well.
        code = re.sub(r'"trade_cooldown":\s*\d+', f'"trade_cooldown": {squid_trade_cooldown}', code)
        
        # Create unique filename for this trial
        temp_filename = f"temp_optuna_{trial.number}.py"
        with open(temp_filename, 'w') as file:
            file.write(code)
        
        # Run backtest (using your backtester 'prosperity3bt')
        process = subprocess.Popen(
            f"prosperity3bt {temp_filename} 1",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        stdout, stderr = process.communicate(timeout=60)
        
        # Parse results (assuming the backtest outputs profits in the pattern specified)
        day_sections = re.split(r'Backtesting .+ on round 1 day', stdout)
        
        squid_profits = [0, 0, 0]  # Default values for 3 days
        # Extract profits for each day
        for i, day in enumerate(day_sections[:3]):
            profit_match = re.search(r'SQUID_INK: ([+-]?[\d,]+)', day)
            if profit_match:
                squid_profits[i] = int(profit_match.group(1).replace(',', ''))
        
        total_squid_profit = sum(squid_profits)
        
        # Clean up temporary file
        os.remove(temp_filename)
        
        # Log results
        print(f"\nTrial {trial.number}:")
        print(f"  squid_ma_window = {squid_ma_window}")
        print(f"  squid_entry_threshold = {squid_entry_threshold}")
        print(f"  squid_exit_threshold = {squid_exit_threshold}")
        print(f"  squid_stop_loss_threshold = {squid_stop_loss_threshold}")
        print(f"  squid_price_stop_loss_pct = {squid_price_stop_loss_pct*100:.5f}%")
        print(f"  squid_trend_window = {squid_trend_window}")
        print(f"  squid_trend_threshold = {squid_trend_threshold}")
        print(f"  squid_trade_cooldown = {squid_trade_cooldown}")
        print(f"  squid_volatility_lookback = {squid_volatility_lookback}")
        print(f"  SQUID_INK profits: {squid_profits}, Total: {total_squid_profit}")
        
        return total_squid_profit
        
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        return -100000  # Return a poor result on error

# Create a study object and optimize the parameters
study = optuna.create_study(direction='maximize', 
                           study_name=f'trading_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

print("Starting parameter optimization with Bayesian search...")
print("Each trial adapts based on previous results to focus on promising areas.")

study.optimize(objective, n_trials=50)  # Run 50 trials

# Print best parameters
best_params = study.best_params
best_value = study.best_value

print("\n" + "="*50)
print("OPTIMAL PARAMETERS:")
print(f"squid_ma_window = {best_params['squid_ma_window']}")
print(f"squid_entry_threshold = {best_params['squid_entry_threshold']:.2f}")
print(f"squid_exit_threshold = {best_params['squid_exit_threshold']:.2f}")
print(f"squid_stop_loss_threshold = {best_params['squid_stop_loss_threshold']:.2f}")
print(f"squid_price_stop_loss_pct = {best_params['squid_price_stop_loss_pct']*100:.5f}%")
print(f"squid_trend_window = {best_params['squid_trend_window']}")
print(f"squid_trend_threshold = {best_params['squid_trend_threshold']:.3f}")
print(f"squid_trade_cooldown = {best_params['squid_trade_cooldown']}")
print(f"squid_volatility_lookback = {best_params['squid_volatility_lookback']}")
print(f"Total SQUID_INK profit: {best_value}")
print("="*50)

# Save the results to a text file
with open('optuna_results.txt', 'w') as f:
    f.write("OPTIMAL PARAMETERS:\n")
    for param, value in best_params.items():
        if param == 'squid_price_stop_loss_pct':
            f.write(f"{param} = {value*100:.5f}%\n")
        else:
            f.write(f"{param} = {value}\n")
    f.write(f"Total SQUID_INK profit: {best_value}\n")
    
    f.write("\nTOP 10 TRIALS:\n")
    best_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -float('inf'), reverse=True)[:10]
    for i, trial in enumerate(best_trials):
        f.write(f"{i+1}. ")
        for param, value in trial.params.items():
            if param == 'squid_price_stop_loss_pct':
                f.write(f"{param}={value*100:.5f}%, ")
            else:
                f.write(f"{param}={value}, ")
        f.write(f"profit={trial.value}\n")

print(f"\nResults saved to optuna_results.txt")
