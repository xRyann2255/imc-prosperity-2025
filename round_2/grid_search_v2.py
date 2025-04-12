import subprocess
import pandas as pd
import os
import re
import numpy as np
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import time
from datetime import timedelta

# Define parameter ranges to search - OPTIMIZED FOR SPEED
long_ma_values = range(5, 30, 3)  # Reduced: 5 to 29 with step of 3
z_threshold_values = [round(1.0 + i * 0.2, 1) for i in range(30)]  # Reduced: 1.0 to 6.8 with step of 0.2
# Exponentially increasing stop loss values from 0.05% to ~4.3%
stop_loss_values = [round(0.00005 * (2.6**i), 5) for i in range(13)]

# Function to test a single parameter combination
def test_parameters(params):
    long_ma, z_threshold, stop_loss = params
    
    # Create a modified version of the algorithm with these parameters
    try:
        with open('ryan_r2_v2.py', 'r') as file:  # Assuming your file is now named paste.txt
            code = file.read()
        
        # Replace the parameter values
        modified_code = re.sub(
            r'self\.squid_ma_long\s*=\s*\d+', 
            f'self.squid_ma_long = {long_ma}', 
            code
        )
        modified_code = re.sub(
            r'self\.squid_z_threshold\s*=\s*\d+(?:\.\d+)?', 
            f'self.squid_z_threshold = {z_threshold}', 
            modified_code
        )
        modified_code = re.sub(
            r'self\.price_stop_loss_pct\s*=\s*\d+(?:\.\d+)?', 
            f'self.price_stop_loss_pct = {stop_loss}', 
            modified_code
        )
        
        # Write the modified code to a temporary file - use underscore for decimal
        z_str = str(z_threshold).replace(".", "_")
        stop_loss_str = str(stop_loss).replace(".", "_")
        temp_filename = f"temp_{long_ma}_{z_str}_{stop_loss_str}.py"
        with open(temp_filename, 'w') as file:
            file.write(modified_code)
        
        result = {
            'long_ma': int(long_ma),
            'z_threshold': float(z_threshold),
            'stop_loss': float(stop_loss),
            'squid_profit_day_minus2': 0,
            'squid_profit_day_minus1': 0,
            'squid_profit_day_0': 0,
            'total_squid_profit': 0,
            'total_profit': 0
        }
        
        # Run the backtest
        try:
            # Run the command and capture both stdout and stderr
            process = subprocess.Popen(
                f"prosperity3bt {temp_filename} 1",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=60)
            
            # Check if command was successful
            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
                print(f"Error output: {stderr}")
                return result
            
            output = stdout
            
            # Parse the full output to extract all SQUID_INK profits
            # Split by the backtest headers to get results for each day
            day_sections = re.split(r'Backtesting .+ on round 1 day', output)
            
            # First section is day -2
            squid_profit_match_day_minus2 = re.search(r'SQUID_INK: ([+-]?[\d,]+)', day_sections[0])
            if squid_profit_match_day_minus2:
                result['squid_profit_day_minus2'] = int(squid_profit_match_day_minus2.group(1).replace(',', ''))
            
            # Check if we have results for day -1
            if len(day_sections) > 1:
                squid_profit_match_day_minus1 = re.search(r'SQUID_INK: ([+-]?[\d,]+)', day_sections[1])
                if squid_profit_match_day_minus1:
                    result['squid_profit_day_minus1'] = int(squid_profit_match_day_minus1.group(1).replace(',', ''))
            
            # Check if we have results for day 0
            if len(day_sections) > 2:
                squid_profit_match_day_0 = re.search(r'SQUID_INK: ([+-]?[\d,]+)', day_sections[2])
                if squid_profit_match_day_0:
                    result['squid_profit_day_0'] = int(squid_profit_match_day_0.group(1).replace(',', ''))
                    
            # Calculate the total SQUID_INK profit across all days
            squid_profits = [
                result['squid_profit_day_minus2'],
                result['squid_profit_day_minus1'],
                result['squid_profit_day_0']
            ]
            result['total_squid_profit'] = sum(squid_profits)
            
            # Extract overall total profit from the summary
            total_profit_match = re.search(r'Total profit: ([+-]?[\d,]+)$', output, re.MULTILINE)
            if total_profit_match:
                result['total_profit'] = int(total_profit_match.group(1).replace(',', ''))
            
        except subprocess.TimeoutExpired:
            print(f"Timeout for long_ma={long_ma}, z_threshold={z_threshold}, stop_loss={stop_loss:.5f}")
        except Exception as e:
            print(f"Error running backtest for long_ma={long_ma}, z_threshold={z_threshold}, stop_loss={stop_loss:.5f}: {e}")
            print(f"Command: prosperity3bt {temp_filename} 1")
        
        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except:
            pass
            
        return result
    except Exception as e:
        print(f"Fatal error for long_ma={long_ma}, z_threshold={z_threshold}, stop_loss={stop_loss:.5f}: {e}")
        return {
            'long_ma': int(long_ma),
            'z_threshold': float(z_threshold),
            'stop_loss': float(stop_loss),
            'squid_profit_day_minus2': 0,
            'squid_profit_day_minus1': 0,
            'squid_profit_day_0': 0,
            'total_squid_profit': 0,
            'total_profit': 0
        }

# Generate all valid parameter combinations
param_combinations = [(long_ma, z_threshold, stop_loss) 
                     for long_ma in long_ma_values 
                     for z_threshold in z_threshold_values
                     for stop_loss in stop_loss_values]

total_combinations = len(param_combinations)
print(f"Testing {total_combinations} parameter combinations across 3 dimensions")
print(f"Long MA: {min(long_ma_values)}-{max(long_ma_values)} (step 3)")
print(f"Z-threshold: {min(z_threshold_values)}-{max(z_threshold_values)} (step 0.2)")
print(f"Stop Loss values: {', '.join([f'{sl*100:.5f}%' for sl in stop_loss_values])}")
print(f"Stop Loss range: {min(stop_loss_values)*100:.5f}% to {max(stop_loss_values)*100:.5f}% (exponential increase)")

# Determine number of workers (use fewer workers to avoid overloading)
num_workers = max(1, 30)
print(f"Running with {num_workers} parallel workers")

# Enhanced progress tracking
start_time = time.time()
completed_tasks = 0

# Run tests in parallel using ThreadPoolExecutor
results = []
with tqdm(total=total_combinations, desc="Grid Search Progress", 
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_params = {executor.submit(test_parameters, params): params for params in param_combinations}
        
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result:  # Only add if we got a valid result
                    results.append(result)
                    long_ma, z_threshold, stop_loss = params
                    
                    # Update time estimates
                    completed_tasks += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed_tasks
                    estimated_remaining = avg_time_per_task * (total_combinations - completed_tasks)
                    
                    # Format as HH:MM:SS
                    remaining_time = str(timedelta(seconds=int(estimated_remaining)))
                    
                    # Set postfix with detailed progress info
                    pbar.set_postfix({
                        'current': f"({long_ma},{z_threshold},{stop_loss*100:.5f}%)",
                        'squid_profit': result['total_squid_profit'],
                        'ETA': remaining_time
                    })
                    
                    # Print detailed results occasionally
                    if completed_tasks % max(1, num_workers // 2) == 0 or completed_tasks <= 5:
                        print(f"\nResults for long_ma={long_ma}, z_threshold={z_threshold}, stop_loss={stop_loss*100:.5f}%:")
                        print(f"  SQUID_INK day -2: {result['squid_profit_day_minus2']}")
                        print(f"  SQUID_INK day -1: {result['squid_profit_day_minus1']}")
                        print(f"  SQUID_INK day 0: {result['squid_profit_day_0']}")
                        print(f"  Total SQUID_INK: {result['total_squid_profit']}")
                        print(f"  Overall total profit: {result['total_profit']}")
                        
                        # Show current best results
                        if len(results) > 5:
                            # Create a fresh DataFrame with proper numeric types
                            temp_df = pd.DataFrame(results).astype({
                                'long_ma': int,
                                'z_threshold': float,
                                'stop_loss': float,
                                'squid_profit_day_minus2': int,
                                'squid_profit_day_minus1': int,
                                'squid_profit_day_0': int,
                                'total_squid_profit': int,
                                'total_profit': int
                            })
                            
                            # Now find the best result so far
                            best_so_far = temp_df.nlargest(1, 'total_squid_profit').iloc[0]
                            print(f"\nBest configuration so far:")
                            print(f"  long_ma={best_so_far['long_ma']}, z_threshold={best_so_far['z_threshold']}, stop_loss={best_so_far['stop_loss']*100:.5f}%")
                            print(f"  Total SQUID_INK profit: {best_so_far['total_squid_profit']}")
            except Exception as e:
                long_ma, z_threshold, stop_loss = params
                print(f"\nError processing result for long_ma={long_ma}, z_threshold={z_threshold}, stop_loss={stop_loss*100:.5f}%: {e}")
            
            pbar.update(1)

# Convert results to DataFrame for analysis
if results:
    # Create DataFrame with explicit type conversion
    results_df = pd.DataFrame(results).astype({
        'long_ma': int,
        'z_threshold': float,
        'stop_loss': float,
        'squid_profit_day_minus2': int,
        'squid_profit_day_minus1': int,
        'squid_profit_day_0': int,
        'total_squid_profit': int,
        'total_profit': int
    })
    
    if not results_df.empty:
        # Sort by total SQUID_INK profit (descending)
        best_squid_params = results_df.sort_values('total_squid_profit', ascending=False)
        print("\n=== Best parameters for total SQUID_INK profit across all three days ===")
        top_results = best_squid_params.head(15)
        # Format stop_loss as percentage for readability
        top_results_display = top_results.copy()
        top_results_display['stop_loss'] = top_results_display['stop_loss'] * 100
        print(top_results_display[['long_ma', 'z_threshold', 'stop_loss', 'squid_profit_day_minus2', 
                                  'squid_profit_day_minus1', 'squid_profit_day_0', 
                                  'total_squid_profit', 'total_profit']])
        
        # Print the absolute best result
        best_result = best_squid_params.iloc[0]
        print("\n=== OPTIMAL PARAMETERS ===")
        print(f"long_ma = {best_result['long_ma']}, z_threshold = {best_result['z_threshold']}, stop_loss = {best_result['stop_loss']*100:.5f}%")
        print(f"Total SQUID_INK profit: {best_result['total_squid_profit']}")
        print(f"SQUID_INK day -2: {best_result['squid_profit_day_minus2']}")
        print(f"SQUID_INK day -1: {best_result['squid_profit_day_minus1']}")
        print(f"SQUID_INK day 0: {best_result['squid_profit_day_0']}")
        print(f"Overall total profit: {best_result['total_profit']}")
        
        # Create visualizations for 3D data
        # We can't do a simple 2D heatmap with 3 parameters, so we'll provide different views
        
        # 1. Find best stop_loss for each (long_ma, z_threshold) combination
        try:
            # Group by long_ma and z_threshold, find max profit
            best_by_ma_z = results_df.groupby(['long_ma', 'z_threshold'])['total_squid_profit'].max().reset_index()
            # Create a pivot table to visualize
            pivot_ma_z = best_by_ma_z.pivot(index='long_ma', columns='z_threshold', values='total_squid_profit')
            print("\nBest SQUID_INK profit heatmap (long_ma × z_threshold, optimized for stop_loss):")
            print(pivot_ma_z)
            
            # 2. Show top stop_loss values
            print("\nPerformance by stop_loss value (averaged across all other parameters):")
            by_stop_loss = results_df.groupby('stop_loss')['total_squid_profit'].mean().reset_index()
            by_stop_loss['stop_loss_pct'] = by_stop_loss['stop_loss'] * 100
            by_stop_loss = by_stop_loss.sort_values('total_squid_profit', ascending=False)
            print(by_stop_loss[['stop_loss_pct', 'total_squid_profit']].head(10))
            
            # 3. Find best configuration for different ranges of stop_loss
            # Small stop_loss (0-0.5%)
            small_sl = results_df[results_df['stop_loss'] <= 0.005]
            if not small_sl.empty:
                best_small_sl = small_sl.sort_values('total_squid_profit', ascending=False).iloc[0]
                print(f"\nBest config with small stop_loss (≤0.5%): long_ma={best_small_sl['long_ma']}, z_threshold={best_small_sl['z_threshold']}, stop_loss={best_small_sl['stop_loss']*100:.5f}%")
                print(f"Profit: {best_small_sl['total_squid_profit']}")
            
            # Medium stop_loss (0.5-2%)
            medium_sl = results_df[(results_df['stop_loss'] > 0.005) & (results_df['stop_loss'] <= 0.02)]
            if not medium_sl.empty:
                best_medium_sl = medium_sl.sort_values('total_squid_profit', ascending=False).iloc[0]
                print(f"Best config with medium stop_loss (0.5-2%): long_ma={best_medium_sl['long_ma']}, z_threshold={best_medium_sl['z_threshold']}, stop_loss={best_medium_sl['stop_loss']*100:.5f}%")
                print(f"Profit: {best_medium_sl['total_squid_profit']}")
            
            # Large stop_loss (>2%)
            large_sl = results_df[results_df['stop_loss'] > 0.02]
            if not large_sl.empty:
                best_large_sl = large_sl.sort_values('total_squid_profit', ascending=False).iloc[0]
                print(f"Best config with large stop_loss (>2%): long_ma={best_large_sl['long_ma']}, z_threshold={best_large_sl['z_threshold']}, stop_loss={best_large_sl['stop_loss']*100:.5f}%")
                print(f"Profit: {best_large_sl['total_squid_profit']}")
                
        except Exception as e:
            print(f"\nError creating visualizations: {e}")
        
        # Save results to CSV
        results_df.to_csv('grid_search_results_3d.csv', index=False)
        print("\nComplete results saved to grid_search_results_3d.csv")
    else:
        print("No valid results found after filtering")
else:
    print("No results collected")