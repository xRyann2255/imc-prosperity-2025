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

# Define parameter ranges to search
long_ma_values = range(1, 51, 1)  # 1 to 50 with step of 1
z_threshold_values = [round(1.0 + i * 0.1, 1) for i in range(21)]  # 1.0 to 3.0 with step of 0.1

# Function to test a single parameter combination
def test_parameters(params):
    long_ma, z_threshold = params
    
    # Create a modified version of the algorithm with these parameters
    try:
        with open('ryan_r2_v2.py', 'r') as file:
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
        
        # Write the modified code to a temporary file - use underscore for decimal
        z_str = str(z_threshold).replace(".", "_")
        temp_filename = f"paste_temp_{long_ma}_{z_str}.py"
        with open(temp_filename, 'w') as file:
            file.write(modified_code)
        
        result = {
            'long_ma': int(long_ma),
            'z_threshold': float(z_threshold),
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
            print(f"Timeout for long_ma={long_ma}, z_threshold={z_threshold}")
        except Exception as e:
            print(f"Error running backtest for long_ma={long_ma}, z_threshold={z_threshold}: {e}")
            print(f"Command: prosperity3bt {temp_filename} 1")
        
        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except:
            pass
            
        return result
    except Exception as e:
        print(f"Fatal error for long_ma={long_ma}, z_threshold={z_threshold}: {e}")
        return {
            'long_ma': int(long_ma),
            'z_threshold': float(z_threshold),
            'squid_profit_day_minus2': 0,
            'squid_profit_day_minus1': 0,
            'squid_profit_day_0': 0,
            'total_squid_profit': 0,
            'total_profit': 0
        }

# Generate all valid parameter combinations
param_combinations = [(long_ma, z_threshold) 
                     for long_ma in long_ma_values 
                     for z_threshold in z_threshold_values]

total_combinations = len(param_combinations)
print(f"Testing {total_combinations} parameter combinations across 2 dimensions")
print(f"Long MA: {min(long_ma_values)}-{max(long_ma_values)} (step 1)")
print(f"Z-threshold: {min(z_threshold_values)}-{max(z_threshold_values)} (step 0.1)")

# Determine number of workers (use fewer workers to avoid overloading)
num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
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
                    long_ma, z_threshold = params
                    
                    # Update time estimates
                    completed_tasks += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed_tasks
                    estimated_remaining = avg_time_per_task * (total_combinations - completed_tasks)
                    
                    # Format as HH:MM:SS
                    remaining_time = str(timedelta(seconds=int(estimated_remaining)))
                    
                    # Set postfix with detailed progress info
                    pbar.set_postfix({
                        'current': f"({long_ma},{z_threshold})",
                        'squid_profit': result['total_squid_profit'],
                        'ETA': remaining_time
                    })
                    
                    # Print detailed results occasionally
                    if completed_tasks % max(1, num_workers // 2) == 0 or completed_tasks <= 5:
                        print(f"\nResults for long_ma={long_ma}, z_threshold={z_threshold}:")
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
                                'squid_profit_day_minus2': int,
                                'squid_profit_day_minus1': int,
                                'squid_profit_day_0': int,
                                'total_squid_profit': int,
                                'total_profit': int
                            })
                            
                            # Now find the best result so far
                            best_so_far = temp_df.nlargest(1, 'total_squid_profit').iloc[0]
                            print(f"\nBest configuration so far:")
                            print(f"  long_ma={best_so_far['long_ma']}, z_threshold={best_so_far['z_threshold']}")
                            print(f"  Total SQUID_INK profit: {best_so_far['total_squid_profit']}")
            except Exception as e:
                long_ma, z_threshold = params
                print(f"\nError processing result for long_ma={long_ma}, z_threshold={z_threshold}: {e}")
            
            pbar.update(1)

# Convert results to DataFrame for analysis
if results:
    # Create DataFrame with explicit type conversion
    results_df = pd.DataFrame(results).astype({
        'long_ma': int,
        'z_threshold': float,
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
        print(top_results[['long_ma', 'z_threshold', 'squid_profit_day_minus2', 
                          'squid_profit_day_minus1', 'squid_profit_day_0', 
                          'total_squid_profit', 'total_profit']])
        
        # Print the absolute best result
        best_result = best_squid_params.iloc[0]
        print("\n=== OPTIMAL PARAMETERS ===")
        print(f"long_ma = {best_result['long_ma']}, z_threshold = {best_result['z_threshold']}")
        print(f"Total SQUID_INK profit: {best_result['total_squid_profit']}")
        print(f"SQUID_INK day -2: {best_result['squid_profit_day_minus2']}")
        print(f"SQUID_INK day -1: {best_result['squid_profit_day_minus1']}")
        print(f"SQUID_INK day 0: {best_result['squid_profit_day_0']}")
        print(f"Overall total profit: {best_result['total_profit']}")
        
        # Create a heatmap visualization
        try:
            pivot_table = results_df.pivot(index='long_ma', columns='z_threshold', values='total_squid_profit')
            print("\nTotal SQUID_INK profit heatmap (long_ma × z_threshold):")
            print(pivot_table)
        except:
            print("\nCouldn't create heatmap due to missing values")
        
        # Save results to CSV
        results_df.to_csv('grid_search_results_2d.csv', index=False)
        print("\nComplete results saved to grid_search_results_2d.csv")
    else:
        print("No valid results found after filtering")
else:
    print("No results collected")