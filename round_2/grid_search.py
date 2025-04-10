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
short_ma_values = range(2, 20, 2)  # 5, 8, 11, 14, 17, 20
long_ma_values = range(21, 50, 3)  # 30, 35, 40, 45, 50, 55, 60
z_threshold_values = [round(1.5 + i * 0.25, 2) for i in range(11)]  # 2.0, 2.25, 2.5, ..., 4.5

# Function to test a single parameter combination
def test_parameters(params):
    short_ma, long_ma, z_threshold = params
    
    # Create a modified version of the algorithm with these parameters
    try:
        with open('ryan_r2_v1.py', 'r') as file:
            code = file.read()
        
        # Replace the parameter values
        modified_code = re.sub(
            r'self\.squid_ma_short\s*=\s*\d+', 
            f'self.squid_ma_short = {short_ma}', 
            code
        )
        modified_code = re.sub(
            r'self\.squid_ma_long\s*=\s*\d+', 
            f'self.squid_ma_long = {long_ma}', 
            modified_code
        )
        modified_code = re.sub(
            r'self\.squid_z_threshold\s*=\s*\d+(?:\.\d+)?', 
            f'self.squid_z_threshold = {z_threshold}', 
            modified_code
        )
        
        # Write the modified code to a temporary file - use underscore for decimal
        z_str = str(z_threshold).replace(".", "_")
        temp_filename = f"ryan_r2_v1_temp_{short_ma}_{long_ma}_{z_str}.py"
        with open(temp_filename, 'w') as file:
            file.write(modified_code)
        
        result = {
            'short_ma': int(short_ma),
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
            print(f"Timeout for short_ma={short_ma}, long_ma={long_ma}, z_threshold={z_threshold}")
        except Exception as e:
            print(f"Error running backtest for short_ma={short_ma}, long_ma={long_ma}, z_threshold={z_threshold}: {e}")
            print(f"Command: prosperity3bt {temp_filename} 1")
        
        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except:
            pass
            
        return result
    except Exception as e:
        print(f"Fatal error for short_ma={short_ma}, long_ma={long_ma}, z_threshold={z_threshold}: {e}")
        return {
            'short_ma': int(short_ma),
            'long_ma': int(long_ma),
            'z_threshold': float(z_threshold),
            'squid_profit_day_minus2': 0,
            'squid_profit_day_minus1': 0,
            'squid_profit_day_0': 0,
            'total_squid_profit': 0,
            'total_profit': 0
        }

# Generate all valid parameter combinations
param_combinations = [(short_ma, long_ma, z_threshold) 
                     for short_ma in short_ma_values 
                     for long_ma in long_ma_values 
                     for z_threshold in z_threshold_values
                     if short_ma < long_ma]

total_combinations = len(param_combinations)
print(f"Testing {total_combinations} parameter combinations across 3 dimensions")
print(f"Short MA: {list(short_ma_values)}")
print(f"Long MA: {list(long_ma_values)}")
print(f"Z-threshold: {z_threshold_values}")

# Determine number of workers (use fewer workers to avoid overloading)
num_workers = max(1, int(multiprocessing.cpu_count() * 0.5))
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
                    short_ma, long_ma, z_threshold = params
                    
                    # Update time estimates
                    completed_tasks += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed_tasks
                    estimated_remaining = avg_time_per_task * (total_combinations - completed_tasks)
                    
                    # Format as HH:MM:SS
                    remaining_time = str(timedelta(seconds=int(estimated_remaining)))
                    
                    # Set postfix with detailed progress info
                    pbar.set_postfix({
                        'current': f"({short_ma},{long_ma},{z_threshold})",
                        'squid_profit': result['total_squid_profit'],
                        'ETA': remaining_time
                    })
                    
                    # Print detailed results occasionally
                    if completed_tasks % max(1, num_workers // 2) == 0 or completed_tasks <= 5:
                        print(f"\nResults for short_ma={short_ma}, long_ma={long_ma}, z_threshold={z_threshold}:")
                        print(f"  SQUID_INK day -2: {result['squid_profit_day_minus2']}")
                        print(f"  SQUID_INK day -1: {result['squid_profit_day_minus1']}")
                        print(f"  SQUID_INK day 0: {result['squid_profit_day_0']}")
                        print(f"  Total SQUID_INK: {result['total_squid_profit']}")
                        print(f"  Overall total profit: {result['total_profit']}")
                        
                        # Show current best results
                        if len(results) > 5:
                            # Create a fresh DataFrame with proper numeric types
                            temp_df = pd.DataFrame(results).astype({
                                'short_ma': int,
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
                            print(f"  short_ma={best_so_far['short_ma']}, long_ma={best_so_far['long_ma']}, z_threshold={best_so_far['z_threshold']}")
                            print(f"  Total SQUID_INK profit: {best_so_far['total_squid_profit']}")
            except Exception as e:
                short_ma, long_ma, z_threshold = params
                print(f"\nError processing result for short_ma={short_ma}, long_ma={long_ma}, z_threshold={z_threshold}: {e}")
            
            pbar.update(1)

# Convert results to DataFrame for analysis
if results:
    # Create DataFrame with explicit type conversion
    results_df = pd.DataFrame(results).astype({
        'short_ma': int,
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
        print(top_results[['short_ma', 'long_ma', 'z_threshold', 'squid_profit_day_minus2', 
                          'squid_profit_day_minus1', 'squid_profit_day_0', 
                          'total_squid_profit', 'total_profit']])
        
        # Print the absolute best result
        best_result = best_squid_params.iloc[0]
        print("\n=== OPTIMAL PARAMETERS ===")
        print(f"short_ma = {best_result['short_ma']}, long_ma = {best_result['long_ma']}, z_threshold = {best_result['z_threshold']}")
        print(f"Total SQUID_INK profit: {best_result['total_squid_profit']}")
        print(f"SQUID_INK day -2: {best_result['squid_profit_day_minus2']}")
        print(f"SQUID_INK day -1: {best_result['squid_profit_day_minus1']}")
        print(f"SQUID_INK day 0: {best_result['squid_profit_day_0']}")
        print(f"Overall total profit: {best_result['total_profit']}")
        
        # Create pivot tables for analysis (one for each z-threshold value)
        print("\n=== Generating 2D heatmaps for each z-threshold value ===")
        for z in z_threshold_values:
            z_subset = results_df[results_df['z_threshold'] == z]
            if not z_subset.empty:
                try:
                    pivot = z_subset.pivot(index='short_ma', columns='long_ma', values='total_squid_profit')
                    print(f"\nHeatmap for z_threshold = {z}:")
                    print(pivot)
                except:
                    print(f"\nCouldn't create heatmap for z_threshold = {z}")
        
        # Save results to CSV
        results_df.to_csv('grid_search_results_3d.csv', index=False)
        print("\nComplete results saved to grid_search_results_3d.csv")
        
        # Create a summary table showing the best parameters for each z-threshold
        print("\n=== Best parameters for each z-threshold value ===")
        z_summary = []
        for z in z_threshold_values:
            z_subset = results_df[results_df['z_threshold'] == z]
            if not z_subset.empty:
                best_for_z = z_subset.nlargest(1, 'total_squid_profit').iloc[0]
                z_summary.append({
                    'z_threshold': z,
                    'short_ma': best_for_z['short_ma'],
                    'long_ma': best_for_z['long_ma'],
                    'total_squid_profit': best_for_z['total_squid_profit']
                })
        
        z_summary_df = pd.DataFrame(z_summary)
        print(z_summary_df.sort_values('total_squid_profit', ascending=False))
    else:
        print("No valid results found after filtering")
else:
    print("No results collected")