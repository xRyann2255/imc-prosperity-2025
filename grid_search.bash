#!/bin/bash
# grid_search_baskets.sh -- Grid search for Picnic Basket Spread Strategy parameters.
# This script builds a parameter file with combinations of:
#   default_spread_mean, strong_entry_zscore, weak_entry_zscore,
#   exit_zscore, spread_std_window, strong_target, weak_target
# It then calls the backtester (prosperity3bt with your experiment file) with these parameters 
# passed via environment variables.
# The script extracts the overall Total profit from the output and writes a CSV record.

RESULTS_FILE="grid_search_baskets_results.csv"
PARAMS_FILE="grid_params_baskets.csv"

# Write CSV header for results.
echo "default_spread_mean,strong_entry_zscore,weak_entry_zscore,exit_zscore,spread_std_window,strong_target,weak_target,total_profit" > $RESULTS_FILE

# Remove any previous parameter file.
rm -f $PARAMS_FILE

# Build the parameter file.
# For PICNIC_BASKET1, we try default_spread_mean values around its typical value.
for dsm in 48.76; do
  for se in 1.6; do
    for we in 1.8; do
      for ez in 0.2; do
        for ssw in 15 20 25; do
          for st in 60; do
            for wt in 30; do
              echo "$dsm,$se,$we,$ez,$ssw,$st,$wt" >> $PARAMS_FILE
            done
          done
        done
      done
    done
  done
done

# Define a function that runs one grid job.
run_job() {
  Write-Output "Running grid job with parameters: $line"
  Write-Output "Environment variables set: DEFAULT_SPREAD_MEAN=$env:DEFAULT_SPREAD_MEAN, STRONG_ENTRY_ZSCORE=$env:STRONG_ENTRY_ZSCORE, ..."
  Write-Output "Backtester OUTPUT: $OUTPUT"

  # Read the comma-delimited parameters into variables.
  IFS=',' read -r dsm se_zscore we_zscore ez ssw st wt <<< "$1"
  # Export the parameters as environment variables.
  export DEFAULT_SPREAD_MEAN=$dsm
  export STRONG_ENTRY_ZSCORE=$se_zscore
  export WEAK_ENTRY_ZSCORE=$we_zscore
  export EXIT_ZSCORE=$ez
  export SPREAD_STD_WINDOW=$ssw
  export STRONG_TARGET=$st
  export WEAK_TARGET=$wt
  export SYMBOL="PICNIC_BASKET1"   # Change this to PICNIC_BASKET2 for basket 2 grid search.
  export LIMIT=60                # For PICNIC_BASKET2, set LIMIT accordingly (e.g. 100).
  
  # Run the backtester using your experiment file.
  OUTPUT=$(prosperity3bt C:\\Users\\Jackson\\Desktop\\IMC Prosperity\\imc-prosperity-2025\\algo.py 2 --merge-pnl)
  # Extract the overall Total profit from the output.
  total_profit=$(echo "$OUTPUT" | grep "Total profit:" | tail -n 1 | awk '{gsub(/,/, "", $3); print $3}')
  echo "$dsm,$se_zscore,$we_zscore,$ez,$ssw,$st,$wt,$total_profit"
}

export -f run_job

# Run the grid jobs in parallel (adjust -j as appropriate for your machine).
parallel -a $PARAMS_FILE -j 4 --no-notice run_job {} >> $RESULTS_FILE

echo "Grid search complete. Results saved to $RESULTS_FILE"