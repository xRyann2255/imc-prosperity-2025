# grid_search_baskets.ps1 -- Grid search for Picnic Basket Spread Strategy parameters.
# This script builds a parameter file with combinations of:
#   DEFAULT_SPREAD_MEAN, STRONG_ENTRY_ZSCORE, WEAK_ENTRY_ZSCORE,
#   EXIT_ZSCORE, SPREAD_STD_WINDOW, STRONG_TARGET, WEAK_TARGET
# It then calls the backtester (prosperity3bt with your experiment file) with these 
# parameters (passed via environment variables).
# The script extracts the overall Total profit from the output and writes a CSV record.

# Define file names.
$RESULTS_FILE = "grid_search_baskets_results.csv"
$PARAMS_FILE = "grid_params_baskets.csv"

# Write CSV header for results.
"DEFAULT_SPREAD_MEAN,STRONG_ENTRY_ZSCORE,WEAK_ENTRY_ZSCORE,EXIT_ZSCORE,SPREAD_STD_WINDOW,STRONG_TARGET,WEAK_TARGET,TOTAL_PROFIT" | Out-File -FilePath $RESULTS_FILE -Encoding ascii

# Remove any previous parameter file.
if (Test-Path $PARAMS_FILE) {
    Remove-Item $PARAMS_FILE -Force
}

# Build the parameter file.
# For PICNIC_BASKET1, we try DEFAULT_SPREAD_MEAN values around its typical value.
# The loops here mirror the Bash nested loops.
foreach ($dsm in 48.76) {
    foreach ($se in 1.6) {
        foreach ($we in 1.8) {
            foreach ($ez in 0.2) {
                foreach ($ssw in 15, 20, 25) {
                    foreach ($st in 60) {
                        foreach ($wt in 30) {
                            "$dsm,$se,$we,$ez,$ssw,$st,$wt" | Add-Content -Path $PARAMS_FILE
                        }
                    }
                }
            }
        }
    }
}

# Define a script block to run one grid job.
# In PowerShell 7 and higher, you can use ForEach-Object -Parallel. 
# Alternatively, you could use Start-Job if you’re using an earlier version.
$runJobScript = {
    param($line)
    # Remove or comment out debug output:
    # $line = $_.Trim()
    $line = $_.Trim()  # Use the incoming pipeline object
    # (Optional) Write-Output "DEBUG: Received line => '$line'"
    $params = $line -split ","
    # (Optional) Write-Output "DEBUG: Params array: $($params -join ';')"
    
    $dsm        = $params[0]
    $se_zscore  = $params[1]
    $we_zscore  = $params[2]
    $ez         = $params[3]
    $ssw        = $params[4]
    $st         = $params[5]
    $wt         = $params[6]

    $env:DEFAULT_SPREAD_MEAN = $dsm
    $env:STRONG_ENTRY_ZSCORE = $se_zscore
    $env:WEAK_ENTRY_ZSCORE   = $we_zscore
    $env:EXIT_ZSCORE         = $ez
    $env:SPREAD_STD_WINDOW   = $ssw
    $env:STRONG_TARGET       = $st
    $env:WEAK_TARGET         = $wt
    $env:SYMBOL              = "PICNIC_BASKET1"
    $env:LIMIT               = "60"

    $fullPath = Join-Path $using:PSScriptRoot "algo.py"
    $OUTPUT = & "prosperity3bt" $fullPath "2" "--merge-pnl" 2>&1

    $profitLine = ($OUTPUT | Select-String -Pattern "Total profit:" | Select-Object -Last 1).Line
    if ($profitLine -match 'Total profit:\s+([\d,\.]+)') {
        $total_profit = $matches[1] -replace ",", ""
    }
    else {
        $total_profit = "NotFound"
    }
    # Return only the final CSV line.
    "$dsm,$se_zscore,$we_zscore,$ez,$ssw,$st,$wt,$total_profit"
}

# Run the grid jobs in parallel.
# Requires PowerShell 7+ for the -Parallel parameter.
# Adjust the ThrottleLimit to match the number of jobs you want to run concurrently.
# $resultLines = Get-Content $PARAMS_FILE | ForEach-Object -Parallel $runJobScript -ThrottleLimit 4
"48.76,1.6,1.8,0.2,15,60,30" | ForEach-Object -Parallel $runJobScript -ThrottleLimit 1

# Append the results to the results file.
$resultLines | Out-File -FilePath $RESULTS_FILE -Append -Encoding ascii

Write-Output "Grid search complete. Results saved to $RESULTS_FILE"
