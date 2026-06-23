# Stage-1 concurvity lambda sweep for NAM v6 (Windows / PowerShell).
#
# Mirrors scripts/run_concurvity_sweep.sh.  Each lambda writes to its own
# subdirectory; completed runs (metrics_summary.txt present) are skipped.
#
# Usage (from project root):
#   .\scripts\run_concurvity_sweep.ps1

$ErrorActionPreference = "Stop"

$TrainScript = "scripts/train_nam_v6_final.py"
$SweepRoot   = "reports/nam/v6_concurvity_sweep"
$Stage1Seed  = 42
$Lambdas     = @(0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
$DoneFile    = "metrics_summary.txt"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (Test-Path ".venv/Scripts/python.exe") {
    $Python = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
} elseif (Test-Path ".venv/bin/python") {
    $Python = Join-Path $ProjectRoot ".venv/bin/python"
} else {
    $Python = "python"
}

$env:PYTHONIOENCODING = "utf-8"

Write-Host "================================================================"
Write-Host "NAM v6 concurvity sweep - Stage 1"
Write-Host "  trainer : $TrainScript"
Write-Host "  python  : $Python"
Write-Host "  output  : $SweepRoot/"
Write-Host "  seed    : $Stage1Seed"
Write-Host "  lambdas : $($Lambdas -join ' ')"
Write-Host "================================================================"
Write-Host ""

$nRun = 0
$nSkip = 0

foreach ($lam in $Lambdas) {
    $outDir = Join-Path $SweepRoot "lambda_$lam"
    $donePath = Join-Path $outDir $DoneFile
    $logPath = Join-Path $outDir "run.log"

    if (Test-Path $donePath) {
        Write-Host "[skip  ] lambda=$lam - $DoneFile already present in $outDir"
        $nSkip++
        continue
    }

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "[start ] lambda=$lam -> $outDir"
    Write-Host "         log: $logPath"

    & $Python $TrainScript `
        --concurvity_lambda $lam `
        --out_dir $outDir `
        --seed $Stage1Seed `
        2>&1 | Tee-Object -FilePath $logPath

    if (-not (Test-Path $donePath)) {
        Write-Error "[ERROR ] lambda=$lam finished but $DoneFile not found - check $logPath"
    }

    Write-Host "[done  ] lambda=$lam"
    $nRun++
    Write-Host ""
}

Write-Host "================================================================"
Write-Host "Stage 1 complete.  Ran: $nRun  Skipped: $nSkip  Total: $($Lambdas.Count)"
Write-Host "Results in: $SweepRoot/"
Write-Host ""
Write-Host "Next steps:"
Write-Host ""
Write-Host "  1. Inspect the trade-off curve:"
Write-Host "     python scripts/plot_concurvity_tradeoff.py"
Write-Host ""
Write-Host "  2. Choose lambda at the elbow (script prints a recommendation)."
Write-Host ""
Write-Host "  3. Run Stage 2 - 5-seed final training with chosen lambda:"
Write-Host "     python $TrainScript --concurvity_lambda <LAMBDA> --out_dir reports/nam/v6_final_lambda<LAMBDA>"
Write-Host ""
Write-Host "  4. Inspect shape functions:"
Write-Host "     python scripts/inspect_shape_functions_v6_final.py"
Write-Host "================================================================"
