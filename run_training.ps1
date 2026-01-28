# PowerShell script to run DQN training, PPO training, and evaluation
# Activate virtual environment and run all three training/evaluation scripts

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "UWB NODE SELECTION - TRAINING AND EVALUATION PIPELINE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found at .\.venv" -ForegroundColor Red
    Write-Host "Please create it first with: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Virtual environment activated successfully!" -ForegroundColor Green
Write-Host ""

# Step 1: Train DQN
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "STEP 1: Training DQN Agent" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host ""
python .\src\rl\trainer_dqn.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: DQN training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "✓ DQN training completed successfully!" -ForegroundColor Green
Write-Host ""

# Step 2: Train PPO
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "STEP 2: Training PPO Agent" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host ""
python .\src\rl\trainer_ppo.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PPO training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "✓ PPO training completed successfully!" -ForegroundColor Green
Write-Host ""

# Step 3: Evaluate
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "STEP 3: Evaluating All Methods" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host ""
python .\src\evaluation\evaluate.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "✓ Evaluation completed successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "✓ ALL STEPS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - DQN Model: src\models\dqn_model.pt" -ForegroundColor White
Write-Host "  - PPO Model: src\models\ppo_model.pt" -ForegroundColor White
Write-Host "  - Evaluation: src\evaluation\results\" -ForegroundColor White
Write-Host ""
