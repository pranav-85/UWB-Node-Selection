@echo off
REM Batch script to run DQN training, PPO training, and evaluation
REM Activate virtual environment and run all three training/evaluation scripts

setlocal enabledelayedexpansion

echo ======================================================================
echo UWB NODE SELECTION - TRAINING AND EVALUATION PIPELINE
echo ======================================================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at .venv
    echo Please create it first with: python -m venv .venv
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Virtual environment activated successfully!
echo.

REM Step 1: Train DQN
echo ======================================================================
echo STEP 1: Training DQN Agent
echo ======================================================================
echo.
python .\src\rl\trainer_dqn.py
if %errorlevel% neq 0 (
    echo ERROR: DQN training failed with exit code %errorlevel%
    exit /b 1
)
echo.
echo [OK] DQN training completed successfully!
echo.

REM Step 2: Train PPO
echo ======================================================================
echo STEP 2: Training PPO Agent
echo ======================================================================
echo.
python .\src\rl\trainer_ppo.py
if %errorlevel% neq 0 (
    echo ERROR: PPO training failed with exit code %errorlevel%
    exit /b 1
)
echo.
echo [OK] PPO training completed successfully!
echo.

REM Step 3: Evaluate
echo ======================================================================
echo STEP 3: Evaluating All Methods
echo ======================================================================
echo.
python .\src\evaluation\evaluate.py
if %errorlevel% neq 0 (
    echo ERROR: Evaluation failed with exit code %errorlevel%
    exit /b 1
)
echo.
echo [OK] Evaluation completed successfully!
echo.

echo ======================================================================
echo [OK] ALL STEPS COMPLETED SUCCESSFULLY!
echo ======================================================================
echo.
echo Results saved to:
echo   - DQN Model: src\models\dqn_model.pt
echo   - PPO Model: src\models\ppo_model.pt
echo   - Evaluation: src\evaluation\results\
echo.
pause
