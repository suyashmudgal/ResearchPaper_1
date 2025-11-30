Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   SASKC Research Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Install dependencies
Write-Host "[1/4] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Run experiments
Write-Host "[2/4] Running experiments (100 cycles)..." -ForegroundColor Yellow
python src/experiment_runs.py

# Train final model
Write-Host "[3/4] Training final SASKC model..." -ForegroundColor Yellow
python src/train_best_model.py

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the Dashboard:" -ForegroundColor Green
Write-Host "    streamlit run dashboard/streamlit_app.py"
Write-Host ""
Write-Host "To run the API:" -ForegroundColor Green
Write-Host "    python src/api/flask_api.py"
Write-Host ""
