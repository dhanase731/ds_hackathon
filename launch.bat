@echo off
echo ============================================================
echo  StockSense - IntelliData 2026
echo ============================================================

cd /d "%~dp0"

echo.
echo [Step 1] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [Step 2] Running ML pipeline (feature engineering + models)...
python run_pipeline.py

echo.
echo [Step 3] Launching dashboard...
streamlit run dashboard/app.py
