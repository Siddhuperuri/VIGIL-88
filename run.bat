@echo off
REM Quick launch for development — no build needed
echo  Starting Urban Safety AI...
python main_app.py
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] Launch failed. Check:
    echo    1. Python installed
    echo    2. Run: pip install -r requirements.txt
    echo    3. For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pause
)
