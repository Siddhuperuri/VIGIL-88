@echo off
REM ══════════════════════════════════════════════════════════
REM  VIGIL-88 AI  —  Windows Build Script
REM  Builds a standalone .exe using PyInstaller
REM ══════════════════════════════════════════════════════════

echo.
echo  ██████╗ ██╗   ██╗██╗██╗      ██████╗
echo  ██╔══██╗██║   ██║██║██║      ██╔══██╗
echo  ██████╔╝██║   ██║██║██║      ██║  ██║
echo  ██╔══██╗██║   ██║██║██║      ██║  ██║
echo  ██████╔╝╚██████╔╝██║███████╗ ██████╔╝
echo.
echo  VIGIL-88 AI  v4.0  ^|  Build Script
echo.

REM — Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH. Install from python.org
    pause
    exit /b 1
)

REM — Check pip
pip --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip not found.
    pause
    exit /b 1
)

echo [1/5] Installing dependencies...
pip install -r requirements.txt --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency install failed.
    pause
    exit /b 1
)

echo [2/5] Installing PyInstaller...
pip install pyinstaller>=6.0 --quiet

echo [3/5] Creating required directories...
if not exist models mkdir models
if not exist assets mkdir assets
if not exist dataset\fire mkdir dataset\fire
if not exist dataset\accident mkdir dataset\accident
if not exist dataset\normal mkdir dataset\normal
if not exist logs mkdir logs

echo [4/5] Running PyInstaller...
pyinstaller urban_safety.spec --noconfirm --clean
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyInstaller build failed. Check the output above.
    pause
    exit /b 1
)

echo [5/5] Build complete!
echo.
echo  Executable location:
echo    dist\Vigil88\Vigil88.exe
echo.
echo  To run the app now:
echo    dist\Vigil88\Vigil88.exe
echo.
echo  NOTE: Copy your trained model (.pt) into the models\ folder
echo        before distributing.
echo.
pause
