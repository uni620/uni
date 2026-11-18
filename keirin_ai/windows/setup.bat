@echo off
rem === First-time setup: creates .venv and installs libraries ===
cd /d "%~dp0.."
where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not installed. Install Python 3.12 from https://www.python.org/ and check "Add python.exe to PATH".
  pause
  exit /b 1
)
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.11 or newer is required.
  pause
  exit /b 1
)
if not exist .venv py -3 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Library install failed. Check your internet connection.
  pause
  exit /b 1
)
if not exist config.toml copy config.example.toml config.toml
echo.
echo Setup finished. Next: double-click windows\run_demo.bat
pause
