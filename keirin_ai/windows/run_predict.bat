@echo off
rem === Predict races starting soon and record virtual bets (run every 20 min) ===
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
.venv\Scripts\python -m keirin predict >> logs\task_predict.txt 2>&1
