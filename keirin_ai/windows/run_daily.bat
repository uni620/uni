@echo off
rem === Results - settle - retrain - update Excel (run once a day) ===
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
.venv\Scripts\python -m keirin daily >> logs\task_daily.txt 2>&1
