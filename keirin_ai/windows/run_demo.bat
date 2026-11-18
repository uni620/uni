@echo off
rem === Demo with fake data: learn - backtest - Excel (no internet needed) ===
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -m keirin demo
pause
