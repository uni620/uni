@echo off
rem === Remove the automatic runs ===
schtasks /Delete /F /TN "KeirinAI\predict"
schtasks /Delete /F /TN "KeirinAI\daily"
echo Removed.
pause
