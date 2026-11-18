@echo off
rem === Register automatic runs in Windows Task Scheduler ===
rem   predict : every 20 minutes from 08:00 to 23:30
rem   daily   : every day at 23:50
cd /d "%~dp0.."
if not exist logs mkdir logs
schtasks /Create /F /TN "KeirinAI\predict" /SC MINUTE /MO 20 /ST 08:00 /DU 15:30 /TR "\"%~dp0run_predict.bat\""
schtasks /Create /F /TN "KeirinAI\daily" /SC DAILY /ST 23:50 /TR "\"%~dp0run_daily.bat\""
echo.
echo Registered. The PC must be ON and you must be signed in at those times.
echo To check: open "Task Scheduler" and look at the KeirinAI folder.
pause
