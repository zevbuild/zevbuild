@echo off
title Kalyan Matka - Update Predictions & Push to Web
echo ======================================================================
echo   Updating Kalyan Matka predictions and deploying to web...
echo ======================================================================
cd /d "%~dp0"
python sync_and_push.py
echo.
echo ======================================================================
echo Process finished. Press any key to exit.
pause >nul
