@echo off
title Kalyan Realtime Auto-Predictor Web Server
echo Starting Kalyan Web Predictor...
start http://127.0.0.1:8080
python app.py --port 8080
pause
