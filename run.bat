@echo off
title CamScanner Pro Server
cd /d "%~dp0"

echo ===================================================
echo   Starting CamScanner Pro Full-Stack Server...
echo ===================================================

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python app.py
pause
