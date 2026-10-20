@echo off
title Push CamScanner Pro to GitHub
cd /d "%~dp0"

echo ===================================================
echo   Pushing CamScanner Pro to GitHub for Render...
echo ===================================================

git add .
git commit -m "Deploy CamScanner Pro full-stack to Render"

set /p REPO_URL="Enter your GitHub Repo URL (e.g. https://github.com/zhonghanteo-gif/camscanner-pro.git): "

if "%REPO_URL%"=="" (
    echo No repository URL entered. Exiting.
    pause
    exit /b
)

git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main
git push -u origin main

echo.
echo ===================================================
echo   Code pushed successfully!
echo   Go to https://dashboard.render.com to deploy.
echo ===================================================
pause
