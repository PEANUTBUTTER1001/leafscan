@echo off
chcp 949 >nul
cd /d "%~dp0"
title LeafScan Lab

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "lab_gui.py"
    exit /b 0
)

where python >nul 2>nul
if %errorlevel%==0 (
    start "" python "lab_gui.py"
    exit /b 0
)

echo.
echo  [오류] 파이썬을 찾을 수 없습니다.
echo  python.org 에서 설치할 때 Add Python to PATH 를 체크해야 합니다.
echo.
pause
