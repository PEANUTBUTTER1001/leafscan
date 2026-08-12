@echo off
chcp 949 >nul
cd /d "%~dp0"
title LeafScan Lab - 바탕화면 아이콘 만들기

echo.
echo  바탕화면에 LeafScan Lab 아이콘을 만듭니다...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcut.ps1"

echo.
pause
