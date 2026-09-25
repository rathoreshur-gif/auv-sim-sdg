@echo off
rem SDG App: recipe editor, render control and training launcher for the synthetic data pipeline.
rem Double-click to start. The app opens in its own window; close THIS console to stop the app.
rem Python: %SDG_PYTHON% if set, else SegPipeline\.venv, else %USERPROFILE%\seg-train\.venv, else py -3.
title SDG App - close this window to stop the app
cd /d "%~dp0"
set "PY=%SDG_PYTHON%"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\seg-train\.venv\Scripts\python.exe" set "PY=%USERPROFILE%\seg-train\.venv\Scripts\python.exe"
if defined PY (
  "%PY%" "%~dp0app\server.py" %*
) else (
  py -3 "%~dp0app\server.py" %*
)
if errorlevel 1 pause
