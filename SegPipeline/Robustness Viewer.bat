@echo off
rem Robustness viewer: a .pt model + an image, live predictions while you distort the image with sliders.
rem Double-click to start; it opens in the browser. Close THIS console to stop it.
rem Drop a .pt and/or an image onto this .bat to preload them.
rem Python (needs ultralytics): %SDG_PYTHON% if set, else SegPipeline\.venv, else %USERPROFILE%\seg-train\.venv, else py -3.
title Robustness Viewer - close this window to stop it
cd /d "%~dp0"
set "PY=%SDG_PYTHON%"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\seg-train\.venv\Scripts\python.exe" set "PY=%USERPROFILE%\seg-train\.venv\Scripts\python.exe"

set "MODEL="
set "IMAGE="
:args
if "%~1"=="" goto run
if /i "%~x1"==".pt" (set "MODEL=%~f1") else (set "IMAGE=%~f1")
shift
goto args

:run
set "ARGS="
if defined MODEL set ARGS=--model "%MODEL%"
if defined IMAGE set ARGS=%ARGS% --image "%IMAGE%"
if defined PY (
  "%PY%" "%~dp0tools\robustness_viewer.py" %ARGS%
) else (
  py -3 "%~dp0tools\robustness_viewer.py" %ARGS%
)
if errorlevel 1 pause
