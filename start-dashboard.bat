@echo off
REM AegisMind: double-click to start the dashboard (Windows).
REM Uses the project's .venv, builds the frontend the first time, picks a free port and opens the browser.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [AegisMind] No .venv found. Run these once in PowerShell:
  echo   py -3.12 -m venv .venv
  echo   .venv\Scripts\Activate.ps1
  echo   pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist "frontend\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [AegisMind] Dashboard not built and Node.js is missing. Install Node LTS from https://nodejs.org
    pause
    exit /b 1
  )
  echo [AegisMind] Building the dashboard the first time...
  pushd frontend
  call npm install || (popd & pause & exit /b 1)
  call npm run build || (popd & pause & exit /b 1)
  popd
)

set PORT=8000
:findport
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
  set /a PORT=%PORT%+1
  goto findport
)

echo [AegisMind] Starting on http://localhost:%PORT%  (close this window or press Ctrl+C to stop)
start "" "http://localhost:%PORT%"
".venv\Scripts\python.exe" -m aegismind.cli serve --port %PORT%
pause
