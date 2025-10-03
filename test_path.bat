@echo off
echo Script location: %~dp0
echo Current directory before cd: %CD%
cd /d "%~dp0frontend"
echo Current directory after cd: %CD%
echo Checking if package.json exists:
if exist "package.json" (
    echo ✓ package.json found
) else (
    echo ❌ package.json NOT found
)
pause