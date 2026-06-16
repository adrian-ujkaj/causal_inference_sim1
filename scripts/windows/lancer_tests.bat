@echo off
REM Runs the tests with the project's Python (.conda). Log: runs\tests.log
cd /d "%~dp0..\.."
if not exist runs mkdir runs
"%CD%\.conda\python.exe" -m pytest -q > runs\tests.log 2>&1
type runs\tests.log
pause
