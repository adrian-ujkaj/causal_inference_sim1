@echo off
REM Full causal study with the project's Python (.conda, created by setup_windows.bat).
REM   Double-click                      : full study (24 flights x 3 campaigns, 1 to 3 h)
REM   etude_causale.bat --runs 6        : quick version
REM   etude_causale.bat --skip_sim      : analyses only, on flights already simulated
REM No %%PY%% inside a parenthesised block: a ")" in the path would break the block.
cd /d "%~dp0..\.."
set "PY=%CD%\.conda\python.exe"
if not exist "%PY%" goto :nopython
if not exist runs mkdir runs
set "LOG=runs\etude_causale.log"
echo Journal : %LOG%> "%LOG%"

echo === 1/3 Bibliotheques de l'analyse causale ===
"%PY%" -c "import torch, networkx, statsmodels, sklearn" 2>nul
if not errorlevel 1 goto :tests
echo Installation de torch, networkx, statsmodels, scikit-learn (quelques minutes)...
"%PY%" -m pip install torch networkx statsmodels scikit-learn >> "%LOG%" 2>&1
if errorlevel 1 goto :fail

:tests
echo === 2/3 Tests ===
"%PY%" -m pytest tests\test_causal.py -q >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
echo Tests OK

echo === 3/3 Etude causale ===
"%PY%" analysis\run_causal_study.py %* >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
type "%LOG%"
echo.
echo Termine. Les 3 figures a regarder sont dans runs\causal\RESULTATS
pause
exit /b 0

:nopython
echo Python du projet introuvable : lance d'abord setup_windows.bat
pause
exit /b 1

:fail
type "%LOG%"
echo.
echo ECHEC, voir le journal ci-dessus (%LOG%)
pause
exit /b 1
