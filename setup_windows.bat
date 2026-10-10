@echo off
REM ==================================================================
REM  Installs everything needed to run main.py on Windows,
REM  then starts the simulation.
REM  - Miniforge (conda) if missing -> pybullet has no prebuilt
REM    Windows build on pip, so it is taken from conda-forge
REM  - Local Python environment in .conda\ (in this folder)
REM  - Dependencies + gym-pybullet-drones + causal analysis and tests
REM ==================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "ENV=%~dp0.conda"
set PYTHONUTF8=1

REM ---------- 1. Find conda (or install Miniforge) ----------
call :find_conda
if not defined CONDA (
    echo [1/4] Installation de Miniforge via winget...
    winget install -e --id CondaForge.Miniforge3 --scope user --accept-package-agreements --accept-source-agreements
    call :find_conda
)
if not defined CONDA (
    echo.
    echo ERREUR : conda introuvable. Installe Miniforge depuis
    echo https://github.com/conda-forge/miniforge puis relance ce script.
    pause
    exit /b 1
)
echo conda : %CONDA%

REM ---------- 2. Create the environment ----------
if not exist "%ENV%\python.exe" (
    echo [2/4] Creation de l'environnement Python dans .conda ...
    "%CONDA%" create -y -p "%ENV%" -c conda-forge --override-channels ^
        python=3.12 pybullet numpy scipy pyzmq pyyaml gymnasium matplotlib pandas
    if errorlevel 1 goto :fail
) else (
    echo [2/4] Environnement .conda deja present.
)

set "PATH=%ENV%;%ENV%\Library\bin;%ENV%\Scripts;%PATH%"

REM ---------- 3. pip packages ----------
echo [3/4] Installation de pathfinding, gym-pybullet-drones et des outils d'analyse...
"%ENV%\python.exe" -m pip install --upgrade pathfinding
if errorlevel 1 goto :fail
"%ENV%\python.exe" -m pip install --no-deps https://github.com/utiasDSL/gym-pybullet-drones/archive/7ebad1ecabd28a7000add2d05f888aa2e837c2cc.zip
if errorlevel 1 goto :fail
echo      Bibliotheques de l'analyse causale et des tests (torch, statsmodels...)
"%ENV%\python.exe" -m pip install torch networkx statsmodels scikit-learn pytest
if errorlevel 1 goto :fail

"%ENV%\python.exe" -c "import pybullet, zmq, yaml, pathfinding, scipy, gymnasium; from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl; print('Toutes les dependances sont OK')"
if errorlevel 1 goto :fail

REM ---------- 4. Start the simulation ----------
echo [4/4] Lancement de main.py ...
"%ENV%\python.exe" main.py
echo.
pause
exit /b 0

:find_conda
set "CONDA="
for %%P in ("%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniforge3" "%ProgramData%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3") do (
    if not defined CONDA if exist "%%~P\Scripts\conda.exe" set "CONDA=%%~P\Scripts\conda.exe"
)
if not defined CONDA for /f "delims=" %%C in ('where conda.exe 2^>nul') do if not defined CONDA set "CONDA=%%C"
exit /b 0

:fail
echo.
echo ERREUR pendant l'installation (voir les messages ci-dessus).
pause
exit /b 1
