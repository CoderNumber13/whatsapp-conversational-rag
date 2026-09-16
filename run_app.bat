@echo off
REM Launch the Conversation Memory demo with the correct interpreter.
REM
REM Why this file exists: Anaconda's BASE environment has streamlit but not
REM faiss, and base is Python 3.14 which has no faiss wheel at all. So running
REM "streamlit run app.py" without activating convmem starts the app happily and
REM then fails at ingestion with ModuleNotFoundError: No module named 'faiss'.
REM Naming the environment's python directly removes that whole class of
REM mistake -- there is nothing to remember and nothing to activate.
REM
REM Usage: double-click, or from cmd:  run_app.bat

setlocal

REM Override by setting CONVMEM_PYTHON before calling, e.g. for a different
REM machine or a renamed environment.
REM Find the environment's interpreter. Order: an explicit override, then
REM conda's own resolution, then the common install locations. Nothing is
REM hardcoded to one machine's drive letter.
if not "%CONVMEM_PYTHON%"=="" goto :have_python

for /f "delims=" %%i in ('conda run -n convmem python -c "import sys;print(sys.executable)" 2^>nul') do set "CONVMEM_PYTHON=%%i"
if not "%CONVMEM_PYTHON%"=="" goto :have_python

for %%R in ("%CONDA_PREFIX%\..\.." "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\Miniforge3" "C:\ProgramData\anaconda3" "D:\anaconda3") do (
    if exist "%%~R\envs\convmem\python.exe" (
        set "CONVMEM_PYTHON=%%~R\envs\convmem\python.exe"
        goto :have_python
    )
)

:have_python

if not exist "%CONVMEM_PYTHON%" (
    echo.
    echo   ERROR: interpreter not found:
    echo     %CONVMEM_PYTHON%
    echo.
    echo   Create the environment first:
    echo     conda create -n convmem python=3.12 -y
    echo     conda activate convmem
    echo     pip install -r requirements.txt
    echo.
    echo   Or point this script at an existing one:
    echo     set CONVMEM_PYTHON=C:\path\to\env\python.exe
    echo.
    echo   (searched: CONVMEM_PYTHON, conda run, and the usual install paths)
    echo.
    exit /b 1
)

REM cd /d because the project and Anaconda may sit on different drives.
cd /d "%~dp0"

"%CONVMEM_PYTHON%" -c "import faiss, torch, streamlit" 2>nul
if errorlevel 1 (
    echo.
    echo   ERROR: %CONVMEM_PYTHON%
    echo   is missing faiss, torch or streamlit.
    echo.
    echo   Install them:
    echo     "%CONVMEM_PYTHON%" -m pip install -r requirements.txt
    echo.
    exit /b 1
)

echo Starting Conversation Memory  (Ctrl+C to stop)
echo   interpreter: %CONVMEM_PYTHON%
echo.
"%CONVMEM_PYTHON%" -m streamlit run app.py %*
