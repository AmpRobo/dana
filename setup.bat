@echo off
set VENV_DIR=.venv

:: Find Python 3.11+
set PYTHON_CMD=
py -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
) else (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON_CMD=python
    )
)

if "%PYTHON_CMD%"=="" (
    echo Error: Python ^>= 3.11 is required. Please install Python 3.11+ from https://www.python.org/
    exit /b 1
)

:: Check if venv exists and is valid
set RECREATE=0
if not exist "%VENV_DIR%\Scripts\python.exe" (
    set RECREATE=1
) else (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
    if errorlevel 1 set RECREATE=1
)

if %RECREATE%==1 (
    echo Creating virtual environment with %PYTHON_CMD%...
    if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
    %PYTHON_CMD% -m venv "%VENV_DIR%"
) else (
    echo Virtual environment already exists.
)

call "%VENV_DIR%\Scripts\activate.bat"

echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -e .

:: On native Windows, JAX GPU wheels do not exist.
:: Only CPU mode is supported natively on Windows.
echo Installing native Windows JAX (CPU mode)...
python -m pip install -U jax jaxlib

:: Kokoro TTS, used by scripts\voice_finetune.py to read voice commands aloud. Installs the
:: kokoro-onnx package and downloads the full-precision model (~325 MB) and voices (~28 MB) into
:: checkpoints\kokoro\ unless they are already there. set DANA_KOKORO=0 skips this.
set KOKORO_DIR=checkpoints\kokoro
set KOKORO_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
if not "%DANA_KOKORO%"=="0" (
    echo Installing Kokoro TTS...
    python -m pip install -e ".[voice]"
    if not exist "%KOKORO_DIR%" mkdir "%KOKORO_DIR%"
    for %%F in (kokoro-v1.0.onnx voices-v1.0.bin) do (
        if exist "%KOKORO_DIR%\%%F" (
            echo   %KOKORO_DIR%\%%F already exists.
        ) else (
            echo   Downloading %%F...
            curl.exe -fL -o "%KOKORO_DIR%\%%F.part" "%KOKORO_URL%/%%F" && move /y "%KOKORO_DIR%\%%F.part" "%KOKORO_DIR%\%%F" >nul || (
                if exist "%KOKORO_DIR%\%%F.part" del "%KOKORO_DIR%\%%F.part"
                echo   Warning: could not download %%F; get it from %KOKORO_URL%/%%F
            )
        )
    )
)

:: Sanity check
python -c "import jax; print('JAX devices:', jax.devices())" || echo Warning: jax import failed; check the install above.

:: Prompt for W&B key
if "%WANDB_API_KEY%"=="" (
    echo.
    echo   Enter your W^&B API key ^(https://wandb.ai/authorize^)
    echo   Press Enter to skip.
    set /p WANDB_API_KEY="  > "
)

dana --help