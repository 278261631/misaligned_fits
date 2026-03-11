@echo off
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 ^<fits_path^> [out_stars_npz]
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "FITS_PATH=%~1"

if "%~2"=="" (
    python "%SCRIPT_DIR%export_fits_stars.py" --fits "%FITS_PATH%"
) else (
    python "%SCRIPT_DIR%export_fits_stars.py" --fits "%FITS_PATH%" --out "%~2"
)

exit /b %ERRORLEVEL%
