@echo off
setlocal

if "%~3"=="" (
    echo Usage: %~nx0 ^<a_fits^> ^<b_fits^> ^<align_npz^> [outdir]
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
if "%~4"=="" (
    python "%SCRIPT_DIR%render_alignment_outputs.py" --a "%~1" --b "%~2" --align "%~3"
) else (
    python "%SCRIPT_DIR%render_alignment_outputs.py" --a "%~1" --b "%~2" --align "%~3" --outdir "%~4"
)
exit /b %ERRORLEVEL%
