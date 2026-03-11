@echo off
setlocal

if "%~3"=="" (
    echo Usage: %~nx0 ^<a_stars_npz^> ^<b_stars_npz^> ^<out_align_npz^>
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%solve_alignment_from_stars.py" --a-stars "%~1" --b-stars "%~2" --out "%~3"
exit /b %ERRORLEVEL%
