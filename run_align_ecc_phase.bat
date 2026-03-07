@echo off
setlocal

set "BASE=E:\github\test_align"
if not "%~1"=="" (
    set "BASE=%~1"
)

python "E:\github\misaligned_fits\align_ecc_phase.py" --base "%BASE%" --batch --pattern "*.fit" "*.fits" "*.FIT" "*.FITS"
exit /b %ERRORLEVEL%
