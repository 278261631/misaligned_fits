@echo off
python "D:\github\misaligned_fits\direct_fit_no_wcs.py" --base "%~1" --batch --pattern "*.fit" "*.fits" "*.FIT" "*.FITS"
exit /b %ERRORLEVEL%
