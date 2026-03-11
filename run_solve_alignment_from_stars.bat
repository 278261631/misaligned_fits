@echo off
python "D:\github\misaligned_fits\solve_alignment_from_stars.py" --a-stars "%~1" --b-stars "%~2" --out "%~3"
exit /b %ERRORLEVEL%
