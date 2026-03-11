@echo off
python "D:\github\misaligned_fits\render_alignment_outputs.py" --a "%~1" --b "%~2" --align "%~3" --outdir "%~4"
exit /b %ERRORLEVEL%
