@echo off
python "D:\github\misaligned_fits\render_alignment_outputs.py" --a "D:/align/gy3_K043-8.fits" --b "D:/align/GY3_K043-8_UTC20260310_163817_-25C_.fit" --align "D:/align/GY3_K043-8_UTC20260310_163817_-25C_.align.npz" --outdir "D:/align/output/"
exit /b %ERRORLEVEL%
