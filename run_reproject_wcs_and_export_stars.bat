@echo off
python "D:\github\misaligned_fits\reproject_wcs_and_export_stars.py" --a "D:/missalign/gy2_K043-7.fits" --b "D:/missalign/GY2_K043-7_UTC20260310_163625_-25C_.fit" --out-fits "D:/missalign/GY2_K043-7_UTC20260310_163625_-25C_.rp.fit" --out-stars "D:/missalign/GY2_K043-7_UTC20260310_163625_-25C_.stars.npz" --median-size "3" --max-stars "5000"
exit /b %ERRORLEVEL%
