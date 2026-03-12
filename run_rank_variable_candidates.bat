@echo off
python "D:\github\misaligned_fits\rank_variable_candidates.py" --base "D:/align" --ref "D:/align/gy3_K043-8.fits" --out-csv "D:/align/output/variable_candidates_rank.csv" --out-png "D:/align/output/variable_candidates_rank.png"
exit /b %ERRORLEVEL%
