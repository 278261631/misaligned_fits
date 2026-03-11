@echo off
python "D:\github\misaligned_fits\solve_alignment_from_stars.py" --a-stars "D:/github/test_flow_data/stars/K024-6.stars.npz" --b-stars "D:/github/test_flow_data/stars/GY1_K024-6_NoFilter_60S_Bin2_UTC20260303_162922_-29.9C_stars.npz" --out "D:/github/test_flow_data/stars/GY1_K024-6_NoFilter_60S_Bin2_UTC20260303_162922_-29.9C_align.npz"
exit /b %ERRORLEVEL%
