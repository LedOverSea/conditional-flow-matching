@echo off
echo Starting all experiments...
echo.

echo Running eb_pca experiments...
call ".\scripts\batch_experiments_eb_pca.bat"

echo.
echo Running eb_phate experiments...
call ".\scripts\batch_experiments_eb_phate.bat"

echo.
echo Running multi experiments...
call ".\scripts\batch_experiments_multi.bat"

echo.
echo Running cite experiments...
call ".\scripts\batch_experiments_cite.bat"

echo.
echo All experiments completed!
pause