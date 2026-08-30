@echo off
setlocal
set "PROJ=%~dp0.."
for %%I in ("%PROJ%") do set "PROJ=%%~fI"
for %%I in ("%PROJ%\..") do set "ROOT=%%~fI"
set "BIN=%ROOT%\cmake-build-debug\bin"
if not exist "%BIN%\ddscat_cuda.exe" (echo ERROR: %BIN%\ddscat_cuda.exe not found.& exit /b 1)
if not exist "%BIN%\ddscat_matvec_cuda.dll" (echo ERROR: CUDA DLL not found in %BIN%.& exit /b 2)
copy /Y "%PROJ%\tests\ddscat_qmrccg.par" "%BIN%\ddscat.par" >nul
cd /d "%BIN%"
set "DDSCAT_CUDA_DLL=%BIN%\ddscat_matvec_cuda.dll"
ddscat_cuda.exe
endlocal
