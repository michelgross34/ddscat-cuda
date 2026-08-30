@echo off
setlocal
set "PROJ=%~dp0.."
for %%I in ("%PROJ%") do set "PROJ=%%~fI"
for %%I in ("%PROJ%\..") do set "ROOT=%%~fI"
set "BIN=%ROOT%\cmake-build-debug\bin"
if not exist "%BIN%\ddscat_cuda_slice.exe" (echo ERROR: %BIN%\ddscat_cuda_slice.exe not found& exit /b 1)
if not exist "%BIN%\ddscat_matvec_cuda_slice.dll" (echo ERROR: %BIN%\ddscat_matvec_cuda_slice.dll not found& exit /b 2)
pushd "%BIN%"
ddscat_cuda_slice.exe
set RC=%ERRORLEVEL%
popd
exit /b %RC%
