@echo off
setlocal
set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "PROJ=%%~fI"
call "%HERE%build_cuda_backend.bat" SLICES "%PROJ%\build_cuda_dll"
exit /b %ERRORLEVEL%
