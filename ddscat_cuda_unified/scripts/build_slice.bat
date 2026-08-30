@echo off
setlocal EnableExtensions
set "PROJ=%~dp0.."
for %%I in ("%PROJ%") do set "PROJ=%%~fI"
for %%I in ("%PROJ%\..") do set "ROOT=%%~fI"
set "BUILD=%ROOT%\cmake-build-debug"
if not exist "%BUILD%\CMakeCache.txt" (echo ERROR: configure the root CMake project with CLion first.& exit /b 1)
cmake --build "%BUILD%" --target ddscat_cuda_slice -j
exit /b %ERRORLEVEL%
