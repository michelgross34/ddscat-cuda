@echo off
setlocal EnableExtensions
set "PROJ=%~dp0.."
for %%I in ("%PROJ%") do set "PROJ=%%~fI"
for %%I in ("%PROJ%\..") do set "ROOT=%%~fI"
set "BUILD=%ROOT%\cmake-build-debug"
if not exist "%BUILD%\CMakeCache.txt" (
  echo CLion Debug cache not found. Configuring root project in %BUILD% ...
  cmake -S "%ROOT%" -B "%BUILD%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DDDSCAT_BUILD_CUDA_UNIFIED=ON
  if errorlevel 1 exit /b 1
)
echo Building CPU DDSCAT plus both unified CUDA variants...
cmake --build "%BUILD%" -j
if errorlevel 1 exit /b 2
echo Build complete. Executables and CUDA DLLs are under %BUILD%\bin
endlocal
