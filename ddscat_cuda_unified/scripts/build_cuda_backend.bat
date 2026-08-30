@echo off
setlocal EnableExtensions
set "BACKEND=%~1"
set "OUT=%~2"
if "%BACKEND%"=="" goto :usage
if "%OUT%"=="" goto :usage

set "CUDA118=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "NVCC=%CUDA118%\bin\nvcc.exe"
if not exist "%NVCC%" (echo ERROR: CUDA 11.8 nvcc not found at "%NVCC%"& exit /b 1)

if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
 for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
 if defined VSROOT if exist "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
)
where cl.exe >nul 2>nul
if errorlevel 1 (echo ERROR: cl.exe required by nvcc 11.8 was not found.& exit /b 2)

set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "PROJ=%%~fI"
if not exist "%OUT%" mkdir "%OUT%"

set "DEF="
set "DLL=ddscat_matvec_cuda.dll"
if /I "%BACKEND%"=="SLICES" (
 set "DEF=-DDDSCAT_CUDA_BACKEND_SLICES=1"
 set "DLL=ddscat_matvec_cuda_slice.dll"
) else if /I not "%BACKEND%"=="FFT3D" (
 echo ERROR: backend must be FFT3D or SLICES
 exit /b 3
)

echo Building %BACKEND% from the common CUDA source...
"%NVCC%" -O3 -std=c++14 -lineinfo -shared %DEF% ^
 -gencode arch=compute_70,code=sm_70 ^
 -gencode arch=compute_70,code=compute_70 ^
 -Xcompiler "/O2 /MD" ^
 "%PROJ%\cuda\ddscat_matvec_cuda.cu" ^
 -I"%PROJ%\cuda" -I"%CUDA118%\include" -L"%CUDA118%\lib\x64" ^
 -lcufft -lcublas -lcudart -o "%OUT%\%DLL%"
if errorlevel 1 exit /b %ERRORLEVEL%
if not exist "%OUT%\%DLL%" (echo ERROR: %DLL% not produced.& exit /b 4)
echo Produced: %OUT%\%DLL%
exit /b 0

:usage
echo Usage: build_cuda_backend.bat FFT3D^|SLICES output_directory
exit /b 64
