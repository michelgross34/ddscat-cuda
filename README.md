# DDSCAT-CUDA

This repository provides a CUDA-enabled version of [DDSCAT](https://github.com/DDSCAT), the Discrete Dipole Approximation code for electromagnetic light-scattering calculations.

The CUDA implementation accelerates the matrix-vector products used by the iterative solvers and provides two GPU backends:

- **FFT3D**: the standard three-dimensional FFT backend;
- **SLICES**: a lower-memory sliced FFT backend.

## Main characteristics

The current CUDA configuration uses **single precision**. This choice reduces GPU memory consumption and makes it possible to process larger computational grids than a double-precision implementation on the same GPU.

Single precision is the default for the CUDA targets. Numerical results should be compared with the CPU implementation when changing solver, grid size, material contrast, or convergence tolerance.

The repository also includes diagnostic and validation material for solver residuals, GPU memory usage, sliced FFT processing, and final scattering cross sections.

## Windows build environment

The provided Windows build setup uses the MinGW-w64 toolchain:

- `gcc` for C sources;
- `g++` for C++/CUDA-related host compilation where required;
- `gfortran` for the DDSCAT Fortran sources;
- CMake and the supplied Windows batch build scripts.

MPI is not required. The code uses the non-MPI DDSCAT configuration.

## FFTW installation

The project expects a directory named `fftw` (also written `FFTW`; Windows
paths are case-insensitive) at the repository root:

```text
ddscat-cuda/
├── fftw/
├── ddscat_cuda_unified/
├── CMakeLists.txt
└── ...
```

For the default single-precision MinGW-w64 build, the essential files are:

```text
fftw/
├── fftw3.h
├── libfftw3f.a
├── libfftw3f.dll.a       # MinGW-w64 import library
└── libfftw3f-3.dll       # runtime DLL
```

The supplied FFTW directory also contains the following optional/reference
files:

```text
fftw3.f
fftw3.f03
fftw3l.f03
fftw3q.f03
libfftw3.a
libfftw3.dll.a
libfftw3-3.dll
libfftw3f-3.exp
libfftw3f-3.lib
libfftw3f_omp.a
libfftw3f_omp.dll.a
libfftw3f_omp-3.dll
libfftw3f_threads.a
libfftw3f_threads.dll.a
libfftw3f_threads-3.dll
libfftw3l.a
libfftw3l.dll.a
libfftw3l-3.dll
libfftw3l_omp.a
libfftw3l_omp.dll.a
libfftw3l_omp-3.dll
libfftw3l_threads.a
libfftw3l_threads.dll.a
libfftw3l_threads-3.dll
libfftw3q.a
libfftw3q_omp.a
libfftw3q_threads.a
libfftw3_omp.a
libfftw3_omp.dll.a
libfftw3_omp-3.dll
libfftw3_threads.a
libfftw3_threads.dll.a
libfftw3_threads-3.dll
```

The `fftw-wisdom-to-conf`, `fftw-wisdom.exe`, `fftwf-wisdom.exe`,
`fftwl-wisdom.exe`, and `fftwq-wisdom.exe` utilities are not required to
build DDSCAT. The CUDA backend itself uses cuFFT for GPU FFT operations,
while the CPU/reference path uses FFTW3 where configured.

If FFTW is installed elsewhere, configure CMake with:

```powershell
cmake -S . -B cmake-build-debug `
  -DDDSCAT_CUDA_FFTW_ROOT="C:/path/to/FFTW"
```

## Build

From a Windows PowerShell prompt at the repository root:

```powershell
cmake -S . -B cmake-build-debug -G Ninja
cmake --build cmake-build-debug --parallel 4
```

The CUDA build requires a CUDA toolkit/NVCC installation and a CUDA-capable NVIDIA GPU for runtime validation. The generated programs and DLLs are placed in:

```text
cmake-build-debug/bin/
```

Typical CUDA outputs are:

```text
ddscat_cuda.exe
ddscat_cuda_slice.exe
ddscat_matvec_cuda.dll
ddscat_matvec_cuda_slice.dll
```

## Running DDSCAT

Run DDSCAT from a directory containing the required parameter file and material data files, for example:

```powershell
cd path/to/your/calculation
path/to/ddscat_cuda.exe
```

The file `ddscat.par` is read from the current working directory. Material files, target files, and other input paths are also resolved relative to that directory.

## Repository layout

```text
ddscat_cuda_unified/
├── cuda/       CUDA backend and loader
├── fortran/    CUDA-aware Fortran bridge and DDSCAT variants
├── scripts/    Windows build, run, and validation scripts
└── tests/      Example parameter files
```

The original DDSCAT project and documentation are available at [github.com/DDSCAT](https://github.com/DDSCAT).

## Status

The repository is intended for research and development. Final performance and numerical validation should be performed on the target NVIDIA GPU, comparing CPU and GPU results for solver residuals, iteration counts, `Qext`, `Qabs`, `Qsca`, and the corresponding physical scattering cross sections.
