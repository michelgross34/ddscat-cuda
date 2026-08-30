# DDSCAT CUDA float32 — QMRCCG GPU-resident

This project is a separate `ddscat_cuda` CMake build. It keeps the original CPU
DDSCAT source tree unchanged.

## Added in this version

`CMDSOL='QMRCCG'` is routed from `GETFML` to `CUDA_QMRCCG_SOLVE` in the CUDA DLL.
The original CPU `PIMQMRCG` call remains under the preprocessor fallback.

QMRCCG keeps the following 10 vectors resident on the GPU for the whole solve:

`r, p, Ap, q, d, s, vtilde, wtilde, Atw, x`

The QMR scalar recurrence (`lambda`, `kappa`, `theta`, `gamma`, `ksi`, `rho`,
`epsilon`, `mu`, `tau`) also remains device resident. cuBLAS is used for norms
and dot products. The unconjugated QMR products from the original Fortran are
implemented with `chunked cublasZdotu`, not `chunked cublasZdotc`.

The original DDSCAT QMR implementation assumes the DDA matrix is symmetric and
uses the normal `MATVEC` routine for `A^T*w`. The CUDA port intentionally keeps
that same assumption: the second matrix-vector product in each QMR iteration is
another device-resident normal MATVEC.

## CPU/GPU transfers

At solver entry only:

- `b` H2D;
- initial `x` H2D;
- `CXADIA` H2D;
- `CXAOFF` H2D.

During the QMR iteration loop:

- H2D `cudaMemcpy`: **0**;
- D2H `cudaMemcpy`: **0**.

The convergence residual is one mapped pinned scalar (CUDA zero-copy). The
solution vector is copied D2H once when QMR converges.

## MATVEC

MATVEC uses one persistent rank-3 `cufftPlanMany`, `batch=3`, for X/Y/Z. Thus
there is one batched 3-D cuFFT execution in each transform direction, rather
than three separate component calls.

Each MATVEC prints GPU time and `previous MATVEC end -> current MATVEC end`.

## Build — Windows / CUDA 11.8 / sm_70

Run from `ddscat_cuda`:

```bat
scripts\build_all.bat
```

CUDA is compiled separately with:

`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe`

The DLL links `cufft`, `cublas`, and `cudart`, targets `sm_70`, and is copied to:

`ddscat_cuda\cmake-build-debug\bin\ddscat_matvec_cuda.dll`

The Fortran/C executable remains GCC/GFortran/MinGW through CMake.

## Test parameter file

`tests\ddscat_qmrccg.par` is derived from the supplied `ddscat(2).par`; only
`CMDSOL` is changed from `GPBICG` to `QMRCCG`.

For the first numerical validation, compare CPU QMRCCG and CUDA QMRCCG residual
history and final `Qext/Qabs/Qsca`. Float32 reduction order differs between CPU
and cuBLAS, so bitwise equality is not expected.
