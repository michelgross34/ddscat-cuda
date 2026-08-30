# DDSCAT CUDA unified: FFT3D + SLICES from one codebase

Target installation root:

`E:\msna6\ddscat7.3.4_250505`

This overlay adds one new source directory:

`ddscat_cuda_unified\`

and updates only the root `CMakeLists.txt` so CLion sees the CUDA targets.
Existing `ddscat_cuda\` and `pkg\` are left untouched and can be retained as
rollback copies until the unified build is validated.

## Architecture

A single CUDA source is maintained:

`ddscat_cuda_unified\cuda\ddscat_matvec_cuda.cu`

It contains all common functionality:

- CUDA-resident Green construction;
- Green lifetime/allocation ordering;
- GPBICG, QMRCCG, PBCGST, PBCGS2, PETRKP;
- BICGS2/BICGS4/GPBGS2/GPBGS4 selectors from the later CUDA branch;
- chunked 1,000,000-element float32->complex<double> conversion;
- cublasZdotc/cublasZdotu reductions and CPU double accumulation;
- all elementwise solver kernels;
- reliable true-residual control every 20 solver iterations with vector-gap
  detection and Krylov restart only when required;
- timing and transfer diagnostics;
- compact Krylov storage on occupied `IOCC` sites only (`3*NAT0` instead of `3*NAT`), with GPU gather/scatter around MATVEC;
- GPU final `EVALQ` (`Cabs/Cext/Cpha`) with chunked complex-double cuBLAS reductions;
- GPU final `SCAT` for `NOTORQ`: fused phase/dipole block reductions, `cublasZgemv` block summation and `cublasDdot` angular integration;
- release of the 13 compact Krylov work vectors before final postprocessing, with lazy reallocation before a later solve.

The same source is compiled twice by CUDA 11.8:

1. `FFT3D` with no backend macro -> `ddscat_matvec_cuda.dll`;
2. `SLICES` with `DDSCAT_CUDA_BACKEND_SLICES=1` ->
   `ddscat_matvec_cuda_slice.dll`.

Only the following code is backend-specific:

- MATVEC explicit workspace allocation;
- cuFFT plan creation/destruction;
- MATVEC transform sequence.

For FFT3D the runtime buffer is `3*(2NX)*(2NY)*(2NZ)`. For SLICES the runtime buffers are `3*(2NZ)*NX*NY` plus a batch buffer of
`4*3*(2NX)*(2NY)` for four XY slices.

Green construction is common. Green-only scratch and its temporary cuFFT plan
are destroyed before either backend allocates its MATVEC/solver buffers.

Solver/Krylov vectors are compacted from the actual `IOCC` mask. Their storage
therefore scales as `NAT0/NAT`; for a sphere tightly inscribed in its box this
is approximately `pi/6 = 0.5236`. The FFT/MATVEC full-grid buffers remain full
size, and conversion between compact and full layouts is performed by CUDA
kernels without full-vector CPU transfers.

## CMake / CLion targets

After extracting this overlay at the DDSCAT root and reloading CMake, CLion
should show at least:

- `ddscat` - original CPU executable;
- `ddscat_cuda` - CUDA FFT3D executable;
- `ddscat_cuda_slice` - CUDA low-memory SLICES executable;
- `ddscat_cuda_all` - builds both CUDA executables;
- `calltarget`, `ddpostprocess`, `vtrconvert` - original utilities.

On Windows the CUDA DLL custom targets are `ALL` targets, so **Build All** from
the root project builds both CUDA 11.8 DLLs as well as the executables. The
NVCC helper finds Visual Studio with `vswhere` and calls `vcvars64.bat`, so it
does not require CLion itself to have `cl.exe` preconfigured in its environment.

Output is placed in the common root build directory:

`cmake-build-debug\bin\`

including:

- `ddscat.exe`
- `ddscat_cuda.exe`
- `ddscat_cuda_slice.exe`
- `ddscat_matvec_cuda.dll`
- `ddscat_matvec_cuda_slice.dll`

The common loader selects the correct DLL from the executable name.

## Installation

1. Back up the root `CMakeLists.txt`.
2. Extract this ZIP into `E:\msna6\ddscat7.3.4_250505`.
3. Accept replacement of the root `CMakeLists.txt`.
4. Reload the CMake project in CLion.
5. Build the root `all` target, or `ddscat_cuda_all` if only the CUDA variants
   are required.

The supplied root `CMakeLists.txt` is based on the 5,953-byte file from the
current DDSCAT tree and only appends the `add_subdirectory(ddscat_cuda_unified)`
integration.

## Validation performed before packaging

- root CMake configure: PASS;
- complete GCC/GFortran build of `ddscat_cuda_core`: PASS;
- link of `ddscat_cuda` and `ddscat_cuda_slice`: PASS;
- root Build All including original CPU DDSCAT: PASS;
- CUDA source syntax/type parsing in both backend modes: PASS using a CUDA
  syntax harness (actual nvcc/GPU unavailable in this environment);
- static unified-backend/layout checks: PASS;
- compact solver-storage audit for all eight CUDA solver entry points: PASS;
- compact component-major gather/scatter round-trip test: PASS.

- GPU final-sections source/bridge/routing audit: PASS;
- direct far-field algebra compared against the original SCAT formulation: PASS.

The remaining target-machine validation is the real NVCC 11.8 compilation and
GPU numerical comparison of FFT3D vs SLICES and CPU vs GPU final cross sections.
