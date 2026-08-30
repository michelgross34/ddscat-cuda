# Additional IFDDA-derived CUDA solvers

This revision adds four selectors to the CUDA DDSCAT build:

- `BICGS2` — BiCGStab(2), deliberately mapped to the existing enhanced PBCGS2/ZBCG2 L=2 CUDA implementation.
- `BICGS4` — BiCGStab(4), L=4 Sleijpen/Fokkema recurrence with modified Gram-Schmidt MR polynomial.
- `GPBGS2` — GPBiCGStab(2), memory-reduced L=2 recurrence from the previous IFDDA/ADDA CUDA conversion.
- `GPBGS4` — GPBiCGStab(4), memory-reduced L=4 recurrence from the previous IFDDA/ADDA CUDA conversion.

The names are six characters because DDSCAT declares `CMDSOL*6`.

## Numerical architecture

The new methods follow the current DDSCAT CUDA precision policy:

- Krylov vectors and MATVEC/FFT: `cufftComplex` / float32.
- Every large-vector scalar product: the single shared `dot_product_double_chunked()` path.
- Each chunk contains at most 1,000,000 float-complex elements, converted by a CUDA kernel to `cuDoubleComplex`.
- Reductions use `cublasZdotc` in FP64 and partial chunk results are summed by the CPU in double precision.
- L=2/L=4 local dense minimization systems are assembled and solved on the CPU in `std::complex<double>` with pivoted Gaussian elimination.
- All elementwise large-vector arithmetic is performed by CUDA kernels. No cuBLAS AXPY/SCAL/COPY path is introduced.
- No full Krylov vector is copied H2D/D2H inside the iterative loops. Full-vector transfers occur at solver boundaries; the central dot-product routine transfers only one FP64 complex scalar result per chunk to the CPU as requested.

## Memory

`d_wrk` is increased from 12 to 13 float-complex vectors. GPBiCGStab(4) uses the already-existing `d_y` staging vector as its transient `work`, giving the 15 resident-vector layout (including x/r/p/work) from the memory-reduced IFDDA port without another full-size allocation.

## GPBiCGStab(2) reliable residual

Every 20 outer cycles, and whenever recursive convergence is claimed, GPBiCGStab(2) evaluates `r_true=b-A*x`. It restarts the Krylov recurrence only when the relative residual gap is at least `1e-3` or recursive convergence is false. The solution `x` is retained.

## Build / CLion

Run `scripts\build_all.bat`, or rebuild the CUDA DLL then reload/rebuild CMake in CLion. `build_cuda_dll.bat` copies the DLL to `cmake-build-debug\bin`.

Test parameter files and CLion scripts are provided for all four selectors.
