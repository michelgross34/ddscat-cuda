# CUDA Green timing and GPU memory diagnostics

This diagnostic is compiled into both CUDA backends produced from
`cuda/ddscat_matvec_cuda.cu`:

- `ddscat_cuda.exe` / `ddscat_matvec_cuda.dll` (full FFT3D MATVEC)
- `ddscat_cuda_slice.exe` / `ddscat_matvec_cuda_slice.dll` (low-memory SLICES MATVEC)

## Green build time

When Green is built on the GPU by `ddscat_cuda_prepare_green_f32`, the complete
wall time of the resident Green construction is printed after the direct Green
kernels, Green FFT(s), trimming/status work, destruction of the temporary cuFFT
plan, and freeing of Green-only scratch memory:

```text
CUDA Green build time: ... s (... ms); Green scratch and temporary cuFFT plan have been freed.
```

Therefore the later memory report contains only the Green tensor that remains
resident during the iterative solver, not the temporary memory needed to build
it.

The compatibility path `ddscat_cuda_prepare_f32`, where Green is supplied by
the CPU, cannot measure Green computation time. It explicitly prints that the
build time is not available and reports only the H2D upload time.

## Memory report location

The report is printed at the end of CUDA preparation: after resident Green,
MATVEC buffers, solver buffers, cuFFT plans/workspaces, cuBLAS handle, events,
and SLICES streams/events (when applicable) have been created, but before the
iterative solver starts.

A `cudaMemGetInfo()` baseline is captured immediately after the previous DDSCAT
CUDA state has been released and before the new DDSCAT device allocations are
made. A second `cudaMemGetInfo()` measurement is made after all persistent
allocations. The difference in free memory is reported as the actual GPU
allocation attributable to this DDSCAT CUDA preparation.

## Reported categories

- **Green tensor resident (net)**: only `d_green`; Green-construction scratch is
  already freed.
- **MATVEC vectors/operator/IOCC**: `d_x`, `d_y`, `d_adia`, `d_aoff`, `d_iocc`.
- **MATVEC FFT data**: `d_work` for FFT3D or `d_zwork + d_slice` for SLICES.
- **Solver vectors**: `d_b` plus the 13 persistent solver work vectors.
- **Solver scalar/dot staging**: double recurrence arrays and the two
  1,000,000-element `cuDoubleComplex` dot-product conversion buffers.
- **Explicit DDSCAT cudaMalloc subtotal**: sum of the preceding explicit
  device buffers.
- **cuFFT workspace reported by plans**: obtained with `cufftGetSize`.
- **DDSCAT actual GPU allocation**: measured free-memory decrease with
  `cudaMemGetInfo`.
- **CUDA/cuFFT/cuBLAS/internal overhead**: actual allocation minus explicit
  DDSCAT `cudaMalloc` buffers. This includes library workspaces/caches and
  allocator overhead; the cuFFT reported workspace is shown separately for
  interpretation and is not double-counted in the explicit subtotal.
- **GPU memory remaining** and **GPU total memory**: current values from
  `cudaMemGetInfo` immediately before iterative solution.

The `cudaMemGetInfo` difference assumes no unrelated process changes its GPU
allocation significantly during the short DDSCAT preparation interval.
