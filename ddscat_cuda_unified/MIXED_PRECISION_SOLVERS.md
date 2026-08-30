# DDSCAT CUDA solver scalar precision policy

This build keeps the large solver vectors, Green tensor and cuFFT workspace in
`float32` / `cufftComplex`, while scalar reductions and recurrence algebra use
double precision.

## One shared double-precision dot-product implementation

Every solver calls the same function:

```text
dot_product_double_chunked(a, b, n, conjugate, host_total, where)
```

The function internally splits the float32 GPU vectors into chunks of at most
**1,000,000 complex values**. Two persistent `cuDoubleComplex` staging buffers
(~16 MB each) are allocated once. A CUDA kernel converts both float32 slices to
double complex, then cuBLAS performs the block reduction:

- conjugated/Hermitian product: `cublasZdotc`;
- unconjugated QMRCCG product: `cublasZdotu`.

The cuBLAS handle uses `CUBLAS_POINTER_MODE_HOST`. Each block therefore returns
one `cuDoubleComplex` to the CPU, and the CPU accumulates all block results in
double precision. Only the completed 16-byte scalar is copied back to GPU when
a recurrence kernel needs it.

There is no `cublasCdotc`, `cublasCdotu`, or `cublasScnrm2` reduction left.
Norms reuse the same `cublasZdotc(a,a)` path and take the square root in CPU
double precision.

## Double precision scalar algebra

The persistent solver scalar arrays are:

```text
cuDoubleComplex d_cs[32]
double          d_rs[8]
```

`alpha`, `beta`, `omega`, `rho`, `eta`, `dzeta`, QMR coefficients, the PBCGS2
3x3 polynomial coefficients, reliable-update thresholds, PETRKP coefficients,
and relative residual arithmetic therefore remain `double` /
`cuDoubleComplex`.

## Elementwise vector arithmetic

All arithmetic over solver vectors is performed by explicit CUDA kernels.
There are no cuBLAS AXPY/SCAL/COPY vector operations. Simple vector copies use
the explicit `vcopy` CUDA kernel. `cudaMemset` is used only for initialization.

## Transfer implications

The full solver vectors remain GPU-resident throughout the iterations. The new
double-dot design intentionally introduces **scalar-only CPU traffic**: one
double-complex partial result per million-element block is returned by cuBLAS
to the CPU for accumulation, and the final 16-byte result is copied back to
the GPU coefficient storage. This replaces the earlier strict scalar
zero-copy design because CPU accumulation was explicitly requested.

## PBCGS2 3x3 algebra

The six Gram-matrix entries over full DDSCAT vectors use the same chunked
`cublasZdotc` function. The subsequent tiny 3x3 polynomial algebra is not a
vector reduction and remains in a CUDA kernel using `cuDoubleComplex` /
`double`.
