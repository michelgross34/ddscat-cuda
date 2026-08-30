# Central double-precision scalar-product path

All five CUDA solvers use one common implementation:

`dot_product_double_chunked(a,b,n,conjugate,host_total,where)`

The solver vectors remain `cufftComplex` / complex float32 on the GPU. The
function splits them internally into blocks of at most **1,000,000 complex
numbers**. Two persistent staging buffers are allocated once:

- `d_dot_chunk_a[1,000,000]` as `cuDoubleComplex` (~16 MB)
- `d_dot_chunk_b[1,000,000]` as `cuDoubleComplex` (~16 MB)

For each block:

1. one CUDA kernel converts both float32 complex input slices to the two
   `cuDoubleComplex` staging buffers;
2. Hermitian scalar products call `cublasZdotc`;
3. QMRCCG unconjugated scalar products call `cublasZdotu`;
4. cuBLAS writes the block result to a CPU `cuDoubleComplex` because the handle
   uses `CUBLAS_POINTER_MODE_HOST`;
5. the CPU adds the block result to the running real/imaginary double sums.

Only after every block has been accumulated is the final 16-byte complex-double
coefficient copied back to the GPU coefficient array. The vector data itself is
never copied to CPU for a scalar product.

Norms use exactly the same path with `dotc(a,a)` followed by a CPU `sqrt` in
double precision, so no `cublasScnrm2` single-precision reduction remains.

This intentionally changes the earlier strict "zero-copy scalar" claim: the
iterative solver vectors still remain GPU-resident, but each scalar product now
returns one **double-complex partial result per million-element block to the
CPU**, because CPU accumulation of the block results is explicitly required.
There are still no full-vector H2D/D2H transfers inside the solver iterations.
