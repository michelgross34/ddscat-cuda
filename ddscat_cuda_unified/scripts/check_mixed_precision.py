from pathlib import Path
import re, sys
p=Path(__file__).resolve().parents[1]/"cuda"/"ddscat_matvec_cuda.cu"
s=p.read_text(encoding="utf-8")
checks={
 "complex solver scalar storage is double": "cuDoubleComplex *d_cs" in s,
 "real solver scalar storage is double": "double *d_rs" in s,
 "mapped convergence residual is double": "double *h_resid" in s and "*d_resid_map" in s,
 "chunk conversion is float complex to double complex": "convert_dot_pair_f32_to_f64_kernel" in s,
 "dotc uses double cuBLAS": "cublasZdotc" in s,
 "dotu uses double cuBLAS": "cublasZdotu" in s,
 "norm uses the same double dot path": "dot_product_double_chunked(a,a,n,true" in s,
 "no single-precision cuBLAS dot": "cublasCdotc(" not in s and "cublasCdotu(" not in s,
 "no single-precision cuBLAS norm": "cublasScnrm2(" not in s,
 "no cuBLAS vector AXPY/SCAL/COPY": not re.search(r"cublas\w*(axpy|scal|copy)",s,re.I),
 "no float solver scalar arrays": "float *d_rs" not in s and "cufftComplex *d_cs" not in s,
}
for k,v in checks.items(): print(("OK  " if v else "FAIL"),k)
if not all(checks.values()): sys.exit(1)
