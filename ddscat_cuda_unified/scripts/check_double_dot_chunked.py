from pathlib import Path
import re, sys
p=Path(__file__).resolve().parents[1]/"cuda"/"ddscat_matvec_cuda.cu"
s=p.read_text(encoding="utf-8")
checks={
 "single central chunked dot function": s.count("int dot_product_double_chunked(")==1,
 "chunk size is exactly 1,000,000 complex": "DOT_CHUNK_COMPLEX=1000000" in s,
 "persistent double chunk A": "cuDoubleComplex *d_dot_chunk_a" in s,
 "persistent double chunk B": "*d_dot_chunk_b" in s,
 "float-to-double pair conversion kernel": "convert_dot_pair_f32_to_f64_kernel" in s,
 "Hermitian products use cublasZdotc": "cublasZdotc" in s,
 "unconjugated products use cublasZdotu": "cublasZdotu" in s,
 "no cublasCdotc remains": "cublasCdotc(" not in s,
 "no cublasCdotu remains": "cublasCdotu(" not in s,
 "no cublasScnrm2 remains": "cublasScnrm2(" not in s,
 "CPU partial accumulation real": "sum_re+=cuCreal(partial)" in s,
 "CPU partial accumulation imag": "sum_im+=cuCimag(partial)" in s,
 "cuBLAS pointer mode is HOST": "CUBLAS_POINTER_MODE_HOST" in s,
 "dotc wrapper uses central function": re.search(r"int dotc\([^\n]+\).*?dot_product_double_chunked\(a,b,n,true",s,re.S) is not None,
 "dotu wrapper uses central function": re.search(r"int dotu\([^\n]+\).*?dot_product_double_chunked\(a,b,n,false",s,re.S) is not None,
 "norm reuses double dot(a,a)": "dot_product_double_chunked(a,a,n,true" in s,
 "no cuBLAS vector AXPY/SCAL/COPY": not re.search(r"cublas\w*(axpy|scal|copy)",s,re.I),
}
for k,v in checks.items(): print(("OK  " if v else "FAIL"),k)
if not all(checks.values()): sys.exit(1)
