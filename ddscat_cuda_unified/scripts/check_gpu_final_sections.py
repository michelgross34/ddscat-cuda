#!/usr/bin/env python3
from pathlib import Path
import re, sys
root=Path(__file__).resolve().parents[1]
cu=(root/'cuda/ddscat_matvec_cuda.cu').read_text(errors='ignore')
h=(root/'cuda/ddscat_matvec_cuda.h').read_text(errors='ignore')
ld=(root/'cuda/ddscat_cuda_loader.c').read_text(errors='ignore')
br=(root/'fortran/cuda_matvec_bridge.f90').read_text(errors='ignore')
gf=(root/'fortran/getfml_cuda.f90').read_text(errors='ignore')
checks={
 'evalq export': 'ddscat_cuda_evalq_f32' in cu and 'ddscat_cuda_evalq_f32' in h,
 'scat export': 'ddscat_cuda_scat_f32' in cu and 'ddscat_cuda_scat_f32' in h,
 'loader evalq': 'ddscat_cuda_loader_evalq_f32' in ld,
 'loader scat': 'ddscat_cuda_loader_scat_f32' in ld,
 'Fortran EVALQ bridge': 'CUDA_EVALQ_GPU' in br,
 'Fortran SCAT bridge': 'CUDA_SCAT_GPU' in br,
 'chunked Zdotc EVALQ': 'GPU EVALQ CABS cublasZdotc' in cu and 'GPU EVALQ CEXT/CPHA cublasZdotc' in cu,
 'block cuBLAS reduction': 'cublasZgemv' in cu,
 'angular cuBLAS reduction': 'cublasDdot' in cu,
 'fused phase': 'sincosf(ph,&sn,&cs)' in cu and 'farfield_partial_kernel' in cu,
 'Krylov release': 'release_krylov_workspace("GPU SCAT")' in cu,
 'NOTORQ GPU route': "CMDTRQ=='NOTORQ'" in gf and 'CALL CUDA_SCAT_GPU' in gf,
 'DOTORQ CPU fallback present': 'CALL SCAT(' in gf,
 'no stale dummy farfield helper': 'int farfield_batch(FarfieldWorkspace' not in cu,
}
for k,v in checks.items(): print(f"{'PASS' if v else 'FAIL'}: {k}")
if not all(checks.values()): sys.exit(1)
print('PASS: GPU final-sections static audit')
