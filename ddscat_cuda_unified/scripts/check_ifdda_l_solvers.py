from pathlib import Path
import re, sys
root=Path(__file__).resolve().parents[1]
cu=(root/'cuda/ddscat_matvec_cuda.cu').read_text()
get=(root/'fortran/getfml_cuda.f90').read_text()
rea=(root/'fortran/reapar_cuda.f90').read_text()
dds=(root/'fortran/DDSCAT_cuda.f90').read_text()
cmk=(root/'CMakeLists.txt').read_text()
checks={
 'DDSCAT top-level selectors': all(("CMDSOL=='"+x+"'") in dds for x in ['BICGS2','BICGS4','GPBGS2','GPBGS4']),
 'CMake uses local DDSCAT_cuda': 'fortran/DDSCAT_cuda.f90' in cmk,
 'BICGS2 selector': "CMDSOL=='BICGS2'" in get and "CMDSOL=='BICGS2'" in rea,
 'BICGS4 selector': "CMDSOL=='BICGS4'" in get and "CMDSOL=='BICGS4'" in rea,
 'GPBGS2 selector': "CMDSOL=='GPBGS2'" in get and "CMDSOL=='GPBGS2'" in rea,
 'GPBGS4 selector': "CMDSOL=='GPBGS4'" in get and "CMDSOL=='GPBGS4'" in rea,
 'BiCGStab4 CUDA export': 'ddscat_cuda_solve_bicgstab4_f32' in cu,
 'GPBiCGStab2 CUDA export': 'ddscat_cuda_solve_gpbicgstab2_f32' in cu,
 'GPBiCGStab4 CUDA export': 'ddscat_cuda_solve_gpbicgstab4_f32' in cu,
 'shared chunked FP64 dot': 'dot_product_double_chunked' in cu and 'cublasZdotc' in cu and 'DOT_CHUNK_COMPLEX=1000000' in cu,
 'no float cuBLAS reductions': all(x not in cu for x in ['cublasCdotc(','cublasCdotu(','cublasScnrm2(']),
 '13 compact work vectors': 'MALR(s.d_wrk,13*cn*sizeof(cufftComplex)' in cu,
 'GPU elementwise kernels': all(x in cu for x in ['l_axpy','l_affine','l_lincomb2','l_self_add2']),
}
for k,v in checks.items(): print(('OK   ' if v else 'FAIL ')+k)
if not all(checks.values()): sys.exit(1)
# The solver bodies may have boundary/final cudaMemcpy, but their iterative
# vector recurrences call CUDA kernels/run_mv. Report direct memcpy occurrences
# for manual audit.
for fn in ['ddscat_cuda_solve_bicgstab4_f32','ddscat_cuda_solve_gpbicgstab2_f32','ddscat_cuda_solve_gpbicgstab4_f32']:
    a=cu.index('extern "C" DDSCAT_CUDA_API int '+fn)
    b=cu.find('\nextern "C" DDSCAT_CUDA_API int ',a+10)
    body=cu[a:b if b!=-1 else len(cu)]
    print(f'{fn}: direct cudaMemcpy occurrences={body.count("cudaMemcpy(")} (boundary/final only expected)')
