from pathlib import Path
import re, sys
p=Path(__file__).resolve().parents[1]/"cuda"/"ddscat_matvec_cuda.cu"
s=p.read_text(encoding="utf-8")
solver_names=[
 'ddscat_cuda_solve_gpbicg_f32','ddscat_cuda_solve_qmrccg_f32','ddscat_cuda_solve_pbcgst_f32',
 'ddscat_cuda_solve_pbcgs2_f32','ddscat_cuda_solve_petrkp_f32','ddscat_cuda_solve_bicgstab4_f32',
 'ddscat_cuda_solve_gpbicgstab2_f32','ddscat_cuda_solve_gpbicgstab4_f32']
checks={
 "occupied map stored on GPU": "int *d_occ_index=nullptr" in s,
 "compact solver x/y exist": "cufftComplex *d_sx=nullptr,*d_sy=nullptr,*d_b=nullptr,*d_wrk=nullptr" in s,
 "compact dimension is 3*NAT0": "cn=(size_t)3*nat0" in s and "s.solver_n=(int)cn" in s,
 "b allocated on compact dimension": 'MALR(s.d_b,cn*sizeof(cufftComplex)' in s,
 "13 work vectors allocated compact": 'MALR(s.d_wrk,13*cn*sizeof(cufftComplex)' in s,
 "compact x/y allocated compact": 'MALR(s.d_sx,cn*sizeof(cufftComplex)' in s and 'MALR(s.d_sy,cn*sizeof(cufftComplex)' in s,
 "mapping built from IOCC": "if(iocc[j]!=0)occ.push_back(j)" in s,
 "NAT0 checked against IOCC": "IOCC contains %zu occupied sites but NAT0=%d" in s,
 "gather kernel exists": "gather_full_to_compact_kernel" in s,
 "scatter kernel exists": "scatter_compact_to_full_kernel" in s,
 "solver MATVEC wrapper exists": "int matvec_solver_device(" in s,
 "solver MATVEC expands compact input": 'scatter_compact_to_full(compact_in,s.d_x' in s,
 "solver MATVEC gathers compact output": 'gather_full_to_compact(s.d_y,compact_out' in s,
 "PBCGST compact diagonal mapping": "pbc_precon" in s and "s.d_occ_index,s.nat,s.nat0,n" in s,
 "solution expands to full host layout": "download_compact_to_full_host" in s,
 "memory report prints occupancy factor": "Solver compact occupancy NAT0/NAT" in s,
 "memory report prints compact solver vectors": "Solver vectors compact (16 vectors)" in s,
 "memory report prints memory saved": "Solver vector memory saved" in s,
}
for fn in solver_names:
    marker='extern "C" DDSCAT_CUDA_API int '+fn
    try:
        a=s.index(marker); b=s.find('\nextern "C" DDSCAT_CUDA_API int ',a+len(marker)); body=s[a:b if b!=-1 else len(s)]
        checks[f"{fn} uses solver_n"]="const int n=s.solver_n" in body
        checks[f"{fn} uses compact MATVEC"]=("matvec_solver_device(" in body or "run_mv(" in body)
        checks[f"{fn} has no direct full MATVEC"]="matvec_device(" not in body
    except ValueError:
        checks[f"{fn} exists"]=False
for k,v in checks.items(): print(("OK   " if v else "FAIL ")+k)
if not all(checks.values()): sys.exit(1)
