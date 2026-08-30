from pathlib import Path
import re, sys
p=Path(__file__).resolve().parents[1]/"cuda"/"ddscat_matvec_cuda.cu"
s=p.read_text(encoding="utf-8")
checks={
 "period is exactly 20": "RELIABLE_RESID_PERIOD=20" in s,
 "gap threshold is exactly 1e-3": "RELIABLE_RESID_GAP=1.e-3" in s,
 "shared reliable function exists": s.count("int reliable_residual_check(")==1,
 "true residual uses b-Ax": "norm_diff_host(rhs,s.d_sy" in s,
 "vector gap uses true-recursive residual": "norm_residual_gap_host(rhs,s.d_sy,rrec" in s,
 "restart on large vector gap": "rr.gap>=RELIABLE_RESID_GAP" in s,
 "restart on false recursive convergence": "(recursive_res<=tol)&&!rr.converged" in s,
 "GPBICG periodic check": 'reliable_residual_check("GPBICG"' in s,
 "QMRCCG periodic check": 'reliable_residual_check("QMRCCG"' in s,
 "PBCGS2 periodic check": 'reliable_residual_check("PBCGS2"' in s,
 "BiCGStab4 periodic check": 'reliable_residual_check("BiCGStab(4)"' in s,
 "GPBiCGStab2 periodic check": 'reliable_residual_check("GPBiCGStab(2)"' in s,
 "GPBiCGStab4 periodic check": 'reliable_residual_check("GPBiCGStab(4)"' in s,
 "GP4 preserves work when no restart": "GP4 restore work after reliable check" in s,
 "PBCGST true residual every iteration retained": 'PBCGST-true-residual' in s,
 "PETRKP exact 10-step refresh retained": 'it!=10*(it/10)' in s and 'PETRKP-A-x-refresh' in s,
 "gap norm is double cuBLAS": "cublasDznrm2" in s,
}
for k,v in checks.items(): print(("OK  " if v else "FAIL"),k)
if not all(checks.values()): sys.exit(1)
