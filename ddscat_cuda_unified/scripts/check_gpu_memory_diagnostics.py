from pathlib import Path
import sys
p=Path(__file__).resolve().parents[1]/"cuda"/"ddscat_matvec_cuda.cu"
s=p.read_text(encoding="utf-8")
checks={
 "Green build wall time stored": "s.green_build_ms=ms" in s,
 "Green build time printed in seconds and ms": "CUDA Green build time: %.6f s (%.3f ms)" in s,
 "Green scratch freed before timing report": "Green scratch and temporary cuFFT plan have been freed" in s,
 "cudaMemGetInfo baseline helper": "capture_gpu_memory_baseline" in s and "cudaMemGetInfo(&free_b,&total_b)" in s,
 "final cudaMemGetInfo": "cudaMemGetInfo(&free_now,&total_now)" in s,
 "Green net category": "Green tensor resident (net)" in s,
 "MATVEC category": "MATVEC full x/y + operator + maps" in s,
 "compact solver vector category": "Solver vectors compact (16 vectors)" in s,
 "full solver comparison": "Solver vectors if full-grid" in s,
 "solver memory saving": "Solver vector memory saved" in s,
 "occupancy factor": "Solver compact occupancy NAT0/NAT" in s,
 "solver scalar category": "Solver scalar/dot staging buffers" in s,
 "explicit subtotal": "Explicit DDSCAT cudaMalloc subtotal" in s,
 "FFT workspace query FFT3D": "cufftGetSize(s.plan,&s.plan_workspace)" in s,
 "FFT workspace query SLICES": "cufftGetSize(s.plan_z,&s.plan_z_workspace)" in s and "cufftGetSize(s.plan_xy,&s.plan_xy_workspace)" in s,
 "actual DDSCAT allocation": "DDSCAT actual GPU allocation" in s,
 "remaining GPU memory": "GPU memory remaining" in s,
 "memory report before solver label": "CUDA GPU memory before iterative solver" in s,
 "report called GPU Green path": 'print_backend_layout(nx,ny,nz);print_gpu_memory_report();return 0;' in s,
 "CPU Green path reports N/A": "Green tensor supplied by CPU" in s,
}
for k,v in checks.items(): print(("OK   " if v else "FAIL ")+k)
if not all(checks.values()): sys.exit(1)
