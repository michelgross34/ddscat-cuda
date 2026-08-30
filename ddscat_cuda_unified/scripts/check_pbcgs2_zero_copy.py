from pathlib import Path
p=Path(__file__).resolve().parents[1]/"cuda"/"ddscat_matvec_cuda.cu"
s=p.read_text(errors="replace")
a=s.index('extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_pbcgs2_f32')
body=s[a:]
loop=body[body.index('while(resid>tol && it<maxit)'):body.index('// ZBCG2 returns X + XP')]
for token in ("cudaMemcpyHostToDevice","cudaMemcpyDeviceToHost","cudaMemcpy("):
    n=loop.count(token)
    print(f"{token}: {n}")
    if n:
        raise SystemExit(1)
print("PBCGS2 loop body has no direct vector cudaMemcpy; scalar dot-product traffic is intentionally encapsulated in the shared chunked-dot function")
