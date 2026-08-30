from pathlib import Path
p=Path(__file__).resolve().parents[1]/"cuda"/"ddscat_matvec_cuda.cu"
s=p.read_text(errors="replace")
a=s.index('extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_petrkp_f32')
body=s[a:]
loop=body[body.index('for(it=1;it<=maxit;++it){'):body.index('// If MAXIT is reached')]
for token in ("cudaMemcpyHostToDevice","cudaMemcpyDeviceToHost","cudaMemcpy("):
    n=loop.count(token)
    print(f"{token}: {n}")
    if n:
        raise SystemExit(1)
print("PETRKP loop body has no direct vector cudaMemcpy; scalar dot-product traffic is intentionally encapsulated in the shared chunked-dot function")
