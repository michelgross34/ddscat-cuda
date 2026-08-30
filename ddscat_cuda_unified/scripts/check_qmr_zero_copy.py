from pathlib import Path
p=Path(__file__).resolve().parents[1]/'cuda'/'ddscat_matvec_cuda.cu'
s=p.read_text()
a=s.index('extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_qmrccg_f32')
b=s.index('auto wall1=',a)
body=s[a:b]
loop=body[body.index('while(it<maxit&&resid>tol)'):]
print('QMR direct-loop cudaMemcpyHostToDevice:',loop.count('cudaMemcpyHostToDevice'))
print('QMR direct-loop cudaMemcpyDeviceToHost:',loop.count('cudaMemcpyDeviceToHost'))
assert 'cudaMemcpyHostToDevice' not in loop
assert 'cudaMemcpyDeviceToHost' not in loop
print('OK: no direct full-vector cudaMemcpy in QMR loop body; reliable residual and chunked scalar traffic are encapsulated in shared helpers')
