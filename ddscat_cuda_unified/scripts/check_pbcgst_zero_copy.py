from pathlib import Path
p=Path(__file__).resolve().parents[1]/"cuda"/"ddscat_matvec_cuda.cu"
s=p.read_text()
a=s.index('extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_pbcgst_f32')
loop=s[s.index('while(total_it<maxit&&!converged)',a):s.index('// Ensure TOLE',a)]
print('PBCGST iterative/restart-loop cudaMemcpyHostToDevice:',loop.count('cudaMemcpyHostToDevice'))
print('PBCGST iterative/restart-loop cudaMemcpyDeviceToHost:',loop.count('cudaMemcpyDeviceToHost'))
print('PBCGST iterative/restart-loop cudaMemcpy total:',loop.count('cudaMemcpy('))
assert 'cudaMemcpyHostToDevice' not in loop
assert 'cudaMemcpyDeviceToHost' not in loop
assert 'cudaMemcpy(' not in loop
print('OK: no direct full-vector cudaMemcpy appears in PBCGST loop body; scalar dot-product traffic occurs inside the shared chunked-dot function')
