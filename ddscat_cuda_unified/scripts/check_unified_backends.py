from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
cu=(root/'cuda'/'ddscat_matvec_cuda.cu').read_text(errors='replace')
cm=(root/'CMakeLists.txt').read_text(errors='replace')
checks={
 'one CUDA source file': (root/'cuda'/'ddscat_matvec_cuda.cu').exists() and not (root/'cuda'/'ddscat_matvec_cuda_slice.cu').exists(),
 'compile-time SLICES backend': '#ifdef DDSCAT_CUDA_BACKEND_SLICES' in cu,
 'FFT3D work path present': 's.d_work' in cu and 'cufftPlanMany(&s.plan,3' in cu,
 'SLICES Z/XY work path present': 's.d_zwork' in cu and 's.d_slice' in cu and 'cufftPlanMany(&s.plan_z,1' in cu and 'cufftPlanMany(&s.plan_xy,2' in cu,
 'shared Green build': 'build_green_cuda_resident' in cu and cu.count('ddscat_cuda_prepare_green_f32')==1,
 'Green scratch freed before runtime allocation': 'int grc=build_green_cuda_resident(gp)' in cu and 'int arc=allocate_runtime_after_green' in cu,
 'same source builds FFT3D DLL': 'FFT3D "${DDSCAT_CUDA_BIN_DIR}"' in cm,
 'same source builds SLICES DLL': 'SLICES "${DDSCAT_CUDA_BIN_DIR}"' in cm,
 'shared Fortran core': 'add_library(ddscat_cuda_core STATIC' in cm,
 'two executables': 'ddscat_add_cuda_executable(ddscat_cuda)' in cm and 'ddscat_add_cuda_executable(ddscat_cuda_slice)' in cm,
 'CLion aggregate target': 'add_custom_target(ddscat_cuda_all' in cm,
}
for k,v in checks.items(): print(('OK   ' if v else 'FAIL ')+k)
if not all(checks.values()): sys.exit(1)
