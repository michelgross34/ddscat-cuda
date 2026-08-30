from pathlib import Path
p=Path(__file__).resolve().parents[1]/'cuda'/'ddscat_matvec_cuda.cu'
t=p.read_text(errors='replace')
checks={
 'single common CUDA source':p.exists() and not p.with_name('ddscat_matvec_cuda_slice.cu').exists(),
 'SLICES compile switch':'DDSCAT_CUDA_BACKEND_SLICES' in t,
 'Z workspace formula':'s.zwork_count=(size_t)3*(2*nz)*(size_t)nx*ny' in t,
 'XY batch4 slice formula':'s.slice_count=(size_t)4*3*(2*nx)*(size_t)(2*ny)' in t,
 '1D Z plan':'cufftPlanMany(&s.plan_z,1' in t,
 '2D XY batch12 plan':'cufftPlanMany(&s.plan_xy,2' in t and 'CUFFT_C2C,12' in t,
 'slice loop reduced x4':'for(int kz0=0;kz0<gz;kz0+=4)' in t,
 'Green per slice':'slice_green_xy_kernel' in t,
 'four Green streams':'green_stream[4]' in t and 'cudaStreamCreateWithFlags' in t,
 'event synchronization':'xy_ready' in t and 'green_done[4]' in t and 'cudaStreamWaitEvent' in t,
 'FFT3D backend also present':'cufftPlanMany(&s.plan,3' in t and 's.d_work' in t,
 'full normalization':'1.f/(float)s.ngrid' in t,
}
for k,v in checks.items(): print(('OK   ' if v else 'FAIL ')+k)
raise SystemExit(0 if all(checks.values()) else 1)
