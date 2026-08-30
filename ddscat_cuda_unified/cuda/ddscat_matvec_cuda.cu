#include "ddscat_matvec_cuda.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <climits>
#include <string>
#include <complex>
#include <algorithm>
#include <vector>

namespace {
using WallClock=std::chrono::steady_clock;
constexpr int DOT_CHUNK_COMPLEX=1000000;

struct State {
 int nx=0,ny=0,nz=0,ipbc=-1,nat=0,nat0=0;
 int solver_n=0,dot_chunk_capacity=0;
 size_t ngrid=0,green_count=0,zwork_count=0,slice_count=0;
 size_t plan_workspace=0,plan_z_workspace=0,plan_xy_workspace=0;
 size_t gpu_free_baseline=0,gpu_total_bytes=0;
 double green_build_ms=0.0; bool green_built_on_gpu=false;
 // Runtime MATVEC storage. Only one backend is allocated in a given DLL:
 // d_work for FFT3D, or d_zwork+d_slice for the low-memory SLICES backend.
 cufftComplex *d_work=nullptr,*d_zwork=nullptr,*d_slice=nullptr,*d_x=nullptr,*d_y=nullptr;
 cufftComplex *d_adia=nullptr,*d_aoff=nullptr,*d_green=nullptr;
 int16_t *d_iocc=nullptr;
 int *d_occ_index=nullptr;
 // Solver-only vectors are compact: 3*NAT0 instead of 3*NAT. d_x/d_y remain
 // full MATVEC expansion/scratch vectors for FFT and the public host MATVEC API.
 cufftComplex *d_sx=nullptr,*d_sy=nullptr,*d_b=nullptr,*d_wrk=nullptr;
 // Solver recurrence scalars are double precision. Vector dot products are
 // converted by chunks and reduced with cuBLAS Zdotc/Zdotu in complex double.
 cuDoubleComplex *d_cs=nullptr;
 double *d_rs=nullptr;
 // Shared double-precision chunk buffers used by every scalar product.
 // Solver vectors remain float32; only up to DOT_CHUNK_COMPLEX elements are
 // converted at a time.
 cuDoubleComplex *d_dot_chunk_a=nullptr,*d_dot_chunk_b=nullptr;
 double *h_resid=nullptr,*d_resid_map=nullptr;
 int *h_status=nullptr,*d_status_map=nullptr;
 cufftHandle plan=0,plan_z=0,plan_xy=0; cublasHandle_t blas=nullptr;
 cudaEvent_t ev0=nullptr,ev1=nullptr;
#ifdef DDSCAT_CUDA_BACKEND_SLICES
 cudaStream_t green_stream[4]={nullptr,nullptr,nullptr,nullptr};
 cudaEvent_t xy_ready=nullptr,green_done[4]={nullptr,nullptr,nullptr,nullptr};
 bool green_streams_ok=false,green_events_ok=false;
#endif
 bool plan_ok=false,plan_z_ok=false,plan_xy_ok=false,blas_ok=false,event_ok=false,ready=false,solution_valid=false;
 unsigned long long matvec_count=0; double matvec_total_ms=0.0;
 bool last_end_valid=false; WallClock::time_point last_end{}; double last_print_ms=0.0;
} s;
std::string last_error;

void set_error(const char*w,const char*m){char b[1024];std::snprintf(b,sizeof(b),"%s: %s",w,m);last_error=b;}
void set_cuda_error(const char*w,cudaError_t e){set_error(w,cudaGetErrorString(e));}
const char*cufft_text(cufftResult r){switch(r){case CUFFT_SUCCESS:return"CUFFT_SUCCESS";case CUFFT_INVALID_PLAN:return"CUFFT_INVALID_PLAN";case CUFFT_ALLOC_FAILED:return"CUFFT_ALLOC_FAILED";case CUFFT_INVALID_TYPE:return"CUFFT_INVALID_TYPE";case CUFFT_INVALID_VALUE:return"CUFFT_INVALID_VALUE";case CUFFT_INTERNAL_ERROR:return"CUFFT_INTERNAL_ERROR";case CUFFT_EXEC_FAILED:return"CUFFT_EXEC_FAILED";case CUFFT_SETUP_FAILED:return"CUFFT_SETUP_FAILED";case CUFFT_INVALID_SIZE:return"CUFFT_INVALID_SIZE";default:return"CUFFT_ERROR";}}
void set_cufft_error(const char*w,cufftResult r){set_error(w,cufft_text(r));}
const char*cublas_text(cublasStatus_t r){switch(r){case CUBLAS_STATUS_SUCCESS:return"CUBLAS_STATUS_SUCCESS";case CUBLAS_STATUS_NOT_INITIALIZED:return"CUBLAS_STATUS_NOT_INITIALIZED";case CUBLAS_STATUS_ALLOC_FAILED:return"CUBLAS_STATUS_ALLOC_FAILED";case CUBLAS_STATUS_INVALID_VALUE:return"CUBLAS_STATUS_INVALID_VALUE";case CUBLAS_STATUS_ARCH_MISMATCH:return"CUBLAS_STATUS_ARCH_MISMATCH";case CUBLAS_STATUS_MAPPING_ERROR:return"CUBLAS_STATUS_MAPPING_ERROR";case CUBLAS_STATUS_EXECUTION_FAILED:return"CUBLAS_STATUS_EXECUTION_FAILED";case CUBLAS_STATUS_INTERNAL_ERROR:return"CUBLAS_STATUS_INTERNAL_ERROR";default:return"CUBLAS_STATUS_ERROR";}}
void set_cublas_error(const char*w,cublasStatus_t r){set_error(w,cublas_text(r));}

void release_state(){
 if(s.event_ok){cudaEventDestroy(s.ev0);cudaEventDestroy(s.ev1);}
#ifdef DDSCAT_CUDA_BACKEND_SLICES
 if(s.green_events_ok){if(s.xy_ready)cudaEventDestroy(s.xy_ready);for(int i=0;i<4;i++)if(s.green_done[i])cudaEventDestroy(s.green_done[i]);}
 if(s.green_streams_ok){for(int i=0;i<4;i++)if(s.green_stream[i])cudaStreamDestroy(s.green_stream[i]);}
#endif
 if(s.blas_ok)cublasDestroy(s.blas);
 if(s.plan_ok)cufftDestroy(s.plan);
 if(s.plan_z_ok)cufftDestroy(s.plan_z); if(s.plan_xy_ok)cufftDestroy(s.plan_xy);
 cudaFree(s.d_work);cudaFree(s.d_zwork);cudaFree(s.d_slice);cudaFree(s.d_x);cudaFree(s.d_y);cudaFree(s.d_adia);cudaFree(s.d_aoff);cudaFree(s.d_green);cudaFree(s.d_iocc);cudaFree(s.d_occ_index);
 cudaFree(s.d_sx);cudaFree(s.d_sy);cudaFree(s.d_b);cudaFree(s.d_wrk);cudaFree(s.d_cs);cudaFree(s.d_rs);cudaFree(s.d_dot_chunk_a);cudaFree(s.d_dot_chunk_b);
 if(s.h_resid)cudaFreeHost(s.h_resid); if(s.h_status)cudaFreeHost(s.h_status);
 s=State{};
}

__device__ __forceinline__ cufftComplex ca(cufftComplex a,cufftComplex b){return make_cuFloatComplex(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ cufftComplex csu(cufftComplex a,cufftComplex b){return make_cuFloatComplex(a.x-b.x,a.y-b.y);}
__device__ __forceinline__ cufftComplex cm(cufftComplex a,cufftComplex b){return cuCmulf(a,b);}
__device__ __forceinline__ cufftComplex cd(cufftComplex a,cufftComplex b){return cuCdivf(a,b);}
__device__ __forceinline__ cufftComplex cj(cufftComplex a){return make_cuFloatComplex(a.x,-a.y);}
__device__ __forceinline__ cufftComplex cr(cufftComplex a,float q){return make_cuFloatComplex(a.x*q,a.y*q);}
__device__ __forceinline__ float cabs2(cufftComplex a){return a.x*a.x+a.y*a.y;}
__device__ __forceinline__ bool cz(cufftComplex a){return a.x==0.f&&a.y==0.f;}

// Double-precision scalar algebra. Vectors remain cufftComplex/float32, but all
// Krylov recurrence coefficients and residual arithmetic use these helpers.
__device__ __forceinline__ cuDoubleComplex za(cuDoubleComplex a,cuDoubleComplex b){return make_cuDoubleComplex(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ cuDoubleComplex zsu(cuDoubleComplex a,cuDoubleComplex b){return make_cuDoubleComplex(a.x-b.x,a.y-b.y);}
__device__ __forceinline__ cuDoubleComplex zm(cuDoubleComplex a,cuDoubleComplex b){return make_cuDoubleComplex(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ __forceinline__ cuDoubleComplex zd(cuDoubleComplex a,cuDoubleComplex b){double d=b.x*b.x+b.y*b.y;return make_cuDoubleComplex((a.x*b.x+a.y*b.y)/d,(a.y*b.x-a.x*b.y)/d);}
__device__ __forceinline__ cuDoubleComplex zj(cuDoubleComplex a){return make_cuDoubleComplex(a.x,-a.y);}
__device__ __forceinline__ cuDoubleComplex zr(cuDoubleComplex a,double q){return make_cuDoubleComplex(a.x*q,a.y*q);}
__device__ __forceinline__ double zabs2(cuDoubleComplex a){return a.x*a.x+a.y*a.y;}
__device__ __forceinline__ bool zz(cuDoubleComplex a){return a.x==0.0&&a.y==0.0;}
__device__ __forceinline__ cufftComplex cmz(cuDoubleComplex a,cufftComplex b){double x=a.x*(double)b.x-a.y*(double)b.y;double y=a.x*(double)b.y+a.y*(double)b.x;return make_cuFloatComplex((float)x,(float)y);}
__device__ __forceinline__ cufftComplex czdiv(cufftComplex a,cuDoubleComplex b){double d=b.x*b.x+b.y*b.y;double x=((double)a.x*b.x+(double)a.y*b.y)/d;double y=((double)a.y*b.x-(double)a.x*b.y)/d;return make_cuFloatComplex((float)x,(float)y);}
__device__ __forceinline__ cufftComplex crd(cufftComplex a,double q){return make_cuFloatComplex((float)((double)a.x*q),(float)((double)a.y*q));}

__global__ void pad_kernel(const cufftComplex*x,cufftComplex*w,int nx,int ny,int nz,int nat,int cwhat){
 size_t gx=(size_t)2*nx,gy=(size_t)2*ny,ng=gx*gy*(size_t)(2*nz);size_t k=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(k>=ng)return;
 int ix=(int)(k%gx);size_t q=k/gx;int iy=(int)(q%gy),iz=(int)(q/gy);cufftComplex z=make_cuFloatComplex(0,0);w[k]=w[k+ng]=w[k+2*ng]=z;
 if(ix<nx&&iy<ny&&iz<nz){size_t j=(size_t)ix+(size_t)nx*((size_t)iy+(size_t)ny*iz);for(int m=0;m<3;m++){cufftComplex v=x[j+(size_t)m*nat];if(cwhat=='C')v=cj(v);w[k+(size_t)m*ng]=v;}}
}
__global__ void green_kernel(cufftComplex*w,const cufftComplex*g,int nx,int ny,int nz,int ipbc){
 int gx=2*nx,gy=2*ny,gz=2*nz;size_t ng=(size_t)gx*gy*gz,k=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(k>=ng)return;
 int ix=(int)(k%gx);size_t q=k/gx;int iy=(int)(q%gy),iz=(int)(q/gy);cufftComplex a11,a12,a13,a22,a23,a33;
 if(ipbc==0){int sx=ix<=nx?1:-1,sy=iy<=ny?1:-1,sz=iz<=nz?1:-1;int ir=min(ix,2*nx-ix),jr=min(iy,2*ny-iy),kr=min(iz,2*nz-iz);int cgx=nx+1,cgy=ny+1,cgz=nz+1;size_t b=(size_t)ir+(size_t)cgx*((size_t)jr+(size_t)cgy*kr),st=(size_t)cgx*cgy*cgz;a11=g[b];a12=cr(g[b+st],(float)(sx*sy));a13=cr(g[b+2*st],(float)(sx*sz));a22=g[b+3*st];a23=cr(g[b+4*st],(float)(sy*sz));a33=g[b+5*st];}
 else{size_t st=ng;a11=g[k];a12=g[k+st];a13=g[k+2*st];a22=g[k+3*st];a23=g[k+4*st];a33=g[k+5*st];}
 cufftComplex px=w[k],py=w[k+ng],pz=w[k+2*ng];w[k]=ca(ca(cm(a11,px),cm(a12,py)),cm(a13,pz));w[k+ng]=ca(ca(cm(a12,px),cm(a22,py)),cm(a23,pz));w[k+2*ng]=ca(ca(cm(a13,px),cm(a23,py)),cm(a33,pz));
}
__global__ void finish_kernel(const cufftComplex*x,cufftComplex*y,const cufftComplex*ad,const cufftComplex*ao,const cufftComplex*w,const int16_t*occ,int nx,int ny,int nz,int nat,int cwhat,float inv){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nat)return;if(occ&&occ[j]==0){y[j]=y[j+nat]=y[j+2*nat]=make_cuFloatComplex(0,0);return;}int ix=j%nx,q=j/nx,iy=q%ny,iz=q/ny;size_t ng=(size_t)(2*nx)*(2*ny)*(2*nz),k=(size_t)ix+(size_t)(2*nx)*((size_t)iy+(size_t)(2*ny)*iz);cufftComplex x1=x[j],x2=x[j+nat],x3=x[j+2*nat];if(cwhat=='C'){x1=cj(x1);x2=cj(x2);x3=cj(x3);}cufftComplex r1=csu(cm(ad[j],x1),cr(w[k],inv)),r2=csu(cm(ad[j+nat],x2),cr(w[k+ng],inv)),r3=csu(cm(ad[j+2*nat],x3),cr(w[k+2*ng],inv));cufftComplex a23=ao[j],a31=ao[j+nat],a12=ao[j+2*nat];r1=ca(r1,ca(cm(a31,x3),cm(a12,x2)));r2=ca(r2,ca(cm(a12,x1),cm(a23,x3)));r3=ca(r3,ca(cm(a23,x2),cm(a31,x1)));if(cwhat=='C'){r1=cj(r1);r2=cj(r2);r3=cj(r3);}y[j]=r1;y[j+nat]=r2;y[j+2*nat]=r3;
}

// ---------------- compact solver storage (occupied dipoles only) ----------------
// Compact layout keeps the DDSCAT component-major ordering:
//   compact[c*NAT0 + q] <-> full[c*NAT + occupied_index[q]].
__global__ void gather_full_to_compact_kernel(const cufftComplex*full,cufftComplex*compact,
                                               const int*occupied_index,int nat,int nat0){
 size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)3*nat0;if(q>=n)return;
 int c=(int)(q/(size_t)nat0),k=(int)(q-(size_t)c*nat0),j=occupied_index[k];
 compact[q]=full[(size_t)c*nat+j];
}
__global__ void scatter_compact_to_full_kernel(const cufftComplex*compact,cufftComplex*full,
                                                const int*occupied_index,int nat,int nat0){
 size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)3*nat0;if(q>=n)return;
 int c=(int)(q/(size_t)nat0),k=(int)(q-(size_t)c*nat0),j=occupied_index[k];
 full[(size_t)c*nat+j]=compact[q];
}

// ---------------- low-memory Z + XY-slice FFT MATVEC ----------------
// d_zwork layout: [component][xy][kz], kz contiguous.  This makes the
// 2*NZ-point FFTs contiguous while keeping only NX*NY transverse points.
__global__ void slice_pad_z_kernel(const cufftComplex*x,cufftComplex*zwork,
                                   int nx,int ny,int nz,int nat,int cwhat){
 size_t nxy=(size_t)nx*ny; int gz=2*nz;
 size_t total=3*nxy*(size_t)gz;
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=total)return;
 int kz=(int)(t%gz); size_t q=t/gz; size_t xy=q%nxy; int m=(int)(q/nxy);
 cufftComplex v=make_cuFloatComplex(0.f,0.f);
 if(kz<nz){size_t j=xy+nxy*(size_t)kz;v=x[j+(size_t)m*nat];if(cwhat=='C')v=cj(v);}
 zwork[t]=v;
}

// Gather up to four kz-frequency planes in one launch. Layout is
// [slice-in-batch][component][xy], so cufftPlanMany can transform all
// 4*3 planes in one 2-D batched execution. Unused tail slots are zeroed.
__global__ void slice_gather_xy_batch4_kernel(const cufftComplex*zwork,cufftComplex*slice,
                                              int nx,int ny,int nz,int kz0,int nb){
 int gx=2*nx,gy=2*ny,gz=2*nz;size_t area=(size_t)gx*gy,nxy=(size_t)nx*ny;
 size_t total=(size_t)4*3*area;size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=total)return;
 size_t plane=t/area,q=t-plane*area;int slot=(int)(plane/3),m=(int)(plane%3);
 int ix=(int)(q%gx),iy=(int)(q/gx);cufftComplex v=make_cuFloatComplex(0.f,0.f);
 if(slot<nb&&ix<nx&&iy<ny){size_t xy=(size_t)ix+(size_t)nx*iy;int kz=kz0+slot;v=zwork[((size_t)m*nxy+xy)*(size_t)gz+kz];}
 slice[t]=v;
}

// Multiply one full XY spectral slice by the already Fourier-transformed
// Green tensor.  IPBC=0 reconstructs the trimmed spectral tensor exactly as
// the original full-volume green_kernel; IPBC=1 indexes the full tensor.
__global__ void slice_green_xy_kernel(cufftComplex*slice,const cufftComplex*g,
                                      int nx,int ny,int nz,int ipbc,int kz){
 int gx=2*nx,gy=2*ny,gz=2*nz;size_t area=(size_t)gx*gy;
 size_t kxy=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(kxy>=area)return;
 int ix=(int)(kxy%gx),iy=(int)(kxy/gx);cufftComplex a11,a12,a13,a22,a23,a33;
 if(ipbc==0){
  int sx=ix<=nx?1:-1,sy=iy<=ny?1:-1,sz=kz<=nz?1:-1;
  int ir=min(ix,2*nx-ix),jr=min(iy,2*ny-iy),kr=min(kz,2*nz-kz);
  int cgx=nx+1,cgy=ny+1,cgz=nz+1;
  size_t b=(size_t)ir+(size_t)cgx*((size_t)jr+(size_t)cgy*kr),st=(size_t)cgx*cgy*cgz;
  a11=g[b];a12=cr(g[b+st],(float)(sx*sy));a13=cr(g[b+2*st],(float)(sx*sz));
  a22=g[b+3*st];a23=cr(g[b+4*st],(float)(sy*sz));a33=g[b+5*st];
 }else{
  size_t fullk=kxy+area*(size_t)kz,st=area*(size_t)gz;
  a11=g[fullk];a12=g[fullk+st];a13=g[fullk+2*st];a22=g[fullk+3*st];a23=g[fullk+4*st];a33=g[fullk+5*st];
 }
 cufftComplex px=slice[kxy],py=slice[kxy+area],pz=slice[kxy+2*area];
 slice[kxy]=ca(ca(cm(a11,px),cm(a12,py)),cm(a13,pz));
 slice[kxy+area]=ca(ca(cm(a12,px),cm(a22,py)),cm(a23,pz));
 slice[kxy+2*area]=ca(ca(cm(a13,px),cm(a23,py)),cm(a33,pz));
}

// Scatter up to four inverse-transformed XY planes in one launch.
__global__ void slice_scatter_xy_batch4_kernel(const cufftComplex*slice,cufftComplex*zwork,
                                               int nx,int ny,int nz,int kz0,int nb){
 size_t nxy=(size_t)nx*ny;int gx=2*nx,gz=2*nz;size_t area=(size_t)gx*(2*ny);
 size_t total=(size_t)nb*3*nxy;size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=total)return;
 size_t plane=t/nxy,xy=t-plane*nxy;int slot=(int)(plane/3),m=(int)(plane%3);
 int ix=(int)(xy%nx),iy=(int)(xy/nx),kz=kz0+slot;
 zwork[((size_t)m*nxy+xy)*(size_t)gz+kz]=slice[((size_t)slot*3+m)*area+(size_t)ix+(size_t)gx*iy];
}

__global__ void slice_finish_kernel(const cufftComplex*x,cufftComplex*y,
                                    const cufftComplex*ad,const cufftComplex*ao,
                                    const cufftComplex*zwork,const int16_t*occ,
                                    int nx,int ny,int nz,int nat,int cwhat,float inv){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nat)return;
 if(occ&&occ[j]==0){y[j]=y[j+nat]=y[j+2*nat]=make_cuFloatComplex(0,0);return;}
 int ix=j%nx,q=j/nx,iy=q%ny,iz=q/ny;size_t nxy=(size_t)nx*ny;int gz=2*nz;
 size_t xy=(size_t)ix+(size_t)nx*iy;
 cufftComplex x1=x[j],x2=x[j+nat],x3=x[j+2*nat];if(cwhat=='C'){x1=cj(x1);x2=cj(x2);x3=cj(x3);}
 cufftComplex w1=zwork[(xy)*(size_t)gz+iz];
 cufftComplex w2=zwork[(nxy+xy)*(size_t)gz+iz];
 cufftComplex w3=zwork[(2*nxy+xy)*(size_t)gz+iz];
 cufftComplex r1=csu(cm(ad[j],x1),cr(w1,inv)),r2=csu(cm(ad[j+nat],x2),cr(w2,inv)),r3=csu(cm(ad[j+2*nat],x3),cr(w3,inv));
 cufftComplex a23=ao[j],a31=ao[j+nat],a12=ao[j+2*nat];
 r1=ca(r1,ca(cm(a31,x3),cm(a12,x2)));r2=ca(r2,ca(cm(a12,x1),cm(a23,x3)));r3=ca(r3,ca(cm(a23,x2),cm(a31,x1)));
 if(cwhat=='C'){r1=cj(r1);r2=cj(r2);r3=cj(r3);}y[j]=r1;y[j+nat]=r2;y[j+2*nat]=r3;
}

__global__ void vcopy(cufftComplex*d,const cufftComplex*a,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=a[i];}
__global__ void vzero(cufftComplex*d,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=make_cuFloatComplex(0,0);}
__global__ void vdiff(cufftComplex*d,const cufftComplex*a,const cufftComplex*b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=csu(a[i],b[i]);}
__global__ void set_int(int*p,int v){if(!blockIdx.x&&!threadIdx.x)*p=v;}
__global__ void residual_kernel(const double*r,const double*b,double*out){if(!blockIdx.x&&!threadIdx.x)*out=(*b>0.0)?*r/ *b:INFINITY;}
constexpr int RELIABLE_RESID_PERIOD=20;
constexpr double RELIABLE_RESID_GAP=1.e-3;

__global__ void convert_dot_pair_f32_to_f64_kernel(const cufftComplex*a,const cufftComplex*b,
                                                     int offset,int count,
                                                     cuDoubleComplex*da,cuDoubleComplex*db){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){
  cufftComplex av=a[offset+i],bv=b[offset+i];
  da[i]=make_cuDoubleComplex((double)av.x,(double)av.y);
  db[i]=make_cuDoubleComplex((double)bv.x,(double)bv.y);
 }
}

// Reliable-residual helpers. They evaluate norms in true FP64 without
// allocating another full-size complex vector. Only a <=1M-element double
// staging chunk is used.
__global__ void convert_diff_f32_to_f64_kernel(const cufftComplex*a,const cufftComplex*b,
                                                int offset,int count,cuDoubleComplex*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){
  cufftComplex av=a[offset+i],bv=b[offset+i];
  out[i]=make_cuDoubleComplex((double)av.x-(double)bv.x,(double)av.y-(double)bv.y);
 }
}
__global__ void convert_residual_gap_f32_to_f64_kernel(const cufftComplex*rhs,const cufftComplex*ax,
                                                        const cufftComplex*rrec,int offset,int count,
                                                        cuDoubleComplex*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){
  cufftComplex bv=rhs[offset+i],av=ax[offset+i],rv=rrec[offset+i];
  out[i]=make_cuDoubleComplex((double)bv.x-(double)av.x-(double)rv.x,
                              (double)bv.y-(double)av.y-(double)rv.y);
 }
}

int launch_ok(const char*w){cudaError_t e=cudaGetLastError();if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}return 0;}

// Central scalar-product implementation used by every CUDA solver.
// Input vectors stay float32 on the GPU. Each block is converted by a CUDA
// kernel to persistent complex<double> staging buffers, reduced by cuBLAS Zdot,
// and the block results are accumulated on the CPU in double precision.
int dot_product_double_chunked(const cufftComplex*a,const cufftComplex*b,int n,bool conjugate,
                               cuDoubleComplex*host_total,const char*w){
 if(n<0||!a||!b||!host_total){set_error(w,"invalid chunked dot-product argument");return 1;}
 double sum_re=0.0,sum_im=0.0;
 const int threads=256;
 for(int offset=0;offset<n;offset+=s.dot_chunk_capacity){
  const int count=(n-offset>s.dot_chunk_capacity)?s.dot_chunk_capacity:(n-offset);
  const int blocks=(count+threads-1)/threads;
  convert_dot_pair_f32_to_f64_kernel<<<blocks,threads>>>(a,b,offset,count,s.d_dot_chunk_a,s.d_dot_chunk_b);
  if(launch_ok("convert float32 complex chunk to float64 complex"))return 1;
  cuDoubleComplex partial=make_cuDoubleComplex(0.0,0.0);
  cublasStatus_t r=conjugate
      ? cublasZdotc(s.blas,count,(const cuDoubleComplex*)s.d_dot_chunk_a,1,
                    (const cuDoubleComplex*)s.d_dot_chunk_b,1,&partial)
      : cublasZdotu(s.blas,count,(const cuDoubleComplex*)s.d_dot_chunk_a,1,
                    (const cuDoubleComplex*)s.d_dot_chunk_b,1,&partial);
  if(r!=CUBLAS_STATUS_SUCCESS){set_cublas_error(w,r);return 1;}
  // Pointer mode HOST makes each block result available here. The final
  // reduction across blocks is intentionally a CPU double-precision sum.
  sum_re+=cuCreal(partial);sum_im+=cuCimag(partial);
 }
 *host_total=make_cuDoubleComplex(sum_re,sum_im);
 return 0;
}
int copy_host_complex_scalar_to_device(const cuDoubleComplex&value,cuDoubleComplex*out,const char*w){
 cudaError_t e=cudaMemcpy(out,&value,sizeof(value),cudaMemcpyHostToDevice);
 if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}return 0;
}
int dotc(const cufftComplex*a,const cufftComplex*b,int n,cuDoubleComplex*out,const char*w){
 cuDoubleComplex total;
 if(dot_product_double_chunked(a,b,n,true,&total,w))return 1;
 return copy_host_complex_scalar_to_device(total,out,"copy accumulated cublasZdotc result H2D");
}
int dotu(const cufftComplex*a,const cufftComplex*b,int n,cuDoubleComplex*out,const char*w){
 cuDoubleComplex total;
 if(dot_product_double_chunked(a,b,n,false,&total,w))return 1;
 return copy_host_complex_scalar_to_device(total,out,"copy accumulated cublasZdotu result H2D");
}
int nrm2(const cufftComplex*a,int n,double*out,const char*w){
 // Norms reuse the same cublasZdotc path, so residuals are also based on a
 // genuine double-precision vector reduction rather than cublasScnrm2.
 cuDoubleComplex total;
 if(dot_product_double_chunked(a,a,n,true,&total,w))return 1;
 double norm2=cuCreal(total);if(norm2<0.0&&norm2>-1.e-12)norm2=0.0;
 const double norm=std::sqrt(norm2>0.0?norm2:0.0);
 cudaError_t e=cudaMemcpy(out,&norm,sizeof(norm),cudaMemcpyHostToDevice);
 if(e!=cudaSuccess){set_cuda_error("copy double norm H2D",e);return 1;}return 0;
}

int norm_diff_host(const cufftComplex*a,const cufftComplex*b,int n,double&out,const char*w){
 if(n<0||!a||!b){set_error(w,"invalid norm-difference argument");return 1;}
 const int threads=256;double sumsq=0.0;
 for(int offset=0;offset<n;offset+=s.dot_chunk_capacity){
  const int count=(n-offset>s.dot_chunk_capacity)?s.dot_chunk_capacity:(n-offset);
  const int blocks=(count+threads-1)/threads;
  convert_diff_f32_to_f64_kernel<<<blocks,threads>>>(a,b,offset,count,s.d_dot_chunk_a);
  if(launch_ok("convert difference float32 to float64"))return 1;
  double partial=0.0;cublasStatus_t r=cublasDznrm2(s.blas,count,s.d_dot_chunk_a,1,&partial);
  if(r!=CUBLAS_STATUS_SUCCESS){set_cublas_error(w,r);return 1;}sumsq+=partial*partial;
 }
 out=std::sqrt(std::max(0.0,sumsq));return 0;
}
int norm_residual_gap_host(const cufftComplex*rhs,const cufftComplex*ax,const cufftComplex*rrec,
                           int n,double&out,const char*w){
 if(n<0||!rhs||!ax||!rrec){set_error(w,"invalid residual-gap argument");return 1;}
 const int threads=256;double sumsq=0.0;
 for(int offset=0;offset<n;offset+=s.dot_chunk_capacity){
  const int count=(n-offset>s.dot_chunk_capacity)?s.dot_chunk_capacity:(n-offset);
  const int blocks=(count+threads-1)/threads;
  convert_residual_gap_f32_to_f64_kernel<<<blocks,threads>>>(rhs,ax,rrec,offset,count,s.d_dot_chunk_a);
  if(launch_ok("convert residual gap float32 to float64"))return 1;
  double partial=0.0;cublasStatus_t r=cublasDznrm2(s.blas,count,s.d_dot_chunk_a,1,&partial);
  if(r!=CUBLAS_STATUS_SUCCESS){set_cublas_error(w,r);return 1;}sumsq+=partial*partial;
 }
 out=std::sqrt(std::max(0.0,sumsq));return 0;
}

using hcomplex=std::complex<double>;
inline hcomplex hc(cuDoubleComplex z){return hcomplex(cuCreal(z),cuCimag(z));}
inline cuDoubleComplex dc(hcomplex z){return make_cuDoubleComplex(z.real(),z.imag());}
int dotc_host(const cufftComplex*a,const cufftComplex*b,int n,hcomplex&out,const char*w){
 cuDoubleComplex z;if(dot_product_double_chunked(a,b,n,true,&z,w))return 1;out=hc(z);return 0;
}
int dotu_host(const cufftComplex*a,const cufftComplex*b,int n,hcomplex&out,const char*w){
 cuDoubleComplex z;if(dot_product_double_chunked(a,b,n,false,&z,w))return 1;out=hc(z);return 0;
}
int norm_host(const cufftComplex*a,int n,double&out,const char*w){
 hcomplex z;if(dotc_host(a,a,n,z,w))return 1;double v=z.real();if(v<0.0&&v>-1.e-12)v=0.0;out=std::sqrt(std::max(0.0,v));return 0;
}
bool solve_small(hcomplex*a,hcomplex*b,int n,int ld){
 for(int k=0;k<n;k++){
  int piv=k;double best=std::abs(a[k*ld+k]);
  for(int i=k+1;i<n;i++){double v=std::abs(a[i*ld+k]);if(v>best){best=v;piv=i;}}
  if(best<1.e-30)return false;
  if(piv!=k){for(int j=k;j<n;j++)std::swap(a[k*ld+j],a[piv*ld+j]);std::swap(b[k],b[piv]);}
  for(int i=k+1;i<n;i++){hcomplex f=a[i*ld+k]/a[k*ld+k];a[i*ld+k]=0.0;for(int j=k+1;j<n;j++)a[i*ld+j]-=f*a[k*ld+j];b[i]-=f*b[k];}
 }
 for(int i=n-1;i>=0;i--){hcomplex v=b[i];for(int j=i+1;j<n;j++)v-=a[i*ld+j]*b[j];if(std::abs(a[i*ld+i])<1.e-30)return false;b[i]=v/a[i*ld+i];}
 return true;
}

#ifdef DDSCAT_CUDA_BACKEND_SLICES
int matvec_device(int cwhat,const cufftComplex*x,cufftComplex*y,float*ms){
 cudaError_t e=cudaEventRecord(s.ev0);if(e!=cudaSuccess){set_cuda_error("event start",e);return 20;}
 const int th=256,gz=2*s.nz,gx=2*s.nx,gy=2*s.ny;
 const size_t nxy=(size_t)s.nx*s.ny,area=(size_t)gx*gy;
 int gbz=(int)((s.zwork_count+th-1)/th);
 slice_pad_z_kernel<<<gbz,th>>>(x,s.d_zwork,s.nx,s.ny,s.nz,s.nat,cwhat);if(launch_ok("slice pad Z"))return 21;
 cufftResult fr=cufftExecC2C(s.plan_z,s.d_zwork,s.d_zwork,CUFFT_INVERSE);
 if(fr!=CUFFT_SUCCESS){set_cufft_error("slice cuFFT Z +",fr);return 22;}
 const size_t batch4_count=(size_t)4*3*area;
 int gbgather=(int)((batch4_count+th-1)/th),gbgreen=(int)((area+th-1)/th);
 for(int kz0=0;kz0<gz;kz0+=4){
  const int nb=(gz-kz0<4)?(gz-kz0):4;
  slice_gather_xy_batch4_kernel<<<gbgather,th>>>(s.d_zwork,s.d_slice,s.nx,s.ny,s.nz,kz0,nb);if(launch_ok("slice gather XY batch4"))return 23;
  fr=cufftExecC2C(s.plan_xy,s.d_slice,s.d_slice,CUFFT_INVERSE);if(fr!=CUFFT_SUCCESS){set_cufft_error("slice cuFFT XY batch4 +",fr);return 24;}

  // The four Green products are independent. Record completion of the batched
  // forward FFT on the default stream, then let four persistent nonblocking
  // streams process kz0..kz0+3 concurrently with the unchanged slice kernel.
  e=cudaEventRecord(s.xy_ready,0);if(e!=cudaSuccess){set_cuda_error("record XY-ready event",e);return 25;}
  for(int slot=0;slot<nb;slot++){
   e=cudaStreamWaitEvent(s.green_stream[slot],s.xy_ready,0);if(e!=cudaSuccess){set_cuda_error("Green stream wait XY-ready",e);return 25;}
   cufftComplex* slice_ptr=s.d_slice+(size_t)slot*3*area;
   slice_green_xy_kernel<<<gbgreen,th,0,s.green_stream[slot]>>>(slice_ptr,s.d_green,s.nx,s.ny,s.nz,s.ipbc,kz0+slot);
   if(launch_ok("slice Green XY stream"))return 25;
   e=cudaEventRecord(s.green_done[slot],s.green_stream[slot]);if(e!=cudaSuccess){set_cuda_error("record Green-done event",e);return 25;}
  }
  // The default stream waits on all active Green kernels without blocking CPU.
  for(int slot=0;slot<nb;slot++){e=cudaStreamWaitEvent(0,s.green_done[slot],0);if(e!=cudaSuccess){set_cuda_error("main stream wait Green-done",e);return 25;}}

  fr=cufftExecC2C(s.plan_xy,s.d_slice,s.d_slice,CUFFT_FORWARD);if(fr!=CUFFT_SUCCESS){set_cufft_error("slice cuFFT XY batch4 -",fr);return 26;}
  const size_t phys=(size_t)nb*3*nxy;int gbphys=(int)((phys+th-1)/th);
  slice_scatter_xy_batch4_kernel<<<gbphys,th>>>(s.d_slice,s.d_zwork,s.nx,s.ny,s.nz,kz0,nb);if(launch_ok("slice scatter XY batch4"))return 27;
 }
 fr=cufftExecC2C(s.plan_z,s.d_zwork,s.d_zwork,CUFFT_FORWARD);
 if(fr!=CUFFT_SUCCESS){set_cufft_error("slice cuFFT Z -",fr);return 28;}
 int gn=(s.nat+th-1)/th;
 slice_finish_kernel<<<gn,th>>>(x,y,s.d_adia,s.d_aoff,s.d_zwork,s.d_iocc,s.nx,s.ny,s.nz,s.nat,cwhat,1.f/(float)s.ngrid);
 if(launch_ok("slice finish"))return 29;
 e=cudaEventRecord(s.ev1);if(e!=cudaSuccess){set_cuda_error("event stop",e);return 30;}
 e=cudaEventSynchronize(s.ev1);if(e!=cudaSuccess){set_cuda_error("event sync",e);return 31;}
 float xms=0;e=cudaEventElapsedTime(&xms,s.ev0,s.ev1);if(e!=cudaSuccess){set_cuda_error("event elapsed",e);return 32;}
 s.matvec_count++;s.matvec_total_ms+=xms;if(ms)*ms=xms;return 0;
}
#else
int matvec_device(int cwhat,const cufftComplex*x,cufftComplex*y,float*ms){
 cudaError_t e=cudaEventRecord(s.ev0);if(e!=cudaSuccess){set_cuda_error("event start",e);return 20;}
 int th=256,gb=(int)((s.ngrid+th-1)/th);
 pad_kernel<<<gb,th>>>(x,s.d_work,s.nx,s.ny,s.nz,s.nat,cwhat);if(launch_ok("pad_kernel"))return 21;
 cufftResult fr=cufftExecC2C(s.plan,s.d_work,s.d_work,CUFFT_INVERSE);if(fr!=CUFFT_SUCCESS){set_cufft_error("batched 3D cuFFT +",fr);return 22;}
 green_kernel<<<gb,th>>>(s.d_work,s.d_green,s.nx,s.ny,s.nz,s.ipbc);if(launch_ok("green_kernel"))return 23;
 fr=cufftExecC2C(s.plan,s.d_work,s.d_work,CUFFT_FORWARD);if(fr!=CUFFT_SUCCESS){set_cufft_error("batched 3D cuFFT -",fr);return 24;}
 int gn=(s.nat+th-1)/th;finish_kernel<<<gn,th>>>(x,y,s.d_adia,s.d_aoff,s.d_work,s.d_iocc,s.nx,s.ny,s.nz,s.nat,cwhat,1.f/(float)s.ngrid);if(launch_ok("finish_kernel"))return 25;
 e=cudaEventRecord(s.ev1);if(e!=cudaSuccess){set_cuda_error("event stop",e);return 26;}e=cudaEventSynchronize(s.ev1);if(e!=cudaSuccess){set_cuda_error("event sync",e);return 27;}float xms=0;e=cudaEventElapsedTime(&xms,s.ev0,s.ev1);if(e!=cudaSuccess){set_cuda_error("event elapsed",e);return 28;}s.matvec_count++;s.matvec_total_ms+=xms;if(ms)*ms=xms;return 0;
}
#endif
void report_matvec(int cwhat,float gpu,const char*path){auto now=WallClock::now();double dt=-1;if(s.last_end_valid){dt=std::chrono::duration<double,std::milli>(now-s.last_end).count()-s.last_print_ms;if(dt<0)dt=0;}auto p0=WallClock::now();if(s.last_end_valid)std::printf("CUDA MATVEC %llu [%c,%s]: GPU=%.6f ms  prev_end->end=%.6f ms  outside=%.6f ms\n",s.matvec_count,(char)cwhat,path,(double)gpu,dt,dt-gpu);else std::printf("CUDA MATVEC %llu [%c,%s]: GPU=%.6f ms  prev_end->end=N/A\n",s.matvec_count,(char)cwhat,path,(double)gpu);std::fflush(stdout);auto p1=WallClock::now();s.last_print_ms=std::chrono::duration<double,std::milli>(p1-p0).count();s.last_end=now;s.last_end_valid=true;}
int sync_status(const char*w){cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error(w,e);return -1;}return *s.h_status;}

// ---------------- GPBICG: float32 vectors, double-complex recurrences ----------------
enum G{GR0RN=0,GBETA,GALPHA,GETA,GDZ,G0,G1,G2,G3,G4,G5,GCOUNT};
__global__ void g_init(const cufftComplex*b,const cufftComplex*ax,cufftComplex*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;cufftComplex z=make_cuFloatComplex(0,0),r=csu(b[i],ax[i]);w[i]=r;w[i+n]=z;w[i+2*(size_t)n]=r;w[i+4*(size_t)n]=z;w[i+7*(size_t)n]=z;w[i+8*(size_t)n]=z;w[i+9*(size_t)n]=z;}
__global__ void g_restart_from_r(const cufftComplex*rt,cufftComplex*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;cufftComplex z=make_cuFloatComplex(0,0),r=rt[i];w[i]=r;w[i+(size_t)n]=z;w[i+2*(size_t)n]=r;w[i+3*(size_t)n]=z;w[i+4*(size_t)n]=z;w[i+5*(size_t)n]=z;w[i+6*(size_t)n]=z;w[i+7*(size_t)n]=z;w[i+8*(size_t)n]=z;w[i+9*(size_t)n]=z;w[i+11*(size_t)n]=z;}
__global__ void g_p(cufftComplex*w,cufftComplex*xi,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex p=ca(w[i+2*(size_t)n],cmz(sc[GBETA],csu(w[i+n],w[i+7*(size_t)n])));w[i+n]=p;xi[i]=p;}}
__global__ void g_alpha(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){if(zz(sc[G0])){*st=1;return;}sc[GALPHA]=zd(sc[GR0RN],sc[G0]);}}
__global__ void g_yt(cufftComplex*w,cufftComplex*xi,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex r=w[i+2*(size_t)n],t=w[i+4*(size_t)n],ap=w[i+5*(size_t)n],ww=w[i+8*(size_t)n];cufftComplex y=ca(csu(t,r),cmz(sc[GALPHA],csu(ap,ww))),tn=csu(r,cmz(sc[GALPHA],ap));w[i+3*(size_t)n]=y;w[i+11*(size_t)n]=t;w[i+4*(size_t)n]=tn;xi[i]=tn;}}
__global__ void g_dz0(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){if(zz(sc[G1])){*st=2;return;}sc[GETA]=make_cuDoubleComplex(0,0);sc[GDZ]=zd(sc[G4],sc[G1]);}}
__global__ void g_dze(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){cuDoubleComplex den=zsu(zm(sc[G1],sc[G2]),zm(sc[G3],zj(sc[G3])));if(zz(den)){*st=3;return;}sc[GDZ]=zd(zsu(zm(sc[G2],sc[G4]),zm(sc[G5],sc[G3])),den);sc[GETA]=zd(zsu(zm(sc[G1],sc[G5]),zm(zj(sc[G3]),sc[G4])),den);}}
__global__ void g_up(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex r=w[i+2*(size_t)n],ap=w[i+5*(size_t)n],at=w[i+6*(size_t)n],uo=w[i+7*(size_t)n],zo=w[i+9*(size_t)n],to=w[i+11*(size_t)n],t=w[i+4*(size_t)n],y=w[i+3*(size_t)n];cufftComplex u=ca(cmz(sc[GDZ],ap),cmz(sc[GETA],ca(csu(to,r),cmz(sc[GBETA],uo))));cufftComplex z=csu(ca(cmz(sc[GDZ],r),cmz(sc[GETA],zo)),cmz(sc[GALPHA],u));w[i+10*(size_t)n]=ca(w[i+10*(size_t)n],ca(cmz(sc[GALPHA],w[i+n]),z));w[i+2*(size_t)n]=csu(csu(t,cmz(sc[GETA],y)),cmz(sc[GDZ],at));w[i+7*(size_t)n]=u;w[i+9*(size_t)n]=z;}}
__global__ void g_beta(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){if(zz(sc[GR0RN])||zz(sc[GDZ])){*st=4;return;}sc[GBETA]=zd(zd(zm(sc[GALPHA],sc[G0]),sc[GDZ]),sc[GR0RN]);sc[GR0RN]=sc[G0];}}
__global__ void g_w(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)w[i+8*(size_t)n]=ca(w[i+6*(size_t)n],cmz(sc[GBETA],w[i+5*(size_t)n]));}
__global__ void set_c(cuDoubleComplex*p,double a,double b){if(!blockIdx.x&&!threadIdx.x)*p=make_cuDoubleComplex(a,b);}

// ---------------- QMRCCG ----------------
enum QCS{QLAMBDA=0,QKAPPA,QTHETA,QGAMMA,QKSI,QRHO,QEPS,QMU,QTAU,QGAMMA0,QKSI0,QRHO0,QTAU0,QKAPPA0,QNEW_RHO,QNEW_EPS,QCOUNT};
enum QRS{QBNORM=0,QGNORM,QKNORM,QRNORM,QRCOUNT};
__global__ void q_init_vec(const cufftComplex*b,const cufftComplex*ax,cufftComplex*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex z=make_cuFloatComplex(0,0),r=csu(b[i],ax[i]);w[i]=r;w[i+n]=z;w[i+2*(size_t)n]=z;w[i+3*(size_t)n]=z;w[i+4*(size_t)n]=z;w[i+5*(size_t)n]=z;w[i+6*(size_t)n]=r;w[i+7*(size_t)n]=r;w[i+8*(size_t)n]=z;}}
__global__ void q_restart_from_r(const cufftComplex*rt,cufftComplex*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex z=make_cuFloatComplex(0,0),r=rt[i];w[i]=r;w[i+(size_t)n]=z;w[i+2*(size_t)n]=z;w[i+3*(size_t)n]=z;w[i+4*(size_t)n]=z;w[i+5*(size_t)n]=z;w[i+6*(size_t)n]=r;w[i+7*(size_t)n]=r;w[i+8*(size_t)n]=z;}}
__global__ void q_init_sc(cuDoubleComplex*sc,const double*rs,int*st){if(!blockIdx.x&&!threadIdx.x){*st=0;sc[QLAMBDA]=make_cuDoubleComplex(1,0);sc[QKAPPA]=make_cuDoubleComplex(-1,0);sc[QTHETA]=make_cuDoubleComplex(-1,0);sc[QGAMMA]=make_cuDoubleComplex(rs[QGNORM],0);sc[QKSI]=make_cuDoubleComplex(rs[QKNORM],0);sc[QMU]=make_cuDoubleComplex(0,0);if(zz(sc[QRHO])){*st=5;return;}if(zz(sc[QGAMMA])){*st=6;return;}if(zz(sc[QKSI])){*st=7;return;}sc[QTAU]=zd(sc[QEPS],sc[QRHO]);}}
__global__ void q_pq(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){w[i+n]=csu(czdiv(w[i+6*(size_t)n],sc[QGAMMA]),cmz(sc[QMU],w[i+n]));w[i+3*(size_t)n]=czdiv(csu(w[i+8*(size_t)n],cmz(zm(sc[QGAMMA],sc[QMU]),w[i+3*(size_t)n])),sc[QKSI]);}}
__global__ void q_update_vw(cufftComplex*w,const cufftComplex*ap,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){w[i+2*(size_t)n]=ap[i];w[i+6*(size_t)n]=csu(ap[i],cmz(zd(sc[QTAU],sc[QGAMMA]),w[i+6*(size_t)n]));w[i+7*(size_t)n]=csu(w[i+3*(size_t)n],cmz(zd(sc[QTAU],sc[QKSI]),w[i+7*(size_t)n]));}}
__global__ void q_recur(cuDoubleComplex*sc,const double*rs,int*st){if(!blockIdx.x&&!threadIdx.x){*st=0;sc[QGAMMA0]=sc[QGAMMA];sc[QKSI0]=sc[QKSI];sc[QRHO0]=sc[QRHO];sc[QTAU0]=sc[QTAU];sc[QKAPPA0]=sc[QKAPPA];sc[QGAMMA]=make_cuDoubleComplex(rs[QGNORM],0);sc[QKSI]=make_cuDoubleComplex(rs[QKNORM],0);sc[QRHO]=sc[QNEW_RHO];sc[QEPS]=sc[QNEW_EPS];if(zz(sc[QGAMMA])){*st=6;return;}if(zz(sc[QKSI])){*st=7;return;}cuDoubleComplex den=zm(zm(sc[QGAMMA],sc[QTAU]),sc[QRHO0]);if(zz(den)){*st=12;return;}sc[QMU]=zd(zm(zm(sc[QGAMMA0],sc[QKSI0]),sc[QRHO]),den);if(zz(sc[QRHO])){*st=13;return;}sc[QTAU]=zsu(zd(sc[QEPS],sc[QRHO]),zm(sc[QGAMMA],sc[QMU]));double at0=zabs2(sc[QTAU0]),ag=zabs2(sc[QGAMMA]);cuDoubleComplex den2=za(zr(sc[QLAMBDA],at0),make_cuDoubleComplex(ag,0));if(zz(den2)){*st=14;return;}sc[QTHETA]=zd(zr(zsu(make_cuDoubleComplex(1,0),sc[QLAMBDA]),at0),den2);sc[QKAPPA]=zd(zr(zm(zm(sc[QGAMMA0],zj(sc[QTAU0])),sc[QKAPPA0]),-1.0),den2);sc[QLAMBDA]=zd(zr(sc[QLAMBDA],at0),den2);}}
__global__ void q_update_xr(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex d=ca(cmz(sc[QTHETA],w[i+4*(size_t)n]),cmz(sc[QKAPPA],w[i+n]));cufftComplex sv=ca(cmz(sc[QTHETA],w[i+5*(size_t)n]),cmz(sc[QKAPPA],w[i+2*(size_t)n]));w[i+4*(size_t)n]=d;w[i+5*(size_t)n]=sv;w[i+9*(size_t)n]=ca(w[i+9*(size_t)n],d);w[i]=csu(w[i],sv);}}

// ---------------- PBCGST ----------------
enum PCS{PRHO=20,PALPHA,POMEGA,PBETA,PXI,PTT,PTS,PNEW_RHO,PCOUNT};
enum PRS{PBNORM=4,PSNORM=5,PTRNORM=6};
__global__ void pbc_precon(cufftComplex*out,const cufftComplex*in,const cufftComplex*adia,const int*occ_index,int nat,int nat0,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int c=i/nat0,k=i-c*nat0,j=occ_index[k];out[i]=cd(in[i],adia[(size_t)c*nat+j]);}}
__global__ void pbc_restart_init(const cufftComplex*rpre,cufftComplex*w,cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex z=make_cuFloatComplex(0,0);w[i]=rpre[i];w[i+(size_t)n]=rpre[i];w[i+2*(size_t)n]=z;w[i+4*(size_t)n]=z;}if(i==0){sc[PRHO]=make_cuDoubleComplex(1,0);sc[PALPHA]=make_cuDoubleComplex(1,0);sc[POMEGA]=make_cuDoubleComplex(1,0);}}
__global__ void pbc_beta(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){*st=0;cuDoubleComplex kap=zm(sc[PRHO],sc[POMEGA]);if(zz(kap)){*st=6;return;}sc[PBETA]=zd(zm(sc[PNEW_RHO],sc[PALPHA]),kap);sc[PRHO]=sc[PNEW_RHO];}}
__global__ void pbc_p(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex r=w[i],p=w[i+2*(size_t)n],v=w[i+4*(size_t)n];w[i+2*(size_t)n]=ca(r,cmz(sc[PBETA],csu(p,cmz(sc[POMEGA],v))));}}
__global__ void pbc_alpha(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){*st=0;if(zz(sc[PXI])){*st=10;return;}sc[PALPHA]=zd(sc[PRHO],sc[PXI]);}}
__global__ void pbc_s(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)w[i+3*(size_t)n]=csu(w[i],cmz(sc[PALPHA],w[i+4*(size_t)n]));}
__global__ void pbc_soft_update(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)w[i+6*(size_t)n]=ca(w[i+6*(size_t)n],cmz(sc[PALPHA],w[i+2*(size_t)n]));}
__global__ void pbc_omega(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){*st=0;if(zz(sc[PTT])){*st=14;return;}sc[POMEGA]=zd(sc[PTS],sc[PTT]);}}
__global__ void pbc_update(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex p=w[i+2*(size_t)n],sv=w[i+3*(size_t)n],t=w[i+5*(size_t)n];w[i+6*(size_t)n]=ca(w[i+6*(size_t)n],ca(cmz(sc[PALPHA],p),cmz(sc[POMEGA],sv)));w[i]=csu(sv,cmz(sc[POMEGA],t));}}
__global__ void real_to_mapped(const double*in,double*out){if(!blockIdx.x&&!threadIdx.x)*out=*in;}

// ---------------- PBCGS2 / ZBCG2, BiCGStab(2) ----------------
enum P2CS {P2_ALPHA=0,P2_BETA,P2_OMEGA,P2_RHO0,P2_RHO1,P2_SIGMA,P2_Z11,P2_Z21,P2_Z22,P2_Z31,P2_Z32,P2_Z33,P2_Y1,P2_Y2,P2CS_COUNT};
enum P2RS {P2_RNRM0=0,P2_RNRM,P2_MXNRMX,P2_MXNRMR,P2RS_COUNT};
__global__ void p2_init_vectors(const cufftComplex*rhs,cufftComplex*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;cufftComplex z=make_cuFloatComplex(0,0),b=rhs[i],x0=w[i+9*(size_t)n];w[i]=b;w[i+(size_t)n]=b;w[i+2*(size_t)n]=z;w[i+3*(size_t)n]=z;w[i+4*(size_t)n]=z;w[i+5*(size_t)n]=z;w[i+6*(size_t)n]=z;w[i+7*(size_t)n]=x0;w[i+8*(size_t)n]=b;w[i+9*(size_t)n]=z;}
__global__ void p2_init_scalars(cuDoubleComplex*sc){if(!blockIdx.x&&!threadIdx.x){sc[P2_ALPHA]=make_cuDoubleComplex(0,0);sc[P2_OMEGA]=make_cuDoubleComplex(1,0);sc[P2_SIGMA]=make_cuDoubleComplex(1,0);sc[P2_RHO0]=make_cuDoubleComplex(1,0);}}
__global__ void p2_init_reals(double*rs){if(!blockIdx.x&&!threadIdx.x){rs[P2_RNRM]=rs[P2_RNRM0];rs[P2_MXNRMX]=rs[P2_RNRM0];rs[P2_MXNRMR]=rs[P2_RNRM0];}}
__global__ void p2_outer_start(cuDoubleComplex*sc){if(!blockIdx.x&&!threadIdx.x)sc[P2_RHO0]=zr(zm(sc[P2_OMEGA],sc[P2_RHO0]),-1.0);}
__global__ void p2_beta(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){*st=0;if(zz(sc[P2_RHO0])){*st=1;return;}sc[P2_BETA]=zm(sc[P2_ALPHA],zd(sc[P2_RHO1],sc[P2_RHO0]));sc[P2_RHO0]=sc[P2_RHO1];}}
__global__ void p2_update_u(cufftComplex*w,const cuDoubleComplex*sc,int n,int k){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;for(int j=0;j<k;j++)w[i+(size_t)(4+j)*n]=csu(w[i+(size_t)(1+j)*n],cmz(sc[P2_BETA],w[i+(size_t)(4+j)*n]));}
__global__ void p2_alpha(cuDoubleComplex*sc,int*st){if(!blockIdx.x&&!threadIdx.x){*st=0;if(zz(sc[P2_SIGMA])){*st=2;return;}sc[P2_ALPHA]=zd(sc[P2_RHO1],sc[P2_SIGMA]);}}
__global__ void p2_update_x_r(cufftComplex*w,const cuDoubleComplex*sc,int n,int k){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;w[i+9*(size_t)n]=ca(w[i+9*(size_t)n],cmz(sc[P2_ALPHA],w[i+4*(size_t)n]));for(int j=0;j<k;j++)w[i+(size_t)(1+j)*n]=csu(w[i+(size_t)(1+j)*n],cmz(sc[P2_ALPHA],w[i+(size_t)(5+j)*n]));}
__global__ void p2_update_max(double*rs){if(!blockIdx.x&&!threadIdx.x){rs[P2_MXNRMX]=fmax(rs[P2_MXNRMX],rs[P2_RNRM]);rs[P2_MXNRMR]=fmax(rs[P2_MXNRMR],rs[P2_RNRM]);}}
__device__ cuDoubleComplex p2_inner3_coeff(const cuDoubleComplex*a,const cuDoubleComplex*b){cuDoubleComplex z=make_cuDoubleComplex(0,0);for(int i=0;i<3;i++)z=za(z,zm(zj(a[i]),b[i]));return z;}
__global__ void p2_convex_scalars(cuDoubleComplex*sc,double*rs,int*st){if(!blockIdx.x&&!threadIdx.x){*st=0;cuDoubleComplex Z[3][3];Z[0][0]=sc[P2_Z11];Z[1][0]=sc[P2_Z21];Z[2][0]=sc[P2_Z31];Z[0][1]=zj(sc[P2_Z21]);Z[1][1]=sc[P2_Z22];Z[2][1]=sc[P2_Z32];Z[0][2]=zj(sc[P2_Z31]);Z[1][2]=zj(sc[P2_Z32]);Z[2][2]=sc[P2_Z33];if(zz(Z[1][1])){*st=3;return;}cuDoubleComplex y0[3]={make_cuDoubleComplex(-1,0),zd(Z[1][0],Z[1][1]),make_cuDoubleComplex(0,0)};cuDoubleComplex yl[3]={make_cuDoubleComplex(0,0),zd(Z[1][2],Z[1][1]),make_cuDoubleComplex(-1,0)};cuDoubleComplex zy0[3],zyl[3];for(int i=0;i<3;i++){zy0[i]=make_cuDoubleComplex(0,0);zyl[i]=make_cuDoubleComplex(0,0);for(int j=0;j<3;j++){zy0[i]=za(zy0[i],zm(Z[i][j],y0[j]));zyl[i]=za(zyl[i],zm(Z[i][j],yl[j]));}}cuDoubleComplex q0=p2_inner3_coeff(y0,zy0),ql=p2_inner3_coeff(yl,zyl);double k0=sqrt(sqrt(zabs2(q0))),kl=sqrt(sqrt(zabs2(ql)));if(k0==0.0||kl==0.0){*st=4;return;}cuDoubleComplex vr=zd(p2_inner3_coeff(yl,zy0),make_cuDoubleComplex(k0*kl,0));double avr=sqrt(zabs2(vr));if(avr==0.0){*st=5;return;}cuDoubleComplex hg=zr(vr,fmax(avr,0.7)/avr),fac=zr(hg,k0/kl);for(int i=0;i<3;i++)y0[i]=zsu(y0[i],zm(fac,yl[i]));sc[P2_Y1]=y0[1];sc[P2_Y2]=y0[2];sc[P2_OMEGA]=y0[2];for(int i=0;i<3;i++){zy0[i]=make_cuDoubleComplex(0,0);for(int j=0;j<3;j++)zy0[i]=za(zy0[i],zm(Z[i][j],y0[j]));}cuDoubleComplex qr=p2_inner3_coeff(y0,zy0);rs[P2_RNRM]=sqrt(sqrt(zabs2(qr)));rs[P2_MXNRMX]=fmax(rs[P2_MXNRMX],rs[P2_RNRM]);rs[P2_MXNRMR]=fmax(rs[P2_MXNRMR],rs[P2_RNRM]);}}
__global__ void p2_convex_update(cufftComplex*w,const cuDoubleComplex*sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;cufftComplex r0=w[i+(size_t)n],r1=w[i+2*(size_t)n],r2=w[i+3*(size_t)n],u0=w[i+4*(size_t)n],u1=w[i+5*(size_t)n],u2=w[i+6*(size_t)n];w[i+4*(size_t)n]=csu(csu(u0,cmz(sc[P2_Y1],u1)),cmz(sc[P2_Y2],u2));w[i+9*(size_t)n]=ca(w[i+9*(size_t)n],ca(cmz(sc[P2_Y1],r0),cmz(sc[P2_Y2],r1)));w[i+(size_t)n]=csu(csu(r0,cmz(sc[P2_Y1],r1)),cmz(sc[P2_Y2],r2));}
__global__ void p2_flags(const double*rs,double*resid,int*flags){if(!blockIdx.x&&!threadIdx.x){const double delta=1.e-2,r0=rs[P2_RNRM0],r=rs[P2_RNRM];bool xpdt=(r<delta*r0&&r0<rs[P2_MXNRMX]);bool rcmp=((r<delta*rs[P2_MXNRMR]&&r0<rs[P2_MXNRMR])||xpdt);*flags=(rcmp?1:0)|(xpdt?2:0);*resid=(r0>0.0)?r/r0:0.0;}}
__global__ void p2_reliable_r(cufftComplex*w,const cufftComplex*ax,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)w[i+(size_t)n]=csu(w[i+8*(size_t)n],ax[i]);}
__global__ void p2_xpdt(cufftComplex*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){w[i+7*(size_t)n]=ca(w[i+9*(size_t)n],w[i+7*(size_t)n]);w[i+9*(size_t)n]=make_cuFloatComplex(0,0);w[i+8*(size_t)n]=w[i+(size_t)n];}}
__global__ void p2_reliable_reals(double*rs,int xpdt){if(!blockIdx.x&&!threadIdx.x){rs[P2_MXNRMR]=rs[P2_RNRM];if(xpdt)rs[P2_MXNRMX]=rs[P2_RNRM];}}
__global__ void p2_finalize_x(cufftComplex*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)w[i+9*(size_t)n]=ca(w[i+9*(size_t)n],w[i+7*(size_t)n]);}
__global__ void p2_restart_vectors(cufftComplex*w,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){cufftComplex r=w[i+(size_t)n],z=make_cuFloatComplex(0,0);w[i]=r;w[i+2*(size_t)n]=z;w[i+3*(size_t)n]=z;w[i+4*(size_t)n]=z;w[i+5*(size_t)n]=z;w[i+6*(size_t)n]=z;}}
__global__ void p2_restart_reals(double*rs,double rnorm){if(!blockIdx.x&&!threadIdx.x){rs[P2_RNRM]=rnorm;rs[P2_MXNRMX]=rnorm;rs[P2_MXNRMR]=rnorm;}}

// ---------------- PETRKP ----------------
enum PC {PC_BB=0,PC_GG,PC_OLDGG,PC_QQ,PC_RR};
enum PR {PR_ALPHA=0,PR_BETA};
__global__ void petr_init_vectors(const cufftComplex*ace_in,cufftComplex*wrk,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;cufftComplex v=ace_in[i];wrk[i]=v;wrk[i+(size_t)n]=v;wrk[i+2*(size_t)n]=v;}
__global__ void petr_save_gigi(cuDoubleComplex*cs){if(!blockIdx.x&&!threadIdx.x)cs[PC_OLDGG]=cs[PC_GG];}
__global__ void petr_alpha(const cuDoubleComplex*cs,double*rs,int*status){if(blockIdx.x||threadIdx.x)return;double q=cs[PC_QQ].x,g=cs[PC_GG].x;if(q==0.0){*status=1;return;}rs[PR_ALPHA]=g/q;}
__global__ void petr_beta(const cuDoubleComplex*cs,double*rs,int*status){if(blockIdx.x||threadIdx.x)return;double old=cs[PC_OLDGG].x,g=cs[PC_GG].x;if(old==0.0){*status=2;return;}rs[PR_BETA]=g/old;}
__global__ void petr_residual(const cuDoubleComplex*cs,double*out){if(blockIdx.x||threadIdx.x)return;double b=cs[PC_BB].x,r=cs[PC_RR].x;*out=(b>0.0)?sqrt(fmax(r,0.0)/b):INFINITY;}
__global__ void petr_x_update(cufftComplex*x,const cufftComplex*p,const double*rs,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=ca(x[i],crd(p[i],rs[PR_ALPHA]));}
__global__ void petr_g_update(cufftComplex*g,const cufftComplex*ace,const cufftComplex*ahax,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)g[i]=csu(ace[i],ahax[i]);}
__global__ void petr_p_update(cufftComplex*p,const cufftComplex*g,const double*rs,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=ca(g[i],crd(p[i],rs[PR_BETA]));}
__global__ void petr_axi_update(cufftComplex*axi,const cufftComplex*qi,const double*rs,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)axi[i]=ca(axi[i],crd(qi[i],rs[PR_ALPHA]));}

// ---------------- IFDDA-derived BiCGStab(L) / GPBiCGStab(L) helpers ----------------
__global__ void l_scale(cufftComplex*d,const cufftComplex*a,cuDoubleComplex z,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=cmz(z,a[i]);}
__global__ void l_scale_self(cufftComplex*d,cuDoubleComplex z,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=cmz(z,d[i]);}
__global__ void l_axpy(cufftComplex*d,const cufftComplex*a,cuDoubleComplex z,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=ca(d[i],cmz(z,a[i]));}
__global__ void l_affine(cufftComplex*d,const cufftComplex*a,cuDoubleComplex z,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=ca(a[i],cmz(z,d[i]));}
__global__ void l_lincomb2(cufftComplex*d,const cufftComplex*a,const cufftComplex*b,cuDoubleComplex za,cuDoubleComplex zb,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=ca(cmz(za,a[i]),cmz(zb,b[i]));}
__global__ void l_axpy2(cufftComplex*d,const cufftComplex*a,const cufftComplex*b,cuDoubleComplex za,cuDoubleComplex zb,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=ca(d[i],ca(cmz(za,a[i]),cmz(zb,b[i])));}
__global__ void l_self_add2(cufftComplex*d,const cufftComplex*a,const cufftComplex*b,cuDoubleComplex zd0,cuDoubleComplex za,cuDoubleComplex zb,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=ca(cmz(zd0,d[i]),ca(cmz(za,a[i]),cmz(zb,b[i])));}

int op_copy(cufftComplex*d,const cufftComplex*a,int n,const char*w){int th=256,gr=(n+th-1)/th;vcopy<<<gr,th>>>(d,a,n);return launch_ok(w);}
int op_zero(cufftComplex*d,int n,const char*w){int th=256,gr=(n+th-1)/th;vzero<<<gr,th>>>(d,n);return launch_ok(w);}
int op_scale(cufftComplex*d,const cufftComplex*a,hcomplex z,int n,const char*w){int th=256,gr=(n+th-1)/th;l_scale<<<gr,th>>>(d,a,dc(z),n);return launch_ok(w);}
int op_scale_self(cufftComplex*d,hcomplex z,int n,const char*w){int th=256,gr=(n+th-1)/th;l_scale_self<<<gr,th>>>(d,dc(z),n);return launch_ok(w);}
int op_axpy(cufftComplex*d,const cufftComplex*a,hcomplex z,int n,const char*w){int th=256,gr=(n+th-1)/th;l_axpy<<<gr,th>>>(d,a,dc(z),n);return launch_ok(w);}
int op_affine(cufftComplex*d,const cufftComplex*a,hcomplex z,int n,const char*w){int th=256,gr=(n+th-1)/th;l_affine<<<gr,th>>>(d,a,dc(z),n);return launch_ok(w);}
int op_lincomb2(cufftComplex*d,const cufftComplex*a,const cufftComplex*b,hcomplex za,hcomplex zb,int n,const char*w){int th=256,gr=(n+th-1)/th;l_lincomb2<<<gr,th>>>(d,a,b,dc(za),dc(zb),n);return launch_ok(w);}
int op_axpy2(cufftComplex*d,const cufftComplex*a,const cufftComplex*b,hcomplex za,hcomplex zb,int n,const char*w){int th=256,gr=(n+th-1)/th;l_axpy2<<<gr,th>>>(d,a,b,dc(za),dc(zb),n);return launch_ok(w);}
int op_self_add2(cufftComplex*d,const cufftComplex*a,const cufftComplex*b,hcomplex zs,hcomplex za,hcomplex zb,int n,const char*w){int th=256,gr=(n+th-1)/th;l_self_add2<<<gr,th>>>(d,a,b,dc(zs),dc(za),dc(zb),n);return launch_ok(w);}

int gather_full_to_compact(const cufftComplex*full,cufftComplex*compact,const char*w){
 const int n=s.solver_n,th=256,gr=(n+th-1)/th;
 gather_full_to_compact_kernel<<<gr,th>>>(full,compact,s.d_occ_index,s.nat,s.nat0);return launch_ok(w);
}
int scatter_compact_to_full(const cufftComplex*compact,cufftComplex*full,const char*w){
 const size_t full_bytes=(size_t)3*s.nat*sizeof(cufftComplex);cudaError_t e=cudaMemset(full,0,full_bytes);
 if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}const int n=s.solver_n,th=256,gr=(n+th-1)/th;
 scatter_compact_to_full_kernel<<<gr,th>>>(compact,full,s.d_occ_index,s.nat,s.nat0);return launch_ok(w);
}
int upload_full_host_to_compact(const void*host,cufftComplex*compact,const char*w){
 const size_t full_bytes=(size_t)3*s.nat*sizeof(cufftComplex);cudaError_t e=cudaMemcpy(s.d_x,host,full_bytes,cudaMemcpyHostToDevice);
 if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}return gather_full_to_compact(s.d_x,compact,w);
}
int download_compact_to_full_host(const cufftComplex*compact,void*host,const char*w){
 if(scatter_compact_to_full(compact,s.d_x,w))return 1;const size_t full_bytes=(size_t)3*s.nat*sizeof(cufftComplex);
 cudaError_t e=cudaMemcpy(host,s.d_x,full_bytes,cudaMemcpyDeviceToHost);if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}return 0;
}
int upload_full_operators(const void*adia,const void*aoff,const char*w){
 const size_t full_bytes=(size_t)3*s.nat*sizeof(cufftComplex);cudaError_t e=cudaMemcpy(s.d_adia,adia,full_bytes,cudaMemcpyHostToDevice);
 if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}e=cudaMemcpy(s.d_aoff,aoff,full_bytes,cudaMemcpyHostToDevice);
 if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}return 0;
}
int matvec_solver_device(int cwhat,const cufftComplex*compact_in,cufftComplex*compact_out,float*ms){
 if(scatter_compact_to_full(compact_in,s.d_x,"solver compact->full MATVEC"))return 1;
 int rc=matvec_device(cwhat,s.d_x,s.d_y,ms);if(rc)return rc;
 return gather_full_to_compact(s.d_y,compact_out,"solver full->compact MATVEC");
}
int run_mv(const cufftComplex*in,cufftComplex*out,const char*label,int&nc){float ms=0.f;int rc=matvec_solver_device('N',in,out,&ms);if(rc)return rc;++nc;report_matvec('N',ms,label);return 0;}

struct ReliableResidualResult{double true_res=INFINITY,gap=INFINITY;bool converged=false,restart=false;};
int reliable_residual_check(const char*solver,const char*mvlabel,const cufftComplex*rhs,const cufftComplex*xcur,
                            const cufftComplex*rrec,int n,double rhsnorm,double recursive_res,double tol,
                            int&nc,ReliableResidualResult&rr){
 if(rhsnorm<=0.0){set_error(solver,"reliable residual: zero RHS norm");return 1;}
 int rc=run_mv(xcur,s.d_sy,mvlabel,nc);if(rc)return rc;
 double tr=0.0,gapn=0.0;if(norm_diff_host(rhs,s.d_sy,n,tr,"reliable true-residual norm"))return 1;
 if(norm_residual_gap_host(rhs,s.d_sy,rrec,n,gapn,"reliable residual-gap norm"))return 1;
 rr.true_res=tr/rhsnorm;rr.gap=(tr>0.0)?gapn/tr:((gapn==0.0)?0.0:INFINITY);
 rr.converged=(rr.true_res<=tol);
 rr.restart=(rr.gap>=RELIABLE_RESID_GAP)||((recursive_res<=tol)&&!rr.converged);
 std::printf(" %s reliable: true=%14.6E gap=%14.6E%s\n",solver,rr.true_res,rr.gap,rr.restart?" RESTART":"");std::fflush(stdout);
 return 0;
}
int materialize_true_residual(const cufftComplex*rhs,int n,const char*w){
 int th=256,gr=(n+th-1)/th;vdiff<<<gr,th>>>(s.d_sy,rhs,s.d_sy,n);return launch_ok(w);
}
int ensure_krylov_workspace(const char*w);
void release_krylov_workspace(const char*why);
int publish_solution(const cufftComplex*src,const char*w);
int init_solver_boundary(const void*b,void*x,const void*adia,const void*aoff,int full_n,const char*name){
 if(full_n!=3*s.nat){set_error(name,"unexpected full solver dimension");return 1;}
 if(ensure_krylov_workspace(name))return 1;s.solution_valid=false;
 if(upload_full_host_to_compact(b,s.d_b,name))return 1;if(upload_full_host_to_compact(x,s.d_sx,name))return 1;
 if(upload_full_operators(adia,aoff,name))return 1;cudaError_t e=cudaMemset(s.d_wrk,0,13*(size_t)s.solver_n*sizeof(cufftComplex));
 if(e!=cudaSuccess){set_cuda_error(name,e);return 1;}return 0;
}
int initial_true_residual(cufftComplex*r,int n,int&nc,const char*label){int rc=run_mv(s.d_sx,s.d_sy,label,nc);if(rc)return rc;int th=256,gr=(n+th-1)/th;vdiff<<<gr,th>>>(r,s.d_b,s.d_sy,n);return launch_ok("initial true residual");}

int ensure_krylov_workspace(const char*w){
 if(s.d_wrk)return 0;
 const size_t bytes=(size_t)13*s.solver_n*sizeof(cufftComplex);
 cudaError_t e=cudaMalloc((void**)&s.d_wrk,bytes);
 if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}
 std::printf("CUDA solver Krylov workspace reallocated: %.3f MiB\n",(double)bytes/(1024.0*1024.0));std::fflush(stdout);
 return 0;
}
void release_krylov_workspace(const char*why){
 if(!s.d_wrk)return;
 const size_t bytes=(size_t)13*s.solver_n*sizeof(cufftComplex);
 cudaFree(s.d_wrk);s.d_wrk=nullptr;
 std::printf("CUDA postprocess: released Krylov workspace %.3f MiB (%s).\n",(double)bytes/(1024.0*1024.0),why?why:"postprocess");std::fflush(stdout);
}
int publish_solution(const cufftComplex*src,const char*w){
 if(!src){set_error(w,"null solution pointer");return 1;}
 if(src!=s.d_sx){
  cudaError_t e=cudaMemcpy(s.d_sx,src,(size_t)s.solver_n*sizeof(cufftComplex),cudaMemcpyDeviceToDevice);
  if(e!=cudaSuccess){set_cuda_error(w,e);return 1;}
 }
 s.solution_valid=true;return 0;
}


// ================= CUDA-resident Green tensor construction =================
// This code is used only during prepare. s.d_green is the only Green allocation
// that survives. Green-only scratch and the temporary cuFFT plan are destroyed
// before d_zwork/d_slice or any solver buffer is allocated.
struct GreenBuildParams {
 int nx,ny,nz,ipbc,idipint;
 double gamma,pyd,pzd,akx,aky,akz,dx1,dx2,dx3,akd,akd2,pyddx,pzddx;
};
__device__ __forceinline__ cuDoubleComplex gz_add(cuDoubleComplex a,cuDoubleComplex b){return make_cuDoubleComplex(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ cuDoubleComplex gz_sub(cuDoubleComplex a,cuDoubleComplex b){return make_cuDoubleComplex(a.x-b.x,a.y-b.y);}
__device__ __forceinline__ cuDoubleComplex gz_mul(cuDoubleComplex a,cuDoubleComplex b){return make_cuDoubleComplex(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ __forceinline__ cuDoubleComplex gz_scale(cuDoubleComplex a,double q){return make_cuDoubleComplex(a.x*q,a.y*q);}
__device__ __forceinline__ cuDoubleComplex gz_div(cuDoubleComplex a,cuDoubleComplex b){double d=b.x*b.x+b.y*b.y;return make_cuDoubleComplex((a.x*b.x+a.y*b.y)/d,(a.y*b.x-a.x*b.y)/d);}
__device__ __forceinline__ cuDoubleComplex gz_exp_i(double phase,double damp){double e=exp(-damp);return make_cuDoubleComplex(e*cos(phase),e*sin(phase));}
__device__ bool green_cisi(double x,double&ci,double&si){
 const double EPS=6.e-8,EULER=0.57721566,PIBY2=1.5707963,FPMIN=1.e-30,TMIN=2.0;const int MAXIT=100;
 double t=fabs(x);if(t==0.0){si=0.0;ci=-1.0/FPMIN;return true;}
 if(t>TMIN){
  cuDoubleComplex b=make_cuDoubleComplex(1.0,t),c=make_cuDoubleComplex(1.0/FPMIN,0.0),d=gz_div(make_cuDoubleComplex(1.0,0.0),b),h=d;
  bool ok=false;for(int ii=2;ii<=MAXIT;ii++){double a=-(double)(ii-1)*(double)(ii-1);b=gz_add(b,make_cuDoubleComplex(2.0,0.0));d=gz_div(make_cuDoubleComplex(1.0,0.0),gz_add(gz_scale(d,a),b));c=gz_add(b,gz_div(make_cuDoubleComplex(a,0.0),c));cuDoubleComplex del=gz_mul(c,d);h=gz_mul(h,del);cuDoubleComplex dm=gz_sub(del,make_cuDoubleComplex(1.0,0.0));if(fabs(dm.x)+fabs(dm.y)<EPS){ok=true;break;}}
  if(!ok)return false;h=gz_mul(make_cuDoubleComplex(cos(t),-sin(t)),h);ci=-h.x;si=PIBY2+h.y;
 }else{
  if(t<sqrt(FPMIN)){ci=log(t)+EULER;si=t;}else{double su=0.0,sumc=0.0,sums=0.0,sgn=1.0,fact=1.0;bool odd=true,ok=false;for(int k=1;k<=MAXIT;k++){fact=fact*t/(double)k;double term=fact/(double)k;su+=sgn*term;double err=(su!=0.0)?fabs(term/su):fabs(term);if(odd){sgn=-sgn;sums=su;su=sumc;}else{sumc=su;su=sums;}if(err<EPS){ok=true;break;}odd=!odd;}if(!ok)return false;si=sums;ci=sumc+log(t)+EULER;}
 }
 if(x<0.0)si=-si;return true;
}
__device__ void green_accumulate_interaction(double x,double y,double z,double phasyz,double damp,const GreenBuildParams&p,cuDoubleComplex c[6],int*status){
 double r2=x*x+y*y+z*z;if(r2<=1.e-6)return;double r=sqrt(r2),r3=r*r2,kr=p.akd*r;
 if(p.idipint==0){
  cuDoubleComplex ph=gz_scale(gz_exp_i(kr+phasyz,damp),1.0/r3);cuDoubleComplex fac=make_cuDoubleComplex(1.0/r2,-kr/r2);double xx[3]={x,y,z};int m=0;
  for(int ii=0;ii<3;ii++){double q=xx[ii]*xx[ii];cuDoubleComplex term=gz_add(make_cuDoubleComplex(p.akd2*(q-r2),0.0),gz_scale(fac,r2-3.0*q));c[m]=gz_sub(c[m],gz_mul(ph,term));m++;if(ii<2)for(int jj=ii+1;jj<3;jj++){cuDoubleComplex term2=gz_sub(make_cuDoubleComplex(p.akd2,0.0),gz_scale(fac,3.0));c[m]=gz_sub(c[m],gz_scale(gz_mul(ph,term2),xx[ii]*xx[jj]));m++;}}
 }else if(p.idipint==1){
  const double pi=4.0*atan(1.0),kf=pi,kfr=kf*r;double cip,sip,cim,sim;if(!green_cisi(kfr+kr,cip,sip)||!green_cisi(kfr-kr,cim,sim)){atomicCAS(status,0,11);return;}
  double sinkr=sin(kr),coskr=cos(kr),sinkfr=sin(kfr),coskfr=cos(kfr);double a0=sinkr*(cip-cim)+coskr*(pi-sip-sim);double a1=p.akd*sinkr*(-pi+sip+sim)+p.akd*coskr*(cip-cim)-2.0*sinkfr/r;double a2=p.akd2*(sinkr*(cim-cip)+coskr*(sip+sim-pi))+2.0*(sinkfr-kfr*coskfr)/r2;
  cuDoubleComplex eikr=make_cuDoubleComplex(coskr,sinkr),ph=gz_exp_i(phasyz,damp);cuDoubleComplex g0=gz_scale(gz_sub(eikr,make_cuDoubleComplex(a0/pi,0.0)),1.0/r);cuDoubleComplex g1=gz_scale(gz_add(gz_mul(eikr,make_cuDoubleComplex(-1.0,kr)),make_cuDoubleComplex((a0-a1*r)/pi,0.0)),1.0/r2);cuDoubleComplex g2=gz_scale(gz_sub(gz_mul(eikr,make_cuDoubleComplex(2.0-kr*kr,-2.0*kr)),make_cuDoubleComplex((2.0*(a0-r*a1)+r2*a2)/pi,0.0)),1.0/r3);double h2=(sinkfr-kfr*coskfr)/(pi*r3);cuDoubleComplex radial=gz_sub(gz_scale(g2,1.0/r2),gz_scale(g1,1.0/r3));double xx[3]={x,y,z};int m=0;
  for(int ii=0;ii<3;ii++){cuDoubleComplex base=gz_add(gz_add(gz_scale(g0,p.akd2),gz_scale(g1,1.0/r)),make_cuDoubleComplex((2.0/3.0)*h2,0.0));base=gz_add(base,gz_scale(radial,xx[ii]*xx[ii]));c[m]=gz_add(c[m],gz_mul(ph,base));m++;if(ii<2)for(int jj=ii+1;jj<3;jj++){c[m]=gz_add(c[m],gz_scale(gz_mul(ph,radial),xx[ii]*xx[jj]));m++;}}
 }else atomicCAS(status,0,12);
}
__device__ void green_sum_lag(int lx,int ly,int lz,const GreenBuildParams&p,cuDoubleComplex c[6],int*status){
 for(int m=0;m<6;m++)c[m]=make_cuDoubleComplex(0.0,0.0);double x0=(double)lx*p.dx1,y0=(double)ly*p.dx2,z0=(double)lz*p.dx3;int jpbc=0,jpy1=0,jpy2=0,jpz1=0,jpz2=0;double kperp=0.0,krf=0.0,phasf=0.0,term0=0.0,term1=0.0,term2y=0.0,term2z=0.0;
 if(p.ipbc>0){double kp2;if(p.pyddx<=0.0){jpbc=2;kp2=p.akx*p.akx+p.aky*p.aky;}else if(p.pzddx<=0.0){jpbc=1;kp2=p.akx*p.akx+p.akz*p.akz;}else{jpbc=3;kp2=p.akx*p.akx;}if(kp2<=0.0){atomicCAS(status,0,13);return;}kperp=sqrt(kp2);double phimax=1.0/p.gamma;term0=p.akd2*phimax*phimax/kp2;term1=2.0*p.akd*phimax;term2y=p.aky*phimax/kp2;term2z=p.akz*phimax/kp2;double rperp,yf=0.0,zf=0.0,term3;
  if(jpbc==1){rperp=sqrt(x0*x0+z0*z0);krf=p.akd2*rperp/kperp;yf=y0-p.aky*rperp/kperp;phasf=p.aky*yf;term3=sqrt(term0+term1*rperp)/kperp;jpy1=(int)((yf-term2y-term3)/p.pyddx+0.999999);jpy2=(int)((yf-term2y+term3)/p.pyddx);}
  else if(jpbc==2){rperp=sqrt(x0*x0+y0*y0);krf=p.akd2*rperp/kperp;zf=z0-p.akz*rperp/kperp;phasf=p.akz*zf;term3=sqrt(term0+term1*rperp)/kperp;jpz1=(int)((zf-term2z-term3)/p.pzddx+0.999999);jpz2=(int)((zf-term2z+term3)/p.pzddx);}
  else{rperp=fabs(x0);krf=p.akd2*rperp/kperp;yf=y0-p.aky*rperp/kperp;zf=z0-p.akz*rperp/kperp;phasf=p.aky*yf+p.akz*zf;term3=sqrt(term0+term1*rperp)/kperp;jpy1=(int)((yf-term2y-term3)/p.pyddx+0.999999);jpy2=(int)((yf-term2y+term3)/p.pyddx);jpz1=(int)((zf-term2z-term3)/p.pzddx+0.999999);jpz2=(int)((zf-term2z+term3)/p.pzddx);}
 }
 double phimax=(p.gamma>0.0)?1.0/p.gamma:0.0,beta=(p.ipbc>0)?16.0/(phimax*phimax*phimax*phimax):0.0;
 for(int jpy=jpy1;jpy<=jpy2;jpy++){double rjpy=(double)jpy*p.pyddx*p.dx2,y=y0-rjpy,phasy=p.aky*rjpy;for(int jpz=jpz1;jpz<=jpz2;jpz++){double rjpz=(double)jpz*p.pzddx*p.dx3,z=z0-rjpz,r2=x0*x0+y*y+z*z;if(r2<=1.e-6)continue;double r=sqrt(r2),kr=p.akd*r,phasyz=phasy+p.akz*rjpz,damp=0.0;if(jpbc!=0){double phi=kr-krf+phasyz-phasf;damp=beta*phi*phi*phi*phi;}green_accumulate_interaction(x0,y,z,phasyz,damp,p,c,status);}}
}
__global__ void green_fill_component_kernel(cufftComplex*dst,int component,GreenBuildParams p,int*status){size_t ng=(size_t)(2*p.nx)*(2*p.ny)*(2*p.nz),k=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(k>=ng)return;int gx=2*p.nx,gy=2*p.ny,ix=(int)(k%gx);size_t q=k/gx;int iy=(int)(q%gy),iz=(int)(q/gy);if(ix==p.nx||iy==p.ny||iz==p.nz){dst[k]=make_cuFloatComplex(0,0);return;}int lx=(ix<p.nx)?ix:ix-2*p.nx,ly=(iy<p.ny)?iy:iy-2*p.ny,lz=(iz<p.nz)?iz:iz-2*p.nz;cuDoubleComplex c[6];green_sum_lag(lx,ly,lz,p,c,status);dst[k]=make_cuFloatComplex((float)c[component].x,(float)c[component].y);}
__global__ void green_fill_all_kernel(cufftComplex*dst,GreenBuildParams p,int*status){size_t ng=(size_t)(2*p.nx)*(2*p.ny)*(2*p.nz),k=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(k>=ng)return;int gx=2*p.nx,gy=2*p.ny,ix=(int)(k%gx);size_t q=k/gx;int iy=(int)(q%gy),iz=(int)(q/gy);if(ix==p.nx||iy==p.ny||iz==p.nz){for(int m=0;m<6;m++)dst[k+(size_t)m*ng]=make_cuFloatComplex(0,0);return;}int lx=(ix<p.nx)?ix:ix-2*p.nx,ly=(iy<p.ny)?iy:iy-2*p.ny,lz=(iz<p.nz)?iz:iz-2*p.nz;cuDoubleComplex c[6];green_sum_lag(lx,ly,lz,p,c,status);for(int m=0;m<6;m++)dst[k+(size_t)m*ng]=make_cuFloatComplex((float)c[m].x,(float)c[m].y);}
__global__ void green_trim_kernel(const cufftComplex*full,cufftComplex*trim,int nx,int ny,int nz){size_t ox=(size_t)nx+1,oy=(size_t)ny+1,oz=(size_t)nz+1,n=ox*oy*oz,id=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;int ix=(int)(id%ox);size_t q=id/ox;int iy=(int)(q%oy),iz=(int)(q/oy);size_t fi=(size_t)ix+(size_t)(2*nx)*((size_t)iy+(size_t)(2*ny)*iz);trim[id]=full[fi];}
int build_green_cuda_resident(const GreenBuildParams&p){
 auto t0=WallClock::now();cudaError_t e;int*d_status=nullptr;int h_status=0;e=cudaMalloc((void**)&d_status,sizeof(int));if(e!=cudaSuccess){set_cuda_error("Green temporary status allocation",e);return 1;}cudaMemset(d_status,0,sizeof(int));const size_t ng=(size_t)(2*p.nx)*(2*p.ny)*(2*p.nz);const int th=256,gb=(int)((ng+th-1)/th);cufftHandle gp=0;cufftResult fr;
 if(p.ipbc==0){cufftComplex*tmp=nullptr;e=cudaMalloc((void**)&tmp,ng*sizeof(cufftComplex));if(e!=cudaSuccess){cudaFree(d_status);set_cuda_error("Green temporary full-grid allocation",e);return 2;}fr=cufftPlan3d(&gp,2*p.nz,2*p.ny,2*p.nx,CUFFT_C2C);if(fr!=CUFFT_SUCCESS){cudaFree(tmp);cudaFree(d_status);set_cufft_error("Green cufftPlan3d",fr);return 3;}size_t oct=(size_t)(p.nx+1)*(p.ny+1)*(p.nz+1);int gt=(int)((oct+th-1)/th);for(int m=0;m<6;m++){green_fill_component_kernel<<<gb,th>>>(tmp,m,p,d_status);if(launch_ok("Green direct kernel")){cufftDestroy(gp);cudaFree(tmp);cudaFree(d_status);return 4;}fr=cufftExecC2C(gp,tmp,tmp,CUFFT_INVERSE);if(fr!=CUFFT_SUCCESS){cufftDestroy(gp);cudaFree(tmp);cudaFree(d_status);set_cufft_error("Green forward 3D FFT",fr);return 5;}green_trim_kernel<<<gt,th>>>(tmp,s.d_green+(size_t)m*oct,p.nx,p.ny,p.nz);if(launch_ok("Green trim kernel")){cufftDestroy(gp);cudaFree(tmp);cudaFree(d_status);return 6;}}e=cudaDeviceSynchronize();cufftDestroy(gp);cudaFree(tmp);if(e!=cudaSuccess){cudaFree(d_status);set_cuda_error("Green isolated synchronize",e);return 7;}}
 else{cudaMemset(s.d_green,0,6*ng*sizeof(cufftComplex));green_fill_all_kernel<<<gb,th>>>(s.d_green,p,d_status);if(launch_ok("Green PBC direct kernel")){cudaFree(d_status);return 8;}if(ng>(size_t)INT_MAX){cudaFree(d_status);set_error("Green cufftPlanMany","FFT grid exceeds CUFFT int distance");return 9;}int dims[3]={2*p.nz,2*p.ny,2*p.nx},dist=(int)ng;fr=cufftPlanMany(&gp,3,dims,dims,1,dist,dims,1,dist,CUFFT_C2C,6);if(fr!=CUFFT_SUCCESS){cudaFree(d_status);set_cufft_error("Green cufftPlanMany batch=6",fr);return 10;}fr=cufftExecC2C(gp,s.d_green,s.d_green,CUFFT_INVERSE);if(fr!=CUFFT_SUCCESS){cufftDestroy(gp);cudaFree(d_status);set_cufft_error("Green batched 3D FFT",fr);return 11;}e=cudaDeviceSynchronize();cufftDestroy(gp);if(e!=cudaSuccess){cudaFree(d_status);set_cuda_error("Green PBC synchronize",e);return 12;}}
 e=cudaMemcpy(&h_status,d_status,sizeof(int),cudaMemcpyDeviceToHost);cudaFree(d_status);if(e!=cudaSuccess){set_cuda_error("Green status D2H",e);return 13;}if(h_status){char msg[160];std::snprintf(msg,sizeof(msg),"Green kernel status=%d (11=CISI,12=IDIPINT,13=KPERP)",h_status);set_error("CUDA Green",msg);return 14;}double ms=std::chrono::duration<double,std::milli>(WallClock::now()-t0).count();s.green_build_ms=ms;s.green_built_on_gpu=true;std::printf("CUDA Green build time: %.6f s (%.3f ms); Green scratch and temporary cuFFT plan have been freed.\n",ms/1000.0,ms);std::fflush(stdout);return 0;
}
// ================= end CUDA-resident Green construction =================


int allocate_runtime_after_green(int nx,int ny,int nz,int nat,int nat0){
 const size_t n=(size_t)3*nat,cn=(size_t)3*nat0;cudaError_t e;if(cn>(size_t)INT_MAX){set_error("allocate runtime","3*NAT0 exceeds solver int range");return 204;}s.solver_n=(int)cn;s.dot_chunk_capacity=(int)std::min((size_t)DOT_CHUNK_COMPLEX,cn);if(s.dot_chunk_capacity<1){set_error("allocate runtime","NAT0 must be positive");return 204;}
#define MALR(ptr,bytes,label) do{cudaError_t me=cudaMalloc((void**)&(ptr),(bytes));if(me!=cudaSuccess){set_cuda_error(label,me);return 204;}}while(0)
#ifdef DDSCAT_CUDA_BACKEND_SLICES
 s.zwork_count=(size_t)3*(2*nz)*(size_t)nx*ny;s.slice_count=(size_t)4*3*(2*nx)*(size_t)(2*ny);
 MALR(s.d_zwork,s.zwork_count*sizeof(cufftComplex),"malloc slice Z workspace");MALR(s.d_slice,s.slice_count*sizeof(cufftComplex),"malloc slice XY slice");
#else
 MALR(s.d_work,3*s.ngrid*sizeof(cufftComplex),"malloc FFT3D work");
#endif
 MALR(s.d_x,n*sizeof(cufftComplex),"malloc full MATVEC x");MALR(s.d_y,n*sizeof(cufftComplex),"malloc full MATVEC y");MALR(s.d_adia,n*sizeof(cufftComplex),"malloc adia");MALR(s.d_aoff,n*sizeof(cufftComplex),"malloc aoff");MALR(s.d_iocc,(size_t)nat*sizeof(int16_t),"malloc iocc");MALR(s.d_occ_index,(size_t)nat0*sizeof(int),"malloc occupied-index map");MALR(s.d_sx,cn*sizeof(cufftComplex),"malloc compact solver x");MALR(s.d_sy,cn*sizeof(cufftComplex),"malloc compact solver y");MALR(s.d_b,cn*sizeof(cufftComplex),"malloc compact solver b");MALR(s.d_wrk,13*cn*sizeof(cufftComplex),"malloc compact solver work");MALR(s.d_cs,32*sizeof(cuDoubleComplex),"malloc scalar complex double");MALR(s.d_rs,8*sizeof(double),"malloc scalar real double");MALR(s.d_dot_chunk_a,(size_t)s.dot_chunk_capacity*sizeof(cuDoubleComplex),"malloc double dot chunk A");MALR(s.d_dot_chunk_b,(size_t)s.dot_chunk_capacity*sizeof(cuDoubleComplex),"malloc double dot chunk B");
#undef MALR
 e=cudaHostAlloc((void**)&s.h_resid,sizeof(double),cudaHostAllocMapped);if(e!=cudaSuccess){set_cuda_error("mapped residual",e);return 205;}e=cudaHostGetDevicePointer((void**)&s.d_resid_map,s.h_resid,0);if(e!=cudaSuccess){set_cuda_error("map residual",e);return 205;}e=cudaHostAlloc((void**)&s.h_status,sizeof(int),cudaHostAllocMapped);if(e!=cudaSuccess){set_cuda_error("mapped status",e);return 205;}e=cudaHostGetDevicePointer((void**)&s.d_status_map,s.h_status,0);if(e!=cudaSuccess){set_cuda_error("map status",e);return 205;}
#ifdef DDSCAT_CUDA_BACKEND_SLICES
 if((size_t)(2*nz)>(size_t)INT_MAX||(size_t)3*nx*ny>(size_t)INT_MAX||(size_t)(2*nx)*(2*ny)>(size_t)INT_MAX){set_error("cufftPlanMany","slice dimensions too large");return 206;}
 int zdim=2*nz,zdist=2*nz,zbatch=3*nx*ny;cufftResult fr=cufftPlanMany(&s.plan_z,1,&zdim,&zdim,1,zdist,&zdim,1,zdist,CUFFT_C2C,zbatch);if(fr!=CUFFT_SUCCESS){set_cufft_error("cufftPlanMany(slice Z,batch=3*NX*NY)",fr);return 206;}s.plan_z_ok=true;
 int xydims[2]={2*ny,2*nx},xydist=(2*nx)*(2*ny);fr=cufftPlanMany(&s.plan_xy,2,xydims,xydims,1,xydist,xydims,1,xydist,CUFFT_C2C,12);if(fr!=CUFFT_SUCCESS){set_cufft_error("cufftPlanMany(slice XY,batch=12=4 slices*3 components)",fr);return 206;}s.plan_xy_ok=true;
 fr=cufftGetSize(s.plan_z,&s.plan_z_workspace);if(fr!=CUFFT_SUCCESS){set_cufft_error("cufftGetSize(slice Z)",fr);return 206;}fr=cufftGetSize(s.plan_xy,&s.plan_xy_workspace);if(fr!=CUFFT_SUCCESS){set_cufft_error("cufftGetSize(slice XY)",fr);return 206;}
#else
 if(s.ngrid>(size_t)INT_MAX){set_error("cufftPlanMany","grid too large");return 206;}int dims[3]={2*nz,2*ny,2*nx},dist=(int)s.ngrid;cufftResult fr=cufftPlanMany(&s.plan,3,dims,dims,1,dist,dims,1,dist,CUFFT_C2C,3);if(fr!=CUFFT_SUCCESS){set_cufft_error("cufftPlanMany(rank=3,batch=3)",fr);return 206;}s.plan_ok=true;fr=cufftGetSize(s.plan,&s.plan_workspace);if(fr!=CUFFT_SUCCESS){set_cufft_error("cufftGetSize(FFT3D)",fr);return 206;}
#endif
 cublasStatus_t br=cublasCreate(&s.blas);if(br!=CUBLAS_STATUS_SUCCESS){set_cublas_error("cublasCreate",br);return 206;}s.blas_ok=true;br=cublasSetPointerMode(s.blas,CUBLAS_POINTER_MODE_HOST);if(br!=CUBLAS_STATUS_SUCCESS){set_cublas_error("cublas pointer mode HOST",br);return 206;}
 e=cudaEventCreate(&s.ev0);if(e!=cudaSuccess){set_cuda_error("event create",e);return 206;}e=cudaEventCreate(&s.ev1);if(e!=cudaSuccess){set_cuda_error("event create",e);return 206;}s.event_ok=true;
#ifdef DDSCAT_CUDA_BACKEND_SLICES
 s.green_streams_ok=true;for(int i=0;i<4;i++){e=cudaStreamCreateWithFlags(&s.green_stream[i],cudaStreamNonBlocking);if(e!=cudaSuccess){set_cuda_error("create Green stream",e);return 206;}}
 s.green_events_ok=true;e=cudaEventCreateWithFlags(&s.xy_ready,cudaEventDisableTiming);if(e!=cudaSuccess){set_cuda_error("create XY-ready event",e);return 206;}
 for(int i=0;i<4;i++){e=cudaEventCreateWithFlags(&s.green_done[i],cudaEventDisableTiming);if(e!=cudaSuccess){set_cuda_error("create Green-done event",e);return 206;}}
#endif
 return 0;
}
int upload_occupied_index_map(const int16_t*iocc){
 if(!iocc||s.nat0<=0||s.nat0>s.nat){set_error("occupied-index map","invalid NAT/NAT0/IOCC");return 1;}
 std::vector<int> occ;occ.reserve((size_t)s.nat0);for(int j=0;j<s.nat;j++)if(iocc[j]!=0)occ.push_back(j);
 if((int)occ.size()!=s.nat0){char msg[160];std::snprintf(msg,sizeof(msg),"IOCC contains %zu occupied sites but NAT0=%d",occ.size(),s.nat0);set_error("occupied-index map",msg);return 1;}
 cudaError_t e=cudaMemcpy(s.d_occ_index,occ.data(),(size_t)s.nat0*sizeof(int),cudaMemcpyHostToDevice);
 if(e!=cudaSuccess){set_cuda_error("occupied-index map H2D",e);return 1;}return 0;
}

int capture_gpu_memory_baseline(const char*where){
 cudaError_t e=cudaFree(0);if(e!=cudaSuccess){set_cuda_error(where,e);return 1;}
 e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error(where,e);return 1;}
 size_t free_b=0,total_b=0;e=cudaMemGetInfo(&free_b,&total_b);if(e!=cudaSuccess){set_cuda_error(where,e);return 1;}
 s.gpu_free_baseline=free_b;s.gpu_total_bytes=total_b;return 0;
}

void print_gpu_memory_report(){
 const double mib=1024.0*1024.0,gib=1024.0*mib;
 const size_t n=(size_t)3*s.nat,cn=(size_t)s.solver_n;
 const size_t green_bytes=s.green_count*sizeof(cufftComplex);
 const size_t occupancy_map_bytes=(size_t)s.nat*sizeof(int16_t)+(size_t)s.nat0*sizeof(int);
 const size_t matvec_io_operator_bytes=(size_t)4*n*sizeof(cufftComplex)+occupancy_map_bytes;
#ifdef DDSCAT_CUDA_BACKEND_SLICES
 const size_t fft_data_bytes=(s.zwork_count+s.slice_count)*sizeof(cufftComplex);
 const size_t fft_workspace_reported=s.plan_z_workspace+s.plan_xy_workspace;
#else
 const size_t fft_data_bytes=(size_t)3*s.ngrid*sizeof(cufftComplex);
 const size_t fft_workspace_reported=s.plan_workspace;
#endif
 const size_t solver_vector_bytes=(size_t)16*cn*sizeof(cufftComplex); // sx,sy,b + 13 work
 const size_t solver_vector_full_equiv=(size_t)16*n*sizeof(cufftComplex);
 const size_t solver_scalar_bytes=32*sizeof(cuDoubleComplex)+8*sizeof(double)+
     (size_t)2*s.dot_chunk_capacity*sizeof(cuDoubleComplex);
 const size_t explicit_bytes=green_bytes+matvec_io_operator_bytes+fft_data_bytes+
     solver_vector_bytes+solver_scalar_bytes;
 cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("memory report synchronize",e);return;}
 size_t free_now=0,total_now=0;e=cudaMemGetInfo(&free_now,&total_now);if(e!=cudaSuccess){set_cuda_error("cudaMemGetInfo(memory report)",e);return;}
 const size_t actual_alloc=(s.gpu_free_baseline>free_now)?(s.gpu_free_baseline-free_now):0;
 const size_t internal_overhead=(actual_alloc>explicit_bytes)?(actual_alloc-explicit_bytes):0;
 std::printf("\nCUDA GPU memory before iterative solver (Green build temporaries already freed):\n");
 std::printf("  Green tensor resident (net)          : %10.2f MiB\n",(double)green_bytes/mib);
 std::printf("  MATVEC full x/y + operator + maps    : %10.2f MiB\n",(double)matvec_io_operator_bytes/mib);
#ifdef DDSCAT_CUDA_BACKEND_SLICES
 std::printf("  MATVEC FFT data (Z + XY batch4)      : %10.2f MiB\n",(double)fft_data_bytes/mib);
#else
 std::printf("  MATVEC FFT3D data buffer             : %10.2f MiB\n",(double)fft_data_bytes/mib);
#endif
 const double fill=(s.nat>0)?(double)s.nat0/(double)s.nat:0.0;
 std::printf("  Solver compact occupancy NAT0/NAT    : %10.5f  (%d / %d)\n",fill,s.nat0,s.nat);
 std::printf("  Solver vectors compact (16 vectors)  : %10.2f MiB\n",(double)solver_vector_bytes/mib);
 std::printf("  Solver vectors if full-grid          : %10.2f MiB\n",(double)solver_vector_full_equiv/mib);
 std::printf("  Solver vector memory saved           : %10.2f MiB  (%5.1f%%)\n",(double)(solver_vector_full_equiv-solver_vector_bytes)/mib,100.0*(1.0-fill));
 std::printf("  Solver scalar/dot staging buffers    : %10.2f MiB\n",(double)solver_scalar_bytes/mib);
 std::printf("  Explicit DDSCAT cudaMalloc subtotal  : %10.2f MiB\n",(double)explicit_bytes/mib);
 std::printf("  cuFFT workspace reported by plans    : %10.2f MiB\n",(double)fft_workspace_reported/mib);
 std::printf("  DDSCAT actual GPU allocation         : %10.2f MiB  (%6.3f GiB)\n",(double)actual_alloc/mib,(double)actual_alloc/gib);
 std::printf("  CUDA/cuFFT/cuBLAS/internal overhead  : %10.2f MiB\n",(double)internal_overhead/mib);
 std::printf("  GPU memory remaining                 : %10.2f MiB  (%6.3f GiB)\n",(double)free_now/mib,(double)free_now/gib);
 std::printf("  GPU total memory                     : %10.2f MiB  (%6.3f GiB)\n",(double)total_now/mib,(double)total_now/gib);
 if(s.gpu_free_baseline)
  std::printf("  GPU free at DDSCAT allocation start  : %10.2f MiB  (%6.3f GiB)\n",(double)s.gpu_free_baseline/mib,(double)s.gpu_free_baseline/gib);
 std::printf("  NOTE: actual allocation is measured with cudaMemGetInfo; internal overhead includes library workspaces/caches and allocator overhead.\n\n");
 std::fflush(stdout);
}

void print_backend_layout(int nx,int ny,int nz){
 const double mib=1024.0*1024.0;
#ifdef DDSCAT_CUDA_BACKEND_SLICES
 std::printf("CUDA backend: SLICES (FFT1D-Z + FFT2D-XY batch=4 slices + 4 concurrent Green streams)\n");
 std::printf("CUDA SLICES memory: Z=%.2f MiB XY-batch4=%.2f MiB; cuFFT work Z=%.2f MiB XY=%.2f MiB\n",(double)(s.zwork_count*sizeof(cufftComplex))/mib,(double)(s.slice_count*sizeof(cufftComplex))/mib,(double)s.plan_z_workspace/mib,(double)s.plan_xy_workspace/mib);
#else
 std::printf("CUDA backend: FFT3D batched, batch=3, grid=%dx%dx%d\n",2*nx,2*ny,2*nz);
 std::printf("CUDA FFT3D MATVEC work: %.2f MiB; cuFFT work=%.2f MiB\n",(double)(3*s.ngrid*sizeof(cufftComplex))/mib,(double)s.plan_workspace/mib);
#endif
 std::printf("CUDA solver precision: vectors/FFT=float32; scalar products=cublasZdotc/Zdotu chunks<=%d; recurrence scalars=double/cuDoubleComplex.\n",DOT_CHUNK_COMPLEX);
 std::printf("CUDA solver compact storage: solverN=3*NAT0=%d versus fullN=3*NAT=%d; factor=%.6f.\n",s.solver_n,3*s.nat,(s.nat>0)?(double)s.nat0/(double)s.nat:0.0);std::fflush(stdout);
}


// ================= GPU final EVALQ / SCAT postprocessing =================
// Heavy O(NAT0) and O(NAT0*Ndir) summations are performed on the GPU.
// EVALQ uses the existing chunked complex-double cuBLAS dot-product path.
// SCAT fuses phase generation with a block reduction, then uses cublasZgemv
// to sum block partials. This avoids writing one complex term per dipole and
// rereading it only to sum it.

__global__ void evalq_local_apply_kernel(const cufftComplex*p,cufftComplex*q,
                                         const cufftComplex*ad,const cufftComplex*ao,
                                         const int*occ,int nat,int nat0,
                                         float rabs_im,int compact_op){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nat0)return;
 int jf=compact_op?j:occ[j];int stride=compact_op?nat0:nat;
 cufftComplex p1=p[j],p2=p[j+nat0],p3=p[j+2*nat0];
 cufftComplex d1=ad[jf],d2=ad[jf+stride],d3=ad[jf+2*stride];
 cufftComplex a23=ao[jf],a31=ao[jf+stride],a12=ao[jf+2*stride];
 cufftComplex ri=make_cuFloatComplex(0.f,rabs_im);
 q[j]=ca(ca(cm(ca(d1,ri),p1),cm(a31,p3)),cm(a12,p2));
 q[j+nat0]=ca(ca(cm(ca(d2,ri),p2),cm(a12,p1)),cm(a23,p3));
 q[j+2*nat0]=ca(ca(cm(ca(d3,ri),p3),cm(a23,p2)),cm(a31,p1));
}

constexpr int SCAT_THREADS=256;
constexpr int SCAT_ITEMS=4;
constexpr int SCAT_BATCH=16;

__global__ void fill_ones_z_kernel(cuDoubleComplex*x,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=make_cuDoubleComplex(1.0,0.0);
}

__global__ void farfield_partial_kernel(const cufftComplex*p,const int*occ,int nat0,int nx,int ny,
                                         float dx1,float dx2,float dx3,float x01,float x02,float x03,
                                         const float*dirs,int ndir,cuDoubleComplex*partials,int nblocks){
 int idir=(int)blockIdx.y,ib=(int)blockIdx.x,tid=(int)threadIdx.x;
 if(idir>=ndir||ib>=nblocks)return;
 double ar0=0,ai0=0,ar1=0,ai1=0,ar2=0,ai2=0;
 float kx=dirs[3*idir],ky=dirs[3*idir+1],kz=dirs[3*idir+2];
 int base=ib*(blockDim.x*SCAT_ITEMS)+tid;
 #pragma unroll
 for(int it=0;it<SCAT_ITEMS;it++){
  int j=base+it*blockDim.x;if(j>=nat0)continue;
  int full=occ[j];int ix=full%nx;int qq=full/nx;int iy=qq%ny;int iz=qq/ny;
  float rx=((float)(ix+1)+x01)*dx1,ry=((float)(iy+1)+x02)*dx2,rz=((float)(iz+1)+x03)*dx3;
  float ph=kx*rx+ky*ry+kz*rz;float sn,cs;sincosf(ph,&sn,&cs);
  cufftComplex px=p[j],py=p[j+nat0],pz=p[j+2*nat0];
  // (a+ib)*exp(-i*ph) = (a*c+b*s)+i(b*c-a*s)
  ar0+=(double)px.x*cs+(double)px.y*sn; ai0+=(double)px.y*cs-(double)px.x*sn;
  ar1+=(double)py.x*cs+(double)py.y*sn; ai1+=(double)py.y*cs-(double)py.x*sn;
  ar2+=(double)pz.x*cs+(double)pz.y*sn; ai2+=(double)pz.y*cs-(double)pz.x*sn;
 }
 extern __shared__ double sm[];double*s0r=sm,*s0i=sm+blockDim.x,*s1r=sm+2*blockDim.x,*s1i=sm+3*blockDim.x,*s2r=sm+4*blockDim.x,*s2i=sm+5*blockDim.x;
 s0r[tid]=ar0;s0i[tid]=ai0;s1r[tid]=ar1;s1i[tid]=ai1;s2r[tid]=ar2;s2i[tid]=ai2;__syncthreads();
 for(int off=blockDim.x/2;off>0;off>>=1){if(tid<off){s0r[tid]+=s0r[tid+off];s0i[tid]+=s0i[tid+off];s1r[tid]+=s1r[tid+off];s1i[tid]+=s1i[tid+off];s2r[tid]+=s2r[tid+off];s2i[tid]+=s2i[tid+off];}__syncthreads();}
 if(tid==0){
  int rows=3*ndir;
  partials[(size_t)ib*rows+3*idir  ]=make_cuDoubleComplex(s0r[0],s0i[0]);
  partials[(size_t)ib*rows+3*idir+1]=make_cuDoubleComplex(s1r[0],s1i[0]);
  partials[(size_t)ib*rows+3*idir+2]=make_cuDoubleComplex(s2r[0],s2i[0]);
 }
}

__global__ void farfield_es2_kernel(const cuDoubleComplex*sums,const float*dirs,double akk,double*es2,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 double nx=(double)dirs[3*i]/akk,ny=(double)dirs[3*i+1]/akk,nz=(double)dirs[3*i+2]/akk;
 cuDoubleComplex sx=sums[3*i],sy=sums[3*i+1],sz=sums[3*i+2];
 cuDoubleComplex ndotp=make_cuDoubleComplex(nx*sx.x+ny*sy.x+nz*sz.x,nx*sx.y+ny*sy.y+nz*sz.y);
 double exr=sx.x-nx*ndotp.x,exi=sx.y-nx*ndotp.y;
 double eyr=sy.x-ny*ndotp.x,eyi=sy.y-ny*ndotp.y;
 double ezr=sz.x-nz*ndotp.x,ezi=sz.y-nz*ndotp.y;
 es2[i]=exr*exr+exi*exi+eyr*eyr+eyi*eyi+ezr*ezr+ezi*ezi;
}

__global__ void selected_amp_kernel(const cuDoubleComplex*sums,const float*em1,const float*em2,double term,
                                    cufftComplex*f1,cufftComplex*f2,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 cuDoubleComplex sx=sums[3*i],sy=sums[3*i+1],sz=sums[3*i+2];
 double a1r=em1[3*i]*sx.x+em1[3*i+1]*sy.x+em1[3*i+2]*sz.x;
 double a1i=em1[3*i]*sx.y+em1[3*i+1]*sy.y+em1[3*i+2]*sz.y;
 double a2r=em2[3*i]*sx.x+em2[3*i+1]*sy.x+em2[3*i+2]*sz.x;
 double a2i=em2[3*i]*sx.y+em2[3*i+1]*sy.y+em2[3*i+2]*sz.y;
 f1[i]=make_cuFloatComplex((float)(term*a1r),(float)(term*a1i));
 f2[i]=make_cuFloatComplex((float)(term*a2r),(float)(term*a2i));
}

struct ScatIntDir {float k[3];double w[5];};

int make_scat_integration_dirs(const float ak[3],const cufftComplex e01[3],float etasca,std::vector<ScatIntDir>&dirs,int&navg){
 const double pi=4.0*atan(1.0);
 double ak2=(double)ak[0]*ak[0]+(double)ak[1]*ak[1]+(double)ak[2]*ak[2],akk=sqrt(ak2);
 double a1[3]={(double)e01[0].x,(double)e01[1].x,(double)e01[2].x};
 double einc=sqrt(a1[0]*a1[0]+a1[1]*a1[1]+a1[2]*a1[2]);
 if(!(einc>0.0)){set_error("GPU SCAT","CXE01_TF is pure imaginary");return 1;}
 for(int k=0;k<3;k++)a1[k]*=akk/einc;
 double a2[3]={(ak[1]*a1[2]-ak[2]*a1[1])/akk,(ak[2]*a1[0]-ak[0]*a1[2])/akk,(ak[0]*a1[1]-ak[1]*a1[0])/akk};
 double x=akk*cbrt(3.0*(double)s.nat0/(4.0*pi)),theta0=2.0*pi/(1.0+x),afac=1.0;
 int ntheta=1+(int)llround((2.0*(3.0+x)/(double)etasca)*(1.0+afac/(pi+theta0))/(1.0+afac*theta0/((pi+theta0)*(pi+theta0))));
 double theta=0.0,thetau=0.0;dirs.clear();navg=0;
 for(int ic=1;ic<=ntheta;ic++){
  double thetal=theta;theta=thetau;
  if(ic<ntheta){double si=(double)ic*(pi+afac*pi/(theta0+pi))/(double)(ntheta-1),term=si-afac-theta0;thetau=0.5*(term+sqrt(term*term+4.0*theta0*si));}
  else thetau=pi;
  double domega=2.0*pi*(cos(0.5*(thetal+theta))-cos(0.5*(theta+thetau)));
  double ct=cos(theta),st=sin(theta),ct2=ct*ct;int nphi=(ic==1||ic==ntheta)?1:(int)llround(4.0*pi*st/(thetau-thetal));if(nphi<3&&ic!=1&&ic!=ntheta)nphi=3;
  double w=domega/(double)nphi,dphi=2.0*pi/(double)nphi;navg+=nphi;
  for(int ip=1;ip<=nphi;ip++){
   double phi=dphi*((double)ip-0.5*(double)(ic%2)),sp=sin(phi),cp=cos(phi);
   ScatIntDir d{};
   for(int k=0;k<3;k++)d.k[k]=(float)(ct*(double)ak[k]+st*(cp*a1[k]+sp*a2[k]));
   d.w[0]=w;d.w[1]=w*ct;d.w[2]=w*st*cp;d.w[3]=w*st*sp;d.w[4]=w*ct2;dirs.push_back(d);
  }
 }
 return 0;
}

struct FarfieldWorkspace{
 int nblocks=0,maxbatch=SCAT_BATCH;
 float*d_dirs=nullptr,*d_em1=nullptr,*d_em2=nullptr;
 double*d_es2=nullptr,*d_weights=nullptr;
 cuDoubleComplex*d_partials=nullptr,*d_ones=nullptr,*d_sums=nullptr;
 cufftComplex*d_f1=nullptr,*d_f2=nullptr;
};
void free_farfield_workspace(FarfieldWorkspace&w){
 cudaFree(w.d_dirs);cudaFree(w.d_em1);cudaFree(w.d_em2);cudaFree(w.d_es2);cudaFree(w.d_weights);
 cudaFree(w.d_partials);cudaFree(w.d_ones);cudaFree(w.d_sums);cudaFree(w.d_f1);cudaFree(w.d_f2);w=FarfieldWorkspace{};
}
int alloc_farfield_workspace(FarfieldWorkspace&w){
 w.nblocks=(s.nat0+SCAT_THREADS*SCAT_ITEMS-1)/(SCAT_THREADS*SCAT_ITEMS);if(w.nblocks<1)return 1;
 cudaError_t e;
#define PMAL(ptr,bytes,label) do{e=cudaMalloc((void**)&(ptr),(bytes));if(e!=cudaSuccess){set_cuda_error(label,e);free_farfield_workspace(w);return 1;}}while(0)
 PMAL(w.d_dirs,(size_t)3*w.maxbatch*sizeof(float),"SCAT dirs");
 PMAL(w.d_em1,(size_t)3*w.maxbatch*sizeof(float),"SCAT em1");PMAL(w.d_em2,(size_t)3*w.maxbatch*sizeof(float),"SCAT em2");
 PMAL(w.d_es2,(size_t)w.maxbatch*sizeof(double),"SCAT es2");PMAL(w.d_weights,(size_t)5*w.maxbatch*sizeof(double),"SCAT weights");
 PMAL(w.d_partials,(size_t)w.nblocks*3*w.maxbatch*sizeof(cuDoubleComplex),"SCAT block partials");
 PMAL(w.d_ones,(size_t)w.nblocks*sizeof(cuDoubleComplex),"SCAT reduction ones");
 PMAL(w.d_sums,(size_t)3*w.maxbatch*sizeof(cuDoubleComplex),"SCAT sums");
 PMAL(w.d_f1,(size_t)w.maxbatch*sizeof(cufftComplex),"SCAT f1");PMAL(w.d_f2,(size_t)w.maxbatch*sizeof(cufftComplex),"SCAT f2");
#undef PMAL
 int th=256,gr=(w.nblocks+th-1)/th;fill_ones_z_kernel<<<gr,th>>>(w.d_ones,w.nblocks);if(launch_ok("SCAT ones")){free_farfield_workspace(w);return 1;}return 0;
}
int farfield_batch_geom(FarfieldWorkspace&w,const float*dirs,int count,const float dx[3],const float x0[3]){
 cudaError_t e=cudaMemcpy(w.d_dirs,dirs,(size_t)3*count*sizeof(float),cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("SCAT dirs H2D",e);return 1;}
 dim3 grid((unsigned)w.nblocks,(unsigned)count,1);size_t sh=(size_t)6*SCAT_THREADS*sizeof(double);
 farfield_partial_kernel<<<grid,SCAT_THREADS,sh>>>(s.d_sx,s.d_occ_index,s.nat0,s.nx,s.ny,dx[0],dx[1],dx[2],x0[0],x0[1],x0[2],w.d_dirs,count,w.d_partials,w.nblocks);
 if(launch_ok("SCAT farfield partial"))return 1;
 const int rows=3*count;cuDoubleComplex alpha=make_cuDoubleComplex(1.0,0.0),beta=make_cuDoubleComplex(0.0,0.0);
 cublasStatus_t br=cublasZgemv(s.blas,CUBLAS_OP_N,rows,w.nblocks,&alpha,w.d_partials,rows,w.d_ones,1,&beta,w.d_sums,1);
 if(br!=CUBLAS_STATUS_SUCCESS){set_cublas_error("SCAT cublasZgemv block reduction",br);return 1;}return 0;
}
// ================= end GPU final EVALQ / SCAT postprocessing =================

} // namespace

extern "C" DDSCAT_CUDA_API int ddscat_cuda_release(void){release_state();last_error.clear();return 0;}
extern "C" DDSCAT_CUDA_API const char*ddscat_cuda_last_error(void){return last_error.c_str();}


extern "C" DDSCAT_CUDA_API int ddscat_cuda_evalq_f32(int use_resident,
    const void*e_compact,const void*p_compact,const void*adia_compact,const void*aoff_compact,
    float akx,float aky,float akz,float e02,double*cabs,double*cext,double*cpha){
 last_error.clear();
 if(!s.ready||!cabs||!cext||!cpha||!(e02>0.f)){set_error("GPU EVALQ","invalid argument");return 300;}
 release_krylov_workspace("EVALQ/SCAT starts after iterative solve");
 auto t0=WallClock::now();cudaError_t e;const cufftComplex *ad=s.d_adia,*ao=s.d_aoff;int compact_op=0;
 cufftComplex *tmpad=nullptr,*tmpao=nullptr;const size_t cb=(size_t)s.solver_n*sizeof(cufftComplex);
 if(use_resident){
  if(!s.solution_valid){set_error("GPU EVALQ","no resident final polarization published by solver");return 301;}
 }else{
  if(!e_compact||!p_compact||!adia_compact||!aoff_compact){set_error("GPU EVALQ","missing compact host arrays");return 302;}
  e=cudaMemcpy(s.d_b,e_compact,cb,cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("GPU EVALQ E H2D",e);return 303;}
  e=cudaMemcpy(s.d_sx,p_compact,cb,cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("GPU EVALQ P H2D",e);return 303;}s.solution_valid=true;
  e=cudaMalloc((void**)&tmpad,cb);if(e!=cudaSuccess){set_cuda_error("GPU EVALQ compact ADIA alloc",e);return 304;}
  e=cudaMalloc((void**)&tmpao,cb);if(e!=cudaSuccess){cudaFree(tmpad);set_cuda_error("GPU EVALQ compact AOFF alloc",e);return 304;}
  e=cudaMemcpy(tmpad,adia_compact,cb,cudaMemcpyHostToDevice);if(e!=cudaSuccess){cudaFree(tmpad);cudaFree(tmpao);set_cuda_error("GPU EVALQ ADIA H2D",e);return 305;}
  e=cudaMemcpy(tmpao,aoff_compact,cb,cudaMemcpyHostToDevice);if(e!=cudaSuccess){cudaFree(tmpad);cudaFree(tmpao);set_cuda_error("GPU EVALQ AOFF H2D",e);return 305;}
  ad=tmpad;ao=tmpao;compact_op=1;
 }
 double ak2=(double)akx*akx+(double)aky*aky+(double)akz*akz,akk=sqrt(ak2),ak3=akk*ak2;
 int th=256,gr=(s.nat0+th-1)/th;
 evalq_local_apply_kernel<<<gr,th>>>(s.d_sx,s.d_sy,ad,ao,s.d_occ_index,s.nat,s.nat0,(float)(ak3/1.5),compact_op);
 if(launch_ok("GPU EVALQ local alpha^-1 P")){cudaFree(tmpad);cudaFree(tmpao);return 306;}
 cuDoubleComplex zabs,zep;
 if(dot_product_double_chunked(s.d_sx,s.d_sy,s.solver_n,true,&zabs,"GPU EVALQ CABS cublasZdotc")){cudaFree(tmpad);cudaFree(tmpao);return 307;}
 // Zdotc(E,P) = sum conj(E_j)*P_j, exactly the scalar used for Cext/Cpha.
 if(dot_product_double_chunked(s.d_b,s.d_sx,s.solver_n,true,&zep,"GPU EVALQ CEXT/CPHA cublasZdotc")){cudaFree(tmpad);cudaFree(tmpao);return 307;}
 const double pi=4.0*atan(1.0);
 *cabs=-4.0*pi*akk*cuCimag(zabs)/(double)e02;
 *cext= 4.0*pi*akk*cuCimag(zep)/(double)e02;
 *cpha= 2.0*pi*akk*cuCreal(zep)/(double)e02;
 cudaFree(tmpad);cudaFree(tmpao);
 double ms=std::chrono::duration<double,std::milli>(WallClock::now()-t0).count();
 std::printf("CUDA EVALQ GPU: Cabs/Cext/Cpha in %.3f ms (complex-double cuBLAS reductions).\\n",ms);std::fflush(stdout);
 return 0;
}

extern "C" DDSCAT_CUDA_API int ddscat_cuda_scat_f32(int jpbc,int ndir,
    const float*ak,const float*aks,const float*dx,const float*em1,const float*em2,
    float e02,float etasca,const void*e01v,const float*x0,
    double*cbksca,double*csca,double*cscag,double*cscag2,int*navg,
    void*f1v,void*f2v){
 last_error.clear();
 if(!s.ready||!s.solution_valid||!ak||!dx||!x0||!cbksca||!csca||!cscag||!cscag2||!navg||!(e02>0.f)){
  set_error("GPU SCAT","invalid state/argument");return 320;
 }
 if(ndir>0&&(!aks||!em1||!em2||!f1v||!f2v)){set_error("GPU SCAT","missing selected-direction arrays");return 321;}
 release_krylov_workspace("GPU SCAT");
 auto t0=WallClock::now();double ak2=(double)ak[0]*ak[0]+(double)ak[1]*ak[1]+(double)ak[2]*ak[2],akk=sqrt(ak2),ak3=akk*ak2;
 FarfieldWorkspace w;if(alloc_farfield_workspace(w))return 322;
 *cbksca=0.0;*csca=0.0;for(int k=0;k<3;k++)cscag[k]=0.0;*cscag2=0.0;*navg=0;
 if(jpbc==0){
  if(!e01v||!(etasca>0.f)){free_farfield_workspace(w);set_error("GPU SCAT","finite-target integration needs CXE01 and ETASCA>0");return 323;}
  const cufftComplex*e01=(const cufftComplex*)e01v;std::vector<ScatIntDir> idirs;int navi=0;
  if(make_scat_integration_dirs(ak,e01,etasca,idirs,navi)){free_farfield_workspace(w);return 324;}*navg=navi;
  double accum[5]={0,0,0,0,0},back_es2=0.0;
  std::vector<float> hdirs((size_t)3*SCAT_BATCH);std::vector<double> hweights((size_t)5*SCAT_BATCH);
  for(size_t off=0;off<idirs.size();off+=SCAT_BATCH){
   int cnt=(int)std::min((size_t)SCAT_BATCH,idirs.size()-off);
   for(int i=0;i<cnt;i++){for(int k=0;k<3;k++)hdirs[3*i+k]=idirs[off+i].k[k];for(int k=0;k<5;k++)hweights[(size_t)k*SCAT_BATCH+i]=idirs[off+i].w[k];}
   if(farfield_batch_geom(w,hdirs.data(),cnt,dx,x0)){free_farfield_workspace(w);return 325;}
   int th=128,gr=(cnt+th-1)/th;farfield_es2_kernel<<<gr,th>>>(w.d_sums,w.d_dirs,akk,w.d_es2,cnt);if(launch_ok("GPU SCAT ES2")){free_farfield_workspace(w);return 326;}
   for(int k=0;k<5;k++){cudaError_t e=cudaMemcpy(w.d_weights+(size_t)k*SCAT_BATCH,hweights.data()+(size_t)k*SCAT_BATCH,(size_t)cnt*sizeof(double),cudaMemcpyHostToDevice);if(e!=cudaSuccess){free_farfield_workspace(w);set_cuda_error("GPU SCAT weights H2D",e);return 327;}double part=0.0;cublasStatus_t br=cublasDdot(s.blas,cnt,w.d_es2,1,w.d_weights+(size_t)k*SCAT_BATCH,1,&part);if(br!=CUBLAS_STATUS_SUCCESS){free_farfield_workspace(w);set_cublas_error("GPU SCAT cublasDdot angular integral",br);return 328;}accum[k]+=part;}
   if(off+(size_t)cnt==idirs.size()){cudaError_t e=cudaMemcpy(&back_es2,w.d_es2+(cnt-1),sizeof(double),cudaMemcpyDeviceToHost);if(e!=cudaSuccess){free_farfield_workspace(w);set_cuda_error("GPU SCAT backscatter D2H",e);return 329;}}
  }
  double fac=ak2*ak2/(double)e02;*csca=fac*accum[0];cscag[0]=fac*accum[1];cscag[1]=fac*accum[2];cscag[2]=fac*accum[3];*cscag2=fac*accum[4];*cbksca=fac*back_es2;
 }
 if(ndir>0){
  cufftComplex*h1=(cufftComplex*)f1v,*h2=(cufftComplex*)f2v;std::vector<float> hdirs((size_t)3*SCAT_BATCH);
  for(int off=0;off<ndir;off+=SCAT_BATCH){
   int cnt=std::min(SCAT_BATCH,ndir-off);for(int i=0;i<cnt;i++)for(int k=0;k<3;k++)hdirs[3*i+k]=aks[3*(off+i)+k];
   if(farfield_batch_geom(w,hdirs.data(),cnt,dx,x0)){free_farfield_workspace(w);return 330;}
   cudaError_t e=cudaMemcpy(w.d_em1,em1+3*off,(size_t)3*cnt*sizeof(float),cudaMemcpyHostToDevice);if(e!=cudaSuccess){free_farfield_workspace(w);set_cuda_error("GPU SCAT EM1 H2D",e);return 331;}
   e=cudaMemcpy(w.d_em2,em2+3*off,(size_t)3*cnt*sizeof(float),cudaMemcpyHostToDevice);if(e!=cudaSuccess){free_farfield_workspace(w);set_cuda_error("GPU SCAT EM2 H2D",e);return 331;}
   int th=128,gr=(cnt+th-1)/th;selected_amp_kernel<<<gr,th>>>(w.d_sums,w.d_em1,w.d_em2,ak3/sqrt((double)e02),w.d_f1,w.d_f2,cnt);if(launch_ok("GPU SCAT selected amplitudes")){free_farfield_workspace(w);return 332;}
   e=cudaMemcpy(h1+off,w.d_f1,(size_t)cnt*sizeof(cufftComplex),cudaMemcpyDeviceToHost);if(e!=cudaSuccess){free_farfield_workspace(w);set_cuda_error("GPU SCAT F1 D2H",e);return 333;}
   e=cudaMemcpy(h2+off,w.d_f2,(size_t)cnt*sizeof(cufftComplex),cudaMemcpyDeviceToHost);if(e!=cudaSuccess){free_farfield_workspace(w);set_cuda_error("GPU SCAT F2 D2H",e);return 333;}
  }
 }
 free_farfield_workspace(w);double ms=std::chrono::duration<double,std::milli>(WallClock::now()-t0).count();
 std::printf("CUDA SCAT GPU: NAT0=%d NAVG=%d NDIR=%d time=%.3f ms; phase+dipole sums fused, block sums=cublasZgemv, angular sums=cublasDdot.\\n",s.nat0,*navg,ndir,ms);std::fflush(stdout);
 return 0;
}


extern "C" DDSCAT_CUDA_API int ddscat_cuda_prepare_green_f32(int nx,int ny,int nz,int ipbc,int nat,int nat0,int idipint,
    float gamma,float pyd,float pzd,float akx,float aky,float akz,float dx1,float dx2,float dx3,const int16_t*iocc){
 last_error.clear();if(nx<=0||ny<=0||nz<=0||nat<=0||nat0<=0||nat0>nat||(ipbc!=0&&ipbc!=1)||(idipint!=0&&idipint!=1)||!iocc){set_error("prepare_green","invalid argument");return 201;}if(ipbc&&gamma<=0.f){set_error("prepare_green","GAMMA must be >0 for PBC");return 202;}
 cudaError_t e=cudaSetDeviceFlags(cudaDeviceMapHost);if(e!=cudaSuccess&&e!=cudaErrorSetOnActiveProcess){set_cuda_error("cudaSetDeviceFlags",e);return 203;}if(e==cudaErrorSetOnActiveProcess)cudaGetLastError();
 // Shared ordering for both backends: resident Green -> temporary Green build -> free Green scratch -> backend/runtime buffers.
 release_state();if(capture_gpu_memory_baseline("cudaMemGetInfo baseline before DDSCAT allocations"))return 203;s.nx=nx;s.ny=ny;s.nz=nz;s.ipbc=ipbc;s.nat=nat;s.nat0=nat0;s.ngrid=(size_t)(2*nx)*(2*ny)*(2*nz);s.green_count=(ipbc==0)?(size_t)(nx+1)*(ny+1)*(nz+1)*6:s.ngrid*6;
 e=cudaMalloc((void**)&s.d_green,s.green_count*sizeof(cufftComplex));if(e!=cudaSuccess){set_cuda_error("malloc resident Green",e);release_state();return 204;}
 GreenBuildParams gp{nx,ny,nz,ipbc,idipint,(double)gamma,(double)pyd,(double)pzd,(double)akx,(double)aky,(double)akz,(double)dx1,(double)dx2,(double)dx3,0.0,0.0,0.0,0.0};gp.akd=std::sqrt(gp.akx*gp.akx+gp.aky*gp.aky+gp.akz*gp.akz);gp.akd2=gp.akd*gp.akd;gp.pyddx=gp.pyd*gp.dx2;gp.pzddx=gp.pzd*gp.dx3;
 int grc=build_green_cuda_resident(gp);if(grc){release_state();return 220+grc;}
 int arc=allocate_runtime_after_green(nx,ny,nz,nat,nat0);if(arc){release_state();return arc;}
 e=cudaMemcpy(s.d_iocc,iocc,(size_t)nat*sizeof(int16_t),cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("iocc H2D",e);release_state();return 207;}if(upload_occupied_index_map(iocc)){release_state();return 208;}s.ready=true;
 std::printf("CUDA Green residency: d_green built on GPU and remains resident; Green scratch/temporary plan released before runtime buffers.\n");print_backend_layout(nx,ny,nz);print_gpu_memory_report();return 0;
}
extern "C" DDSCAT_CUDA_API int ddscat_cuda_prepare_f32(int nx,int ny,int nz,int ipbc,int nat,int nat0,const void*green,const int16_t*iocc){
 last_error.clear();if(nx<=0||ny<=0||nz<=0||nat<=0||nat0<=0||nat0>nat||(ipbc!=0&&ipbc!=1)||!green||!iocc){set_error("prepare","invalid argument");return 1;}
 const size_t ng=(size_t)(2*nx)*(2*ny)*(2*nz),gc=(ipbc==0)?(size_t)(nx+1)*(ny+1)*(nz+1)*6:ng*6;cudaError_t e;
 e=cudaSetDeviceFlags(cudaDeviceMapHost);if(e!=cudaSuccess&&e!=cudaErrorSetOnActiveProcess){set_cuda_error("cudaSetDeviceFlags",e);return 2;}if(e==cudaErrorSetOnActiveProcess)cudaGetLastError();
 release_state();if(capture_gpu_memory_baseline("cudaMemGetInfo baseline before DDSCAT allocations"))return 2;s.nx=nx;s.ny=ny;s.nz=nz;s.ipbc=ipbc;s.nat=nat;s.nat0=nat0;s.ngrid=ng;s.green_count=gc;
 e=cudaMalloc((void**)&s.d_green,gc*sizeof(cufftComplex));if(e!=cudaSuccess){set_cuda_error("malloc green",e);release_state();return 2;}
 int arc=allocate_runtime_after_green(nx,ny,nz,nat,nat0);if(arc){release_state();return arc;}
 auto green_upload_t0=WallClock::now();e=cudaMemcpy(s.d_green,green,gc*sizeof(cufftComplex),cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("green H2D",e);release_state();return 4;}e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("green H2D synchronize",e);release_state();return 4;}double green_upload_ms=std::chrono::duration<double,std::milli>(WallClock::now()-green_upload_t0).count();e=cudaMemcpy(s.d_iocc,iocc,(size_t)nat*sizeof(int16_t),cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("iocc H2D",e);release_state();return 5;}if(upload_occupied_index_map(iocc)){release_state();return 6;}s.ready=true;std::printf("CUDA Green build time: not measured in this path (Green tensor supplied by CPU); Green H2D upload time %.3f ms.\n",green_upload_ms);print_backend_layout(nx,ny,nz);print_gpu_memory_report();return 0;
}
extern "C" DDSCAT_CUDA_API int ddscat_cuda_apply_f32(int cwhat,const void*x,void*y,const void*adia,const void*aoff){
 if(!s.ready){set_error("apply","not prepared");return 10;}if((cwhat!='N'&&cwhat!='C')||!x||!y||!adia||!aoff){set_error("apply","invalid argument");return 11;}size_t b=(size_t)3*s.nat*sizeof(cufftComplex);cudaError_t e=cudaMemcpy(s.d_x,x,b,cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("x H2D",e);return 12;}e=cudaMemcpy(s.d_adia,adia,b,cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("adia H2D",e);return 13;}e=cudaMemcpy(s.d_aoff,aoff,b,cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("aoff H2D",e);return 14;}float ms;int rc=matvec_device(cwhat,s.d_x,s.d_y,&ms);if(rc)return rc;e=cudaMemcpy(y,s.d_y,b,cudaMemcpyDeviceToHost);if(e!=cudaSuccess){set_cuda_error("y D2H",e);return 15;}report_matvec(cwhat,ms,"host-copy");return 0;
}

extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_gpbicg_f32(const void*b,void*x,const void*adia,const void*aoff,int ndim,int maxit,float tol,int*itout,float*tole,int*ncout){
 if(!s.ready||!b||!x||!adia||!aoff||!itout||!tole||!ncout||ndim!=3*s.nat||maxit<=0||tol<=0.f){set_error("GPBICG","invalid argument");return 30;}
 if(ensure_krylov_workspace("GPBICG Krylov workspace"))return 29;s.solution_valid=false;
 const int n=s.solver_n,th=256,gr=(n+th-1)/th;const size_t bytes=(size_t)n*sizeof(cufftComplex),full_bytes=(size_t)ndim*sizeof(cufftComplex);cudaError_t e;
 if(upload_full_host_to_compact(b,s.d_b,"GPBICG b compact H2D"))return 31;
 if(upload_full_host_to_compact(x,s.d_sx,"GPBICG x compact H2D"))return 31;
 if(upload_full_operators(adia,aoff,"GPBICG operator H2D"))return 31;
 cudaMemset(s.d_wrk,0,13*bytes);cudaMemset(s.d_cs,0,32*sizeof(cuDoubleComplex));
 cufftComplex*r0=s.d_wrk,*p=s.d_wrk+n,*r=s.d_wrk+2*(size_t)n,*yv=s.d_wrk+3*(size_t)n,*t=s.d_wrk+4*(size_t)n,*ap=s.d_wrk+5*(size_t)n,*at=s.d_wrk+6*(size_t)n,*xsol=s.d_wrk+10*(size_t)n;
 vcopy<<<gr,th>>>(xsol,s.d_sx,n);if(launch_ok("GPBICG copy x"))return 32;
 double bnorm=0.0,rnorm=0.0;if(norm_host(s.d_b,n,bnorm,"GPBICG norm b"))return 32;if(bnorm==0.0){set_error("GPBICG","zero RHS norm");return 32;}
 int nc=0;float ms=0.f;int rc=matvec_solver_device('N',s.d_sx,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"GPBICG-device");
 g_init<<<gr,th>>>(s.d_b,s.d_sy,s.d_wrk,n);if(launch_ok("GPBICG init"))return 32;
 if(dotc(r0,r0,n,s.d_cs+GR0RN,"GPBICG dot r0"))return 32;set_c<<<1,1>>>(s.d_cs+GBETA,0,0);if(cudaDeviceSynchronize()!=cudaSuccess){set_error("GPBICG","initial sync failed");return 32;}
 if(norm_host(r,n,rnorm,"GPBICG initial residual norm"))return 32;double resid=rnorm/bnorm;int it=0;bool first_cycle=true;
 std::printf(" GPBICG iteration=%6d residual=%14.6E\n",0,resid);std::fflush(stdout);
 while(it<maxit&&resid>tol){
  ++it;
  g_p<<<gr,th>>>(s.d_wrk,s.d_sx,s.d_cs,n);if(launch_ok("GPBICG p"))return 33;
  rc=matvec_solver_device('N',s.d_sx,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"GPBICG-device");vcopy<<<gr,th>>>(ap,s.d_sy,n);
  if(dotc(r0,ap,n,s.d_cs+G0,"GPBICG dot alpha"))return 33;set_int<<<1,1>>>(s.d_status_map,0);g_alpha<<<1,1>>>(s.d_cs,s.d_status_map);if(sync_status("GPBICG alpha")!=0){set_error("GPBICG","alpha breakdown");return 34;}
  g_yt<<<gr,th>>>(s.d_wrk,s.d_sx,s.d_cs,n);if(launch_ok("GPBICG y/t"))return 34;
  rc=matvec_solver_device('N',s.d_sx,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"GPBICG-device");vcopy<<<gr,th>>>(at,s.d_sy,n);
  if(dotc(at,at,n,s.d_cs+G1,"GPBICG c1")||dotc(at,t,n,s.d_cs+G4,"GPBICG c4"))return 35;
  set_int<<<1,1>>>(s.d_status_map,0);if(first_cycle)g_dz0<<<1,1>>>(s.d_cs,s.d_status_map);else{if(dotc(yv,yv,n,s.d_cs+G2,"GPBICG c2")||dotc(at,yv,n,s.d_cs+G3,"GPBICG c3")||dotc(yv,t,n,s.d_cs+G5,"GPBICG c5"))return 35;g_dze<<<1,1>>>(s.d_cs,s.d_status_map);}if(sync_status("GPBICG dzeta")!=0){set_error("GPBICG","dzeta/eta breakdown");return 36;}
  g_up<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("GPBICG update"))return 36;
  if(norm_host(r,n,rnorm,"GPBICG recursive residual norm"))return 37;resid=rnorm/bnorm;
  std::printf(" GPBICG iteration=%6d residual=%14.6E\n",it,resid);std::fflush(stdout);
  if((it%RELIABLE_RESID_PERIOD)==0||resid<=tol){
   ReliableResidualResult rr;if(reliable_residual_check("GPBICG","GPBICG-reliable",s.d_b,xsol,r,n,bnorm,resid,tol,nc,rr))return 37;
   if(rr.converged){resid=rr.true_res;break;}
   if(rr.restart){if(materialize_true_residual(s.d_b,n,"GPBICG materialize true residual"))return 37;g_restart_from_r<<<gr,th>>>(s.d_sy,s.d_wrk,n);if(launch_ok("GPBICG reliable restart vectors"))return 37;if(dotc(r0,r0,n,s.d_cs+GR0RN,"GPBICG restart dot r0"))return 37;set_c<<<1,1>>>(s.d_cs+GBETA,0,0);if(cudaDeviceSynchronize()!=cudaSuccess){set_error("GPBICG","restart sync failed");return 37;}first_cycle=true;resid=rr.true_res;continue;}
  }
  if(dotc(r0,r,n,s.d_cs+G0,"GPBICG rho"))return 39;set_int<<<1,1>>>(s.d_status_map,0);g_beta<<<1,1>>>(s.d_cs,s.d_status_map);if(sync_status("GPBICG beta")!=0){set_error("GPBICG","beta breakdown");return 39;}g_w<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("GPBICG w"))return 39;first_cycle=false;
 }
 if(resid>tol){ReliableResidualResult rr;if(reliable_residual_check("GPBICG","GPBICG-final-true",s.d_b,xsol,r,n,bnorm,resid,tol,nc,rr))return 40;resid=rr.true_res;}
 if(publish_solution(xsol,"GPBICG publish solution"))return 40;if(download_compact_to_full_host(s.d_sx,x,"GPBICG solution compact D2H")){return 40;}*itout=it;*tole=(float)resid;*ncout=nc;
 std::printf("CUDA GPBICG reliable residual: period=%d gap_restart=%g; no full-vector host transfers.\n",RELIABLE_RESID_PERIOD,RELIABLE_RESID_GAP);std::fflush(stdout);return 0;
}

extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_qmrccg_f32(const void*b,void*x,const void*adia,const void*aoff,int ndim,int maxit,float tol,int*itout,float*tole,int*ncout){
 last_error.clear();if(!s.ready||!b||!x||!adia||!aoff||!itout||!tole||!ncout||ndim!=3*s.nat||maxit<=0||tol<=0.f){set_error("QMRCCG","invalid argument");return 50;}
 if(ensure_krylov_workspace("QMRCCG Krylov workspace"))return 29;s.solution_valid=false;
 const int n=s.solver_n,th=256,gr=(n+th-1)/th;const size_t bytes=(size_t)n*sizeof(cufftComplex),full_bytes=(size_t)ndim*sizeof(cufftComplex);cudaError_t e;
 if(upload_full_host_to_compact(b,s.d_b,"QMR b compact H2D"))return 51;if(upload_full_operators(adia,aoff,"QMR operator H2D"))return 51;cudaMemset(s.d_wrk,0,13*bytes);cudaMemset(s.d_cs,0,32*sizeof(cuDoubleComplex));cudaMemset(s.d_rs,0,8*sizeof(double));
 cufftComplex*r=s.d_wrk,*p=s.d_wrk+n,*ap=s.d_wrk+2*(size_t)n,*q=s.d_wrk+3*(size_t)n,*vtilde=s.d_wrk+6*(size_t)n,*wtilde=s.d_wrk+7*(size_t)n,*atw=s.d_wrk+8*(size_t)n,*xs=s.d_wrk+9*(size_t)n;
 if(upload_full_host_to_compact(x,xs,"QMR x0 compact H2D"))return 51;
 double bnorm=0.0,rnorm=0.0;if(norm_host(s.d_b,n,bnorm,"QMR norm b"))return 52;if(bnorm==0.0){set_error("QMRCCG","zero RHS norm");return 52;}e=cudaMemcpy(s.d_rs+QBNORM,&bnorm,sizeof(double),cudaMemcpyHostToDevice);if(e!=cudaSuccess){set_cuda_error("QMR bnorm H2D",e);return 52;}
 auto wall0=WallClock::now();double mv0=s.matvec_total_ms;unsigned long long mc0=s.matvec_count;int nc=0;float ms=0.f;int rc=matvec_solver_device('N',xs,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"QMRCCG-device");q_init_vec<<<gr,th>>>(s.d_b,s.d_sy,s.d_wrk,n);if(launch_ok("QMR init vectors"))return 53;
 auto init_scalars=[&]()->int{
  if(nrm2(vtilde,n,s.d_rs+QGNORM,"QMR gamma")||nrm2(wtilde,n,s.d_rs+QKNORM,"QMR ksi")||dotu(vtilde,wtilde,n,s.d_cs+QRHO,"QMR rho"))return 54;
  int irc=matvec_solver_device('N',wtilde,s.d_sy,&ms);if(irc)return irc;nc++;report_matvec('N',ms,"QMRCCG-At=A");vcopy<<<gr,th>>>(atw,s.d_sy,n);if(launch_ok("QMR Atw copy"))return 54;if(dotu(vtilde,atw,n,s.d_cs+QEPS,"QMR epsilon"))return 54;
  set_int<<<1,1>>>(s.d_status_map,0);q_init_sc<<<1,1>>>(s.d_cs,s.d_rs,s.d_status_map);int ist=sync_status("QMR init scalars");if(ist!=0){char msg[80];std::snprintf(msg,sizeof(msg),"breakdown at step %d",ist);set_error("QMRCCG",msg);return 55;}return 0;};
 if((rc=init_scalars()))return rc;if(norm_host(r,n,rnorm,"QMR initial residual norm"))return 55;double resid=rnorm/bnorm;int it=0;
 std::printf(" QMRCCG iteration=%6d residual=%14.6E\n",0,resid);std::fflush(stdout);
 while(it<maxit&&resid>tol){
  ++it;q_pq<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("QMR p/q"))return 56;
  rc=matvec_solver_device('N',p,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"QMRCCG-device");q_update_vw<<<gr,th>>>(s.d_wrk,s.d_sy,s.d_cs,n);if(launch_ok("QMR update v/w"))return 56;
  if(nrm2(vtilde,n,s.d_rs+QGNORM,"QMR gamma next")||nrm2(wtilde,n,s.d_rs+QKNORM,"QMR ksi next")||dotu(vtilde,wtilde,n,s.d_cs+QNEW_RHO,"QMR rho next"))return 57;
  rc=matvec_solver_device('N',wtilde,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"QMRCCG-At=A");vcopy<<<gr,th>>>(atw,s.d_sy,n);if(launch_ok("QMR Atw copy"))return 57;if(dotu(vtilde,atw,n,s.d_cs+QNEW_EPS,"QMR epsilon next"))return 57;
  set_int<<<1,1>>>(s.d_status_map,0);q_recur<<<1,1>>>(s.d_cs,s.d_rs,s.d_status_map);int st=sync_status("QMR scalar recurrence");if(st!=0){char msg[80];std::snprintf(msg,sizeof(msg),"breakdown at step %d",st);set_error("QMRCCG",msg);return 58;}
  q_update_xr<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("QMR x/r update"))return 59;if(norm_host(r,n,rnorm,"QMR recursive residual norm"))return 59;resid=rnorm/bnorm;std::printf(" QMRCCG iteration=%6d residual=%14.6E\n",it,resid);std::fflush(stdout);
  if((it%RELIABLE_RESID_PERIOD)==0||resid<=tol){
   ReliableResidualResult rr;if(reliable_residual_check("QMRCCG","QMRCCG-reliable",s.d_b,xs,r,n,bnorm,resid,tol,nc,rr))return 59;
   if(rr.converged){resid=rr.true_res;break;}
   if(rr.restart){if(materialize_true_residual(s.d_b,n,"QMR materialize true residual"))return 59;q_restart_from_r<<<gr,th>>>(s.d_sy,s.d_wrk,n);if(launch_ok("QMR reliable restart vectors"))return 59;if((rc=init_scalars()))return rc;resid=rr.true_res;}
  }
 }
 if(resid>tol){ReliableResidualResult rr;if(reliable_residual_check("QMRCCG","QMRCCG-final-true",s.d_b,xs,r,n,bnorm,resid,tol,nc,rr))return 60;resid=rr.true_res;}
 auto wall1=WallClock::now();double wall=std::chrono::duration<double,std::milli>(wall1-wall0).count(),mvs=s.matvec_total_ms-mv0;unsigned long long mnc=s.matvec_count-mc0;
 std::printf("CUDA QMRCCG timing: iterative wall(no boundary D2H)=%.6f ms\n",wall);std::printf("CUDA QMRCCG timing: MATVEC GPU sum=%.6f ms over %llu MATVEC calls\n",mvs,mnc);std::printf("CUDA QMRCCG reliable residual: period=%d gap_restart=%g.\n",RELIABLE_RESID_PERIOD,RELIABLE_RESID_GAP);std::fflush(stdout);
 if(publish_solution(xs,"QMRCCG publish solution"))return 61;if(download_compact_to_full_host(s.d_sx,x,"solver solution compact D2H")){return 61;}*itout=it;*tole=(float)resid;*ncout=nc;return 0;
}

extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_pbcgst_f32(const void*b,void*x,const void*adia,const void*aoff,int ndim,int maxit,float tol,int*itout,float*tole,int*ncout){
 last_error.clear();
 if(!s.ready||!b||!x||!adia||!aoff||!itout||!tole||!ncout||ndim!=3*s.nat||maxit<=0||tol<=0.f){set_error("PBCGST","invalid argument");return 70;}
 if(ensure_krylov_workspace("PBCGST Krylov workspace"))return 29;s.solution_valid=false;
 const int n=s.solver_n,th=256,gr=(n+th-1)/th,restart_len=5;const size_t bytes=(size_t)n*sizeof(cufftComplex),full_bytes=(size_t)ndim*sizeof(cufftComplex);cudaError_t e;
 cufftComplex*r=s.d_wrk,*rt=s.d_wrk+(size_t)n,*p=s.d_wrk+2*(size_t)n,*sv=s.d_wrk+3*(size_t)n,*v=s.d_wrk+4*(size_t)n,*t=s.d_wrk+5*(size_t)n,*xs=s.d_wrk+6*(size_t)n,*rtrue=s.d_wrk+7*(size_t)n;
 // Boundary transfers only: the PBCGST iteration/restart loops contain no H2D/D2H cudaMemcpy.
 if(upload_full_host_to_compact(b,s.d_b,"PBCGST b compact H2D"))return 71;
 if(upload_full_operators(adia,aoff,"PBCGST operator H2D"))return 71;
 e=cudaMemset(s.d_wrk,0,13*bytes);if(e!=cudaSuccess){set_cuda_error("PBCGST clear work",e);return 71;}
 if(upload_full_host_to_compact(x,xs,"PBCGST x0 compact H2D"))return 71;
 if(nrm2(s.d_b,n,s.d_rs+PBNORM,"PBCGST norm(b)"))return 72;
 auto wall0=WallClock::now();double mv0=s.matvec_total_ms;unsigned long long mc0=s.matvec_count;
 int total_it=0,nc=0,restart=0;double resid=1.0;float ms=0.f;bool converged=false;
 std::printf(" PBCGST iteration=%6d residual=%14.6E\n",0,1.0);std::fflush(stdout);
 while(total_it<maxit&&!converged){
  if(restart>0){std::printf(" restart PBCGST CUDA: %d\n",restart);std::fflush(stdout);}
  // rtrue=b-Ax ; r=DIAGL(rtrue) ; rtilde=r ; p=v=0 ; rho=alpha=omega=1.
  int rc=matvec_solver_device('N',xs,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PBCGST-restart-residual");
  vdiff<<<gr,th>>>(rtrue,s.d_b,s.d_sy,n);if(launch_ok("PBCGST true residual init"))return 73;
  pbc_precon<<<gr,th>>>(r,rtrue,s.d_adia,s.d_occ_index,s.nat,s.nat0,n);if(launch_ok("PBCGST DIAGL init"))return 73;
  pbc_restart_init<<<gr,th>>>(r,s.d_wrk,s.d_cs,n);if(launch_ok("PBCGST restart init"))return 73;
  bool soft_restart=false;
  for(int local_it=0;local_it<restart_len&&total_it<maxit&&!converged;++local_it){
   if(dotc(rt,r,n,s.d_cs+PNEW_RHO,"PBCGST rho"))return 74;
   set_int<<<1,1>>>(s.d_status_map,0);pbc_beta<<<1,1>>>(s.d_cs,s.d_status_map);int st=sync_status("PBCGST beta");if(st<0)return 74;if(st!=0){set_error("PBCGST","hard breakdown at step 6 (rho0*omega == 0)");return 75;}
   pbc_p<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("PBCGST p update"))return 76;
   rc=matvec_solver_device('N',p,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PBCGST-v");
   pbc_precon<<<gr,th>>>(v,s.d_sy,s.d_adia,s.d_occ_index,s.nat,s.nat0,n);if(launch_ok("PBCGST DIAGL v"))return 76;
   if(dotc(rt,v,n,s.d_cs+PXI,"PBCGST xi"))return 77;set_int<<<1,1>>>(s.d_status_map,0);pbc_alpha<<<1,1>>>(s.d_cs,s.d_status_map);st=sync_status("PBCGST alpha");if(st<0)return 77;if(st!=0){set_error("PBCGST","hard breakdown at step 10 (rtilde^H v == 0)");return 78;}
   pbc_s<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("PBCGST s update"))return 79;if(nrm2(sv,n,s.d_rs+PSNORM,"PBCGST norm(s)"))return 79;real_to_mapped<<<1,1>>>(s.d_rs+PSNORM,s.d_resid_map);e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("PBCGST soft-breakdown sync",e);return 79;}
   if(*s.h_resid<1.19209e-7){
    // PIM step 12: soft breakdown; accept x <- x + alpha p and restart.
    pbc_soft_update<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("PBCGST soft update"))return 79;++total_it;soft_restart=true;break;
   }
   rc=matvec_solver_device('N',sv,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PBCGST-t");
   pbc_precon<<<gr,th>>>(t,s.d_sy,s.d_adia,s.d_occ_index,s.nat,s.nat0,n);if(launch_ok("PBCGST DIAGL t"))return 80;
   if(dotc(t,t,n,s.d_cs+PTT,"PBCGST dot(t,t)")||dotc(t,sv,n,s.d_cs+PTS,"PBCGST dot(t,s)"))return 81;set_int<<<1,1>>>(s.d_status_map,0);pbc_omega<<<1,1>>>(s.d_cs,s.d_status_map);st=sync_status("PBCGST omega");if(st<0)return 81;if(st!=0){set_error("PBCGST","hard breakdown at step 14 (t^H t == 0)");return 82;}
   pbc_update<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("PBCGST x/r update"))return 83;++total_it;
   // STOPTYPE=2: explicitly compute the true residual b-Ax, exactly as STOPCRIT does.
   rc=matvec_solver_device('N',xs,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PBCGST-true-residual");
   vdiff<<<gr,th>>>(rtrue,s.d_b,s.d_sy,n);if(launch_ok("PBCGST true residual"))return 84;if(nrm2(rtrue,n,s.d_rs+PTRNORM,"PBCGST norm true residual"))return 84;residual_kernel<<<1,1>>>(s.d_rs+PTRNORM,s.d_rs+PBNORM,s.d_resid_map);e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("PBCGST residual sync",e);return 84;}resid=*s.h_resid;
   std::printf(" PBCGST iteration=%6d residual=%14.6E\n",total_it,(double)resid);std::fflush(stdout);
   if(resid<=tol)converged=true;
  }
  ++restart;
  if(soft_restart&&!converged){
   // The original PIM routine returns STATUS=-2 and GETFML restarts it.
   std::printf(" PBCGST CUDA soft breakdown: restarting from updated x at iteration %d\n",total_it);std::fflush(stdout);
  }
 }
 // Ensure TOLE is the true relative residual even if MAXIT was reached immediately after a soft restart.
 if(!converged){
  int rc=matvec_solver_device('N',xs,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PBCGST-final-residual");vdiff<<<gr,th>>>(rtrue,s.d_b,s.d_sy,n);if(launch_ok("PBCGST final residual"))return 85;if(nrm2(rtrue,n,s.d_rs+PTRNORM,"PBCGST final norm"))return 85;residual_kernel<<<1,1>>>(s.d_rs+PTRNORM,s.d_rs+PBNORM,s.d_resid_map);e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("PBCGST final residual sync",e);return 85;}resid=*s.h_resid;
 }
 auto wall1=WallClock::now();double wall=std::chrono::duration<double,std::milli>(wall1-wall0).count(),mvs=s.matvec_total_ms-mv0;unsigned long long mnc=s.matvec_count-mc0;
 std::printf("CUDA PBCGST timing: iterative wall(no boundary D2H)=%.6f ms\n",wall);std::printf("CUDA PBCGST timing: MATVEC GPU sum=%.6f ms over %llu MATVEC calls\n",mvs,mnc);std::printf("CUDA PBCGST timing: wall minus MATVEC GPU=%.6f ms\n",wall-mvs);std::printf("CUDA PBCGST transfers: no full-vector H2D/D2H in iteration/restart loops; chunked cublasZdot partials are accumulated on CPU; convergence residual remains mapped.\n");std::fflush(stdout);
 if(publish_solution(xs,"PBCGST publish solution"))return 86;if(download_compact_to_full_host(s.d_sx,x,"solver solution compact D2H")){return 86;}*itout=total_it;*tole=(float)resid;*ncout=nc;
 if(!converged){set_error("PBCGST","MAXIT reached before requested tolerance");return 87;}
 return 0;
}


extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_pbcgs2_f32(const void*b,void*x,const void*adia,const void*aoff,int ndim,int maxit,float tol,int*itout,float*tole,int*ncout){
 last_error.clear();
 if(!s.ready||!b||!x||!adia||!aoff||!itout||!tole||!ncout||ndim!=3*s.nat||maxit<=0||tol<=0.f){set_error("PBCGS2","invalid argument");return 90;}
 if(ensure_krylov_workspace("PBCGS2 Krylov workspace"))return 29;s.solution_valid=false;
 const int n=s.solver_n,th=256,gr=(n+th-1)/th;const size_t bytes=(size_t)n*sizeof(cufftComplex),full_bytes=(size_t)ndim*sizeof(cufftComplex);cudaError_t e;
 cufftComplex *rr=s.d_wrk+0*(size_t)n,*r0=s.d_wrk+1*(size_t)n,*r1=s.d_wrk+2*(size_t)n,*r2=s.d_wrk+3*(size_t)n;
 cufftComplex *u0=s.d_wrk+4*(size_t)n,*u1=s.d_wrk+5*(size_t)n,*u2=s.d_wrk+6*(size_t)n,*xs=s.d_wrk+9*(size_t)n;
 // DDSCAT calls ZBCG2 with NONZERO_X=.FALSE.; X is zero but is still copied once
 // at the solver boundary so this entry point remains robust to a future caller.
 if(upload_full_host_to_compact(b,s.d_b,"PBCGS2 b compact H2D"))return 91;
 if(upload_full_operators(adia,aoff,"PBCGS2 operator H2D"))return 91;
 e=cudaMemset(s.d_wrk,0,13*bytes);if(e!=cudaSuccess){set_cuda_error("PBCGS2 clear work",e);return 91;}
 e=cudaMemset(s.d_cs,0,32*sizeof(cuDoubleComplex));if(e!=cudaSuccess){set_cuda_error("PBCGS2 clear complex scalars",e);return 91;}
 e=cudaMemset(s.d_rs,0,8*sizeof(double));if(e!=cudaSuccess){set_cuda_error("PBCGS2 clear real scalars",e);return 91;}
 if(upload_full_host_to_compact(x,xs,"PBCGS2 x0 compact H2D"))return 91;

 auto wall0=WallClock::now();double mv0=s.matvec_total_ms;unsigned long long mc0=s.matvec_count;
 p2_init_vectors<<<gr,th>>>(s.d_b,s.d_wrk,n);if(launch_ok("PBCGS2 init vectors"))return 92;
 if(nrm2(r0,n,s.d_rs+P2_RNRM0,"PBCGS2 initial norm"))return 92;
 p2_init_reals<<<1,1>>>(s.d_rs);p2_init_scalars<<<1,1>>>(s.d_cs);real_to_mapped<<<1,1>>>(s.d_rs+P2_RNRM0,s.d_resid_map);
 e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("PBCGS2 initial sync",e);return 92;}
 double rnrm0=*s.h_resid,resid=(rnrm0>0.0)?1.0:0.0;float ms=0.f;int it=0,nc=0;
 std::printf(" PBCGS2 iteration=%6d residual=%14.6E\n",0,(double)resid);std::fflush(stdout);

 while(resid>tol && it<maxit){
  ++it;p2_outer_start<<<1,1>>>(s.d_cs);if(launch_ok("PBCGS2 outer start"))return 93;
  for(int k=1;k<=2;k++){
   cufftComplex *rkminus=(k==1)?r0:r1;
   cufftComplex *ukminus=(k==1)?u0:u1;
   cufftComplex *uk=(k==1)?u1:u2;
   cufftComplex *rk=(k==1)?r1:r2;
   if(dotc(rr,rkminus,n,s.d_cs+P2_RHO1,"PBCGS2 rho1"))return 94;
   set_int<<<1,1>>>(s.d_status_map,0);p2_beta<<<1,1>>>(s.d_cs,s.d_status_map);int st=sync_status("PBCGS2 beta");
   if(st<0)return 94;if(st!=0){set_error("PBCGS2","ZBCG2 breakdown: RHO0 == 0");return 95;}
   p2_update_u<<<gr,th>>>(s.d_wrk,s.d_cs,n,k);if(launch_ok("PBCGS2 U recurrence"))return 96;
   int rc=matvec_solver_device('N',ukminus,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,k==1?"PBCGS2-U1":"PBCGS2-U2");
   vcopy<<<gr,th>>>(uk,s.d_sy,n);if(launch_ok("PBCGS2 copy U(k)"))return 96;
   if(dotc(rr,uk,n,s.d_cs+P2_SIGMA,"PBCGS2 sigma"))return 97;
   set_int<<<1,1>>>(s.d_status_map,0);p2_alpha<<<1,1>>>(s.d_cs,s.d_status_map);st=sync_status("PBCGS2 alpha");
   if(st<0)return 97;if(st!=0){set_error("PBCGS2","ZBCG2 breakdown: SIGMA == 0");return 98;}
   p2_update_x_r<<<gr,th>>>(s.d_wrk,s.d_cs,n,k);if(launch_ok("PBCGS2 X/R recurrence"))return 99;
   rc=matvec_solver_device('N',rkminus,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,k==1?"PBCGS2-R1":"PBCGS2-R2");
   vcopy<<<gr,th>>>(rk,s.d_sy,n);if(launch_ok("PBCGS2 copy R(k)"))return 99;
   if(nrm2(r0,n,s.d_rs+P2_RNRM,"PBCGS2 norm R0"))return 100;p2_update_max<<<1,1>>>(s.d_rs);
  }

  // Hermitian 3x3 Gram matrix Z = R^H R for R0,R1,R2.
  if(dotc(r0,r0,n,s.d_cs+P2_Z11,"PBCGS2 Z11")||
     dotc(r1,r0,n,s.d_cs+P2_Z21,"PBCGS2 Z21")||
     dotc(r1,r1,n,s.d_cs+P2_Z22,"PBCGS2 Z22")||
     dotc(r2,r0,n,s.d_cs+P2_Z31,"PBCGS2 Z31")||
     dotc(r2,r1,n,s.d_cs+P2_Z32,"PBCGS2 Z32")||
     dotc(r2,r2,n,s.d_cs+P2_Z33,"PBCGS2 Z33"))return 101;
  set_int<<<1,1>>>(s.d_status_map,0);p2_convex_scalars<<<1,1>>>(s.d_cs,s.d_rs,s.d_status_map);int st=sync_status("PBCGS2 convex polynomial");
  if(st<0)return 101;if(st!=0){set_error("PBCGS2","ZBCG2 convex-polynomial breakdown");return 102;}
  p2_convex_update<<<gr,th>>>(s.d_wrk,s.d_cs,n);if(launch_ok("PBCGS2 convex update"))return 103;
  p2_flags<<<1,1>>>(s.d_rs,s.d_resid_map,s.d_status_map);
  e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("PBCGS2 flags sync",e);return 103;}
  resid=*s.h_resid;int flags=*s.h_status;bool rcmp=(flags&1)!=0,xpdt=(flags&2)!=0;

  // Mandatory true-residual control every 20 outer BiCGStab(2) cycles and on
  // recursive convergence.  BP-A*X is the true residual of the physical
  // solution XP+X in the original ZBCG2 reliable-update representation.
  if((it%RELIABLE_RESID_PERIOD)==0||resid<=tol){
   ReliableResidualResult chk;if(reliable_residual_check("PBCGS2","PBCGS2-periodic-reliable",s.d_wrk+8*(size_t)n,xs,r0,n,rnrm0,resid,tol,nc,chk))return 104;
   if(chk.converged){resid=chk.true_res;break;}
   if(chk.restart){
    // s.d_sy still contains A*X from the check. Reuse the original ZBCG2
    // reliable-update machinery, consolidate XP<-XP+X, then reset all Krylov
    // histories so the next outer cycle starts consistently from r_true.
    p2_reliable_r<<<gr,th>>>(s.d_wrk,s.d_sy,n);if(launch_ok("PBCGS2 periodic true R"))return 104;
    p2_xpdt<<<gr,th>>>(s.d_wrk,n);if(launch_ok("PBCGS2 periodic consolidate XP"))return 104;
    p2_restart_vectors<<<gr,th>>>(s.d_wrk,n);if(launch_ok("PBCGS2 periodic restart vectors"))return 104;
    p2_restart_reals<<<1,1>>>(s.d_rs,chk.true_res*rnrm0);p2_init_scalars<<<1,1>>>(s.d_cs);if(launch_ok("PBCGS2 periodic restart scalars"))return 104;
    resid=chk.true_res;std::printf(" PBCGS2 reliable restart at iteration %d\n",it);std::fflush(stdout);continue;
   }
  }

  // Reliable update from the original ZBCG2 implementation. This MATVEC also
  // remains fully device-to-device; only the two mapped status/residual scalars
  // are observed by the host.
  if(rcmp){
   int rc=matvec_solver_device('N',xs,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PBCGS2-reliable");
   p2_reliable_r<<<gr,th>>>(s.d_wrk,s.d_sy,n);if(launch_ok("PBCGS2 reliable R"))return 104;
   if(xpdt){p2_xpdt<<<gr,th>>>(s.d_wrk,n);if(launch_ok("PBCGS2 XPDT"))return 104;}
   p2_reliable_reals<<<1,1>>>(s.d_rs,xpdt?1:0);if(launch_ok("PBCGS2 reliable scalar update"))return 104;
  }
  std::printf(" PBCGS2 iteration=%6d residual=%14.6E\n",it,(double)resid);std::fflush(stdout);
 }

 // ZBCG2 returns X + XP and always computes one final true residual.
 p2_finalize_x<<<gr,th>>>(s.d_wrk,n);if(launch_ok("PBCGS2 final X+XP"))return 105;
 int rc=matvec_solver_device('N',xs,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PBCGS2-final-residual");
 vdiff<<<gr,th>>>(r0,s.d_b,s.d_sy,n);if(launch_ok("PBCGS2 final true residual"))return 105;
 if(nrm2(r0,n,s.d_rs+P2_RNRM,"PBCGS2 final norm"))return 106;
 residual_kernel<<<1,1>>>(s.d_rs+P2_RNRM,s.d_rs+P2_RNRM0,s.d_resid_map);
 e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("PBCGS2 final residual sync",e);return 106;}resid=*s.h_resid;

 auto wall1=WallClock::now();double wall=std::chrono::duration<double,std::milli>(wall1-wall0).count(),mvs=s.matvec_total_ms-mv0;unsigned long long mnc=s.matvec_count-mc0;
 std::printf("CUDA PBCGS2 timing: iterative wall(no boundary D2H)=%.6f ms\n",wall);
 std::printf("CUDA PBCGS2 timing: MATVEC GPU sum=%.6f ms over %llu MATVEC calls\n",mvs,mnc);
 std::printf("CUDA PBCGS2 timing: wall minus MATVEC GPU=%.6f ms\n",wall-mvs);
 std::printf("CUDA PBCGS2 transfers: no full-vector H2D/D2H in iterative/reliable-update loops; chunked cublasZdot partials are accumulated on CPU; residual/status remain mapped.\n");
 if(resid>tol)std::printf("CUDA PBCGS2: MAXIT reached; final true relative residual=%14.6E\n",(double)resid);
 std::fflush(stdout);
 if(publish_solution(xs,"PBCGS2 publish solution"))return 107;if(download_compact_to_full_host(s.d_sx,x,"PBCGS2 solution compact D2H")){return 107;}
 *itout=it;*tole=(float)resid;*ncout=nc;
 // Original ZBCG2 reports INFO=1 on nonconvergence but GETFML does not abort.
 // Match that behavior by returning success with the achieved TOLE.
 return 0;
}

extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_petrkp_f32(const void*b,void*x,const void*adia,const void*aoff,int ndim,int maxit,float tol,int*itout,float*tole,int*ncout){
 last_error.clear();
 if(!s.ready||!b||!x||!adia||!aoff||!itout||!tole||!ncout||ndim!=3*s.nat||maxit<=0||tol<=0.f){set_error("PETRKP","invalid argument");return 110;}
 if(ensure_krylov_workspace("PETRKP Krylov workspace"))return 29;s.solution_valid=false;
 const int n=s.solver_n,th=256,gr=(n+th-1)/th;const size_t bytes=(size_t)n*sizeof(cufftComplex),full_bytes=(size_t)ndim*sizeof(cufftComplex);cudaError_t e;
 // work layout: ACE, GI, PI, QI, AXI, R, X. CR from PETR90VER2 is unused.
 cufftComplex *ace=s.d_wrk+0*(size_t)n,*gi=s.d_wrk+1*(size_t)n,*pi=s.d_wrk+2*(size_t)n;
 cufftComplex *qi=s.d_wrk+3*(size_t)n,*axi=s.d_wrk+4*(size_t)n,*rv=s.d_wrk+5*(size_t)n,*xs=s.d_wrk+6*(size_t)n;
 // Solver-boundary transfers only. The PETRKP iteration loop below contains no cudaMemcpy.
 if(upload_full_host_to_compact(b,s.d_b,"PETRKP b compact H2D"))return 111;
 if(upload_full_operators(adia,aoff,"PETRKP operator H2D"))return 111;
 e=cudaMemset(s.d_wrk,0,13*bytes);if(e!=cudaSuccess){set_cuda_error("PETRKP clear work",e);return 111;}
 e=cudaMemset(s.d_rs,0,8*sizeof(double));if(e!=cudaSuccess){set_cuda_error("PETRKP clear real scalars",e);return 111;}
 e=cudaMemset(s.d_cs,0,32*sizeof(cuDoubleComplex));if(e!=cudaSuccess){set_cuda_error("PETRKP clear complex scalars",e);return 111;}
 if(upload_full_host_to_compact(x,xs,"PETRKP x0 compact H2D"))return 111;
 if(dotc(s.d_b,s.d_b,n,s.d_cs+PC_BB,"PETRKP dot(b,b)"))return 112;

 auto wall0=WallClock::now();double mv0=s.matvec_total_ms;unsigned long long mc0=s.matvec_count;
 int nc=0,it=0;float ms=0.f;double resid=1.0;
 // ACE=A^H B, GI=ACE, PI=GI.
 int rc=matvec_solver_device('C',s.d_b,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('C',ms,"PETRKP-AH-b");
 petr_init_vectors<<<gr,th>>>(s.d_sy,s.d_wrk,n);if(launch_ok("PETRKP init ACE/GI/PI"))return 113;
 // QI=A PI ; alpha=||GI||^2/||QI||^2 ; X=X+alpha PI.
 rc=matvec_solver_device('N',pi,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PETRKP-A-p-initial");
 vcopy<<<gr,th>>>(qi,s.d_sy,n);if(launch_ok("PETRKP initial QI"))return 113;
 if(dotc(gi,gi,n,s.d_cs+PC_GG,"PETRKP dot(GI,GI)")||dotc(qi,qi,n,s.d_cs+PC_QQ,"PETRKP dot(QI,QI)"))return 114;
 set_int<<<1,1>>>(s.d_status_map,0);petr_alpha<<<1,1>>>(s.d_cs,s.d_rs,s.d_status_map);int st=sync_status("PETRKP initial alpha");
 if(st<0)return 114;if(st!=0){set_error("PETRKP","breakdown: ||QI|| == 0 while computing initial alpha");return 115;}
 petr_x_update<<<gr,th>>>(xs,pi,s.d_rs,n);if(launch_ok("PETRKP initial X update"))return 115;
 // AXI=A X after the initial step.
 rc=matvec_solver_device('N',xs,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PETRKP-A-x-initial");
 vcopy<<<gr,th>>>(axi,s.d_sy,n);if(launch_ok("PETRKP initial AXI"))return 116;
 std::printf(" PETRKP iteration=%6d residual=%14.6E\n",0,1.0);std::fflush(stdout);

 for(it=1;it<=maxit;++it){
  // GI = A^H B - A^H AXI = ACE - A^H AXI.
  petr_save_gigi<<<1,1>>>(s.d_cs);if(launch_ok("PETRKP save GIGI"))return 117;
  rc=matvec_solver_device('C',axi,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('C',ms,"PETRKP-AH-Ax");
  petr_g_update<<<gr,th>>>(gi,ace,s.d_sy,n);if(launch_ok("PETRKP GI update"))return 117;
  if(dotc(gi,gi,n,s.d_cs+PC_GG,"PETRKP dot new GI"))return 118;
  set_int<<<1,1>>>(s.d_status_map,0);petr_beta<<<1,1>>>(s.d_cs,s.d_rs,s.d_status_map);st=sync_status("PETRKP beta");
  if(st<0)return 118;if(st!=0){set_error("PETRKP","breakdown: previous ||GI|| == 0 while computing beta");return 119;}
  petr_p_update<<<gr,th>>>(pi,gi,s.d_rs,n);if(launch_ok("PETRKP PI update"))return 119;

  rc=matvec_solver_device('N',pi,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PETRKP-A-p");
  vcopy<<<gr,th>>>(qi,s.d_sy,n);if(launch_ok("PETRKP QI update"))return 120;
  if(dotc(qi,qi,n,s.d_cs+PC_QQ,"PETRKP dot QI"))return 120;
  set_int<<<1,1>>>(s.d_status_map,0);petr_alpha<<<1,1>>>(s.d_cs,s.d_rs,s.d_status_map);st=sync_status("PETRKP alpha");
  if(st<0)return 120;if(st!=0){set_error("PETRKP","breakdown: ||QI|| == 0 while computing alpha");return 121;}
  petr_x_update<<<gr,th>>>(xs,pi,s.d_rs,n);if(launch_ok("PETRKP X update"))return 121;

  // PETR90VER2 recursively updates AXI except every 10th iteration, where it
  // recomputes A*X exactly to control accumulated roundoff.
  if(it!=10*(it/10)){
   petr_axi_update<<<gr,th>>>(axi,qi,s.d_rs,n);if(launch_ok("PETRKP AXI recurrence"))return 122;
  }else{
   rc=matvec_solver_device('N',xs,s.d_sy,&ms);if(rc)return rc;nc++;report_matvec('N',ms,"PETRKP-A-x-refresh");
   vcopy<<<gr,th>>>(axi,s.d_sy,n);if(launch_ok("PETRKP AXI refresh"))return 122;
  }
  // True residual in the original code is R=AXI-B and sqrt(R^H R / B^H B).
  vdiff<<<gr,th>>>(rv,axi,s.d_b,n);if(launch_ok("PETRKP residual vector"))return 123;
  if(dotc(rv,rv,n,s.d_cs+PC_RR,"PETRKP dot residual"))return 123;
  petr_residual<<<1,1>>>(s.d_cs,s.d_resid_map);
  e=cudaDeviceSynchronize();if(e!=cudaSuccess){set_cuda_error("PETRKP residual sync",e);return 123;}
  resid=*s.h_resid;std::printf(" PETRKP iteration=%6d residual=%14.6E\n",it,(double)resid);std::fflush(stdout);
  if(resid<tol)break;
 }
 // If MAXIT is reached, match PETR90VER2: return the last iterate rather than abort.
 auto wall1=WallClock::now();double wall=std::chrono::duration<double,std::milli>(wall1-wall0).count(),mvs=s.matvec_total_ms-mv0;unsigned long long mnc=s.matvec_count-mc0;
 std::printf("CUDA PETRKP timing: iterative wall(no boundary D2H)=%.6f ms\n",wall);
 std::printf("CUDA PETRKP timing: MATVEC GPU sum=%.6f ms over %llu MATVEC/CMATVEC calls\n",mvs,mnc);
 std::printf("CUDA PETRKP timing: wall minus MATVEC GPU=%.6f ms\n",wall-mvs);
 std::printf("CUDA PETRKP transfers: no full-vector H2D/D2H in iteration loop; chunked cublasZdot partials are accumulated on CPU; convergence residual remains mapped.\n");
 if(resid>=tol)std::printf("CUDA PETRKP: MAXIT reached; achieved relative residual=%14.6E\n",(double)resid);
 std::fflush(stdout);
 if(publish_solution(xs,"PETRKP publish solution"))return 124;if(download_compact_to_full_host(s.d_sx,x,"solver solution compact D2H")){return 124;}
 *itout=(it<=maxit)?it:maxit;*tole=(float)resid;*ncout=nc;return 0;
}

// ============================================================================
// IFDDA/ADDA-derived additional L solvers
// Selector BICGS2 is handled as an alias of the existing PBCGS2/ZBCG2 solver.
// ============================================================================
extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_bicgstab4_f32(const void*b,void*x,const void*adia,const void*aoff,
    int ndim,int maxit,float tol,int*itout,float*tole,int*ncout){
 last_error.clear();
 if(!s.ready||!b||!x||!adia||!aoff||!itout||!tole||!ncout||ndim!=3*s.nat||maxit<=0||tol<=0.f){set_error("BiCGStab4","invalid argument");return 130;}
 const int n=s.solver_n;const size_t bytes=(size_t)n*sizeof(cufftComplex),full_bytes=(size_t)ndim*sizeof(cufftComplex);int nc=0,rc=0;
 if(init_solver_boundary(b,x,adia,aoff,ndim,"BiCGStab4 boundary H2D"))return 131;
 // r0,u0,r~,r1..r4,u1..u4 = 11 auxiliary vectors; x is s.d_sx.
 cufftComplex*r[5]={s.d_wrk+0*(size_t)n,s.d_wrk+3*(size_t)n,s.d_wrk+4*(size_t)n,s.d_wrk+5*(size_t)n,s.d_wrk+6*(size_t)n};
 cufftComplex*u[5]={s.d_wrk+1*(size_t)n,s.d_wrk+7*(size_t)n,s.d_wrk+8*(size_t)n,s.d_wrk+9*(size_t)n,s.d_wrk+10*(size_t)n};
 cufftComplex*rtilda=s.d_wrk+2*(size_t)n;
 if(initial_true_residual(r[0],n,nc,"BiCGStab4-initial-Ax"))return 132;
 double bnorm=0.0,rnorm=0.0;if(norm_host(s.d_b,n,bnorm,"BiCGStab4 norm b")||norm_host(r[0],n,rnorm,"BiCGStab4 norm r0"))return 132;
 if(bnorm==0.0){set_error("BiCGStab4","zero RHS norm");return 133;}
 double resid=rnorm/bnorm;if(resid<=tol){s.solution_valid=true;if(download_compact_to_full_host(s.d_sx,x,"solver solution compact D2H")){return 133;}*itout=0;*tole=(float)resid;*ncout=nc;return 0;}
 if(rnorm<1.e-30){set_error("BiCGStab4","cannot start from zero residual");return 133;}
 if(op_scale(rtilda,r[0],hcomplex(1.0/rnorm,0),n,"BiCGStab4 normalize rtilde")||op_zero(u[0],n,"BiCGStab4 zero u0"))return 134;
 hcomplex rho(1,0),alpha(0,0),omega(1,0),rho1,beta,den,temp;
 hcomplex tau[4][4],gp[5],gam[5],gpp[5];double sigma[5]={0,0,0,0,0};
 auto wall0=WallClock::now();double mv0=s.matvec_total_ms;unsigned long long mc0=s.matvec_count;
 int it=0;
 for(it=1;it<=maxit;it++){
  rho=-omega*rho;
  for(int j=0;j<4;j++){
   if(std::abs(rho)<1.e-30){set_error("BiCGStab4","breakdown: rho is zero");return 135;}
   if(dotc_host(rtilda,r[j],n,rho1,"BiCGStab4 rho"))return 135;
   beta=alpha*rho1/rho;rho=rho1;temp=-beta;
   for(int i=0;i<=j;i++)if(op_affine(u[i],r[i],temp,n,"BiCGStab4 u_i=r_i-beta*u_i"))return 136;
   if((rc=run_mv(u[j],u[j+1],"BiCGStab4-u",nc)))return rc;
   if(dotc_host(rtilda,u[j+1],n,den,"BiCGStab4 denominator"))return 137;
   if(std::abs(den)<1.e-30){set_error("BiCGStab4","breakdown: <rtilde,u> is zero");return 137;}
   alpha=rho/den;temp=-alpha;
   for(int i=0;i<=j;i++)if(op_axpy(r[i],u[i+1],temp,n,"BiCGStab4 r_i-alpha*u"))return 138;
   if((rc=run_mv(r[j],r[j+1],"BiCGStab4-r",nc)))return rc;
   if(op_axpy(s.d_sx,u[0],alpha,n,"BiCGStab4 x+=alpha*u0"))return 139;
  }
  for(auto&row:tau)for(auto&z:row)z=hcomplex(0,0);for(auto&z:gp)z=hcomplex(0,0);for(auto&z:gam)z=hcomplex(0,0);for(auto&z:gpp)z=hcomplex(0,0);
  for(int j=1;j<=4;j++){
   for(int i=1;i<j;i++){
    hcomplex d;if(dotc_host(r[i],r[j],n,d,"BiCGStab4 tau dot"))return 140;
    tau[j-1][i-1]=d/sigma[i];
    if(op_axpy(r[j],r[i],-tau[j-1][i-1],n,"BiCGStab4 MGS"))return 140;
   }
   hcomplex sj;if(dotc_host(r[j],r[j],n,sj,"BiCGStab4 sigma"))return 141;sigma[j]=sj.real();
   if(!(sigma[j]>1.e-30)){set_error("BiCGStab4","breakdown in MR polynomial");return 141;}
   hcomplex d;if(dotc_host(r[j],r[0],n,d,"BiCGStab4 gamma prime"))return 141;gp[j]=d/sigma[j];
  }
  omega=gam[4]=gp[4];
  for(int j=3;j>=1;j--){gam[j]=gp[j];for(int i=j+1;i<=4;i++)gam[j]-=tau[i-1][j-1]*gam[i];}
  for(int j=1;j<4;j++){gpp[j]=gam[j+1];for(int i=j+1;i<4;i++)gpp[j]+=tau[i-1][j-1]*gam[i+1];}
  if(op_axpy(s.d_sx,r[0],gam[1],n,"BiCGStab4 MR x r0"))return 142;
  if(op_axpy(r[0],r[4],-gp[4],n,"BiCGStab4 MR r4")||op_axpy(u[0],u[4],-gam[4],n,"BiCGStab4 MR u4"))return 142;
  for(int j=1;j<4;j++){
   if(op_axpy(s.d_sx,r[j],gpp[j],n,"BiCGStab4 MR x")||op_axpy(r[0],r[j],-gp[j],n,"BiCGStab4 MR r")||op_axpy(u[0],u[j],-gam[j],n,"BiCGStab4 MR u"))return 142;
  }
  if(norm_host(r[0],n,rnorm,"BiCGStab4 residual norm"))return 143;resid=rnorm/bnorm;
  std::printf(" BiCGStab(4) iteration=%6d residual=%14.6E\n",it,resid);std::fflush(stdout);
  if((it%RELIABLE_RESID_PERIOD)==0||resid<=tol){
   ReliableResidualResult rr;if(reliable_residual_check("BiCGStab(4)","BiCGStab4-reliable",s.d_b,s.d_sx,r[0],n,bnorm,resid,tol,nc,rr))return 144;
   if(rr.converged){resid=rr.true_res;break;}
   if(rr.restart){if(materialize_true_residual(s.d_b,n,"BiCGStab4 materialize true residual"))return 144;if(op_copy(r[0],s.d_sy,n,"BiCGStab4 restart r"))return 144;rnorm=rr.true_res*bnorm;if(!(rnorm>0.0)){set_error("BiCGStab4","reliable restart produced zero residual");return 144;}if(op_scale(rtilda,r[0],hcomplex(1.0/rnorm,0),n,"BiCGStab4 restart rtilde")||op_zero(u[0],n,"BiCGStab4 restart u0"))return 144;rho=hcomplex(1,0);alpha=hcomplex(0,0);omega=hcomplex(1,0);resid=rr.true_res;}
  }
 }
 // Final true residual is reported without changing the returned x.
 if((rc=run_mv(s.d_sx,s.d_sy,"BiCGStab4-final-true",nc)))return rc;int th=256,gr=(n+th-1)/th;vdiff<<<gr,th>>>(r[0],s.d_b,s.d_sy,n);if(launch_ok("BiCGStab4 final true residual"))return 144;
 if(norm_host(r[0],n,rnorm,"BiCGStab4 final true norm"))return 144;resid=rnorm/bnorm;
 auto wall1=WallClock::now();double wall=std::chrono::duration<double,std::milli>(wall1-wall0).count(),mvs=s.matvec_total_ms-mv0;unsigned long long mnc=s.matvec_count-mc0;
 std::printf("CUDA BiCGStab(4) timing: wall=%.6f ms MATVEC_GPU=%.6f ms (%llu calls) outside=%.6f ms\n",wall,mvs,mnc,wall-mvs);
 std::printf("CUDA BiCGStab(4): vectors=float32; dot products=chunked cublasZdotc FP64; MR coefficients=complex<double>; full-vector loop copies H2D/D2H=0\n");std::fflush(stdout);
 s.solution_valid=true;if(download_compact_to_full_host(s.d_sx,x,"solver solution compact D2H")){return 145;}
 *itout=(it<=maxit)?it:maxit;*tole=(float)resid;*ncout=nc;return 0;
}

extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_gpbicgstab2_f32(const void*b,void*x,const void*adia,const void*aoff,
    int ndim,int maxit,float tol,int*itout,float*tole,int*ncout){
 last_error.clear();
 if(!s.ready||!b||!x||!adia||!aoff||!itout||!tole||!ncout||ndim!=3*s.nat||maxit<=0||tol<=0.f){set_error("GPBiCGStab2","invalid argument");return 150;}
 const int n=s.solver_n;const size_t bytes=(size_t)n*sizeof(cufftComplex),full_bytes=(size_t)ndim*sizeof(cufftComplex);int nc=0,rc=0;
 if(init_solver_boundary(b,x,adia,aoff,ndim,"GPBiCGStab2 boundary H2D"))return 151;
 cufftComplex*r0=s.d_wrk+0*(size_t)n,*p0=s.d_wrk+1*(size_t)n,*rt=s.d_wrk+2*(size_t)n,*z=s.d_wrk+3*(size_t)n,*yv=s.d_wrk+4*(size_t)n,*u=s.d_wrk+5*(size_t)n,*hs=s.d_wrk+6*(size_t)n,*hq0=s.d_wrk+7*(size_t)n,*hq1=s.d_wrk+8*(size_t)n,*work=s.d_wrk+9*(size_t)n;
 if(initial_true_residual(r0,n,nc,"GPBiCGStab2-initial-Ax"))return 152;
 double bnorm=0,rnorm=0;if(norm_host(s.d_b,n,bnorm,"GPBiCGStab2 norm b")||norm_host(r0,n,rnorm,"GPBiCGStab2 norm r"))return 152;if(bnorm==0.0){set_error("GPBiCGStab2","zero RHS norm");return 153;}
 if(op_copy(rt,r0,n,"GP2 rt")||op_copy(p0,r0,n,"GP2 p0")||op_zero(z,n,"GP2 z"))return 153;
 bool fresh=true;double resid=rnorm/bnorm;hcomplex rho,sigma,alpha,beta,zeta1,zeta2,eta,temp;hcomplex mat[3][3],rhs[3];
 auto wall0=WallClock::now();double mv0=s.matvec_total_ms;unsigned long long mc0=s.matvec_count;int it=0;
 for(it=1;it<=maxit && resid>tol;it++){
  if(fresh){
   if(dotc_host(rt,r0,n,rho,"GP2 rho1"))return 154;if((rc=run_mv(p0,work,"GPBiCGStab2-p1",nc)))return rc;if(dotc_host(rt,work,n,sigma,"GP2 sigma1"))return 154;if(std::abs(sigma)<1.e-30){set_error("GP2","sigma1 zero");return 154;}alpha=rho/sigma;
   if(op_axpy(s.d_sx,p0,alpha,n,"GP2 x")||op_axpy(r0,work,-alpha,n,"GP2 r0"))return 155;if((rc=run_mv(r0,hs,"GPBiCGStab2-r1",nc)))return rc;if(dotc_host(rt,hs,n,rho,"GP2 rho r1"))return 155;beta=rho/sigma;
   if(op_affine(p0,r0,-beta,n,"GP2 p0")||op_affine(work,hs,-beta,n,"GP2 p1"))return 156;
   if((rc=run_mv(work,hq1,"GPBiCGStab2-p2",nc)))return rc;if(dotc_host(rt,hq1,n,sigma,"GP2 sigma2"))return 156;if(std::abs(sigma)<1.e-30){set_error("GP2","sigma2 zero");return 156;}alpha=rho/sigma;
   if(op_axpy(s.d_sx,p0,alpha,n,"GP2 x2")||op_axpy(r0,work,-alpha,n,"GP2 r0 2")||op_axpy(hs,hq1,-alpha,n,"GP2 r1"))return 157;
   if((rc=run_mv(hs,hq0,"GPBiCGStab2-r2",nc)))return rc;if(dotc_host(rt,hq0,n,rho,"GP2 rho r2"))return 157;beta=rho/sigma;
   if(op_affine(p0,r0,-beta,n,"GP2 p0b")||op_affine(work,hs,-beta,n,"GP2 p1b")||op_affine(hq1,hq0,-beta,n,"GP2 p2b"))return 158;
   if(dotc_host(hs,hs,n,mat[0][0],"GP2 m00")||dotc_host(hs,hq0,n,mat[0][1],"GP2 m01")||dotc_host(hq0,hq0,n,mat[1][1],"GP2 m11")||dotc_host(hs,r0,n,rhs[0],"GP2 rhs0")||dotc_host(hq0,r0,n,rhs[1],"GP2 rhs1"))return 159;mat[1][0]=std::conj(mat[0][1]);
   if(!solve_small(&mat[0][0],rhs,2,3)){set_error("GP2","initial 2x2 minimization breakdown");return 159;}zeta1=rhs[0];zeta2=rhs[1];
   if(op_lincomb2(z,r0,hs,zeta1,zeta2,n,"GP2 z init")||op_axpy(s.d_sx,z,hcomplex(1,0),n,"GP2 x z")||op_lincomb2(yv,hs,hq0,zeta1,zeta2,n,"GP2 y")||op_lincomb2(u,work,hq1,zeta1,zeta2,n,"GP2 u")||op_axpy(r0,yv,hcomplex(-1,0),n,"GP2 r-y")||op_axpy(p0,u,hcomplex(-1,0),n,"GP2 p-u")||op_copy(hq0,work,n,"GP2 history q0"))return 160;
   fresh=false;
  }else{
   if(dotc_host(rt,r0,n,rho,"GP2 rho"))return 161;if((rc=run_mv(p0,work,"GPBiCGStab2-p1",nc)))return rc;if(dotc_host(rt,work,n,sigma,"GP2 sigma1"))return 161;if(std::abs(sigma)<1.e-30){set_error("GP2","sigma1 zero");return 161;}alpha=rho/sigma;
   if(op_axpy(s.d_sx,p0,alpha,n,"GP2 x1")||op_axpy(z,u,-alpha,n,"GP2 z1")||op_axpy2(yv,hq0,work,-alpha,alpha,n,"GP2 y1")||op_axpy(r0,work,-alpha,n,"GP2 r01")||op_axpy(hs,hq1,-alpha,n,"GP2 s0"))return 162;
   if((rc=run_mv(r0,hq1,"GPBiCGStab2-r1",nc)))return rc;if(dotc_host(rt,hq1,n,rho,"GP2 rho1 full"))return 162;beta=rho/sigma;
   if(op_affine(p0,r0,-beta,n,"GP2 p0 full")||op_affine(work,hq1,-beta,n,"GP2 p1 full")||op_affine(hq0,hs,-beta,n,"GP2 q0 full")||op_affine(u,yv,-beta,n,"GP2 u full"))return 163;
   if((rc=run_mv(work,hs,"GPBiCGStab2-p2",nc)))return rc;if(dotc_host(rt,hs,n,sigma,"GP2 sigma2 full"))return 163;if(std::abs(sigma)<1.e-30){set_error("GP2","sigma2 zero");return 163;}alpha=rho/sigma;
   if(op_axpy(s.d_sx,p0,alpha,n,"GP2 x2 full")||op_axpy(z,u,-alpha,n,"GP2 z2")||op_axpy2(yv,hq0,work,-alpha,alpha,n,"GP2 y2")||op_axpy(r0,work,-alpha,n,"GP2 r02")||op_axpy(hq1,hs,-alpha,n,"GP2 r1 upd"))return 164;
   if((rc=run_mv(hq1,hq0,"GPBiCGStab2-r2",nc)))return rc;if(dotc_host(rt,hq0,n,rho,"GP2 rho2 full"))return 164;beta=rho/sigma;
   if(op_affine(p0,r0,-beta,n,"GP2 p0 end")||op_affine(work,hq1,-beta,n,"GP2 p1 end")||op_affine(hs,hq0,-beta,n,"GP2 p2 end")||op_affine(u,yv,-beta,n,"GP2 u end"))return 165;
   if(dotc_host(hq1,hq1,n,mat[0][0],"GP2 m00 full")||dotc_host(hq1,hq0,n,mat[0][1],"GP2 m01 full")||dotc_host(hq1,yv,n,mat[0][2],"GP2 m02 full")||dotc_host(hq0,hq0,n,mat[1][1],"GP2 m11 full")||dotc_host(hq0,yv,n,mat[1][2],"GP2 m12 full")||dotc_host(yv,yv,n,mat[2][2],"GP2 m22 full")||dotc_host(hq1,r0,n,rhs[0],"GP2 rhs0 full")||dotc_host(hq0,r0,n,rhs[1],"GP2 rhs1 full")||dotc_host(yv,r0,n,rhs[2],"GP2 rhs2 full"))return 166;
   mat[1][0]=std::conj(mat[0][1]);mat[2][0]=std::conj(mat[0][2]);mat[2][1]=std::conj(mat[1][2]);
   if(!solve_small(&mat[0][0],rhs,3,3)){set_error("GP2","3x3 minimization breakdown");return 166;}zeta1=rhs[0];zeta2=rhs[1];eta=rhs[2];
   if(op_self_add2(z,r0,hq1,eta,zeta1,zeta2,n,"GP2 z final")||op_axpy(s.d_sx,z,hcomplex(1,0),n,"GP2 x final")||op_self_add2(yv,hq1,hq0,eta,zeta1,zeta2,n,"GP2 y final")||op_self_add2(u,work,hs,eta,zeta1,zeta2,n,"GP2 u final")||op_axpy(r0,yv,hcomplex(-1,0),n,"GP2 r final")||op_axpy(p0,u,hcomplex(-1,0),n,"GP2 p final"))return 167;
   // Restore fixed boundary histories: hs=r1, hq0=p1, hq1=p2.
   if(op_copy(hq0,hs,n,"GP2 tmp p2")||op_copy(hs,hq1,n,"GP2 hist r1")||op_copy(hq1,hq0,n,"GP2 hist p2")||op_copy(hq0,work,n,"GP2 hist p1"))return 168;
  }
  if(norm_host(r0,n,rnorm,"GP2 recursive residual"))return 169;resid=rnorm/bnorm;
  std::printf(" GPBiCGStab(2) iteration=%6d residual=%14.6E\n",it,resid);std::fflush(stdout);
  // Common reliable-residual policy: every 20 iterations and whenever the
  // recursive residual first claims convergence. Keep Krylov history intact
  // when the gap is small; restart only for gap>=1e-3 or false convergence.
  if((it%RELIABLE_RESID_PERIOD)==0||resid<=tol){
   ReliableResidualResult rr;if(reliable_residual_check("GPBiCGStab(2)","GPBiCGStab2-reliable",s.d_b,s.d_sx,r0,n,bnorm,resid,tol,nc,rr))return 170;
   if(rr.converged){resid=rr.true_res;break;}
   if(rr.restart){if(materialize_true_residual(s.d_b,n,"GP2 materialize true residual"))return 171;if(op_copy(r0,s.d_sy,n,"GP2 restart r")||op_copy(rt,r0,n,"GP2 restart rt")||op_copy(p0,r0,n,"GP2 restart p")||op_zero(z,n,"GP2 restart z"))return 171;fresh=true;resid=rr.true_res;}
  }
 }
 if(resid>tol){if((rc=run_mv(s.d_sx,s.d_sy,"GPBiCGStab2-final-true",nc)))return rc;int th=256,gr=(n+th-1)/th;vdiff<<<gr,th>>>(r0,s.d_b,s.d_sy,n);if(launch_ok("GP2 final residual"))return 172;if(norm_host(r0,n,rnorm,"GP2 final norm"))return 172;resid=rnorm/bnorm;}
 auto wall1=WallClock::now();double wall=std::chrono::duration<double,std::milli>(wall1-wall0).count(),mvs=s.matvec_total_ms-mv0;unsigned long long mnc=s.matvec_count-mc0;
 std::printf("CUDA GPBiCGStab(2) timing: wall=%.6f ms MATVEC_GPU=%.6f ms (%llu calls) outside=%.6f ms\n",wall,mvs,mnc,wall-mvs);std::printf("CUDA GPBiCGStab(2): chunked cublasZdotc FP64 + CPU complex<double> 2x2/3x3 minimization; vector arithmetic CUDA kernels\n");std::fflush(stdout);
 s.solution_valid=true;if(download_compact_to_full_host(s.d_sx,x,"solver solution compact D2H")){return 173;}*itout=(it<=maxit)?it:maxit;*tole=(float)resid;*ncout=nc;return 0;
}

extern "C" DDSCAT_CUDA_API int ddscat_cuda_solve_gpbicgstab4_f32(const void*b,void*x,const void*adia,const void*aoff,
    int ndim,int maxit,float tol,int*itout,float*tole,int*ncout){
 last_error.clear();
 if(!s.ready||!b||!x||!adia||!aoff||!itout||!tole||!ncout||ndim!=3*s.nat||maxit<=0||tol<=0.f){set_error("GPBiCGStab4","invalid argument");return 180;}
 const int n=s.solver_n;const size_t bytes=(size_t)n*sizeof(cufftComplex),full_bytes=(size_t)ndim*sizeof(cufftComplex);int nc=0,rc=0;
 if(init_solver_boundary(b,x,adia,aoff,ndim,"GPBiCGStab4 boundary H2D"))return 181;
 // 15 resident vectors including x/r0/p0/work, matching the memory-reduced IFDDA port.
 cufftComplex*r0=s.d_wrk+0*(size_t)n,*p0=s.d_wrk+1*(size_t)n,*rt=s.d_wrk+2*(size_t)n,*z=s.d_wrk+3*(size_t)n,*yv=s.d_wrk+4*(size_t)n,*u=s.d_wrk+5*(size_t)n;
 cufftComplex*hs[3]={s.d_wrk+6*(size_t)n,s.d_wrk+7*(size_t)n,s.d_wrk+8*(size_t)n};
 cufftComplex*hq[4]={s.d_wrk+9*(size_t)n,s.d_wrk+10*(size_t)n,s.d_wrk+11*(size_t)n,s.d_wrk+12*(size_t)n};
 cufftComplex*work=s.d_sy; // existing persistent MATVEC staging vector, free at outer boundaries
 if(initial_true_residual(r0,n,nc,"GPBiCGStab4-initial-Ax"))return 182;
 double bnorm=0,rnorm=0;if(norm_host(s.d_b,n,bnorm,"GP4 norm b")||norm_host(r0,n,rnorm,"GP4 norm r"))return 182;if(bnorm==0.0){set_error("GP4","zero RHS norm");return 183;}
 if(op_copy(rt,r0,n,"GP4 rt")||op_copy(p0,r0,n,"GP4 p0")||op_zero(z,n,"GP4 z"))return 183;
 bool fresh=true;double resid=rnorm/bnorm;hcomplex rho,sigma,alpha,beta,temp,zeta[4],eta;hcomplex mat[5][5],rhs[5];int it=0;
 auto restore=[&]()->int{
  if(op_copy(hq[0],hs[0],n,"GP4 restore tmp0")||op_copy(hs[0],hq[3],n,"GP4 restore s0")||op_copy(hq[3],hq[0],n,"GP4 restore q3"))return 1;
  if(op_copy(hq[0],hs[1],n,"GP4 restore tmp1")||op_copy(hs[1],hq[2],n,"GP4 restore s1")||op_copy(hq[2],hq[0],n,"GP4 restore q2"))return 1;
  if(op_copy(hq[0],hs[2],n,"GP4 restore tmp2")||op_copy(hs[2],hq[1],n,"GP4 restore s2")||op_copy(hq[1],hq[0],n,"GP4 restore q1")||op_copy(hq[0],work,n,"GP4 restore q0"))return 1;return 0;};
 auto wall0=WallClock::now();double mv0=s.matvec_total_ms;unsigned long long mc0=s.matvec_count;
 for(it=1;it<=maxit && resid>tol;it++){
  if(fresh){
   cufftComplex*rr[5]={r0,hq[3],hq[2],hq[1],hq[0]};cufftComplex*pp[5]={p0,work,hs[2],hs[1],hs[0]};
   if(dotc_host(rt,rr[0],n,rho,"GP4 initial rho"))return 184;
   for(int j=1;j<=4;j++){
    if((rc=run_mv(pp[j-1],pp[j],"GPBiCGStab4-p",nc)))return rc;if(dotc_host(rt,pp[j],n,sigma,"GP4 initial sigma"))return 184;if(std::abs(sigma)<1.e-30){set_error("GP4","initial sigma zero");return 184;}alpha=rho/sigma;
    if(op_axpy(s.d_sx,pp[0],alpha,n,"GP4 init x"))return 185;for(int i=0;i<j;i++)if(op_axpy(rr[i],pp[i+1],-alpha,n,"GP4 init r update"))return 185;
    if((rc=run_mv(rr[j-1],rr[j],"GPBiCGStab4-r",nc)))return rc;if(dotc_host(rt,rr[j],n,rho,"GP4 init rho j"))return 185;beta=rho/sigma;for(int i=0;i<=j;i++)if(op_affine(pp[i],rr[i],-beta,n,"GP4 init p update"))return 186;
   }
   for(auto&row:mat)for(auto&v:row)v=0.0;for(auto&v:rhs)v=0.0;
   for(int i=0;i<4;i++){for(int j=0;j<4;j++)if(dotc_host(rr[i+1],rr[j+1],n,mat[i][j],"GP4 init gram"))return 187;if(dotc_host(rr[i+1],rr[0],n,rhs[i],"GP4 init rhs"))return 187;}
   if(!solve_small(&mat[0][0],rhs,4,5)){set_error("GP4","initial 4x4 minimization breakdown");return 187;}for(int i=0;i<4;i++)zeta[i]=rhs[i];
   if(op_scale(z,rr[0],zeta[0],n,"GP4 init z"))return 188;for(int i=1;i<4;i++)if(op_axpy(z,rr[i],zeta[i],n,"GP4 init z add"))return 188;if(op_axpy(s.d_sx,z,hcomplex(1,0),n,"GP4 init x z"))return 188;
   if(op_scale(yv,rr[1],zeta[0],n,"GP4 init y")||op_scale(u,pp[1],zeta[0],n,"GP4 init u"))return 188;for(int i=1;i<4;i++)if(op_axpy(yv,rr[i+1],zeta[i],n,"GP4 init yadd")||op_axpy(u,pp[i+1],zeta[i],n,"GP4 init uadd"))return 188;
   if(op_axpy(r0,yv,hcomplex(-1,0),n,"GP4 init r-y")||op_axpy(p0,u,hcomplex(-1,0),n,"GP4 init p-u")||restore())return 189;
   fresh=false;
  }else{
   if(dotc_host(rt,r0,n,rho,"GP4 rho"))return 190;
   // j=1
   if((rc=run_mv(p0,work,"GPBiCGStab4-p1",nc)))return rc;if(dotc_host(rt,work,n,sigma,"GP4 sigma1"))return 190;if(std::abs(sigma)<1.e-30){set_error("GP4","sigma1 zero");return 190;}alpha=rho/sigma;
   if(op_axpy(s.d_sx,p0,alpha,n,"GP4 x1")||op_axpy(z,u,-alpha,n,"GP4 z1")||op_axpy2(yv,hq[0],work,-alpha,alpha,n,"GP4 y1")||op_axpy(r0,work,-alpha,n,"GP4 r01"))return 191;for(int i=0;i<3;i++)if(op_axpy(hs[i],hq[i+1],-alpha,n,"GP4 s update1"))return 191;
   if((rc=run_mv(r0,hq[3],"GPBiCGStab4-r1",nc)))return rc;if(dotc_host(rt,hq[3],n,rho,"GP4 rho1"))return 191;beta=rho/sigma;if(op_affine(p0,r0,-beta,n,"GP4 p01")||op_affine(work,hq[3],-beta,n,"GP4 p11"))return 192;for(int i=0;i<3;i++)if(op_affine(hq[i],hs[i],-beta,n,"GP4 q update1"))return 192;if(op_affine(u,yv,-beta,n,"GP4 u1"))return 192;
   // j=2
   if((rc=run_mv(work,hs[2],"GPBiCGStab4-p2",nc)))return rc;if(dotc_host(rt,hs[2],n,sigma,"GP4 sigma2"))return 193;if(std::abs(sigma)<1.e-30){set_error("GP4","sigma2 zero");return 193;}alpha=rho/sigma;
   if(op_axpy(s.d_sx,p0,alpha,n,"GP4 x2")||op_axpy(z,u,-alpha,n,"GP4 z2")||op_axpy2(yv,hq[0],work,-alpha,alpha,n,"GP4 y2")||op_axpy(r0,work,-alpha,n,"GP4 r02")||op_axpy(hq[3],hs[2],-alpha,n,"GP4 r1upd"))return 194;for(int i=0;i<2;i++)if(op_axpy(hs[i],hq[i+1],-alpha,n,"GP4 s update2"))return 194;
   if((rc=run_mv(hq[3],hq[2],"GPBiCGStab4-r2",nc)))return rc;if(dotc_host(rt,hq[2],n,rho,"GP4 rho2"))return 194;beta=rho/sigma;if(op_affine(p0,r0,-beta,n,"GP4 p02")||op_affine(work,hq[3],-beta,n,"GP4 p12")||op_affine(hs[2],hq[2],-beta,n,"GP4 p22"))return 195;for(int i=0;i<2;i++)if(op_affine(hq[i],hs[i],-beta,n,"GP4 q update2"))return 195;if(op_affine(u,yv,-beta,n,"GP4 u2"))return 195;
   // j=3
   if((rc=run_mv(hs[2],hs[1],"GPBiCGStab4-p3",nc)))return rc;if(dotc_host(rt,hs[1],n,sigma,"GP4 sigma3"))return 196;if(std::abs(sigma)<1.e-30){set_error("GP4","sigma3 zero");return 196;}alpha=rho/sigma;
   if(op_axpy(s.d_sx,p0,alpha,n,"GP4 x3")||op_axpy(z,u,-alpha,n,"GP4 z3")||op_axpy2(yv,hq[0],work,-alpha,alpha,n,"GP4 y3")||op_axpy(r0,work,-alpha,n,"GP4 r03")||op_axpy(hq[3],hs[2],-alpha,n,"GP4 r1u3")||op_axpy(hq[2],hs[1],-alpha,n,"GP4 r2u3")||op_axpy(hs[0],hq[1],-alpha,n,"GP4 s0u3"))return 197;
   if((rc=run_mv(hq[2],hq[1],"GPBiCGStab4-r3",nc)))return rc;if(dotc_host(rt,hq[1],n,rho,"GP4 rho3"))return 197;beta=rho/sigma;if(op_affine(p0,r0,-beta,n,"GP4 p03")||op_affine(work,hq[3],-beta,n,"GP4 p13")||op_affine(hs[2],hq[2],-beta,n,"GP4 p23")||op_affine(hs[1],hq[1],-beta,n,"GP4 p33")||op_affine(hq[0],hs[0],-beta,n,"GP4 q03")||op_affine(u,yv,-beta,n,"GP4 u3"))return 198;
   // j=4
   if((rc=run_mv(hs[1],hs[0],"GPBiCGStab4-p4",nc)))return rc;if(dotc_host(rt,hs[0],n,sigma,"GP4 sigma4"))return 199;if(std::abs(sigma)<1.e-30){set_error("GP4","sigma4 zero");return 199;}alpha=rho/sigma;
   if(op_axpy(s.d_sx,p0,alpha,n,"GP4 x4")||op_axpy(z,u,-alpha,n,"GP4 z4")||op_axpy2(yv,hq[0],work,-alpha,alpha,n,"GP4 y4")||op_axpy(r0,work,-alpha,n,"GP4 r04")||op_axpy(hq[3],hs[2],-alpha,n,"GP4 r1u4")||op_axpy(hq[2],hs[1],-alpha,n,"GP4 r2u4")||op_axpy(hq[1],hs[0],-alpha,n,"GP4 r3u4"))return 200;
   if((rc=run_mv(hq[1],hq[0],"GPBiCGStab4-r4",nc)))return rc;if(dotc_host(rt,hq[0],n,rho,"GP4 rho4"))return 200;beta=rho/sigma;if(op_affine(p0,r0,-beta,n,"GP4 p04")||op_affine(work,hq[3],-beta,n,"GP4 p14")||op_affine(hs[2],hq[2],-beta,n,"GP4 p24")||op_affine(hs[1],hq[1],-beta,n,"GP4 p34")||op_affine(hs[0],hq[0],-beta,n,"GP4 p44")||op_affine(u,yv,-beta,n,"GP4 u4"))return 201;
   // Boundary polynomial identities now: rr={r0,hq3,hq2,hq1,hq0}, pp={p0,work,hs2,hs1,hs0}.
   cufftComplex*rr[5]={r0,hq[3],hq[2],hq[1],hq[0]};cufftComplex*pp[5]={p0,work,hs[2],hs[1],hs[0]};for(auto&row:mat)for(auto&v:row)v=0.0;for(auto&v:rhs)v=0.0;
   for(int i=0;i<4;i++){for(int j=0;j<4;j++)if(dotc_host(rr[i+1],rr[j+1],n,mat[i][j],"GP4 Gram rr"))return 202;if(dotc_host(rr[i+1],yv,n,mat[i][4],"GP4 Gram y"))return 202;mat[4][i]=std::conj(mat[i][4]);if(dotc_host(rr[i+1],rr[0],n,rhs[i],"GP4 rhs"))return 202;}
   if(dotc_host(yv,yv,n,mat[4][4],"GP4 yy")||dotc_host(yv,rr[0],n,rhs[4],"GP4 rhs y"))return 202;if(!solve_small(&mat[0][0],rhs,5,5)){set_error("GP4","5x5 minimization breakdown");return 202;}for(int i=0;i<4;i++)zeta[i]=rhs[i];eta=rhs[4];
   if(op_scale_self(z,eta,n,"GP4 z eta"))return 203;for(int i=0;i<4;i++)if(op_axpy(z,rr[i],zeta[i],n,"GP4 z add"))return 203;if(op_axpy(s.d_sx,z,hcomplex(1,0),n,"GP4 x z"))return 203;
   if(op_scale_self(yv,eta,n,"GP4 y eta")||op_scale_self(u,eta,n,"GP4 u eta"))return 203;for(int i=0;i<4;i++)if(op_axpy(yv,rr[i+1],zeta[i],n,"GP4 y add")||op_axpy(u,pp[i+1],zeta[i],n,"GP4 u add"))return 203;
   if(op_axpy(r0,yv,hcomplex(-1,0),n,"GP4 r-y")||op_axpy(p0,u,hcomplex(-1,0),n,"GP4 p-u")||restore())return 204;
  }
  if(norm_host(r0,n,rnorm,"GP4 residual norm"))return 205;resid=rnorm/bnorm;std::printf(" GPBiCGStab(4) iteration=%6d residual=%14.6E\n",it,resid);std::fflush(stdout);
  if((it%RELIABLE_RESID_PERIOD)==0||resid<=tol){
   ReliableResidualResult rr;if(reliable_residual_check("GPBiCGStab(4)","GPBiCGStab4-reliable",s.d_b,s.d_sx,r0,n,bnorm,resid,tol,nc,rr))return 206;
   if(rr.converged){resid=rr.true_res;break;}
   if(rr.restart){if(materialize_true_residual(s.d_b,n,"GP4 materialize true residual"))return 206;if(op_copy(r0,s.d_sy,n,"GP4 restart r")||op_copy(rt,r0,n,"GP4 restart rt")||op_copy(p0,r0,n,"GP4 restart p")||op_zero(z,n,"GP4 restart z"))return 206;fresh=true;resid=rr.true_res;}
   else{ // reliable check used s.d_sy, which is GP4's persistent `work`; hq[0] is its boundary duplicate.
    if(op_copy(work,hq[0],n,"GP4 restore work after reliable check"))return 206;
   }
  }
 }
 // Final true residual check.
 if((rc=run_mv(s.d_sx,s.d_sy,"GPBiCGStab4-final-true",nc)))return rc;int th=256,gr=(n+th-1)/th;vdiff<<<gr,th>>>(r0,s.d_b,s.d_sy,n);if(launch_ok("GP4 final residual"))return 206;if(norm_host(r0,n,rnorm,"GP4 final norm"))return 206;resid=rnorm/bnorm;
 auto wall1=WallClock::now();double wall=std::chrono::duration<double,std::milli>(wall1-wall0).count(),mvs=s.matvec_total_ms-mv0;unsigned long long mnc=s.matvec_count-mc0;
 std::printf("CUDA GPBiCGStab(4) timing: wall=%.6f ms MATVEC_GPU=%.6f ms (%llu calls) outside=%.6f ms\n",wall,mvs,mnc,wall-mvs);std::printf("CUDA GPBiCGStab(4): memory-reduced 15-vector recurrence; chunked cublasZdotc FP64 + CPU complex<double> 4x4/5x5 minimization\n");std::fflush(stdout);
 s.solution_valid=true;if(download_compact_to_full_host(s.d_sx,x,"solver solution compact D2H")){return 207;}*itout=(it<=maxit)?it:maxit;*tole=(float)resid;*ncout=nc;return 0;
}
