#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef int (*prepare_fn)(int,int,int,int,int,int,const void*,const int16_t*);
typedef int (*prepare_green_fn)(int,int,int,int,int,int,int,float,float,float,float,float,float,float,float,float,const int16_t*);
typedef int (*apply_fn)(int,const void*,void*,const void*,const void*);
typedef int (*solver_fn)(const void*,void*,const void*,const void*,int,int,float,int*,float*,int*);
typedef int (*evalq_fn)(int,const void*,const void*,const void*,const void*,float,float,float,float,double*,double*,double*);
typedef int (*scat_fn)(int,int,const float*,const float*,const float*,const float*,const float*,float,float,const void*,const float*,double*,double*,double*,double*,int*,void*,void*);
typedef int (*release_fn)(void);
typedef const char* (*error_fn)(void);
static prepare_fn p_prepare=NULL; static prepare_green_fn p_prepare_green=NULL; static apply_fn p_apply=NULL;
static solver_fn p_gpbicg=NULL, p_qmrccg=NULL, p_pbcgst=NULL, p_pbcgs2=NULL, p_petrkp=NULL;
static solver_fn p_bicgst4=NULL, p_gpbgs2=NULL, p_gpbgs4=NULL;
static evalq_fn p_evalq=NULL; static scat_fn p_scat=NULL;
static release_fn p_release=NULL; static error_fn p_error=NULL;
static char loader_error[1024]="";
#ifdef _WIN32
static HMODULE lib=NULL;
#else
static void *lib=NULL;
#endif

static void print_loaded_path(const char *fallback){
#ifdef _WIN32
 char full[MAX_PATH]; DWORD n=GetModuleFileNameA(lib,full,(DWORD)sizeof(full));
 if(n>0 && n<sizeof(full)) fprintf(stdout,"DDSCAT CUDA DLL loaded: %s\n",full);
 else fprintf(stdout,"DDSCAT CUDA DLL loaded: %s\n",fallback);
#else
 fprintf(stdout,"DDSCAT CUDA library loaded: %s\n",fallback);
#endif
 fflush(stdout);
}
static int load_cuda(void){
 if(lib) return 0;
 const char *path=getenv("DDSCAT_CUDA_DLL");
 if(!path||!*path){
#ifdef _WIN32
  char exe_path[MAX_PATH];DWORD en=GetModuleFileNameA(NULL,exe_path,(DWORD)sizeof(exe_path));
  if(en>0&&en<sizeof(exe_path)&&(strstr(exe_path,"ddscat_cuda_slice")!=NULL)) path="ddscat_matvec_cuda_slice.dll";
  else path="ddscat_matvec_cuda.dll";
#else
  path="libddscat_matvec_cuda.so";
#endif
 }
#ifdef _WIN32
 lib=LoadLibraryA(path);
 if(!lib){snprintf(loader_error,sizeof(loader_error),"LoadLibrary failed for %s (Windows error %lu)",path,(unsigned long)GetLastError());return 1;}
#define SYM(n,t,v) do{v=(t)GetProcAddress(lib,n);if(!v){snprintf(loader_error,sizeof(loader_error),"Missing symbol %s (stale CUDA DLL?)",n);return 2;}}while(0)
#else
 lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
 if(!lib){snprintf(loader_error,sizeof(loader_error),"dlopen failed for %s: %s",path,dlerror());return 1;}
#define SYM(n,t,v) do{v=(t)dlsym(lib,n);if(!v){snprintf(loader_error,sizeof(loader_error),"Missing symbol %s",n);return 2;}}while(0)
#endif
 SYM("ddscat_cuda_prepare_f32",prepare_fn,p_prepare);
#ifdef _WIN32
 p_prepare_green=(prepare_green_fn)GetProcAddress(lib,"ddscat_cuda_prepare_green_f32");
#else
 p_prepare_green=(prepare_green_fn)dlsym(lib,"ddscat_cuda_prepare_green_f32");
#endif
 SYM("ddscat_cuda_apply_f32",apply_fn,p_apply);
 SYM("ddscat_cuda_solve_gpbicg_f32",solver_fn,p_gpbicg);
 SYM("ddscat_cuda_solve_qmrccg_f32",solver_fn,p_qmrccg);
 SYM("ddscat_cuda_solve_pbcgst_f32",solver_fn,p_pbcgst);
 SYM("ddscat_cuda_solve_pbcgs2_f32",solver_fn,p_pbcgs2);
 SYM("ddscat_cuda_solve_petrkp_f32",solver_fn,p_petrkp);
 SYM("ddscat_cuda_solve_bicgstab4_f32",solver_fn,p_bicgst4);
 SYM("ddscat_cuda_solve_gpbicgstab2_f32",solver_fn,p_gpbgs2);
 SYM("ddscat_cuda_solve_gpbicgstab4_f32",solver_fn,p_gpbgs4);
 SYM("ddscat_cuda_evalq_f32",evalq_fn,p_evalq);
 SYM("ddscat_cuda_scat_f32",scat_fn,p_scat);
 SYM("ddscat_cuda_release",release_fn,p_release);
 SYM("ddscat_cuda_last_error",error_fn,p_error);
#undef SYM
 print_loaded_path(path); return 0;
}
static int report_result(int r){
 if(r&&p_error){snprintf(loader_error,sizeof(loader_error),"%s",p_error());fprintf(stderr,"DDSCAT CUDA: %s\n",loader_error);}
 return r;
}
int ddscat_cuda_loader_prepare_green_f32(int nx,int ny,int nz,int ipbc,int nat,int nat0,int idipint,float gamma,float pyd,float pzd,float akx,float aky,float akz,float dx1,float dx2,float dx3,const int16_t*iocc){int r=load_cuda();if(r)return 1000+r;if(!p_prepare_green)return -777;return report_result(p_prepare_green(nx,ny,nz,ipbc,nat,nat0,idipint,gamma,pyd,pzd,akx,aky,akz,dx1,dx2,dx3,iocc));}
int ddscat_cuda_loader_prepare_f32(int nx,int ny,int nz,int ipbc,int nat,int nat0,const void*g,const int16_t*iocc){int r=load_cuda();if(r)return 1000+r;return report_result(p_prepare(nx,ny,nz,ipbc,nat,nat0,g,iocc));}
int ddscat_cuda_loader_apply_f32(int cwhat,const void*x,void*y,const void*adia,const void*aoff){int r=load_cuda();if(r)return 1000+r;return report_result(p_apply(cwhat,x,y,adia,aoff));}
int ddscat_cuda_loader_solve_gpbicg_f32(const void*b,void*x,const void*adia,const void*aoff,int n,int m,float t,int*it,float*to,int*nc){int r=load_cuda();if(r)return 1000+r;return report_result(p_gpbicg(b,x,adia,aoff,n,m,t,it,to,nc));}
int ddscat_cuda_loader_solve_qmrccg_f32(const void*b,void*x,const void*adia,const void*aoff,int n,int m,float t,int*it,float*to,int*nc){int r=load_cuda();if(r)return 1000+r;return report_result(p_qmrccg(b,x,adia,aoff,n,m,t,it,to,nc));}
int ddscat_cuda_loader_solve_pbcgst_f32(const void*b,void*x,const void*adia,const void*aoff,int n,int m,float t,int*it,float*to,int*nc){int r=load_cuda();if(r)return 1000+r;return report_result(p_pbcgst(b,x,adia,aoff,n,m,t,it,to,nc));}
int ddscat_cuda_loader_solve_pbcgs2_f32(const void*b,void*x,const void*adia,const void*aoff,int n,int m,float t,int*it,float*to,int*nc){int r=load_cuda();if(r)return 1000+r;return report_result(p_pbcgs2(b,x,adia,aoff,n,m,t,it,to,nc));}
int ddscat_cuda_loader_solve_petrkp_f32(const void*b,void*x,const void*adia,const void*aoff,int n,int m,float t,int*it,float*to,int*nc){int r=load_cuda();if(r)return 1000+r;return report_result(p_petrkp(b,x,adia,aoff,n,m,t,it,to,nc));}
int ddscat_cuda_loader_solve_bicgstab4_f32(const void*b,void*x,const void*adia,const void*aoff,int n,int m,float t,int*it,float*to,int*nc){int r=load_cuda();if(r)return 1000+r;return report_result(p_bicgst4(b,x,adia,aoff,n,m,t,it,to,nc));}
int ddscat_cuda_loader_solve_gpbicgstab2_f32(const void*b,void*x,const void*adia,const void*aoff,int n,int m,float t,int*it,float*to,int*nc){int r=load_cuda();if(r)return 1000+r;return report_result(p_gpbgs2(b,x,adia,aoff,n,m,t,it,to,nc));}
int ddscat_cuda_loader_solve_gpbicgstab4_f32(const void*b,void*x,const void*adia,const void*aoff,int n,int m,float t,int*it,float*to,int*nc){int r=load_cuda();if(r)return 1000+r;return report_result(p_gpbgs4(b,x,adia,aoff,n,m,t,it,to,nc));}
int ddscat_cuda_loader_evalq_f32(int resident,const void*e,const void*p,const void*ad,const void*ao,float akx,float aky,float akz,float e02,double*cabs,double*cext,double*cpha){int r=load_cuda();if(r)return 1000+r;return report_result(p_evalq(resident,e,p,ad,ao,akx,aky,akz,e02,cabs,cext,cpha));}
int ddscat_cuda_loader_scat_f32(int jpbc,int ndir,const float*ak,const float*aks,const float*dx,const float*em1,const float*em2,float e02,float etasca,const void*e01,const float*x0,double*cbksca,double*csca,double*cscag,double*cscag2,int*navg,void*f1,void*f2){int r=load_cuda();if(r)return 1000+r;return report_result(p_scat(jpbc,ndir,ak,aks,dx,em1,em2,e02,etasca,e01,x0,cbksca,csca,cscag,cscag2,navg,f1,f2));}
int ddscat_cuda_loader_release(void){if(!lib)return 0;return p_release?p_release():0;}
const char *ddscat_cuda_loader_last_error(void){return loader_error;}
