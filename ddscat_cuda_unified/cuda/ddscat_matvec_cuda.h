#pragma once
#include <stdint.h>
#ifdef _WIN32
# define DDSCAT_CUDA_API __declspec(dllexport)
#else
# define DDSCAT_CUDA_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
DDSCAT_CUDA_API int ddscat_cuda_prepare_green_f32(int nx,int ny,int nz,int ipbc,
    int nat,int nat0,int idipint,float gamma,float pyd,float pzd,
    float akx,float aky,float akz,float dx1,float dx2,float dx3,const int16_t *iocc);
DDSCAT_CUDA_API int ddscat_cuda_prepare_f32(int nx,int ny,int nz,int ipbc,
    int nat,int nat0,const void *green,const int16_t *iocc);
DDSCAT_CUDA_API int ddscat_cuda_apply_f32(int cwhat,const void *x,void *y,
    const void *adia,const void *aoff);
DDSCAT_CUDA_API int ddscat_cuda_solve_gpbicg_f32(const void *b,void *x,
    const void *adia,const void *aoff,int ndim,int maxit,float tol,
    int *itno,float *tole,int *ncompte);
DDSCAT_CUDA_API int ddscat_cuda_solve_qmrccg_f32(const void *b,void *x,
    const void *adia,const void *aoff,int ndim,int maxit,float tol,
    int *itno,float *tole,int *ncompte);
DDSCAT_CUDA_API int ddscat_cuda_solve_pbcgst_f32(const void *b,void *x,
    const void *adia,const void *aoff,int ndim,int maxit,float tol,
    int *itno,float *tole,int *ncompte);
DDSCAT_CUDA_API int ddscat_cuda_solve_pbcgs2_f32(const void *b,void *x,
    const void *adia,const void *aoff,int ndim,int maxit,float tol,
    int *itno,float *tole,int *ncompte);
DDSCAT_CUDA_API int ddscat_cuda_solve_petrkp_f32(const void *b,void *x,
    const void *adia,const void *aoff,int ndim,int maxit,float tol,
    int *itno,float *tole,int *ncompte);
DDSCAT_CUDA_API int ddscat_cuda_solve_bicgstab4_f32(const void *b,void *x,
    const void *adia,const void *aoff,int ndim,int maxit,float tol,
    int *itno,float *tole,int *ncompte);
DDSCAT_CUDA_API int ddscat_cuda_solve_gpbicgstab2_f32(const void *b,void *x,
    const void *adia,const void *aoff,int ndim,int maxit,float tol,
    int *itno,float *tole,int *ncompte);
DDSCAT_CUDA_API int ddscat_cuda_solve_gpbicgstab4_f32(const void *b,void *x,
    const void *adia,const void *aoff,int ndim,int maxit,float tol,
    int *itno,float *tole,int *ncompte);
DDSCAT_CUDA_API int ddscat_cuda_evalq_f32(int use_resident,
    const void *e_compact,const void *p_compact,const void *adia_compact,const void *aoff_compact,
    float akx,float aky,float akz,float e02,double *cabs,double *cext,double *cpha);
DDSCAT_CUDA_API int ddscat_cuda_scat_f32(int jpbc,int ndir,
    const float *ak,const float *aks,const float *dx,const float *em1,const float *em2,
    float e02,float etasca,const void *e01,const float *x0,
    double *cbksca,double *csca,double *cscag,double *cscag2,int *navg,
    void *f1,void *f2);
DDSCAT_CUDA_API int ddscat_cuda_release(void);
DDSCAT_CUDA_API const char *ddscat_cuda_last_error(void);
#ifdef __cplusplus
}
#endif
