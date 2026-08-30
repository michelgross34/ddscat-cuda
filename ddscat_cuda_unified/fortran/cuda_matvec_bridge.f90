module cuda_matvec_bridge
  use iso_c_binding, only: c_int,c_short,c_float,c_double,c_float_complex
  use ddprecision, only: WP
  use ddcommon_0, only: IDIPINT,WOLD,AK2OLD,AK3OLD
  implicit none
  private
  public :: CUDA_CPROD,CUDA_GPBICG_SOLVE,CUDA_QMRCCG_SOLVE,CUDA_PBCGST_SOLVE,CUDA_PBCGS2_SOLVE,CUDA_PETRKP_SOLVE
  public :: CUDA_BICGST4_SOLVE,CUDA_GPBGS2_SOLVE,CUDA_GPBGS4_SOLVE,CUDA_MATVEC_RELEASE
  public :: CUDA_EVALQ_GPU,CUDA_SCAT_GPU
  logical,save :: uploaded=.false.
  integer,save :: lnx=-1,lny=-1,lnz=-1,lipbc=-1,lid=-999
  real(WP),save :: lakd=-1._WP,lak2=0._WP,lak3=0._WP,lgamma=0._WP,lpyd=-1._WP,lpzd=-1._WP,ldx(3)=0._WP

  interface
    integer(c_int) function c_prepare(nx,ny,nz,ipbc,nat,nat0,g,occ) bind(C,name='ddscat_cuda_loader_prepare_f32')
      import :: c_int,c_short,c_float_complex
      integer(c_int),value :: nx,ny,nz,ipbc,nat,nat0
      complex(c_float_complex) :: g(*)
      integer(c_short) :: occ(*)
    end function
    integer(c_int) function c_apply(cwhat,x,y,ad,ao) bind(C,name='ddscat_cuda_loader_apply_f32')
      import :: c_int,c_float_complex
      integer(c_int),value :: cwhat
      complex(c_float_complex) :: x(*),y(*),ad(*),ao(*)
    end function
    integer(c_int) function c_gpb(b,x,ad,ao,n,maxit,tol,it,tole,nc) bind(C,name='ddscat_cuda_loader_solve_gpbicg_f32')
      import :: c_int,c_float,c_float_complex
      complex(c_float_complex) :: b(*),x(*),ad(*),ao(*)
      integer(c_int),value :: n,maxit
      real(c_float),value :: tol
      integer(c_int) :: it,nc
      real(c_float) :: tole
    end function
    integer(c_int) function c_qmr(b,x,ad,ao,n,maxit,tol,it,tole,nc) bind(C,name='ddscat_cuda_loader_solve_qmrccg_f32')
      import :: c_int,c_float,c_float_complex
      complex(c_float_complex) :: b(*),x(*),ad(*),ao(*)
      integer(c_int),value :: n,maxit
      real(c_float),value :: tol
      integer(c_int) :: it,nc
      real(c_float) :: tole
    end function
    integer(c_int) function c_pbcgst(b,x,ad,ao,n,maxit,tol,it,tole,nc) bind(C,name='ddscat_cuda_loader_solve_pbcgst_f32')
      import :: c_int,c_float,c_float_complex
      complex(c_float_complex) :: b(*),x(*),ad(*),ao(*)
      integer(c_int),value :: n,maxit
      real(c_float),value :: tol
      integer(c_int) :: it,nc
      real(c_float) :: tole
    end function
    integer(c_int) function c_pbcgs2(b,x,ad,ao,n,maxit,tol,it,tole,nc) bind(C,name='ddscat_cuda_loader_solve_pbcgs2_f32')
      import :: c_int,c_float,c_float_complex
      complex(c_float_complex) :: b(*),x(*),ad(*),ao(*)
      integer(c_int),value :: n,maxit
      real(c_float),value :: tol
      integer(c_int) :: it,nc
      real(c_float) :: tole
    end function
    integer(c_int) function c_petrkp(b,x,ad,ao,n,maxit,tol,it,tole,nc) bind(C,name='ddscat_cuda_loader_solve_petrkp_f32')
      import :: c_int,c_float,c_float_complex
      complex(c_float_complex) :: b(*),x(*),ad(*),ao(*)
      integer(c_int),value :: n,maxit
      real(c_float),value :: tol
      integer(c_int) :: it,nc
      real(c_float) :: tole
    end function
    integer(c_int) function c_bicgst4(b,x,ad,ao,n,maxit,tol,it,tole,nc) bind(C,name='ddscat_cuda_loader_solve_bicgstab4_f32')
      import :: c_int,c_float,c_float_complex
      complex(c_float_complex) :: b(*),x(*),ad(*),ao(*)
      integer(c_int),value :: n,maxit
      real(c_float),value :: tol
      integer(c_int) :: it,nc
      real(c_float) :: tole
    end function
    integer(c_int) function c_gpbgs2(b,x,ad,ao,n,maxit,tol,it,tole,nc) bind(C,name='ddscat_cuda_loader_solve_gpbicgstab2_f32')
      import :: c_int,c_float,c_float_complex
      complex(c_float_complex) :: b(*),x(*),ad(*),ao(*)
      integer(c_int),value :: n,maxit
      real(c_float),value :: tol
      integer(c_int) :: it,nc
      real(c_float) :: tole
    end function
    integer(c_int) function c_gpbgs4(b,x,ad,ao,n,maxit,tol,it,tole,nc) bind(C,name='ddscat_cuda_loader_solve_gpbicgstab4_f32')
      import :: c_int,c_float,c_float_complex
      complex(c_float_complex) :: b(*),x(*),ad(*),ao(*)
      integer(c_int),value :: n,maxit
      real(c_float),value :: tol
      integer(c_int) :: it,nc
      real(c_float) :: tole
    end function
    integer(c_int) function c_evalq(resident,e,p,ad,ao,akx,aky,akz,e02,cabs,cext,cpha) bind(C,name='ddscat_cuda_loader_evalq_f32')
      import :: c_int,c_float,c_double,c_float_complex
      integer(c_int),value :: resident
      complex(c_float_complex) :: e(*),p(*),ad(*),ao(*)
      real(c_float),value :: akx,aky,akz,e02
      real(c_double) :: cabs,cext,cpha
    end function
    integer(c_int) function c_scat(jpbc,ndir,ak,aks,dx,em1,em2,e02,etasca,e01,x0,cbksca,csca,cscag,cscag2,navg,f1,f2) bind(C,name='ddscat_cuda_loader_scat_f32')
      import :: c_int,c_float,c_double,c_float_complex
      integer(c_int),value :: jpbc,ndir
      real(c_float) :: ak(*),aks(*),dx(*),em1(*),em2(*),x0(*)
      real(c_float),value :: e02,etasca
      complex(c_float_complex) :: e01(*),f1(*),f2(*)
      real(c_double) :: cbksca,csca,cscag(*),cscag2
      integer(c_int) :: navg
    end function
    integer(c_int) function c_release() bind(C,name='ddscat_cuda_loader_release')
      import :: c_int
    end function
    integer(c_int) function c_prepare_green(nx,ny,nz,ipbc,nat,nat0,idipint,gamma,pyd,pzd,akx,aky,akz,dx1,dx2,dx3,occ) bind(C,name='ddscat_cuda_loader_prepare_green_f32')
      import :: c_int,c_short,c_float
      integer(c_int),value :: nx,ny,nz,ipbc,nat,nat0,idipint
      real(c_float),value :: gamma,pyd,pzd,akx,aky,akz,dx1,dx2,dx3
      integer(c_short) :: occ(*)
    end function
  end interface
contains
  logical function green_changed(AK,GAMMA,DX,NX,NY,NZ,IPBC,AKD)
    use ddcommon_6, only: PYD,PZD
    real(WP),intent(in)::AK(3),GAMMA,DX(3),AKD
    integer,intent(in)::NX,NY,NZ,IPBC
    real(WP)::t
    green_changed=.not.uploaded
    if(green_changed)return
    if(NX/=lnx.or.NY/=lny.or.NZ/=lnz.or.IPBC/=lipbc.or.IDIPINT/=lid)then;green_changed=.true.;return;endif
    if(GAMMA/=lgamma.or.PYD/=lpyd.or.PZD/=lpzd.or.any(DX/=ldx))then;green_changed=.true.;return;endif
    t=1.e-6_WP*max(AKD,1.e-30_WP)
    if(abs(lakd-AKD)>=t)then;green_changed=.true.;return;endif
    if(IPBC==1)then
      if(PYD/=0._WP.and.abs(lak2-AK(2))>=t)then;green_changed=.true.;return;endif
      if(PZD/=0._WP.and.abs(lak3-AK(3))>=t)then;green_changed=.true.;return;endif
    endif
  end function

  subroutine remember(AK,GAMMA,DX,NX,NY,NZ,IPBC,AKD)
    use ddcommon_6, only: PYD,PZD
    real(WP),intent(in)::AK(3),GAMMA,DX(3),AKD
    integer,intent(in)::NX,NY,NZ,IPBC
    uploaded=.true.;lnx=NX;lny=NY;lnz=NZ;lipbc=IPBC;lid=IDIPINT;lakd=AKD;lak2=AK(2);lak3=AK(3);lgamma=GAMMA;lpyd=PYD;lpzd=PZD;ldx=DX
  end subroutine

  subroutine ensure_prepared(AK,GAMMA,DX,CMETHD,X,Y,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    use ddcommon_6, only: PYD,PZD
    real(WP),intent(in)::AK(3),GAMMA,DX(3)
    character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC
    complex(WP)::X(MXN3),Y(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT)
    real(WP)::AKD
    integer(c_int)::rc
#ifndef sp
    write(IDVOUT,*)'CUDA solver requires DDSCAT_PRECISION=sp';error stop 901
#endif
    AKD=sqrt(sum(AK*AK))
    if(green_changed(AK,GAMMA,DX,NX,NY,NZ,IPBC,AKD))then
      rc=c_prepare_green(int(NX,c_int),int(NY,c_int),int(NZ,c_int),int(IPBC,c_int),int(NAT,c_int),int(NAT0,c_int),int(IDIPINT,c_int), &
           real(GAMMA,c_float),real(PYD,c_float),real(PZD,c_float),real(AK(1),c_float),real(AK(2),c_float),real(AK(3),c_float), &
           real(DX(1),c_float),real(DX(2),c_float),real(DX(3),c_float),IOCC)
      if(rc==-777_c_int)then
        ! Regular ddscat_cuda DLL: preserve the legacy CPU ESELF + Green H2D path.
        call ESELF(CMETHD,X,NX,NY,NZ,IPBC,GAMMA,PYD,PZD,AK,AKD,DX,CXZC,CXZW,Y)
        rc=c_prepare(int(NX,c_int),int(NY,c_int),int(NZ,c_int),int(IPBC,c_int),int(NAT,c_int),int(NAT0,c_int),CXZC,IOCC)
      endif
      if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA Green/prepare failed, status=',rc;error stop 902;endif
      call remember(AK,GAMMA,DX,NX,NY,NZ,IPBC,AKD)
    endif
  end subroutine

  subroutine CUDA_CPROD(AK,GAMMA,DX,CMETHD,CWHAT,AD,AO,X,Y,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,PYD0,PZD0)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),PYD0,PZD0
    character(*),intent(in)::CMETHD;character(1),intent(in)::CWHAT
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC
    complex(WP)::AD(MXN3),AO(MXN3),X(MXN3),Y(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);integer(c_int)::rc
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,Y,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_apply(int(iachar(CWHAT),c_int),X,Y,AD,AO);if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA MATVEC failed=',rc;error stop 903;endif
  end subroutine

  subroutine CUDA_GPBICG_SOLVE(AK,GAMMA,DX,CMETHD,AD,AO,B,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,TOL,MAXIT,TOLE,NLOOP,NCOMPTE)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),TOL;character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,MAXIT
    complex(WP)::AD(MXN3),AO(MXN3),B(MXN3),X(MXN3),SCR(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);real(WP),intent(out)::TOLE;integer,intent(out)::NLOOP,NCOMPTE;integer(c_int)::rc,it,nc;real(c_float)::tf
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_gpb(B,X,AD,AO,int(NAT3,c_int),int(MAXIT,c_int),real(TOL,c_float),it,tf,nc)
    if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA GPBICG failed=',rc;error stop 904;endif
    NLOOP=int(it);NCOMPTE=int(nc);TOLE=real(tf,WP)
  end subroutine

  subroutine CUDA_QMRCCG_SOLVE(AK,GAMMA,DX,CMETHD,AD,AO,B,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,TOL,MAXIT,TOLE,NLOOP,NCOMPTE)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),TOL;character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,MAXIT
    complex(WP)::AD(MXN3),AO(MXN3),B(MXN3),X(MXN3),SCR(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);real(WP),intent(out)::TOLE;integer,intent(out)::NLOOP,NCOMPTE;integer(c_int)::rc,it,nc;real(c_float)::tf
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_qmr(B,X,AD,AO,int(NAT3,c_int),int(MAXIT,c_int),real(TOL,c_float),it,tf,nc)
    if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA QMRCCG failed=',rc;error stop 905;endif
    NLOOP=int(it);NCOMPTE=int(nc);TOLE=real(tf,WP)
  end subroutine

  subroutine CUDA_PBCGST_SOLVE(AK,GAMMA,DX,CMETHD,AD,AO,B,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,TOL,MAXIT,TOLE,NLOOP,NCOMPTE)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),TOL;character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,MAXIT
    complex(WP)::AD(MXN3),AO(MXN3),B(MXN3),X(MXN3),SCR(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);real(WP),intent(out)::TOLE;integer,intent(out)::NLOOP,NCOMPTE;integer(c_int)::rc,it,nc;real(c_float)::tf
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_pbcgst(B,X,AD,AO,int(NAT3,c_int),int(MAXIT,c_int),real(TOL,c_float),it,tf,nc)
    if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA PBCGST failed=',rc;error stop 906;endif
    NLOOP=int(it);NCOMPTE=int(nc);TOLE=real(tf,WP)
  end subroutine


  subroutine CUDA_PBCGS2_SOLVE(AK,GAMMA,DX,CMETHD,AD,AO,B,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,TOL,MAXIT,TOLE,NLOOP,NCOMPTE)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),TOL;character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,MAXIT
    complex(WP)::AD(MXN3),AO(MXN3),B(MXN3),X(MXN3),SCR(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);real(WP),intent(out)::TOLE;integer,intent(out)::NLOOP,NCOMPTE;integer(c_int)::rc,it,nc;real(c_float)::tf
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_pbcgs2(B,X,AD,AO,int(NAT3,c_int),int(MAXIT,c_int),real(TOL,c_float),it,tf,nc)
    if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA PBCGS2 failed=',rc;error stop 907;endif
    NLOOP=int(it);NCOMPTE=int(nc);TOLE=real(tf,WP)
  end subroutine

  subroutine CUDA_PETRKP_SOLVE(AK,GAMMA,DX,CMETHD,AD,AO,B,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,TOL,MAXIT,TOLE,NLOOP,NCOMPTE)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),TOL;character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,MAXIT
    complex(WP)::AD(MXN3),AO(MXN3),B(MXN3),X(MXN3),SCR(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);real(WP),intent(out)::TOLE;integer,intent(out)::NLOOP,NCOMPTE;integer(c_int)::rc,it,nc;real(c_float)::tf
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_petrkp(B,X,AD,AO,int(NAT3,c_int),int(MAXIT,c_int),real(TOL,c_float),it,tf,nc)
    if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA PETRKP failed=',rc;error stop 908;endif
    NLOOP=int(it);NCOMPTE=int(nc);TOLE=real(tf,WP)
  end subroutine

  subroutine CUDA_BICGST4_SOLVE(AK,GAMMA,DX,CMETHD,AD,AO,B,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,TOL,MAXIT,TOLE,NLOOP,NCOMPTE)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),TOL;character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,MAXIT
    complex(WP)::AD(MXN3),AO(MXN3),B(MXN3),X(MXN3),SCR(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);real(WP),intent(out)::TOLE;integer,intent(out)::NLOOP,NCOMPTE;integer(c_int)::rc,it,nc;real(c_float)::tf
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_bicgst4(B,X,AD,AO,int(NAT3,c_int),int(MAXIT,c_int),real(TOL,c_float),it,tf,nc)
    if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA BiCGStab(4) failed=',rc;error stop 909;endif
    NLOOP=int(it);NCOMPTE=int(nc);TOLE=real(tf,WP)
  end subroutine

  subroutine CUDA_GPBGS2_SOLVE(AK,GAMMA,DX,CMETHD,AD,AO,B,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,TOL,MAXIT,TOLE,NLOOP,NCOMPTE)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),TOL;character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,MAXIT
    complex(WP)::AD(MXN3),AO(MXN3),B(MXN3),X(MXN3),SCR(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);real(WP),intent(out)::TOLE;integer,intent(out)::NLOOP,NCOMPTE;integer(c_int)::rc,it,nc;real(c_float)::tf
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_gpbgs2(B,X,AD,AO,int(NAT3,c_int),int(MAXIT,c_int),real(TOL,c_float),it,tf,nc)
    if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA GPBiCGStab(2) failed=',rc;error stop 910;endif
    NLOOP=int(it);NCOMPTE=int(nc);TOLE=real(tf,WP)
  end subroutine

  subroutine CUDA_GPBGS4_SOLVE(AK,GAMMA,DX,CMETHD,AD,AO,B,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,TOL,MAXIT,TOLE,NLOOP,NCOMPTE)
    real(WP),intent(in)::AK(3),GAMMA,DX(3),TOL;character(*),intent(in)::CMETHD
    integer,intent(in)::IDVOUT,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NAT3,NX,NY,NZ,IPBC,MAXIT
    complex(WP)::AD(MXN3),AO(MXN3),B(MXN3),X(MXN3),SCR(MXN3),CXZC(NX+1+IPBC*(NX-1),NY+1+IPBC*(NY-1),NZ+1+IPBC*(NZ-1),6),CXZW(MXNX*MXNY*MXNZ,*)
    integer*2::IOCC(MXNAT);real(WP),intent(out)::TOLE;integer,intent(out)::NLOOP,NCOMPTE;integer(c_int)::rc,it,nc;real(c_float)::tf
    call ensure_prepared(AK,GAMMA,DX,CMETHD,X,SCR,CXZC,CXZW,IDVOUT,IOCC,MXN3,MXNAT,MXNX,MXNY,MXNZ,NAT,NAT0,NX,NY,NZ,IPBC)
    rc=c_gpbgs4(B,X,AD,AO,int(NAT3,c_int),int(MAXIT,c_int),real(TOL,c_float),it,tf,nc)
    if(rc/=0_c_int)then;write(IDVOUT,*)'CUDA GPBiCGStab(4) failed=',rc;error stop 911;endif
    NLOOP=int(it);NCOMPTE=int(nc);TOLE=real(tf,WP)
  end subroutine

  subroutine CUDA_EVALQ_GPU(CXADIA,CXAOFF,AK,NAT03,E02,CXE,CXP,CABS,CEXT,CPHA,RESIDENT)
    integer,intent(in)::NAT03
    real(WP),intent(in)::AK(3),E02
    complex(WP),intent(in)::CXADIA(*),CXAOFF(*),CXE(*),CXP(*)
    real(WP),intent(out)::CABS,CEXT,CPHA
    logical,intent(in)::RESIDENT
    integer(c_int)::rc,ires
    real(c_double)::da,de,dp
#ifndef sp
    error stop 'CUDA_EVALQ_GPU requires DDSCAT_PRECISION=sp'
#endif
    ires=merge(1_c_int,0_c_int,RESIDENT)
    rc=c_evalq(ires,CXE,CXP,CXADIA,CXAOFF,real(AK(1),c_float),real(AK(2),c_float),real(AK(3),c_float),real(E02,c_float),da,de,dp)
    if(rc/=0_c_int)then
      write(*,*)'CUDA EVALQ failed, status=',rc
      error stop 912
    endif
    CABS=real(da,WP);CEXT=real(de,WP);CPHA=real(dp,WP)
  end subroutine

  subroutine CUDA_SCAT_GPU(AK_TF,AKS_TF,DX,EM1_TF,EM2_TF,E02,ETASCA,CXE01_TF,CBKSCA,CSCA,CSCAG,CSCAG2, &
                           CXF1L,CXF2L,MXSCA,JPBC,NAVG,NDIR,X0)
    integer,intent(in)::MXSCA,JPBC,NDIR
    integer,intent(out)::NAVG
    real(WP),intent(in)::AK_TF(3),AKS_TF(3,MXSCA),DX(3),EM1_TF(3,MXSCA),EM2_TF(3,MXSCA),E02,ETASCA,X0(3)
    complex(WP),intent(in)::CXE01_TF(3)
    complex(WP),intent(out)::CXF1L(MXSCA),CXF2L(MXSCA)
    real(WP),intent(out)::CBKSCA,CSCA,CSCAG(3),CSCAG2
    integer(c_int)::rc,navc
    real(c_double)::db,ds,dg(3),dg2
#ifndef sp
    error stop 'CUDA_SCAT_GPU requires DDSCAT_PRECISION=sp'
#endif
    navc=0_c_int
    rc=c_scat(int(JPBC,c_int),int(NDIR,c_int),AK_TF,AKS_TF,DX,EM1_TF,EM2_TF,real(E02,c_float),real(ETASCA,c_float), &
              CXE01_TF,X0,db,ds,dg,dg2,navc,CXF1L,CXF2L)
    if(rc/=0_c_int)then
      write(*,*)'CUDA SCAT failed, status=',rc
      error stop 913
    endif
    CBKSCA=real(db,WP);CSCA=real(ds,WP);CSCAG=real(dg,WP);CSCAG2=real(dg2,WP);NAVG=int(navc)
  end subroutine

  subroutine CUDA_MATVEC_RELEASE()
    integer(c_int)::rc;rc=c_release();uploaded=.false.
  end subroutine
end module
