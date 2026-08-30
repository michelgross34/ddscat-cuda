# DDSCAT CUDA float32 — PBCGST GPU-resident

This version adds CUDA `PBCGST` to the existing GPU `MATVEC`, `GPBICG`, and
`QMRCCG` implementation.

## Algorithm preserved from DDSCAT

DDSCAT calls `PIMCBICGSTAB` with:

- `PRECONTYPE=1`: left preconditioning;
- `STOPTYPE=2`: true residual criterion;
- restart length = 5.

The CUDA port preserves those choices. The left preconditioner `DIAGL` is
implemented on the GPU as elementwise complex division by `CXADIA`:

`z(i) = u(i) / CXADIA(i)`.

For every BiCGStab iteration the recurrence is the PIM recurrence:

`rho`, `beta`, `p`, `v`, `xi`, `alpha`, `s`, `t`, `omega`, `x`, `r`.

The complex scalar products use `chunked cublasZdotc`, matching Fortran `CDOTC`.
The stopping test explicitly recomputes the true residual `b-A*x`, matching
`STOPCRIT` for `STOPTYPE=2`:

`||b-A*x|| / ||b|| <= TOL`.

Because this is the true residual test, PBCGST can execute three GPU MATVECs in
an iteration: `A*p`, `A*s`, and `A*x` for the stopping criterion. A further
`A*x` is done at each 5-iteration restart to reconstruct the preconditioned
residual, just as a new PIM call does.

## Transfers

At solver entry: `b`, initial `x`, `CXADIA`, and `CXAOFF` are copied H2D once.
The Green tensor and `IOCC` are already persistent from CUDA preparation.

Inside the complete iteration/restart loop:

- H2D cudaMemcpy: 0;
- D2H cudaMemcpy: 0;
- D2D cudaMemcpy: 0.

All vector operations, diagonal preconditioning, reductions, and MATVECs are on
the GPU. Only mapped pinned scalars are read by the CPU for convergence and
breakdown decisions. The final polarization vector is copied D2H once.

## MATVEC

The existing persistent `cufftPlanMany(rank=3,batch=3)` is retained. X/Y/Z are
transformed with one batched 3-D cuFFT execution per transform direction.

Timing output includes entries such as:

`CUDA MATVEC ... [N,PBCGST-v]`
`CUDA MATVEC ... [N,PBCGST-t]`
`CUDA MATVEC ... [N,PBCGST-true-residual]`

and solver totals:

`CUDA PBCGST timing: iterative wall(no boundary D2H)=...`
`CUDA PBCGST timing: MATVEC GPU sum=...`
`CUDA PBCGST timing: wall minus MATVEC GPU=...`

## Test

`tests\ddscat_pbcgst.par` uses the current 16x16x16 reference case and selects:

`'PBCGST' = CMDSOL*6`

Build everything:

`scripts\build_all.bat`

Run the CLion Debug executable with the PBCGST test file:

`scripts\run_pbcgst_clion.bat`

The CUDA DLL build still targets CUDA 11.8 / sm_70 and automatically copies the
new DLL into `ddscat_cuda\cmake-build-debug\bin`.
