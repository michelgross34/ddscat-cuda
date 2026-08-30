# PBCGS2 CUDA float32

`CMDSOL='PBCGS2'` now calls a CUDA implementation of the DDSCAT `ZBCG2`
BiCGStab(2) solver (`L=2`). The original Fortran call remains in the `#else`
path for CPU/reference builds.

The port reproduces the two BiCG stages, the 3x3 Hermitian residual Gram matrix,
the convex-polynomial stabilization and the `DELTA=1e-2` reliable-update logic.
DDSCAT's `PRECOND` routine for PBCGS2 is empty, so this solver intentionally does
**not** use the diagonal preconditioner used by PBCGST.

During the iterative and reliable-update path all full vectors remain on the
GPU. There are no explicit H2D/D2H `cudaMemcpy` calls. Residual and reliable
update flags use the already allocated mapped zero-copy scalar storage. A single
D2H copy returns the final solution after the final true-residual evaluation.

Typical MATVEC timing labels are:

```
PBCGS2-U1
PBCGS2-R1
PBCGS2-U2
PBCGS2-R2
PBCGS2-reliable       (only when ZBCG2 requests a reliable update)
PBCGS2-final-residual
```

Build on the target machine:

```bat
scripts\build_all.bat
```

Run the supplied CLion-debug test case (based on `ddscat(2).par`):

```bat
scripts\run_pbcgs2_clion.bat
```

Static transfer audit:

```bat
python scripts\check_pbcgs2_zero_copy.py
```
