# PETRKP CUDA float32

This package adds a GPU-resident implementation of DDSCAT `PETRKP`
(`PETR90VER2`) to the existing CUDA solver package.

## Algorithm preserved

PETRKP solves the normal-equation direction recurrence used by DDSCAT:

- `ACE = A^H b`;
- `GI = ACE`, `PI = GI`;
- `QI = A PI`;
- `alpha = (GI^H GI)/(QI^H QI)`;
- each iteration computes `GI = ACE - A^H AXI`;
- `beta = (GI^H GI)/(GI_old^H GI_old)`;
- `PI = GI + beta PI`;
- `X = X + alpha PI`;
- `AXI` is updated recursively, but recomputed as `A X` every 10th iteration;
- convergence uses the true relative residual `||AXI-b|| / ||b||`.

The `A^H` operation uses the already implemented DDSCAT CUDA `CWHAT='C'`
path, while `A` uses `CWHAT='N'`.

## Transfers

At solver entry only, the RHS, matrix diagonal/off-diagonal blocks and initial
solution are transferred to the GPU. The PETRKP iteration loop contains no
`cudaMemcpy` calls. The convergence residual is exposed through the existing
mapped zero-copy scalar. The final solution is copied GPU -> CPU once.

Run the static audit with:

```bat
python scripts\check_petrkp_zero_copy.py
```

## CLion test

`tests\ddscat_petrkp.par` is based on the current `ddscat(2).par` reference
case with only `CMDSOL` changed to `PETRKP`.

After rebuilding the DLL and the CLion target:

```bat
scripts\run_petrkp_clion.bat
```

Expected timing labels include:

```text
CUDA MATVEC ... [C,PETRKP-AH-b]
CUDA MATVEC ... [N,PETRKP-A-p-initial]
CUDA MATVEC ... [N,PETRKP-A-x-initial]
CUDA MATVEC ... [C,PETRKP-AH-Ax]
CUDA MATVEC ... [N,PETRKP-A-p]
```

and every tenth iteration may also show `PETRKP-A-x-refresh`.
