# Reliable true-residual control every 20 iterations

This policy is common to both executables built from the same CUDA source:

- `ddscat_cuda` (FFT3D MATVEC);
- `ddscat_cuda_slice` (low-memory Z + batched XY-slice MATVEC).


## Policy

For solvers that maintain a recursively updated residual `r_rec`, the code
recomputes

```text
r_true = b - A*x
```

every **20 outer solver iterations**. A true-residual check is also forced
when the recursive residual first claims convergence, so false convergence
cannot be accepted between two periodic controls.

The drift is measured with the vector gap

```text
gap = ||r_true - r_rec||_2 / ||r_true||_2
```

and the fixed restart threshold is

```text
gap_restart = 1.0e-3
```

The true residual and gap norms are evaluated through the existing
float32->complex<double> chunk conversion (maximum 1,000,000 complex values)
and `cublasDznrm2`, with the chunk norms accumulated in CPU `double`.
The full residual vectors remain on the GPU.

## Restart rule

The Krylov history is restarted only when either:

1. `gap >= 1e-3`; or
2. the recursive residual says `res <= tol` but the recomputed true residual
   is still `> tol` (false convergence).

If the gap is below the threshold, the solver keeps the current Krylov
history unchanged. If the true residual itself is already below tolerance,
the solve is accepted immediately.

On restart, `r_rec` is replaced by the recomputed `r_true` and the solver's
algorithm-specific history/direction vectors and recurrence scalars are reset
consistently. The current solution `x` is not discarded.

## Solver coverage

The common 20-step policy is applied to:

- GPBICG;
- QMRCCG;
- PBCGS2 / BiCGStab(2);
- BiCGStab(4);
- GPBiCGStab(2);
- GPBiCGStab(4).

Special cases retained intentionally:

- **PBCGST** already evaluates the true `b-A*x` residual at every iteration
  (`STOPTYPE=2`), which is stricter than a 20-step check.
- **PETRKP** already refreshes `A*x` exactly every 10 iterations as in the
  original PETR90VER2 recurrence; this 10-step protection is retained.

The periodic true-residual MATVEC is counted in `NCOMPTE` / MATVEC timing, so
performance reports include the cost of reliability checks.
