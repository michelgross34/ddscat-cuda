# Compact CUDA solver storage on occupied DDSCAT sites

The CUDA solver state is now stored only on occupied target dipoles (`NAT0`).
The public DDSCAT arrays and the FFT/MATVEC representation keep their original
full-grid layout (`NAT = NX*NY*NZ`), so the physical operator and Fortran API do
not change.

## Compact layout

At CUDA preparation, `IOCC` is scanned once and a persistent GPU map is built:

```text
occupied_index[q] = j,  q=0..NAT0-1
```

where `j` is the corresponding occupied full-grid site. Solver vectors use the
same component-major ordering as DDSCAT:

```text
compact[c*NAT0 + q] <-> full[c*NAT + occupied_index[q]], c=0,1,2
```

Thus every Krylov-vector allocation scales with the actual filling fraction

```text
f = NAT0/NAT.
```

For a sphere tightly inscribed in a cubic box, `f` approaches `pi/6 = 0.523599`.
No geometric assumption is used by the implementation; arbitrary shapes use
their actual `IOCC` occupancy.

## MATVEC boundary

The FFT still requires the full rectangular grid. A solver MATVEC therefore is:

```text
compact solver vector (3*NAT0)
    -> CUDA scatter to full d_x (3*NAT), void sites zeroed
    -> existing FFT3D or SLICES MATVEC, unchanged physics
    -> CUDA gather of occupied sites from full d_y
    -> compact solver vector (3*NAT0)
```

These conversions are GPU kernels. There is no full-vector CPU transfer inside
the iterative loop.

`d_x` and `d_y` remain full-grid buffers because they are part of MATVEC/FFT,
not persistent Krylov storage. `ADIA`/`AOFF` also remain in full MATVEC layout.
PBCGST's diagonal preconditioner maps each compact element through
`occupied_index` before reading the corresponding full-grid diagonal entry.

## Compact allocations

The following solver vectors are compact:

- compact solver scratch x/y;
- RHS `b`;
- 13 persistent Krylov work vectors.

The two double-complex dot-product staging buffers are also capped by
`min(1,000,000, 3*NAT0)` elements.

The GPU memory report prints the exact measured filling factor, compact vector
memory, corresponding full-grid memory, and saved memory before iterations.

## Validation requirement

Static CMake/Fortran/C builds and CUDA syntax checks can be performed without a
GPU. Final validation must still be done with CUDA 11.8 on the target GPU by
comparing compact and non-compact builds on the same target: iteration count,
true residual, and `Qext/Qabs/Qsca` should agree within float32 solver tolerance.
