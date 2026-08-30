# GPU final cross sections and scattering amplitudes

This branch keeps the final compact polarization vector (`3*NAT0`) resident on
CUDA after the iterative solver and moves the heavy final `EVALQ` / `SCAT`
work to the GPU for both `ddscat_cuda` and `ddscat_cuda_slice`.

## EVALQ: Cabs, Cext, Cpha

`EVALQ` is a set of O(NAT0) reductions, not an FFT problem.

- A CUDA kernel computes `(alpha^-1 + i*2/3*k^3) P`, including all symmetric
  3x3 polarizability terms.
- `Cabs` is reduced with the existing chunked complex-double cuBLAS dot path
  (`cublasZdotc`).
- `Cext` and `Cpha` use the same path for `sum conj(E)*P`.
- The vector data are float32; accumulation is complex double, matching the
  mixed-precision solver policy.

For the first solved polarization/orientation the resident GPU solution and
resident incident field/operator are reused. For later linear combinations,
only compact `3*NAT0` arrays are uploaded.

## SCAT: Csca, g moments, backscatter and selected amplitudes

The DDSCAT far-field sum for a direction `ks` contains the phase
`exp(-i ks.rj)`. A naive implementation would launch a kernel that writes one
complex contribution per dipole and then ask cuBLAS to sum the entire vector.
That creates a large extra global-memory write/read for every scattering
direction.

The implemented method is more memory-efficient:

1. a CUDA kernel calculates the complex phase and dipole contribution;
2. each CUDA block accumulates its partial sums in double precision;
3. only block partials are written to global memory;
4. `cublasZgemv` sums the block partials for a batch of scattering directions;
5. a CUDA kernel forms the transverse scattered field and `|E_s|^2`;
6. `cublasDdot` evaluates the angular quadrature sums for `Csca`, the three
   `Csca*g` components, and `Csca*<cos^2(theta)>`.

Selected `F1/F2` amplitudes are obtained from the same batched far-field sums.

No complex square root is required by the far-field sum. The expensive
per-dipole transcendental operation is the complex phase; the kernel uses
`sincosf` to generate sine and cosine together.

## Why standard cuFFT is not used for SCAT

The far field is mathematically a Fourier transform of the polarization, but
DDSCAT requests values on a spherical/angular set of k-vectors (plus arbitrary
selected directions). A standard 3D cuFFT produces values on a regular
Cartesian reciprocal-space grid, so it does not directly produce the required
samples.

Using a standard FFT would therefore require interpolation from a Cartesian
k-grid and would change the numerical method. An exact/controlled accelerated
alternative is a NUFFT, which can be considered later as an optional backend.
The current GPU implementation preserves the direct DDSCAT phase sum.

## GPU memory lifetime

Before final postprocessing, the 13 compact Krylov work vectors are freed.
They are not needed for EVALQ/SCAT and are lazily reallocated before a later
iterative solve. This usually releases much more memory than the SCAT workspace
needs.

The SCAT workspace is batched (`SCAT_BATCH=16`). Its largest allocation stores
only complex-double block partials, approximately

`ceil(NAT0 / 1024) * 3 * 16 * sizeof(complex<double>)`.

The Green tensor and MATVEC FFT workspaces remain resident because GETFML may
start another polarization/orientation solve after the current postprocessing.

## Torque

The GPU SCAT path currently covers `CMDTRQ='NOTORQ'`. `DOTORQ` still calls the
original CPU `SCAT`, because the torque path needs additional position-weighted
complex sums. This preserves the original torque calculation until a dedicated
GPU implementation is added.
