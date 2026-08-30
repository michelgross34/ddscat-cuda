# ddscat_cuda_slice: batch-4 low-memory MATVEC

The SLICES backend keeps the large field storage in complex float32 but avoids
a full doubled 3-D FFT buffer.

For a physical `NX x NY x NZ` vector field with 3 complex components:

1. Pad Z and store `3 x NX x NY x (2*NZ)` complex-float values.
2. Run one batched length-`2*NZ` 1-D cuFFT for every `(component,x,y)` line.
3. Process kz frequencies in groups `kz0 = 0,4,8,...`:
   - gather up to four kz planes in one kernel;
   - zero-pad X/Y into a `4 x 3 x (2*NX) x (2*NY)` buffer;
   - run one 2-D cuFFT with `batch=12`;
   - record an XY-ready event;
   - launch the existing `slice_green_xy_kernel` for `kz0+0...kz0+3` on four
     persistent nonblocking CUDA streams;
   - each Green stream records a completion event;
   - the main/default stream waits on the active completion events;
   - run one inverse 2-D cuFFT with `batch=12`;
   - scatter all active physical `NX x NY` planes back in one kernel.
4. Run the inverse batched 1-D Z cuFFT.
5. Extract the physical field with normalization
   `1/((2NX)(2NY)(2NZ))`.

The Green kernel itself is intentionally unchanged and remains a one-slice
kernel. Parallelism is obtained because each of the four launches operates on a
disjoint `3 x (2NX) x (2NY)` sub-buffer and a different kz value.

For `NX=NY=NZ=128` and complex float32:

- full FFT3D work buffer: `3 x 256^3` = 384 MiB;
- SLICES Z workspace: `3 x 256 x 128^2` = 96 MiB;
- four XY slices: `4 x 3 x 256^2` = 6 MiB;
- explicit SLICES data workspace: 102 MiB;
- saving relative to FFT3D: 282 MiB (~73.4%).

cuFFT can reserve additional internal workspace. The DLL prints the queried Z
and XY plan workspaces at startup, so the real overhead of batch=12 is visible.
