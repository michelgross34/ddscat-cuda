# SLICES batch=4 / four Green streams

Implementation summary:

- outer slice loop: `for (kz0=0; kz0<2*NZ; kz0+=4)`
- XY cuFFT: rank 2, `batch=12` = 4 kz slices x 3 vector components
- Green: existing one-slice kernel, four independent launches
- streams: 4 persistent `cudaStreamNonBlocking` streams
- synchronization: CUDA events only; no `cudaStreamSynchronize` in the kz loop
- buffer: 4 x old single-slice XY buffer

Dependency chain per group:

```text
main/default stream: gather -> FFT2D(batch12) -> record xy_ready
                                      |
                    +-----------------+------------------+
                    v                 v                  v
 green stream 0: Green(kz0+0)   ... Green(kz0+3) : stream 3
                    |                 |                  |
                done[0]           done[...]          done[3]
                    +-----------------+------------------+
                                      v
main/default stream: wait events -> inverse FFT2D(batch12) -> scatter batch4
```

The four Green kernels may overlap when GPU resources permit. If one kernel
already saturates the GPU, CUDA will serialize/partially overlap them naturally;
the batch-4 FFT and loop reduction still provide the primary optimization.
