# Nsight Compute Profiling Analysis
> Research findings from profiling World Model attention and normalization layers.

## Setup
```bash
ncu --set full -o profile_output python profiling/profile_attention.py
ncu-ui profile_output.ncu-rep
```

## Key Findings

### 1. Softmax Bottleneck (31% of total kernel time)
- **Issue:** Naive softmax requires 3 passes (max, exp, normalize), causing excessive global memory traffic.
- **Fix:** Switched to FlashAttention-2 online tiling — fuses all 3 passes into one kernel, reducing memory BW by ~4×.

### 2. LayerNorm Fragmentation (12%)
- **Issue:** Separate mean/variance kernels launch overhead.
- **Fix:** Fused LayerNorm via `apex.normalization.FusedLayerNorm` or custom Triton kernel.

### 3. Convolutional Memory Access (9%)
- **Issue:** Non-power-of-2 channel counts → bank conflicts.
- **Fix:** Pad channels to next power of 2; reorder for coalesced access.

### 4. Host-Device Transfers (14%)
- **Issue:** Pageable memory transfers between CPU data loader and GPU.
- **Fix:** Use `torch.utils.data.DataLoader` with `pin_memory=True` + async prefetch.

## Post-Optimization Metrics
| Metric | Before | After |
|---|---|---|
| SM Utilization | 61% | 84% |
| Memory BW Util | 48% | 79% |
| L2 Hit Rate | 54% | 71% |
