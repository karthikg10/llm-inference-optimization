# LLM Inference Optimization
> 🔬 **Research / Exploratory** — End-to-end exploration of making large models faster: from fused attention kernels and TensorRT quantization, to scalable VLM serving and world model video generation.

Three tightly related projects that each tackle a different layer of the LLM inference stack — kernel-level, serving-level, and generative model-level.

---

## Repository Structure

```
llm-inference-optimization/
├── transformer_inference/   ← Fused attention kernels + TensorRT INT8 on A100
├── vlm_serving/             ← SGLang continuous batching + custom NMS CUDA kernel
└── world_model/             ← Diffusion-based video prediction + Nsight profiling
```

---

## Projects

### 🔷 `transformer_inference/` — GPU-Accelerated Transformer Inference
Optimized transformer attention with CUDA kernel fusion, CUDA streams, and TensorRT post-training quantization.

**Key techniques:** Fused QKV + attention kernel · CUDA stream pipelining · TensorRT INT8 calibration

```
Input Tokens → Fused QKV GEMM → Softmax + Attn (streamed) → TensorRT INT8 Engine → Logits
                     ↑                    ↑                          ↑
              Custom CUDA kernel    CUDA Streams              PTQ calibration
```

| Config | Throughput (tok/s) | Latency (ms) | Memory (GB) |
|---|---|---|---|
| A100 FP32 baseline | 4,200 | 42.1 | 18.2 |
| + Fused kernels | 5,460 | 32.3 | 18.2 |
| + TensorRT INT8 | **6,100** | **28.7** | **13.7** |

> **1.3× throughput gain · 25% memory reduction**

---

### 🔷 `vlm_serving/` — High-Throughput VLM Serving Engine
Scalable serving for Vision-Language Models using SGLang continuous batching, a custom CUDA NMS post-processing kernel, and Dockerized PyTriton deployment.

**Key techniques:** Continuous batching (mixed text+image) · Custom CUDA NMS · PagedAttention KV cache

```
Client Requests → SGLang Continuous Batching → Vision Encoder
                                                      ↓
                                               LLM Decoder (PagedAttn)
                                                      ↓
                                          Custom CUDA NMS (-15ms latency)
                                                      ↓
                                              PyTriton Response
```

| Config | Throughput (req/s) | P50 Latency | P99 Latency |
|---|---|---|---|
| No batching | 12 | 420ms | 810ms |
| SGLang cont. batching | 47 | 198ms | 412ms |
| + Custom NMS kernel | 47 | **183ms** | **397ms** |

```bash
# Docker deployment
docker-compose -f vlm_serving/deployment/docker-compose.yml up
```

---

### 🔷 `world_model/` — World Model Video Generator
Diffusion-based model predicting future video frames, quantized for consumer GPUs and profiled with NVIDIA Nsight Compute to eliminate softmax and LayerNorm bottlenecks.

**Key techniques:** Latent diffusion UNet · PTQ (INT8/FP16) · Nsight-guided optimization · FlashAttention-2

**Nsight Profiling Findings:**
| Bottleneck | Time % | Fix |
|---|---|---|
| Softmax (attention) | 31% | FlashAttention-2 tiling |
| LayerNorm | 12% | Fused Triton kernel |
| Memory transfers | 14% | Pinned memory + prefetch |

| Config | GPU | FPS | Memory |
|---|---|---|---|
| FP32 baseline | A100 | 4.2 | 22.1 GB |
| FP16 | A100 | 9.1 | 11.3 GB |
| INT8 + FlashAttn | RTX 3090 | **11.2** | **7.1 GB** |

---

## Setup

```bash
# Transformer inference
pip install torch>=2.1 tensorrt>=8.6 transformers
cd transformer_inference && python benchmark.py

# VLM serving (Docker)
cd vlm_serving/deployment && docker-compose up

# World model
pip install torch diffusers flash-attn
cd world_model && python train.py --config world_model.yaml
```
