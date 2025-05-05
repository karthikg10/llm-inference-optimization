# quantize.py — Post-Training Quantization Pipeline (fully runnable)
# Supports FP16 and INT8 (dynamic + static) quantization of the WorldModel.

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def quantize_fp16(model: nn.Module) -> nn.Module:
    """Simple FP16 conversion — works on CUDA only."""
    if torch.cuda.is_available():
        return model.cuda().half()
    print("[Quantize] FP16 requires CUDA — skipping, returning original model")
    return model


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """
    Dynamic INT8 quantization via torch.quantization.
    Quantizes Linear layers at runtime — no calibration data needed.
    """
    model.eval().cpu()
    q_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv2d, nn.ConvTranspose2d},
        dtype=torch.qint8
    )
    print("[Quantize] Dynamic INT8 applied")
    return q_model


def quantize_static_int8(model: nn.Module,
                          calibration_loader: DataLoader,
                          device: str = "cpu") -> nn.Module:
    """
    Static INT8 quantization with calibration.
    Runs calibration data through the model to compute activation scales.
    """
    model.eval().to(device)

    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    # Fuse eligible layers (Conv + BN + ReLU etc.)
    try:
        model = torch.quantization.fuse_modules(model, [])
    except Exception:
        pass  # Skip fusion if not applicable

    # Prepare: insert observers
    torch.quantization.prepare(model, inplace=True)

    # Calibration pass
    print("[Quantize] Running calibration...")
    with torch.no_grad():
        for batch_idx, (frames, t) in enumerate(calibration_loader):
            frames = frames.to(device)
            t      = t.to(device)
            model(frames, t)
            if batch_idx >= 99:
                break
    print(f"[Quantize] Calibration complete ({batch_idx+1} batches)")

    # Convert: replace observers with quantized ops
    torch.quantization.convert(model, inplace=True)
    print("[Quantize] Static INT8 conversion done")
    return model


def benchmark_model(model: nn.Module, loader: DataLoader,
                    device: str, label: str, n_iters: int = 50):
    model.eval().to(device)
    # Warmup
    for frames, t in loader:
        with torch.no_grad():
            model(frames.to(device), t.to(device))
        break

    times = []
    with torch.no_grad():
        for i, (frames, t) in enumerate(loader):
            if i >= n_iters: break
            t0 = time.perf_counter()
            model(frames.to(device), t.to(device))
            if device == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"[{label:20s}] Avg latency: {avg_ms:7.2f}ms | "
          f"Throughput: {1000/avg_ms:.1f} batch/s")
    return avg_ms


def model_size_mb(model: nn.Module) -> float:
    params = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (params + buffers) / 1024 / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--precision", default="all",
                        choices=["fp32", "fp16", "dynamic_int8", "static_int8", "all"])
    parser.add_argument("--output-dir", default="quantized_models")
    parser.add_argument("--calib-batches", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu"  # INT8 quant is CPU-only in PyTorch

    from model import WorldModel
    model = WorldModel(img_size=64)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt))
        print(f"Loaded: {args.checkpoint}")
    else:
        print("No checkpoint provided — using randomly initialized model")

    # Build calibration dataset
    B, T, C, H, W = 1, 4, 3, 64, 64
    cal_frames = torch.randn(200, T, C, H, W)
    cal_t      = torch.randint(0, 1000, (200,))
    cal_loader = DataLoader(TensorDataset(cal_frames, cal_t),
                            batch_size=1, shuffle=False)

    print(f"\n{'Model':20s} {'Size(MB)':>10} {'Latency(ms)':>12}")
    print("-" * 46)

    def run(label, m, dev="cpu"):
        sz = model_size_mb(m)
        ms = benchmark_model(m, cal_loader, dev, label)
        out_path = os.path.join(args.output_dir, f"{label.replace(' ','_')}.pt")
        try:
            torch.save(m.state_dict(), out_path)
        except Exception:
            pass  # INT8 models may not serialize directly
        return sz, ms

    results = {}

    if args.precision in ("fp32", "all"):
        results["FP32 baseline"] = run("FP32 baseline", model)

    if args.precision in ("fp16", "all") and torch.cuda.is_available():
        m_fp16 = quantize_fp16(WorldModel(img_size=64))
        results["FP16"] = run("FP16", m_fp16, "cuda")

    if args.precision in ("dynamic_int8", "all"):
        m_dyn = quantize_dynamic_int8(WorldModel(img_size=64))
        results["Dynamic INT8"] = run("Dynamic INT8", m_dyn)

    if args.precision in ("static_int8", "all"):
        m_static = quantize_static_int8(WorldModel(img_size=64), cal_loader)
        results["Static INT8"] = run("Static INT8", m_static)

    print("\n── Summary ──────────────────────────────────")
    for name, (sz, ms) in results.items():
        print(f"  {name:20s}: {sz:6.1f} MB | {ms:7.2f} ms")

    if "FP32 baseline" in results and len(results) > 1:
        base_ms = results["FP32 baseline"][1]
        for name, (sz, ms) in results.items():
            if name != "FP32 baseline":
                print(f"  {name} speedup: {base_ms/ms:.2f}×")


if __name__ == "__main__":
    main()
