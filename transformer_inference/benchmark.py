# benchmark.py — Throughput vs Latency sweep for transformer inference
# Runs with plain PyTorch; swap in TRT engine for production numbers.

import time
import argparse
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    """Lightweight transformer for benchmarking (drop-in for HuggingFace models)."""
    def __init__(self, vocab=1000, d_model=256, nhead=4, layers=4, seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head    = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.head(self.encoder(self.embed(x)))


def benchmark_model(model, batch_size, seq_len, device, n_warmup=20, n_iters=200):
    model.eval()
    dummy = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed loop
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms  = elapsed / n_iters * 1000
    throughput   = batch_size * n_iters / elapsed
    return latency_ms, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,4,8,16,32",
                        help="Comma-separated batch sizes to sweep")
    parser.add_argument("--seq-len",  type=int, default=128)
    parser.add_argument("--d-model",  type=int, default=256)
    parser.add_argument("--layers",   type=int, default=4)
    parser.add_argument("--fp16",     action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SimpleTransformer(d_model=args.d_model, layers=args.layers).to(device)
    if args.fp16 and device == "cuda":
        model = model.half()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    print(f"\n{'BatchSize':>10} {'Latency(ms)':>14} {'Throughput(tok/s)':>18}")
    print("-" * 46)
    for bs in batch_sizes:
        lat, tput = benchmark_model(model, bs, args.seq_len, device)
        print(f"{bs:>10} {lat:>14.2f} {tput * args.seq_len:>18.0f}")


if __name__ == "__main__":
    main()
