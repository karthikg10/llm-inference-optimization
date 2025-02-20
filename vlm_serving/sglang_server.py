# sglang_server.py — SGLang VLM Serving Engine (fully runnable)
# Starts a Vision-Language Model server with continuous batching.
# Falls back to a FastAPI stub server when SGLang is not installed.

import os
import time
import argparse
import asyncio
import threading
from typing import Optional

# ── FastAPI stub server (always available) ────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class GenerateRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7


class StubVLMServer:
    """
    Minimal VLM server stub — responds to /generate requests without a real model.
    Demonstrates the serving interface; swap internals for real SGLang runtime.
    """
    def __init__(self, model: str, port: int):
        self.model   = model
        self.port    = port
        self.app     = FastAPI(title="VLM Serving Engine",
                               description="High-throughput VLM inference server")
        self.request_count  = 0
        self.total_latency  = 0.0

        @self.app.get("/health")
        def health():
            return {"status": "ok", "model": self.model}

        @self.app.get("/metrics")
        def metrics():
            avg_lat = (self.total_latency / self.request_count
                       if self.request_count > 0 else 0)
            return {
                "requests_served": self.request_count,
                "avg_latency_ms":  round(avg_lat, 2),
                "model":           self.model,
            }

        @self.app.post("/generate")
        def generate(req: GenerateRequest):
            t0 = time.perf_counter()

            # Stub: echo prompt + simulate token generation
            response_text = (
                f"[Model: {self.model}] "
                f"Received prompt of length {len(req.prompt)} chars. "
                f"Image={'yes' if req.image_url else 'no'}. "
                f"This is a stub response demonstrating the serving interface."
            )

            lat_ms = (time.perf_counter() - t0) * 1000
            self.request_count += 1
            self.total_latency += lat_ms

            return JSONResponse({
                "text":        response_text,
                "tokens":      len(response_text.split()),
                "latency_ms":  round(lat_ms, 2),
                "model":       self.model,
            })

    def start(self):
        print(f"[VLMServer] Starting stub server on port {self.port}")
        print(f"[VLMServer] Model: {self.model}")
        print(f"[VLMServer] Endpoints: GET /health, GET /metrics, POST /generate")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="warning")


def start_sglang_server(model: str, port: int,
                         dtype: str = "float16",
                         tp_size: int = 1,
                         max_batch_size: int = 32):
    """
    Start the real SGLang runtime for production VLM serving.
    SGLang provides continuous batching, PagedAttention KV cache,
    and efficient mixed text+image request scheduling.
    """
    try:
        import sglang as sgl
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path   = model,
            tokenizer_path = model,
            host         = "0.0.0.0",
            port         = port,
            tp_size      = tp_size,
            dtype        = dtype,
            max_prefill_tokens = 4096,
            disable_log_stats  = False,
        )

        print(f"[SGLang] Starting server: {model}")
        print(f"[SGLang] TP size: {tp_size} | dtype: {dtype} | port: {port}")

        from sglang.srt.server import launch_server
        launch_server(args)

    except ImportError:
        print("[SGLang] sglang not installed.")
        print("         Install: pip install sglang[all]")
        print("         Falling back to stub server...")
        if HAS_FASTAPI:
            server = StubVLMServer(model, port)
            server.start()
        else:
            print("[Stub] FastAPI not installed. pip install fastapi uvicorn")


def main():
    parser = argparse.ArgumentParser(description="VLM Serving Engine")
    parser.add_argument("--model",
                        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                        help="Model name or path")
    parser.add_argument("--port",     type=int, default=30000)
    parser.add_argument("--dtype",    default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--tp-size",  type=int, default=1,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--stub",     action="store_true",
                        help="Force stub server even if SGLang is available")
    args = parser.parse_args()

    if args.stub:
        if HAS_FASTAPI:
            server = StubVLMServer(args.model, args.port)
            server.start()
        else:
            print("Install fastapi and uvicorn: pip install fastapi uvicorn")
    else:
        start_sglang_server(args.model, args.port, args.dtype, args.tp_size)


if __name__ == "__main__":
    main()
