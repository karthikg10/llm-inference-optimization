FROM nvcr.io/nvidia/pytorch:24.01-py3
RUN pip install sglang tritonclient[grpc] transformers accelerate
WORKDIR /app
COPY . .
EXPOSE 30000 8001
CMD ["python", "src/sglang_server.py"]
