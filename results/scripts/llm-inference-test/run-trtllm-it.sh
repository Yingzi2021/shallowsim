docker run -it --rm \
  --name trtllm_tmp \
  --gpus all \
  --ipc=host --shm-size=32g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ./workspace:/app/tensorrt_llm/user_ws \
  -w /app/tensorrt_llm/user_ws \
  nvcr.io/nvidia/tensorrt-llm/release:0.20.0 \