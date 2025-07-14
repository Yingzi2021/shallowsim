docker run --rm --name genai --net=host \
  -v ./workspace/benchmark:/workspace/usr_ws \
  -v ./workspace/model_weight:/workspace/usr_ws/model_weight \
  nvcr.io/nvidia/tritonserver:25.01-py3-sdk \
  genai-perf profile \
      -m qwen \
      --service-kind openai \
      --endpoint-type chat \
      --url http://localhost:8000 \
      --tokenizer /workspace/usr_ws/model_weight/Qwen1.5-MoE-A2.7B \
      --concurrency 16 \
      --synthetic-input-tokens-mean 1024 \
      --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean 100 \
      --output-tokens-stddev 0 \
      --request-count 64 \
      --warmup-request-count 16 \
      --streaming \
      --generate-plots \
      --artifact-dir /workspace/usr_ws/qwen-pp4-tp1-bs16