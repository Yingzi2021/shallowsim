# Evaluation

## 使用TGI进行LLM inference

```bash
model=teknium/OpenHermes-2.5-Mistral-7B # change
volume=$PWD/data

docker run --gpus all --shm-size 64g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.3.4 \
    --model-id $model
```

进入容器命令行

```bash
# 找到容器 ID 或名字
docker ps --format "table {{.Names}}\t{{.Image}}"

# 进入 shell
docker exec -it hopeful_jemison /bin/bash

# 在容器里随意用 CLI
text-generation-launcher --help
text-generation-server   --help
exit
```

使用服务（Consuming Text Generation Inference）

首先：`pip install openai`

```python
from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    base_url="http://localhost:8080/v1/",
    api_key="-"
)

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is deep learning?"}
    ],
    stream=True
)

# iterate and print stream
for message in chat_completion:
    print(message)
```

> No EP/PP. discarded

## 使用TensorRT-LLM + Triton Inference Server进行LLM inference

### TensorRT-LLM安装部署



### 生成随机权重







## 使用genai-perf进行性能测试

When building LLM-based applications, it is critical to understand the performance characteristics of these models on a given hardware. This serves multiple purposes: 

- Identifying the bottleneck and potential optimization opportunities
- Identifying the quality of service and throughput tradeoff
- Infrastructure provisioning

As a client-side LLM-focused benchmarking tool, [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html) provides key metrics: 

- Time to first token (TTFT)
- Inter-token latency (ITL)
- Tokens per second (TPS)
- Requests per second (RPS)
- …and more

> https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/
>
> https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html
>
> https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html

安装：

```bash
export RELEASE="25.01"

docker run -it --net=host --gpus=all  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Validate the genai-perf command works inside the container:
genai-perf --help
```

测试：

```bash
genai-perf profile \
  -m mistral-7b \
  --service-kind openai \
  --endpoint-type chat \
  -u http://localhost:8080 \
  --concurrency 4 \
  --synthetic-input-tokens-mean 256 \
  --output-tokens-mean 256 \
  --measurement-interval 60000 \
  --streaming \
  -v
```

结果：

![](figs/38.png)