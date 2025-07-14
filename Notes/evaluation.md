# Evaluation

[TOC]

## LLM serving准备

(1)命令行登录huggingface

```bash
pip install -U "huggingface_hub[cli]"
# Log in to huggingface-cli
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token *****
```

(2)下载权重到指定目录

```bash
huggingface-cli download "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --local-dir /home/boyingchen/model_weight/TinyLlama-1.1B-Chat-v1.0
```

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

### TensorRT-LLM从docker安装

```
sudo docker run --name trtllm -it --rm \
  --gpus all \
  --ipc=host --shm-size=32g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  -v /home/boyingchen/workspace:/app/tensorrt_llm/user_ws \
  -w /app/tensorrt_llm/user_ws \
  nvcr.io/nvidia/tensorrt-llm/release:0.20.0 \
  bash
```

> https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release
>

![](figs/39.png)

Sanity check the installation by running the following in Python (tested on Python 3.12):

```python
from tensorrt_llm import LLM, SamplingParams

def main():

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")#使用LLM类自动构建engine

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()
```

效果：

![](figs/40.png)

### 构建 TensorRT Engine

> https://blog.csdn.net/scgaliguodong123_/article/details/142531182

(1)定义模型路径和引擎输出路径

这个 HF_MODEL_DIR 就是上次自动下载的路径

```bash
HF_MODEL_DIR="/app/tensorrt_llm/user_ws/model_weight/TinyLlama-1.1B-Chat-v1.0"

WEIGHT_DIR="/app/tensorrt_llm/tinyllama_weight" # 转换后的权重

ENGINE_DIR="/app/tensorrt_llm/tinyllama_engine" 
```

(2)创建输出目录

```bash
mkdir -p ${WEIGHT_DIR}
mkdir -p ${ENGINE_DIR}
```

(3)**将HF格式的权重转换为TensorRT-LLM格式**

```bash
python /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
       --model_dir ${HF_MODEL_DIR} \
       --output_dir ${WEIGHT_DIR} \
       --dtype float16 \
       --pp_size 1  \
       --tp_size 1 
       # expert parallelism: --moe_ep_size(default = 1)
```

> **How to Enable Expert Parallelism**
>
> The default parallel pattern is Tensor Parallel. You can enable Expert Parallel or hybrid parallel by setting `--moe_tp_size` and `--moe_ep_size` when calling `convert_coneckpoint.py`. If only `--moe_tp_size` is provided, TRT-LLM will use Tensor Parallel for the MoE model; if only `--moe_ep_size` is provided, TRT-LLM will use Expert Parallel; if both are provided, the hybrid parallel will be used.
>
> Ensure the product of `moe_tp_size` and `moe_ep_size` is equal to `tp_size`, since the total number of MoE parallelism across all GPUs must match the total number of parallelism in other parts of the model.
>
> The other parameters related to the MoE structure, such as `num_experts_per_tok` (TopK in previous context) and `num_local_experts,` can be found in the model’s configuration file, such as the one for [Mixtral 8x7B model](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json). )
>
> see: https://nvidia.github.io/TensorRT-LLM/latest/advanced/expert-parallelism.html

输出：

![](figs/41.png)

编译出的产物为新的配置文件`config.json`与权重`rank0.safetensors`、`rank1.safetensors`。配置文件中包含了模型的结构信息以及并行配置（**TP/EP/PP均支持！**）

![](figs/42.png)

(4)运行 `trtllm-build` 命令将模型编译为TensorRT engine

```bash
trtllm-build --checkpoint_dir ${WEIGHT_DIR} \
--output_dir ${ENGINE_DIR} \
--max_input_len 1024 \
--max_batch_size 256            
```

`trtllm-build`常见参数说明：

- `--checkpoint_dir`

  The directory path that contains TensorRT-LLM checkpoint.
- `--output_dir`

  The directory path to save the serialized engine files and engine config file.Default: `'engine_outputs'`

- `--max_batch_size`

  Maximum number of requests that the engine can schedule.Default: `2048`
- `--max_input_len`

  Maximum input length of one request.Default: `1024`

- `--max_seq_len`, `--max_decoder_seq_len`

  Maximum total length of one request, **including prompt and outputs**. If unspecified, the value is deduced from the model config.

- `--use_fused_mlp`

  Possible choices: enable, disable. Enable horizontal fusion in Gated-MLP that combines two Matmul operations into a single one followed by a separate SwiGLU kernel.**Default**: `'enable'`

详见：https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html

> simulator config:
>
> ```
> class Config:
>     seq_len = 4383 # prompt length
>     decode_len = 1210 # tokens to generate
>     kv_cache_rate = 0.563
>     bs_list = [16, 32, 64, 128, 256, 512]
>     eplist = [8, 16, 36, 72, 144, 320]
> ```
>
> **Need alignment**: seq_len --> 1024

构建完成之后生成了引擎文件以及配置文件，其中，引擎文件除了模型配置以外，还有很多引擎相关的配置，如：

```json
"build_config": {
        "max_input_len": 1024,
        "max_seq_len": 2048,
        "opt_batch_size": 8,
        "max_batch_size": 256,
        "max_beam_width": 1,
......
```

### LLM serving with `trtllm-serve`

(1) 使用trtllm-serve进行模型部署

```bash
docker run -d --name trtllm_tmp \
  --gpus all \
  --ipc=host --shm-size=32g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/boyingchen/workspace:/app/tensorrt_llm/user_ws \
  -w /app/tensorrt_llm/user_ws \
  nvcr.io/nvidia/tensorrt-llm/release:0.20.0 \
  trtllm-serve serve \
  /app/tensorrt_llm/user_ws/tinyllama_engine \
  --tp_size 1 --pp_size 1 \
  --tokenizer /app/tensorrt_llm/user_ws/model_weight/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 --port 8000
```

![](figs/43.png)

`trtllm-serve`常见参数说明：

- `--max_batch_size` 

  Maximum number of requests that the engine can schedule.

- `--max_num_tokens`

  Maximum number of batched input tokens after padding is removed in each batch.

- `--max_seq_len`

  Maximum total length of one request, **including prompt and outputs**. If unspecified, the value is deduced from the model config.

- `--tp_size` 

  Tensor parallelism size.

- `--pp_size`

  Pipeline parallelism size.

- `--ep_size`

  expert parallelism size

详见：https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html#starting-a-server

(2) 检查：

```
curl http://localhost:8000/v1/models
```

返回

```json
{"object":"list","data":[{"id":"tinyllama_engine","object":"model","created":1750784198,"owned_by":"tensorrt_llm"}]}
```

对话：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "messages":[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Where is New York?"}],
        "max_tokens": 16,
        "temperature": 0
    }'
```

返回

```json
{"id":"chatcmpl-871134b6f5d24e7290586f35f2bdc749","object":"chat.completion","created":1750784404,"model":"TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","content":"New York is a city located in the northeastern part of the United","reasoning_content":null,"tool_calls":[]},"logprobs":null,"finish_reason":"length","stop_reason":null,"disaggregated_params":null}],"usage":{"prompt_tokens":36,"total_tokens":52,"completion_tokens":16}}
```



## 使用GenAI-Perf进行性能测试

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

### 安装

```bash
export RELEASE="25.01"

docker run -it --net=host --gpus=all  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Validate the genai-perf command works inside the container:
genai-perf --help
```

常见参数说明：

- `--tokenizer <str>`: The HuggingFace tokenizer to use to interpret token metrics from prompts and responses. The value can be the name of a tokenizer or the filepath of the tokenizer. The default value is the model name. (default: “<model_value>”)

- `--synthetic-input-tokens-mean <int>`: The mean of number of tokens in the generated prompts when using synthetic data, >= 1.

  > set to 1024 to align with the simulator

- `--output-tokens-mean <int>`: The mean number of tokens in each output. Ensure the `--tokenizer` value is set correctly, >= 1.

- `--streaming`: An option to enable the use of the streaming API. (default: `False`)

- `–batch-size-text` : The text batch size of the requests GenAI-Perf should send. (default: `1`)

- `--concurrency <int>`: The concurrency value to benchmark. (default: `None`)

- `--artifact-dir`: The directory to store all the (output) artifacts generated by GenAI-Perf and Perf Analyzer. (default: `artifacts`)

- `--generate-plots`: An option to enable the generation of plots. (default: False)

- `--profile-export-file <path>`: The path where the perf_analyzer profile export will be generated. By default, the profile export will be to `profile_export.json`. The genai-perf files will be exported to `<profile_export_file>_genai_perf.json` and `<profile_export_file>_genai_perf.csv`. For example, if the profile export file is `profile_export.json`, the genai-perf file will be exported to `profile_export_genai_perf.csv`. (default: `profile_export.json`)

> Note:
>
> For [Large Language Models](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/docs/tutorial.html), there is no batch size **(i.e. batch size is always `1`).** Each request includes the inputs for one individual inference. Other modes such as the [embeddings](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/docs/embeddings.html) and [rankings](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/docs/rankings.html) endpoints support client-side batching, where `--batch-size-text N` means that each request sent will include the inputs for `N` separate inferences, allowing them to be processed together.

### 测试

**基于TGI的LLM推理性能**

> 在另一个终端运行TGI的容器

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

**基于TensorRT-LLM的LLM推理性能**

> LLM inference服务已经按照前文所述部署好

```bash
mkdir -p $PWD/benchmark    

sudo docker run --rm --name genai --net=host \
  -v /home/boyingchen/benchmark:/workspace/usr_ws \
  nvcr.io/nvidia/tritonserver:25.01-py3-sdk \
  genai-perf profile \
      -m tiny-llama-tp2 \
      --service-kind openai \
      --endpoint-type chat \
      -u http://localhost:8000 \
      --tokenizer /workspace/usr_ws/model_weight/TinyLlama-1.1B-Chat-v1.0 \
      --concurrency 1 \
      --synthetic-input-tokens-mean 1024 \
      --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean 100 \
      --output-tokens-stddev 0 \
      --request-count 64 \
      --warmup-request-count 10 \
      --streaming \
      --generate-plots \
      --artifact-dir /workspace/usr_ws/artifacts-bs1-tp1
```

结果：

![](figs/44.png)

## Baseline & simulator result

（1）说明：simulator的结果是比较可信的

> LLMs on 2 RTX 3090 with TP = 2, PP = 1, no EP.
>
> input_len = 1024. prefill batch size = 1
>
> Error = |(Sim − Base)|/ Base × 100 %

Prefill Time analysis:

| Model Name     | Architecture | TP   | Baseline (ms) | Simulator (ms) | Error |
| -------------- | ------------ | ---- | ------------- | -------------- | ----- |
| tinyLlama-1.1B | Dense        | 1    | 47.64         | 12.63          | 73%   |
| tinyLlama-1.1B | Dense        | 2    | 22.86         | 8.02           | 65%   |
| tinyLlama-1.1B | Dense        | 4    |               | 5.09           |       |
|                |              | 8    |               |                |       |

> batch size = 1 fix
>
> 条形图，横坐标TP-degree，纵坐标时间 (per model)，PP/EP不开启，Dense
>
> 条形图，横坐标PP-degree，纵坐标时间 (per model)，EP不开启，TP fix = device num
>
> 条形图，横坐标EP-degree，纵坐标时间 (per model) (MoE)，PP不开启，TP fix = device num
>
> 表格：Error rate / **rank** (TP/PP/EP)

测试pp时统一设置tp?

Decode Time analysis: (tinyLlama-1.1B on 2 RTX 3090 with TP = 2, PP = 1, no EP.)

| Batch Size | Baseline (ms) | Simulator (ms) | Error |
| ---------- | ------------- | -------------- | ----- |
| 1          | 4.35          | 2.144          | 50.7% |
| 4          | 4.53          | 2.193          | 51.5% |
| 8          | 4.99          | 2.258          | 54.7% |
| 16         | 6.01          | 2.388          | 60.2% |

> batch size: 1, 4 8
>
> GenAI-Perf不支持调整batch size，使用concurrency参数模拟。
>
> 折线图（两条线+error条状）：时间随batch size变大而变大；

（2）说明：可以使用simulator来寻找合适的并行配置

> 参照原shallowsim的图

## 对比与联系

| 角色                                | 对应比喻                                | 你要交给它的东西                                             | 它产出的东西                                  | 典型命令                                                     |
| ----------------------------------- | --------------------------------------- | ------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------ |
| **TensorRT-LLM**(builder + SDK)     | C/C++ 的 **编译器 / 链接器 (g++/nvcc)** | PyTorch / HF 权重 (`.safetensors`) + 并行策略 (TP/PP/EP) + 精度 (FP16/INT4…) | **`.engine` / `.plan`** —— GPU 专用二进制     | `trtllm-build …`                                             |
| **Triton Inference Server**         | Linux 的 **运行时 loader + systemd**    | 由 TensorRT-LLM 生成的 plan 文件（model repo）               | **HTTP/gRPC 端口**；自动批处理、监控、热加载  | `tritonserver --model-repository …`                          |
| **TGI (Text-Generation-Inference)** | 一个 **自带模型、自己跑的 Web 服务**    | 直接给它 HuggingFace 模型名(它内部用→ bitsandbytes、Flash-Attn、vLLM kernel) | **HTTP (OpenAI 风格) 端口**；Rust-Router 拼批 | `docker run ghcr.io/huggingface/text-generation-inference …` |

```
          HF 权重
             │
         (convert)
             ▼
      ┌──────────────┐
      │ TensorRT-LLM │  ← 编译/量化/切分
      └──────┬───────┘
             │ plan
   ┌─────────┴─────────┐
   │     Triton        │  ← Router+批处理+监控
   └─────────┬─────────┘
             │ HTTP/gRPC
   ┌─────────┴─────────┐
   │ genAI-Perf / NIM  │  ← 客户端压测 / 商业包装
   └────────────────────┘

TGI 走的是另一条支线：
HF 权重 → Python Kernel (FlashAttn/vLLM) → Rust Router → HTTP
```

```
HF_MODEL_DIR="/app/tensorrt_llm/user_ws/model_weight/TinyLlama-1.1B-Chat-v1.0"

WEIGHT_DIR="/app/tensorrt_llm/user_ws/tinyllama_weight-tp2-pp1"

ENGINE_DIR="/app/tensorrt_llm/user_ws/tinyllama_engine-tp2-pp1"

rm -rf ${WEIGHT_DIR} ${ENGINE_DIR}

mkdir -p ${WEIGHT_DIR}
mkdir -p ${ENGINE_DIR}

# convert the model checkpoint to TensorRT LLM format
python /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
       --model_dir ${HF_MODEL_DIR} \
       --output_dir ${WEIGHT_DIR} \
       --dtype float16 \
       --tp_size 2
```
