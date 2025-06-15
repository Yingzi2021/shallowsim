# Expand `shallowsim`

| 路线                  | 典型模型                  | 关键参数／注意力特色                                         | 许可证            | 为何推荐                                                     |
| --------------------- | ------------------------- | ------------------------------------------------------------ | ----------------- | ------------------------------------------------------------ |
| **Dense（全层密集）** | **Meta Llama 3 8B / 70B** | GQA（`num_key_value_heads < n_heads`）减少 KV-cache，推理显存友好([ai.meta.com](https://ai.meta.com/blog/meta-llama-3/?utm_source=chatgpt.com)) | Llama 3 Community | 成为业界默认 baseline；config 完整，字段与 ShallowSim 一一对应 |
|                       | **Mistral 7B**            | 滑窗注意力 + GQA；`sliding_window=4096`，最长 128 K token([huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/mistral)) | Apache-2.0        | 典型 **局部窗口 + Flash Attn** 组合，验证长上下文加速效果    |
|                       | **Qwen2 7B**              | 结合滑窗/全局注意力、GQA、YARN 长序列稳定器([huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/qwen2)) | Apache-2.0        | 支持 131 K context，可测试超长序列场景                       |
| **MoE（稀疏专家）**   | **Mixtral 8 × 22B**       | 8 路路由、每 token 激活 2 专家；Sparse MoE + MQA([mistral.ai](https://mistral.ai/news/mixtral-8x22b)) | Apache-2.0        | 当下最火高性能 MoE；开箱即用 Flash-Routing 实现              |
|                       | **Qwen2-MoE (32B / 72B)** | 官方文档自带 MoE 变体入口，沿用 Qwen2 长上下文设计([huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/qwen2)) | Apache-2.0        | 便于测试「MoE + 超长序列」双重加速                           |
| **轻量 & Edge**       | **Phi-3-Mini 4 B**        | RoPE-Yarn 扩展；QKV 融合提高 GPU 利用率([huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/phi3)) | MIT               | 手机端也能跑，做显存/延迟缩放曲线首选                        |
|                       | **Gemma 2B / 7B**         | 纯 MQA；RoPE + GeGLU；极简 config([huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/gemma)) | Apache-2.0        | 谷歌官方小模型，结构干净，适合教学/对照                      |
|                       | **Qwen2 0.5B**            | GQA + 滑窗，Context 32 K+；同一家族方便横向比较([huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/qwen2)) | Apache-2.0        | 500 M 参数，分钟级实验利器                                   |

DP SP PP

parallel scheme: **extract out**, use .json to config

attention: more implementation.

architecture: dense / MoE / edge, or use a csv/json file. mode-name, config(attention combinations, MoE or not).

add/specify position encoding & yarn ?