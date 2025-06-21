# My Plan 

(1) background information

> - LLM inference process in general (👌)
>   - prefill (Prompt) & decode (Generation) 
>   - **memory bound(KV cache)**
>
> - Different parallelism strategies (👌)
>   - Data parallelism
>   - Tensor parallelism
>   - sequence parallelism
>   - expert parallelism
>   - pipeline parallelism（⭐）
>
> - Different attention mechanism (👌)
>
>   - Multi-head attention (MHA)
>
>   - Multi-query attention(MQA)
>
>   - Group query attention(GQA)
>
>   - Multi-head latent attention(MLA)
>
> - Architecture for specific LLMs(👌)
>
>   - DeepSeek: MoE & MLA 
>
>   - Llama-3
>
>   - Qwen 3 series
>   
>   - Mistral 7B
>

(2) Run & analyze the code (👌)

(3) Add the implementation for GPT / Llama-like LLM

**Expand `shallowsim`**

| 模型           | 架构        | 注意力                                 | Config                                                       |
| -------------- | ----------- | -------------------------------------- | ------------------------------------------------------------ |
| Deepseek-V3/R1 | Dense + MoE | MLA                                    | https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/tree/main |
| Llama 3 70B    | Pure Dense  | GQA（`num_key_value_heads = 8`）       | https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct/tree/main |
| Qwen 3         | Pure MoE    | GQA（`num_key_value_heads = 4`）       | https://huggingface.co/Qwen/Qwen3-235B-A22B/tree/main        |
| Mistral 7B     | Pure Dense  | GQA（`num_key_value_heads = 8`） + SWA | https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main   |

> **Todo:** 
>
> - add support for model with slide window attention(SWA) (暂缓)
> - MoE comp-comm overlap (all2all & matmul)

