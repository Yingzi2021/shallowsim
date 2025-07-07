import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import copy as copy
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Literal
from pathlib import Path
from typing import Union,List, Tuple

cm = sns.light_palette("red", as_cmap=True)
NVL_GPU_LIST = [72, 144, 576]

class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # mqa
    # NEW ------------------------------
    # for GQA / MQA (KV shared)
    num_key_value_heads: int = n_heads   # default = same as n_heads
    # ----------------------------------

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

    is_moe: bool = True  # whether this model is a MOE model
    attention: Literal["mla", "gqa"] = "mla"  # attention type, default is MLA

    @classmethod
    def load_from_csv(cls,
                 csv_path: Union[str, Path],
                 model_name: str,
                 case_sensitive: bool = False) -> "ModelArgs":
        """
        Create a ModelArgs instance by reading one row from a CSV.

        Parameters
        ----------
        csv_path : str | Path
            CSV file; must contain a 'model_name' column.
        model_name : str
            Which row to load.
        case_sensitive : bool, default False
            Match 'model_name' in a case–insensitive way if False.
        """
        import pandas as pd
        from pathlib import Path

        df = pd.read_csv(Path(csv_path).expanduser().resolve())

        if "model_name" not in df.columns:
            raise ValueError(f"'model_name' column missing in {csv_path}")

        mask = df["model_name"] == model_name if case_sensitive \
               else df["model_name"].str.lower() == model_name.lower()

        if mask.sum() == 0:
            raise KeyError(f"model_name='{model_name}' not found in {csv_path}")

        row = df[mask].iloc[0].to_dict()

        # 1) start from default DeepSeek template
        args = cls()

        # 2) override with CSV values
        for k, v in row.items():
            if k == "model_name" or not hasattr(args, k):
                continue
            try:
                target_type = type(getattr(args, k))
                setattr(args, k, target_type(v))
            except (ValueError, TypeError):
                # blank / NaN cells keep default
                pass
        return args


class Config:
    seq_len = 1024 # prompt length
    decode_len = 100 # tokens to generate
    kv_cache_rate = 0.563
    tp_list = [1, 2, 4, 8]  # Centralized TP configuration
    dp_list=[1, 2, 4, 8]
    bs_list = [1, 8, 16]
    eplist = [8, 16, 36, 72, 144, 320]
    


class GPU_perf:
    def __init__(self, gpu_type, sm, comm_sm, gpu_per_node,
                 fp16_flops, fp8_flops, fp4_flops,
                 mem, mem_bw, nvlink_bw, pcie_bw, discount_rate):
        self.gpu_type = gpu_type
        self.sm = sm
        self.gpu_per_node = gpu_per_node
        self.comm_sm = comm_sm
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.fp4_flops = fp4_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate

    def get_fp16_flops(self):
        return self.fp16_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        return self.fp8_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp4_flops(self):
        return self.fp4_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw * self.discount_rate

    def get_nvlink_bw(self):
        return self.nvlink_bw * self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw * self.discount_rate


def get_gpu_info(filename='./device/gpuinfo.csv',
                 discount_rate=0.85,
                 device_list=[],
                 decoding_mode=False, print_console=False):
    """Get gpu info from csv file.

    Args:
        filename (str, optional): gpu performance datasheet filepath. Defaults to './device/gpuinfo.csv'.
        discount_rate (float, optional): Estimate performance discount from Peak FLOPS and peak BW. Defaults to 0.85.
        device_list (list, optional): select dedicated gpu. Defaults to [].
        decoding_mode (bool, optional): Enable decoding mode to set comm_sm=0. Defaults to False.
        print_console (bool, optional): print result. Defaults to False.

    Returns:
        dict{GPU_perf}: gpu performance dict.
    """
    gpu_dict = {}
    df = pd.read_csv(filename)
    if print_console:
        print(df.set_index('gpu_type').to_markdown())
    if decoding_mode:
        df['comm_sm'] = 0
    for _, c in df.iterrows():
        key = c['gpu_type']
        gpu = GPU_perf(
            gpu_type=c['gpu_type'],
            sm=c['sm'], comm_sm=c['comm_sm'],
            fp16_flops=c['fp16'],
            fp8_flops=c['fp8'],
            fp4_flops=c['fp4'],
            mem=c['mem'],
            mem_bw=c['mem_bw'],
            nvlink_bw=c['nvlink_bw'],
            pcie_bw=c['pcie_bw'],
            gpu_per_node=c['gpu_per_node'],
            discount_rate=discount_rate)
        if (len(device_list) == 0) | (key in device_list):
            gpu_dict[key] = gpu
    return gpu_dict

def gpu_category_idx(gpu_dict):
    gpu_category={}
    i = 0
    for key in gpu_dict.keys():
        gpu_category[key]=i
        i +=1
    return gpu_category


def _uniform_split(total_layers: int, pp: int) -> List[int]:
    """
    Return a list of *layer counts* per pipeline stage.
    e.g. 10 layers, pp = 4  ➜  [3, 3, 2, 2]
    """
    if pp < 1:
        raise ValueError("pp must be ≥ 1")
    base, rem = divmod(total_layers, pp)
    return [base + 1 if i < rem else base for i in range(pp)]


def _gpipe_latency(
    stage_times: List[float], 
    num_micro_batches: int  # M: number of sequence chunks in prefill / number of micro-batches in decode stage
) -> Tuple[float, float]:
    """
    Calculates the total latency and bubble time for a pipeline,
    based on GPipe's batch-splitting algorithm. [cite: 10, 65]

    Parameters
    ----------
    stage_times : List[float]
        A list containing the execution time for each pipeline stage.
    num_micro_batches : int
        The number of micro-batches (or sequence chunks) the input is split into.

    Returns
    -------
    Tuple[float, float]
        (total_latency, bubble_time)
    """
    pp = len(stage_times) # pp
    if pp <= 1:
        return sum(stage_times) * num_micro_batches, 0.0

    # The bottleneck is the slowest stage in the pipeline.
    slowest_stage_time = max(stage_times)

    # The pipeline runs for (M + K - 1) cycles of the slowest stage.
    # M cycles for the steady state, and K - 1 cycles for the fill/drain phase. 
    total_latency = (num_micro_batches + pp - 1) * slowest_stage_time

    # Bubble time is the difference between total pipeline latency and useful computation.
    # Or more simply, it's the idle time during the fill and drain phases.
    bubble_time = (pp - 1) * slowest_stage_time
    
    return total_latency, bubble_time


# 非吸收的版本
def mla_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate):
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_up_proj = q_len * args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = kv_len * args.kv_lora_rank * \
        args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv

    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)
    gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj

    # 把它看成一个标准的args.n_heads的MHA
    mha = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # QK_score_rope
                          + q_len * args.qk_nope_head_dim * kv_len  # QK_score_nope
                          + q_len * kv_len * args.v_head_dim)  # ScoreV
    wo = q_len * args.n_heads * args.v_head_dim * args.dim  # wo
    attn_sum = mha + wo
    # return flops by 2* Sum(MACs)
    GEMM_FP8_FLOPS = gemm_sum * 2/1e9
    ATTN_FP16_FLOPS = attn_sum * 2/1e9

    return GEMM_FP8_FLOPS+ATTN_FP16_FLOPS, GEMM_FP8_FLOPS, ATTN_FP16_FLOPS


# 矩阵吸收的版本
def mla_matabsob_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate=0):
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_rope_up_proj = q_len * args.q_lora_rank * \
        args.n_heads * args.qk_rope_head_dim  # wq_b_rope
    q_absorb = q_len * args.n_heads * (args.q_lora_rank * args.qk_nope_head_dim  # wq_b
                                       + args.qk_nope_head_dim * args.kv_lora_rank)  # w_uk

    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)  # KV-Cache命中率修正
    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj

    # 把它看成一个标准的args.n_heads的MQA
    mqa = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # Score_rope
                          + q_len * args.kv_lora_rank * kv_len  # Score_nope
                          + q_len * kv_len * args.kv_lora_rank)  # Score V

    attn_up_proj = q_len * args.n_heads * args.v_head_dim * args.kv_lora_rank
    o_proj = q_len * args.n_heads * args.v_head_dim * args.dim
    attn_sum = mqa + attn_up_proj + o_proj

    # return flops by 2* Sum(MACs)
    gemm_sum = gemm_sum * 2/1e9
    attn_sum = attn_sum * 2/1e9

    return gemm_sum + attn_sum, gemm_sum, attn_sum


def mla_mem(args: ModelArgs):
    q_down_proj = args.dim * args.q_lora_rank  # wq_a
    q_up_proj = args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv
    wo = args.n_heads * args.v_head_dim * args.dim  # wo
    return (q_down_proj + q_up_proj + k_up_proj + kv_down_proj + v_up_proj + wo)/1024/1024

# has TP/DP 
def mla_elapse_time(args: ModelArgs,
                      gpu: GPU_perf,
                      config: Config,
                      seq_len,
                      kv_cache_rate,
                      decoding_mode=True,
                      batchsize=1,
                      enable_gemm_fp4=False,
                      min_ar_time=0.015,  # Allreduce的静态延迟
                      mla_discount=0.7,  # based on FlashMLA result on H800
                      mla_kernel_static_time=0.05,
                      print_console=False):
    if decoding_mode:
        # Decoding时计算为qlen=1, kv_cache_rate = 1
        _, gemm_flops, attn_fp16_flops = mla_matabsob_flops(
            1, seq_len, args, 1)
        gemm_flops *= batchsize
        attn_fp16_flops *= batchsize
    else:
        # prefill阶段使用非吸收的版本
        _, gemm_flops, attn_fp16_flops = mla_flops(
            seq_len, seq_len, args, kv_cache_rate)
    gemm_fp8_t = gemm_flops / gpu.get_fp8_flops() / mla_discount
    attn_fp16_t = attn_fp16_flops / gpu.get_fp16_flops() / mla_discount

    # load weight
    load_t = mla_mem(args) / gpu.get_mem_bw()

    total = gemm_fp8_t + attn_fp16_t + load_t + mla_kernel_static_time

    if enable_gemm_fp4:
        if gpu.get_fp4_flops() == 0:
            if print_console:
                print('[%8s]This GPU does not support FP4' % gpu.gpu_type)
        else:
            gemm_fp4_t = gemm_flops / gpu.get_fp4_flops()
            total = gemm_fp4_t + attn_fp16_t

    ar_len = batchsize if decoding_mode else seq_len
    all_reduce_comm_size = ar_len * args.dim * 2 / 1024/1024  # fp16 take 2Bytes
    all_reduce_t = all_reduce_comm_size / gpu.get_nvlink_bw() + min_ar_time

    tp_time = {}
    for v in config.tp_list:
        if v == 1:
            tp_time[v] = total
        else:
            tp_time[v] = total / v + all_reduce_t

    if print_console:
        if enable_gemm_fp4 & (gpu.get_fp4_flops() != 0):
            print("[%8s]GEMM_FP4 Elapsed time(ms): %.3f" %
                  (gpu.gpu_type, gemm_fp4_t))
        print("[%8s]GEMM_FP8 Elapsed time(ms): %.3f" %
              (gpu.gpu_type, gemm_fp8_t))
        print("[%8s]ATTN_FP16 Elapsed time(ms): %.3f" %
              (gpu.gpu_type, attn_fp16_t))
        print("[%8s]Total Elapsed time(ms):%.3f" % (gpu.gpu_type, total))
        print("[%8s]AR Elapsed time(ms):%.3f" % (gpu.gpu_type, all_reduce_t))
        for v in config.tp_list:
            if v in tp_time:
                print("[%8s]TP[%2d] Elapsed time(ms):%.3f" %
                      (gpu.gpu_type, v, tp_time[v]))

    return total, tp_time


def prefill_mla(args: ModelArgs, gpu_dict, config: Config, seq_len, kv_cache_rate, print_console=False):
    df = pd.DataFrame(columns=['GPU'] + [f'TP{tp}' for tp in config.tp_list])
    for key in gpu_dict.keys():
        tp1, tp_list_res = mla_elapse_time(args, gpu_dict[key], config,
                                           seq_len, kv_cache_rate,
                                           decoding_mode=False,
                                           enable_gemm_fp4=False,
                                           print_console=print_console)
        
        row = [gpu_dict[key].gpu_type]
        for tp in config.tp_list:
             row.append(tp_list_res.get(tp, None)) # Use .get for safety
        df.loc[len(df)] = row

    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def gqa_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate):
    """
    FLOPs for one dense-GQA layer.

    Parameters
    ----------
    q_len, kv_len : int
        Current query / key-value sequence length.
    kv_cache_rate : float
        Fraction of KV already in cache (1.0 ⇒ full hit).

    Returns
    -------
    total , GEMM_FP8_FLOPS , ATTN_FP16_FLOPS
    """
    n_q  = args.n_heads                       # Query heads = h
    n_kv = args.num_key_value_heads or n_q    # KV heads = g, if g = 1 then MQA, if g = h then MHA.
    d_k  = d_v = args.dim // n_q              # per-head dim = d_k = d_v
    d_model = args.dim                        # full model dim

    # FP8 GEMM  (Q, K, V projection) 
    # Q: h × d_k  ←  d_model
    q_proj_mac  = q_len * n_q  * d_k * d_model
    # K/V: g × d_k  ←  d_model , but only cache-miss part
    kv_proj_mac = kv_len * n_kv * d_k * d_model * 2          # K + V
    kv_proj_mac *= (1.0 - kv_cache_rate)
    gemm_sum = (q_proj_mac + kv_proj_mac)

    # FP16 ATTENTION  (Q⊤K, softmax, Score·V, Wo) 
    # Q⊤K: h × q_len × kv_len × d_k
    qk_mac = n_q * q_len * kv_len * d_k
    # Score·V : h × q_len × kv_len × d_v 
    sv_mac = n_q * q_len * kv_len * d_v
    # Wo : d_model × d_model
    wo_mac = q_len * d_model * d_model
    attn_sum = (qk_mac + sv_mac + wo_mac)

    # return flops by 2* Sum(MACs)
    GEMM_FP8_FLOPS = gemm_sum * 2/1e9
    ATTN_FP16_FLOPS = attn_sum * 2/1e9

    total = GEMM_FP8_FLOPS + ATTN_FP16_FLOPS
    return total, GEMM_FP8_FLOPS, ATTN_FP16_FLOPS



def gqa_mem(args: ModelArgs):
    """
    Parameter footprint (MiB) for one GQA layer (FP16 by default).

    Q  : d_model x d_k         (h heads)      -> dim x dim
    K,V: d_model x d_k x kv_ratio            -> dim x dim x (g/h)
    Wo : d_model x d_model                   -> dim x dim
    """
    g = args.num_key_value_heads or args.n_heads
    d_model = args.dim
    d_k = d_model // args.n_heads # = d_v

    w_q  = d_model * d_k #  = dim × dim
    w_kv = d_model * d_k * (g / args.n_heads) * 2 #  K + V
    w_o  = d_model * d_model

    total_params = w_q + w_kv + w_o

    return total_params / 1024 / 1024         # → MiB


# has TP/DP 
def gqa_elapse_time(args: ModelArgs,
                        gpu: GPU_perf,
                        config: Config,
                        seq_len,
                        kv_cache_rate,
                        decoding_mode: bool=True,
                        batchsize=1,
                        enable_gemm_fp4=False,
                        min_ar_time=0.015,
                        gqa_discount=0.7, # need profile(TODO)
                        gqa_kernel_static_time=0.05,
                        print_console: bool = False):
    """
    Timing model for Dense-GQA Attention (no MoE).

    Returns
    -------
    base_time_tp1_ms, {tp: elapsed_ms}
    """
    if decoding_mode:      # q_len = 1, KV fully cached
        _, gemm_flops, attn_fp16_flops = gqa_flops(
            1, seq_len, args, kv_cache_rate=1.0)
        gemm_flops *= batchsize
        attn_fp16_flops *= batchsize
    else:                  # prefill
        _, gemm_flops, attn_fp16_flops = gqa_flops(
            seq_len, seq_len, args, kv_cache_rate)

    # Compute time
    if enable_gemm_fp4 and gpu.get_fp4_flops() != 0:
        gemm_t = gemm_flops / gpu.get_fp4_flops()
    else:
        gemm_t = gemm_flops / gpu.get_fp8_flops()
    gemm_t /= gqa_discount
    att_t   = attn_fp16_flops / gpu.get_fp16_flops() / gqa_discount

    # load weight
    load_t  = gqa_mem(args) / gpu.get_mem_bw()
    total  = gemm_t + att_t + load_t + gqa_kernel_static_time    # TP=1

    # All-Reduce latency (tensor-parallel)
    ar_len  = batchsize if decoding_mode else seq_len
    ar_size = ar_len * args.dim * 2 / 1024 / 1024      # FP16 = 2 bytes
    ar_t    = ar_size / gpu.get_nvlink_bw() + min_ar_time

    tp_time = {}
    for tp_degree in config.tp_list:
        if tp_degree == 1:
            tp_time[tp_degree] = total
        else:
            tp_time[tp_degree] = total / tp_degree + ar_t

    if print_console:
        print(f"[{gpu.gpu_type}] Dense-GQA total(ms): {total:.3f}")
        for k, v in tp_time.items():
            print(f"[{gpu.gpu_type}] TP{k:2d} time(ms): {v:.3f}")

    return total, tp_time


def prefill_gqa(args: ModelArgs, gpu_dict: dict, config: Config, seq_len, kv_cache_rate, print_console=False):
    """
    Prefill attention latency for Dense-GQA models.
    """
    df = pd.DataFrame(columns=['GPU'] + [f'TP{tp}' for tp in config.tp_list])
    for key, gpu in gpu_dict.items():
        tp1, tp_list_res = gqa_elapse_time(
            args, gpu, config,
            seq_len=seq_len,
            kv_cache_rate=kv_cache_rate,
            decoding_mode=False,
            enable_gemm_fp4=False)
        
        row = [key]
        for tp in config.tp_list:
            row.append(tp_list_res.get(tp, None))
        df.loc[len(df)] = row

    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def densmlp_flops(args: ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.inter_dim * 2/1e9


def densmlp_mem(args: ModelArgs):
    return 3 * args.dim * args.inter_dim / 1024/1024

def densmlp_elapse_time(args: ModelArgs,
                         gpu: GPU_perf,
                         config: Config,
                         seq_len,
                         decoding_mode: bool = False,
                         batchsize: int = 1,
                         min_ar_time=0.015,
                         print_console=False):
    """
    Calculates the elapsed time for a Dense MLP layer, accounting for Tensor Parallelism (TP).

    Returns
    -------
    base_time_tp1_ms (float), {tp: elapsed_ms} (dict)
    """
    # 1. Calculate base time on a single GPU
    # In decoding, the "sequence" processed per-token is the batch size
    effective_seq_len = batchsize if decoding_mode else seq_len
    
    gemm_flops = densmlp_flops(args, effective_seq_len)

    if gpu.get_fp4_flops() != 0:
        compute_time = gemm_flops / gpu.get_fp4_flops()
    else:
        compute_time = gemm_flops / gpu.get_fp8_flops()
    
    load_time = densmlp_mem(args) / gpu.get_mem_bw()
    
    # Total time for a single GPU before parallelism
    total_single_gpu_time = compute_time + load_time

    # 2. Calculate All-Reduce communication time required for TP
    # The activation tensor (seq_len * dim) is communicated
    ar_len = batchsize if decoding_mode else seq_len
    all_reduce_comm_size = ar_len * args.dim * 2 / 1024 / 1024  # fp16 = 2 bytes
    all_reduce_t = all_reduce_comm_size / gpu.get_nvlink_bw() + min_ar_time

    # 3. Calculate latency for each TP degree
    tp_time = {}
    for tp_degree in config.tp_list:
        if tp_degree == 1:
            tp_time[tp_degree] = total_single_gpu_time
        else:
            # Computation and load are split by TP, and communication cost is added
            tp_time[tp_degree] = total_single_gpu_time / tp_degree + all_reduce_t
    
    if print_console:
        print(f"[{gpu.gpu_type}] DenseMLP base time (TP=1): {total_single_gpu_time:.3f} ms")
        print(f"[{gpu.gpu_type}] All-Reduce time: {all_reduce_t:.3f} ms")
        for tp, time in tp_time.items():
            print(f"[{gpu.gpu_type}] --> TP={tp}, Total time: {time:.3f} ms")

    return total_single_gpu_time, tp_time



def moe_expert_flops(args: ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.moe_inter_dim * 2/1e9


def moe_expert_mem(args: ModelArgs):
    return 3 * args.dim * args.moe_inter_dim / 1024 / 1024

# has TP/DP
def _prefill_moe(args: ModelArgs, gpu: GPU_perf, seq_len, tp, dp):
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
    gemm_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops() != 0 else gpu.get_fp8_flops()
    num_device = tp * dp
    if args.n_shared_experts > 0:
        num_shared_token = dp * seq_len / num_device
        shared_flops = moe_expert_flops(args, num_shared_token)
        shared_time = shared_flops / gemm_flops + load_time
    else:
        shared_time = 0.0

    num_routed_token = seq_len * dp * args.n_activated_experts / num_device
    routed_flops = moe_expert_flops(args, num_routed_token)
    expert_num = math.ceil(args.n_routed_experts) / dp / tp
    routed_time = routed_flops / gemm_flops + load_time * expert_num

    return shared_time, routed_time

# has TP/DP 
def prefill_moe(args: ModelArgs, gpu_dict, config: Config, seq_len, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP', 'DP',
                          'Shared Expert', 'Routed Expert'])
    for key in gpu_dict.keys():
        for tp in config.tp_list:
            for dp in config.dp_list:
                s, r = _prefill_moe(args, gpu_dict[key], seq_len, tp, dp)
                df.loc[len(df)] = [gpu_dict[key].gpu_type, tp, dp, s, r]
    if print_console:
        df['TP'] = df['TP'].astype(int).astype(str)
        df['DP'] = df['DP'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df

# has TP/DP
def _prefill_alltoall(args: ModelArgs, gpu, seq_len, tp, static_latency=0.05):
    if gpu.gpu_per_node == 8:
        dp = gpu.gpu_per_node/tp
        dispatch_node = 4
        dispatch_size = (dispatch_node - 1) * dp * seq_len * \
            args.n_activated_experts / gpu.gpu_per_node * args.dim / 1024/1024
        comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node
    else:
        # NVL72
        expert_num = math.ceil(args.n_routed_experts / gpu.gpu_per_node)
        dispatch_prob = (args.n_routed_experts - expert_num) / \
            args.n_routed_experts
        dispatch_size = dispatch_prob * args.n_activated_experts * \
            seq_len/tp * args.dim / 1024/1024
        comm_bw = gpu.get_nvlink_bw()

    combine_size = 2 * dispatch_size  # fp16
    if gpu.get_fp4_flops != 0:
        dispatch_size = dispatch_size / 2
    dispatch_time = dispatch_size / comm_bw + static_latency
    combine_time = combine_size / comm_bw + static_latency
    return dispatch_time, combine_time

# has TP
def prefill_alltoall(args: ModelArgs, gpu_dict, config: Config, seq_len, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP', 'Dispatch', 'Combine'])
    for tp in config.tp_list:
        for key in gpu_dict.keys():
            dispatch_time, combine_time = _prefill_alltoall(
                args, gpu_dict[key], seq_len, tp)
            df.loc[len(df)] = [key, tp, dispatch_time, combine_time]
    if print_console:
        df['TP'] = df['TP'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def _prefill_time(args: ModelArgs, gpu, config: Config, seq_len, kv_cache_rate, tp, dp):
    if args.attention == "mla":      # MoE-MLA
        dense_att, tp_att_dict = mla_elapse_time(
            args, gpu, config,
            seq_len, kv_cache_rate,
            decoding_mode=False,
            enable_gemm_fp4=False)
        tp_att = tp_att_dict.get(tp)
    elif args.attention == "gqa":      # Dense-GQA
        dense_att, tp_att_dict = gqa_elapse_time(
            args, gpu, config,
            seq_len, kv_cache_rate,
            decoding_mode=False,
            enable_gemm_fp4=False)
        tp_att = tp_att_dict.get(tp)
    else:
        raise ValueError(f"Unsupported attention type: {args.attention}")

    _, tp_mlp_dict = densmlp_elapse_time(args, gpu, config, seq_len, decoding_mode=False)
    # Get the specific, TP-aware MLP time
    dense_mlp = tp_mlp_dict.get(tp)

    # MoE-only parts ; skip when not MoE
    if args.is_moe:
        shared, routed = _prefill_moe(args, gpu, seq_len, tp, dp)
        dispatch, combine = _prefill_alltoall(args, gpu, seq_len, tp)
    else:
        shared = routed = dispatch = combine = 0.0

    return dense_att, dense_mlp, tp_att, shared, combine, routed, dispatch


def prefill_time(args: ModelArgs,
                 gpu_dict: dict,
                 config: Config,
                 seq_len: int,
                 kv_cache_rate: float,
                 tp: int,
                 dp: int,
                 moe_prefill_overlap_factor: float = 0.2, # 1. Add a configurable overlap factor
                 print_console: bool = False):
    """
    Calculates prefill latency with a more realistic, configurable overlap factor for MoE layers.
    """
    # 1. Set up columns for the output DataFrames
    detail_cols = ['GPU', 'MLP_TPN', 'Attn_TPN']
    if args.is_moe:
        detail_cols.extend(['Shared Expert', 'Combine', 'Routed Expert', 'Dispatch'])
    
    detail_df = pd.DataFrame(columns=detail_cols)
    # Update summary_df to include the Overlapped column
    summary_df = pd.DataFrame(columns=['GPU', 'Compute', 'Comm', 'Overlapped', 'Sum'])
    
    n_sparse_layers = args.n_layers - args.n_dense_layers

    # --- Add the "Layers" row to the detail_df for better context ---
    layer_row = ['Layers',  # Corresponds to 'GPU' column
                 args.n_dense_layers,  # MLP_TPN is for dense layers
                 args.n_layers  # Attn_TPN is for all layers
                 ]
    if args.is_moe:
        # MoE components apply only to sparse layers
        layer_row.extend([n_sparse_layers] * 4) # For Shared, Combine, Routed, Dispatch
    detail_df.loc[len(detail_df)] = layer_row
    # ----------------------------------------------------------------

    # 2. Loop through each GPU and calculate performance
    for key, gpu in gpu_dict.items():
        # Get timings for individual components from the helper function
        _, mlp_tpn, attn_tpn, shared, combine, routed, dispatch = _prefill_time(
            args, gpu, config, seq_len, kv_cache_rate, tp, dp)

        # Populate the detail table with timing data for the current GPU
        row = [key, mlp_tpn, attn_tpn]
        if args.is_moe:
            row.extend([shared, combine, routed, dispatch])
        detail_df.loc[len(detail_df)] = row

        # 3. Calculate total latency using the refined overlap model
        
        # 3.1 Define time-per-layer for different layer types
        time_per_dense_layer = attn_tpn + mlp_tpn
        
        # 3.2 Calculate the time saved by overlapping communication and computation in MoE layers
        overlapped_reduction = 0
        if args.is_moe and n_sparse_layers > 0:
            # Overlap opportunity 1: Combine (comm) with (Attention + Shared Expert) (comp)
            comp1 = attn_tpn + shared
            overlapped1 = min(comp1, combine) * moe_prefill_overlap_factor
            
            # Overlap opportunity 2: Dispatch (comm) with Routed Expert (comp)
            comp2 = routed
            overlapped2 = min(comp2, dispatch) * moe_prefill_overlap_factor
            
            overlapped_reduction = overlapped1 + overlapped2
        
        # 3.3 Calculate total time for each part of the model
        # Total compute time = compute of dense layers + compute of sparse layers
        comp_time = (time_per_dense_layer * args.n_dense_layers) + \
                    ((attn_tpn + shared + routed) * n_sparse_layers if args.is_moe else 0)
        
        # Total communication time = communication of sparse layers
        comm_time = (combine + dispatch) * n_sparse_layers if args.is_moe else 0

        # Total time saved by overlapping
        total_overlapped_reduction = overlapped_reduction * n_sparse_layers
        
        # Final total latency = Total Compute + Total Communication - Total Overlapped Time
        total_time = comp_time + comm_time - total_overlapped_reduction

        summary_df.loc[len(summary_df)] = [key, comp_time, comm_time, total_overlapped_reduction, total_time]

    # 4. Format and print the final results
    detail_df = detail_df.set_index('GPU').T
    summary_df = summary_df.set_index('GPU').T

    if print_console:
        # The 'Layers' column will contain mixed types (string and numbers), so convert to string for clean printing
        detail_df['Layers'] = detail_df['Layers'].astype(str)
        print("--- Prefill Latency Breakdown (per-layer, ms) ---")
        print(detail_df.to_markdown(floatfmt=".3f"))
        print('\n--- Prefill Latency Summary (total, ms) ---')
        print(summary_df.to_markdown(floatfmt=".3f"))

    return detail_df, summary_df


def prefill_time_pp(args: ModelArgs,
                    config: Config,
                    gpu_dict: dict,
                    kv_cache_rate: float,
                    tp: int, dp: int,
                    pp: int, 
                    num_chunks: int,
                    print_console: bool = False):
    """
    • `pp`                  : pipeline degree (≥1)
    • `num_chunks`         : number of chunks to split the sequence into. (PP -> chunked prefill)
    """

    detail, summary = prefill_time(args, gpu_dict, config, config.seq_len,
                                   kv_cache_rate, tp, dp,
                                   print_console=False) 

    if pp == 1:
        if print_console:
            print(summary.to_markdown(floatfmt='.3f'))
        return detail, summary

    layers_per_stage = _uniform_split(args.n_layers, pp)

    for gpu_name in summary.columns:
        serial_latency_total = float(summary.at['Sum', gpu_name])

        # 假设负载是均匀的，按层数比例分配
        # 注意：这里需要将总时间除以 num_chunks，得到处理一个chunk的时间
        per_chunk_serial_latency = serial_latency_total / num_chunks
        
        stage_times = [per_chunk_serial_latency * (l / args.n_layers)
                       for l in layers_per_stage]

        # 计算流水线总耗时
        # 传入序列切分的总块数 num_chunks
        total_pp_latency, bubble_time = _gpipe_latency(stage_times, num_chunks)

        summary.at[f'Total_PP({pp=},M={num_chunks})', gpu_name] = total_pp_latency
        summary.at[f'Bubble({pp=},M={num_chunks})', gpu_name] = bubble_time
        summary.at[f'Speedup({pp=},M={num_chunks})', gpu_name] = serial_latency_total / total_pp_latency

    if print_console:
        print("\n[Prefill · Pipeline-parallel]")
        print(summary.to_markdown(floatfmt='.3f'))
        
    return detail, summary


# Decoding

def _decoding_batchsize(args: ModelArgs,
                        gpu: GPU_perf,
                        seq_len: int,
                        decode_len: int,
                        tp: int,
                        expert_num: int):
    """
    Parameters
    ----------
    tp          : tensor-parallel degree (heads are sharded)
    expert_num  : routed experts *per device*  (0 for dense models)

    Returns
    -------
    max_bs (float) - largest batch size that fits GPU memory
    """

    # ---------------- 1. CONSTANTS & RESERVES -------------------
    mem_util_rate   = 0.90      # keep 10 % headroom for activations, misc
    others_gib      = 2.91      # torch, embeddings, buffers   (GiB)
    mla_param_mb    = 187.17    # per-layer MLA parameter size  (MiB)
    expert_param_mb = 44.05     # per-layer MoE expert size     (MiB)

    if args.attention == "mla":
        attn_param_mb = mla_param_mb  # per-layer MLA parameter size  (MiB)
    elif args.attention == "gqa":
        attn_param_mb = gqa_mem(args)
    else:
        raise ValueError(f"Unsupported attention type '{args.attention}'")
    # ---------------- 2. PARAMETER FOOTPRINT -------------------
    param_mb = attn_param_mb * args.n_layers / tp           # all MLA/Dense layers
    if args.is_moe and expert_num > 0:
        moe_layers = args.n_layers - args.n_dense_layers
        param_mb  += expert_param_mb * moe_layers * expert_num

    # ---------------- 3. KV-CACHE size / token -----------------
    if args.attention == "mla":
        # MLA kernel stores LoRA-KV  (fp8) + RoPE shift (fp16) per token
        kv_bytes_per_tok = (args.kv_lora_rank + args.qk_rope_head_dim)
    else:   # "gqa" or generic MHA
        head_dim  = args.dim // args.n_heads
        kv_heads  = args.num_key_value_heads or args.n_heads
        kv_bytes_per_tok = head_dim * kv_heads * 2   # K+V, fp16 → 2 bytes

    if kv_bytes_per_tok == 0:
        raise ValueError("KV bytes per token computed as 0 – check model args.")

    total_tokens    = seq_len + decode_len
    kv_cache_bytes  = total_tokens * kv_bytes_per_tok * args.n_layers * tp

    # ---------------- 4. AVAILABLE GPU MEMORY ------------------
    usable_gib      = gpu.mem * mem_util_rate - others_gib - param_mb / 1024
    usable_bytes    = usable_gib * 1024 ** 3            # GiB → Bytes
    if usable_bytes <= 0:
        return 0.0

    # ---------------- 5. MAX BATCH SIZE ------------------------
    max_bs = usable_bytes / kv_cache_bytes
    return max_bs



def decode_batchsize(args: ModelArgs, gpu_dict, config: Config, seq_len, decode_len):
    df = pd.DataFrame(columns=['GPU', 'EP320', 'EP144', 'EP72', 'EP34'])
    for key in gpu_dict.keys():
        item = key
        value = [item]
        # This function seems to have a hardcoded assumption about expert parallelism (EP)
        # and doesn't use tp_list from config. We will leave it as is.
        for exp_num in [2, 3, 5, 9]:
            # Assuming tp=1 for this calculation as it's not specified
            bs = _decoding_batchsize(
                args, gpu_dict[key], seq_len, decode_len, 1, exp_num)
            value.append(bs)
        df.loc[len(df)] = value
    print(df.set_index('GPU').to_markdown(floatfmt=".0f"))
    return df

# has TP
def decode_mla(args: ModelArgs, gpu_dict, config: Config, bs_list, seq_len, decode_len, expert_num=2, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'BatchSize', 'TP', 'LoadKV', 'Attn_TP1', 'Attn_TPN'])
    for key in gpu_dict.keys():
        for bs in bs_list:
            kv_cache = seq_len * (args.kv_lora_rank +
                                  args.qk_rope_head_dim) * bs
            load_kv_time = kv_cache / 1024/1024 / \
                1024 / gpu_dict[key].get_mem_bw() * 1000
            attn_tp1_time, attn_tpn_dict = mla_elapse_time(args, gpu_dict[key], config,
                                                    seq_len, kv_cache_rate=1,
                                                    batchsize=bs,
                                                    decoding_mode=True,
                                                    enable_gemm_fp4=False) # here.
            for tp_num in config.tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], seq_len, decode_len, expert_num=expert_num, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[key].gpu_type, bs, tp_num,
                                       load_kv_time, attn_tp1_time, attn_tpn_dict.get(tp_num)]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df

# has TP
def decode_gqa(args: ModelArgs, gpu_dict, config: Config, bs_list, seq_len, decode_len, print_console=False):
    """
    Return DataFrame : GPU | BatchSize | TP | DenseGQA(ms)
    """
    df = pd.DataFrame(columns=['GPU', 'BatchSize', 'TP', 'LoadKV', 'Attn_TP1', 'Attn_TPN'])

    for key, gpu in gpu_dict.items():
        for bs in bs_list:
            # --- KV-Cache load ---
            kv_bytes = seq_len * (args.dim // args.n_heads) * args.num_key_value_heads
            kv_bytes *= bs            # tokens
            load_kv = kv_bytes / 1024/1024 / 1024 / gpu.get_mem_bw() * 1000  # ms

            # --- Attention kernel ---
            attn_tp1_time, attn_tpn_dict = gqa_elapse_time(
                args, gpu, config, seq_len, 1.0,
                batchsize=bs, decoding_mode=True)
            for tp in config.tp_list:
                max_bs = _decoding_batchsize(
                             args, gpu, seq_len, decode_len,expert_num=0,tp=tp)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [key, bs, tp, load_kv, attn_tp1_time, attn_tpn_dict.get(tp)]
    
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df

# has TP
def decode_dense_mlp(args: ModelArgs, gpu_dict, config: Config, bs_list, seq_len, decode_len, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'BatchSize', 'TP', 'MLP_TPN'])
    for key, gpu in gpu_dict.items():
        for bs in bs_list:
            # Call the TP-aware function once per batch size to get all TP results
            _, tp_mlp_dict = densmlp_elapse_time(
                args, gpu, config,
                seq_len=seq_len,      # This parameter is unused in decoding mode
                decoding_mode=True,
                batchsize=bs
            )
            
            for tp_num in config.tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu, seq_len, decode_len, expert_num=0, tp=tp_num)
                if bs <= max_bs:
                    # Get the correct time for the current tp_num from the dictionary
                    mlp_time = tp_mlp_dict.get(tp_num)
                    df.loc[len(df)] = [gpu.gpu_type, bs, tp_num, mlp_time]
    
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df.to_markdown(floatfmt=".3f"))

    return df


def n_pow2_range(n:int):
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n = n+1
    return n

def _decode_moe_expert(args: ModelArgs, gpu: GPU_perf, bs, 
                       gemm_group_per_device, device_num):
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
    if gpu.get_fp4_flops() != 0:
        load_time = load_time /2
    gpu_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops() != 0 else gpu.get_fp8_flops()
    
    total_expert = gemm_group_per_device * device_num
    m_per_group = bs * args.n_activated_experts * device_num / total_expert
   
    '''
    # TODO: 
    # 基于 group_gemm num 和 m_per_group 调整折扣因子
    # 可以基于Profile实测结果查表, 并将数据放在GPU_Perf结构题中
    # 此处简化以m_per_group估计如下
    '''

    #data from hs's profiling result
    flops_discounts = {
        1: 0.05,
        2: 0.05,
        4: 0.05,
        8: 0.05,
        16: 0.08,
        32: 0.1,
        64: 0.2,
        128: 0.35,
        256: 0.4,
        512: 0.6,
        1024: 0.7,
        2048: 0.7,
        4096: 0.7,
        8192: 0.7,
        16384: 0.7,
        32768: 0.7,
        65536: 0.7
    }

    # H20 exception based on hs's result
    if gpu.gpu_type.find('H20')!= -1 :
        flops_discounts = {
        1: 0.06,
        2: 0.06,
        4: 0.06,
        8: 0.12,
        16: 0.25,
        32: 0.45,
        64: 0.8,
        128: 0.9,
        256: 1.0,
        512: 1.0,
        1024: 1.0,
        2048: 1.0,
        4096: 1.0,
        8192: 1.0,
        16384: 1.0,
        32768: 1.0,
        65536: 1.0
    }

    gpu_flops = gpu_flops * flops_discounts[n_pow2_range(int(m_per_group))]
    
    shared_flops = moe_expert_flops(args, bs)
    shared_time = shared_flops / gpu_flops + load_time

    num_routed_token = bs * args.n_activated_experts
    routed_flops = moe_expert_flops(args, num_routed_token)
    routed_time = routed_flops / gpu_flops + load_time * gemm_group_per_device
    return shared_time, routed_time

# has TP
def decode_moe_expert(args: ModelArgs, gpu_dict, config: Config,
                      bs_list, seq_len, decode_len, 
                      gemm_group_per_device,
                      device_num,
                      mbs=2, 
                      print_console=False):
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                           'TP', 'SharedExpert', 'RoutedExpert'])
    for gpu_key in gpu_dict.keys():
        for bs in bs_list:
            s, r = _decode_moe_expert(
                args, gpu_dict[gpu_key], bs/mbs, 
                gemm_group_per_device=gemm_group_per_device, 
                device_num=device_num)
            s *= mbs
            r *= mbs
            for tp_num in config.tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[gpu_key], seq_len, decode_len, 
                    expert_num= gemm_group_per_device+1, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[gpu_key].gpu_type,
                                       str(bs), tp_num, s, r]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df[df['TP'] == 1][['GPU', 'BatchSize', 'SharedExpert',
              'RoutedExpert']].set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def _moe_a2a(args: ModelArgs, gpu: GPU_perf, bs, expert_num, device_num, fp8_combine=False, static_latency=0.005, mbs=2):
    dispatch_size = bs * args.dim * args.n_activated_experts / 1024/1024
    if fp8_combine & (gpu.get_fp4_flops() != 0):  # 支持FP4GPU才能开启FP8 Combine
        combine_size = dispatch_size
    else:
        combine_size = dispatch_size * 2  # FP16
    if gpu.gpu_per_node == 8:
        comm_bw = gpu.get_pcie_bw()
        # single host deployment
        if args.n_routed_experts / (expert_num - 1) == gpu.gpu_per_node:
            comm_bw = gpu.get_nvlink_bw()
    #NVL72 /144 / 576
    elif (gpu.gpu_per_node in NVL_GPU_LIST) & (device_num >  gpu.gpu_per_node):
            comm_bw = gpu.get_pcie_bw()
    else:
        comm_bw = gpu.get_nvlink_bw()

    dispatch_t = dispatch_size / comm_bw + static_latency * mbs
    combine_t = combine_size / comm_bw + static_latency * mbs
    return dispatch_t, combine_t

# has TP
def decode_a2a(args: ModelArgs, gpu_dict, config: Config,
               bs_list, seq_len, decode_len,
               expert_num, device_num,
               mbs=2,
               print_console=False, fp8_combine=False):
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                           'TP', 'Dispatch', 'Combine'])
    for key in gpu_dict.keys():
        for bs in bs_list:
            dispatch_time, combine_time = _moe_a2a(
                args, gpu_dict[key], bs, 
                expert_num=expert_num, device_num=device_num, 
                mbs=mbs, fp8_combine=fp8_combine)
            for tp_num in config.tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], 
                    seq_len, decode_len, 
                    expert_num=expert_num, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[key].gpu_type, bs,
                                       tp_num, dispatch_time, combine_time]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df[df['TP'] == 1][['GPU', 'BatchSize', 'Dispatch', 'Combine']].set_index(
            'GPU').to_markdown(floatfmt=".3f"))
    return df


def _decode_time(args: ModelArgs,
                 gpu_dict: dict,
                 config: Config,
                 bs_list,
                 seq_len,
                 decode_len,
                 gemm_group_per_device,
                 device_num,
                 mbs: int = 2,
                 fp8_combine: bool = False):
    """
    [Data Aggregator] Calls all component timing functions and merges them into a single DataFrame.
    """
    # 1. Get Attention timings
    if args.attention == 'gqa':
        attn_df = decode_gqa(args, gpu_dict, config, bs_list, seq_len, decode_len)
    else: # default mla
        # expert_num is only for memory calculation, can be 0 for dense parts
        attn_df = decode_mla(args, gpu_dict, config, bs_list, seq_len, decode_len, expert_num=0)

    # 2. Get MLP timings
    dmlp_df = decode_dense_mlp(args, gpu_dict, config, bs_list, seq_len, decode_len)
    
    data_frames = [attn_df, dmlp_df]
    
    # 3. Get MoE timings if applicable
    if args.is_moe:
        expert_per_device = gemm_group_per_device + 1
        moe_expert_df = decode_moe_expert(args, gpu_dict, config, bs_list, seq_len,
                                          decode_len, mbs=mbs,
                                          gemm_group_per_device=gemm_group_per_device,
                                          device_num=device_num)
        
        a2a_df = decode_a2a(args, gpu_dict, config, bs_list, seq_len, decode_len,
                            expert_num=expert_per_device, device_num=device_num,
                            fp8_combine=fp8_combine, mbs=mbs)
        data_frames.extend([moe_expert_df, a2a_df])
        
    # 4. Merge all component DataFrames into one
    # Use an outer merge to prevent data loss if a calculation fails for a specific configuration
    for decode_df in data_frames:
        decode_df['BatchSize'] = decode_df['BatchSize'].astype(int).astype(str)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['GPU', 'BatchSize', 'TP'], how='left'), data_frames)

    return merged_df

def decode_time(args: ModelArgs, gpu_dict, config: Config,
                bs_list, seq_len, decode_len,
                gemm_group_per_device,
                device_num,
                tps_limit=0,
                fp8_combine=False,
                moe_decode_overlap_factor: float = 0.5,
                print_console=False):
    """
    [Calculator] Calculates final per-token latency (TPOT) and throughput (TPS)
    based on the aggregated data from _decode_time and the model architecture.
    Includes a 'Comm_Time_Ratio' metric for bottleneck analysis.
    """
    # 1. Get the aggregated data. This part is correct and does not need changes.
    df = _decode_time(args, gpu_dict, config, bs_list, seq_len, decode_len,
                      gemm_group_per_device=gemm_group_per_device,
                      device_num=device_num,
                      fp8_combine=fp8_combine)
    
    # 2. Define component timings from the DataFrame. This part is also correct.
    attn_time = df['Attn_TPN'] + df['LoadKV']
    mlp_time  = df['MLP_TPN']

    # 3. Calculate latency based on the model's architecture. This logic is also correct.
    time_per_dense_layer = attn_time + mlp_time
    total_time_dense_layers = args.n_dense_layers * time_per_dense_layer
    
    total_time_sparse_layers = 0.0
    overlapped_reduction = 0
    if args.is_moe:
        moe_ffn_time = df['SharedExpert'] + df['RoutedExpert']
        moe_comm_time = df['Dispatch'] + df['Combine']
        n_sparse_layers = args.n_layers - args.n_dense_layers
        if n_sparse_layers > 0:
            comp1 = df['Attn_TPN'] + df['SharedExpert']
            overlapped1 = np.minimum(comp1, df['Combine']) * moe_decode_overlap_factor
            
            comp2 = df['RoutedExpert']
            overlapped2 = np.minimum(comp2, df['Dispatch']) * moe_decode_overlap_factor
            
            overlapped_reduction = overlapped1 + overlapped2
            
            time_per_sparse_layer = (attn_time + moe_ffn_time + moe_comm_time) - overlapped_reduction
            total_time_sparse_layers = time_per_sparse_layer * n_sparse_layers

    df['TPOT'] = total_time_dense_layers + total_time_sparse_layers
    df['TPS'] = 1000 / df['TPOT']
    df['Total'] = df['TPS'] * df['BatchSize'].astype(int)
    
    # --- 4.1. Add detailed components for analysis ---
    n_sparse_layers = args.n_layers - args.n_dense_layers
    
    # Total compute time = compute part of dense layers + compute part of sparse layers
    df['ComputeTime'] = (time_per_dense_layer * args.n_dense_layers) + \
                      ((attn_time + moe_ffn_time) * n_sparse_layers if args.is_moe else 0)
    
    # Total communication time (only from sparse layers)
    df['CommTime'] = (moe_comm_time * n_sparse_layers if args.is_moe else 0)
    
    # Total time saved by overlap (only from sparse layers)
    df['Overlapped'] = overlapped_reduction * n_sparse_layers if args.is_moe else 0

    # --- 4.2. Calculate the new Comm_Time_Ratio metric ---
    # Net communication time is the part that was not hidden by computation
    net_comm_time = df['CommTime'] - df['Overlapped']
    # The ratio of net communication time to the total time
    # This robustly handles division by zero if TPOT is 0
    df['Comm_Time_Ratio'] = np.divide(net_comm_time, df['TPOT'], 
                                    out=np.zeros_like(net_comm_time, dtype=float), 
                                    where=df['TPOT']!=0)

    df = df[df['TPS'] > tps_limit]

    if print_console:
        output_cols = ['GPU', 'TP', 'BatchSize', 'Attn_TPN', 'MLP_TPN']
        if args.is_moe:
            output_cols.extend(['SharedExpert', 'RoutedExpert', 'Dispatch', 'Combine', 'Overlapped'])
        output_cols.extend(['TPOT', 'Comm_Time_Ratio', 'TPS', 'Total'])

        for col in output_cols:
            if col not in df.columns:
                df[col] = 0.0
        print(df[output_cols].set_index(['GPU', 'BatchSize', 'TP']).to_markdown(floatfmt=".3f"))

    return df

# to modify
def decode_time_with_ep_list(args: ModelArgs, gpu_dict,
                             config: Config,
                             tps_limit=0,
                             fp8_combine=False,
                             print_console=False):
    if (not args.is_moe) or (args.is_moe and args.n_routed_experts == 0):
        raise ValueError("n_routed_experts must be set for MoE models.")
    
    attn_type = args.attention.upper()
    
    df_list = []
    for device_num in config.eplist:
        gemm_group_per_device = math.ceil(args.n_routed_experts / device_num)
        df = decode_time(args, gpu_dict, config, config.bs_list, config.seq_len,
                         config.decode_len,
                         gemm_group_per_device=gemm_group_per_device,
                         device_num=device_num,
                         fp8_combine=fp8_combine,
                         tps_limit=tps_limit,
                         print_console=False)
        df['EP'] = device_num
        df_list.append(df)
    dd = pd.concat(df_list)
    dd.reset_index(inplace=True, drop=True)
    order = ['GPU', 'TP', 'EP', 'BatchSize', 'Dense'+attn_type, 'DenseMLP', 'Sparse'+attn_type,
             'Combine', 'SharedExpert', 'RoutedExpert', 'Dispatch', 'COMP_SUM',
             'COMM_SUM', 'Delta', 'TPOT', 'TPOT_O', 'TPS', 'TPS_O', 'Total',
             'Total_O', 'Comm_Impact']
    dd = dd[order]
    dd['BatchSize'] = dd['BatchSize'].astype(int)
    return dd

def decode_time_pp(args: ModelArgs,
                   config: Config,
                   gpu_dict: dict,
                   gemm_group_per_device,
                   device_num,
                   pp: int, micro_bs_num: int,
                   tps_limit: int = 0,
                   fp8_combine: bool = False,
                   print_console: bool = False):

    # Base results without pipeline parallelism
    df = decode_time(args, gpu_dict, config, config.bs_list, config.seq_len, config.decode_len,
                     gemm_group_per_device, device_num,
                     tps_limit=tps_limit,
                     fp8_combine=fp8_combine,
                     print_console=False)

    if pp == 1:                      
        if print_console:
            print(df.set_index('GPU').T.to_markdown(floatfmt='.3f'))
        return df

    layers_per_stage = _uniform_split(args.n_layers, pp)

    for idx, row in df.iterrows():
        
        bs = int(row['BatchSize'])
        serial_latency_total = float(row['TPOT']) * bs
        
        num = min(bs, micro_bs_num)
        per_micro_batch_serial_latency = serial_latency_total / num

        stage_times = [per_micro_batch_serial_latency * (l / args.n_layers)
                       for l in layers_per_stage]

        total_pp_latency, bubble_per_token = _gpipe_latency(stage_times,num)
        tpot_pp = total_pp_latency / bs

        # ---- write back results -------------------------------------
        df.at[idx, 'PP']       = pp
        df.at[idx, 'Bubble']   = bubble_per_token
        df.at[idx, 'TPOT_PP']  = tpot_pp
        df.at[idx, 'Speedup']  = serial_latency_total / total_pp_latency 

    if print_console:
        print("\n[Decode · Pipeline-parallel with PP = {}]".format(pp))
        print(df.set_index('GPU').T.to_markdown(floatfmt='.3f'))

    return df


def df_filter(df,gpu,device_num=0, bs=0,tps_limit=0, value_list=[]):
    df_o = df[df['GPU'] == gpu] 
    if bs > 0:
        df_o = df_o[df_o['BatchSize'] == str(bs)]
    if tps_limit > 0:
        df_o = df_o[df_o['TPS'] > tps_limit]
    if device_num > 0:
        df_o = df_o[df_o['EP'] == device_num]
    if len(value_list) > 0:
        df_o = df_o[value_list]
    return df_o



def df_sort(df,value,ascending=False):
    if ascending:
        df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False)\
            .apply(lambda t: t[t[value]==t[value].min()]).sort_values([value],ascending=True).reset_index(drop=True)
    else:
        df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False)\
            .apply(lambda t: t[t[value]==t[value].max()]).sort_values([value],ascending=False).reset_index(drop=True)
    return df_o

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

def color_positive_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for positive
    strings, black otherwise.
    """
    color = 'red' if val > 0 else 'black'
    return 'color: %s' % color

def gpu_category_color(s,props):
    colors = ['color:darkred','color:steelblue','color:green','color:black','color:m',
              'color:darkgoldenrod','color:darkgreen','color:crimson','color:brwon','color:sienna',
              'color:navy','color:pink','color:gray','color:darkviolet']
    gpu_idx = props[s]
    return colors[gpu_idx]

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)



def draw(df, gpu_dict,
         comp_name,comp_val_list, 
         val_list, val_unit_name,
         title, width=8, height=4,
         filename='',savefig=False):
    def _df_filter(df, gpu, comp_name, comp_val, value_list):
        df1 = df[(df['GPU'] == gpu) & (df[comp_name] == comp_val)][value_list]
        return df1
    
    sns.color_palette("Paired")
    num_gpu = len(gpu_dict)
    fig_height = height * num_gpu
    fig_width = width * len(val_list)
    fig, axs = plt.subplots(nrows=num_gpu, ncols=len(val_list), figsize=(fig_width, fig_height))

    fig.suptitle(title, y=0.9,fontsize='large')
    value_list = val_list + ['index_value']
    cnt = 0
    for key in gpu_dict.keys():
        axt = axs[cnt]
        for i in range(0,len(val_list)):
            axt[i].legend(comp_val_list)
        for comp_v in comp_val_list:
            df1 = _df_filter(df, key , comp_name, comp_v, value_list)
            for i in range(0,len(val_list)):
                sns.lineplot(x='index_value', y=val_list[i],label=str(comp_v), data=df1,  ax=axt[i])
                axt[i].set_ylabel(val_unit_name)
                axt[i].set_xlabel(key+'('+val_list[i]+')')
        cnt += 1

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.2, hspace=0.2)
    if savefig:
        if filename=="":
            filename = './figures/'+title.replace(' ','_')+'.png'
        plt.savefig(filename,bbox_inches='tight', pad_inches=0.05)
    plt.show()