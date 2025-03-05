import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm

world_size = 1  # 世界大小，即设备数量
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


@dataclass
class ModelArgs:

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16

    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0

    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert (
            vocab_size % world_size == 0
        ), f"vocab_size {vocab_size} must be divisible by world_size {world_size}"

        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if weight.element_size() > 1:  # 权重元素大小大于1
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":  # 使用bf16实现 Brain Floating Point
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:  # 使用fp8实现
        x, scale = act_quant(x, block_size)  # 量化激活 将神经网络的激活值量化为8位
        y = fp8_gemm(x, scale, weight, weight.scale)  # 量化矩阵乘法
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):

    dtype = torch.bfloat16  # 默认数据类型

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype or Linear.dtype)
        )  # 初始化权重，特征数为out_features*in_features 矩阵乘法
        if self.weight.element_size() == 1:  # 如果权重的元素大小为1
            scale_out_features = (
                out_features + block_size - 1
            ) // block_size  # 计算输出特征数
            scale_in_features = (
                in_features + block_size - 1
            ) // block_size  # 计算输入特征数
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )  # 初始化缩放因子

        else:
            self.register_parameter(
                "scale", None
            )  # pytorch中的register_parameter方法用于注册模型参数，之后被pytorch自动追踪，用于参数更新（训练），模型保存加载和设备迁移
        if bias:
            self.bias = nn.Parameter()  # 初始化偏置
        else:
            self.register_parameter("bias", None)  # 注册偏置参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """列并行线性层(Column-parallel linear layer)
    这是一种特殊的线性层，用于在多GPU环境下进行线性变换
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        assert (
            out_features % world_size == 0
        ), f"out_features {out_features} must be divisible by world_size {world_size}"  # 保证输出特征数是world_size的整数倍
        self.part_out_features = out_features // world_size  # 每个设备上的输出特征数
        super().__init__(
            in_features, self.part_out_features, bias, dtype
        )  # 调用父类的初始化函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)  # 计算线性变换
        return y


class RowParallelLinear(Linear):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        assert (
            in_features % world_size == 0
        ), f"input features must be divisible by world_size {world_size}"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):

        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:

    dim = args.qk_rope_head_dim  # qk_rope is the same as v_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast  # beta_fast is a parameter for the fast sinusoid
    beta_slow = args.beta_slow  # beta_slow is a parameter for the slow sinusoid
    base = args.rope_theta  # base is the base frequency
    factor = args.rope_factor  # factor is the factor by which the frequency increases

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim=-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_line(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """多头注意力层(Multi-headed attention layers)
    这是Transformer架构中的核心组件之一，用于处理序列中的长距离依赖关系
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 模型的基本维度参数
        self.dim = args.dim  # 输入维度
        self.n_heads = args.n_heads  # 总的注意力头数
        self.n_local_heads = (
            args.n_heads // world_size
        )  # 每个设备上的注意力头数(用于分布式训练)

        # LoRA相关参数 (LoRA是一种参数高效微调方法)
        self.q_lora_rank = args.q_lora_rank  # Query矩阵的LoRA秩
        self.kv_lora_rank = args.kv_lora_rank  # Key-Value矩阵的LoRA秩

        # 注意力头的维度参数
        # nope: no position embedding. rope: rotary position embedding
        self.qk_nope_head_dim = args.qk_nope_head_dim  # 不使用旋转位置编码的Q/K头维度
        self.qk_rope_head_dim = args.qk_rope_head_dim  # 使用旋转位置编码的Q/K头维度
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # Q/K总头维度
        self.v_head_dim = args.v_head_dim  # Value头维度

        # Query变换矩阵的初始化
        if self.q_lora_rank == 0:  # 如果不使用LoRA
            # 直接使用列并行的线性变换
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:  # 如果使用LoRA
            # 使用两个矩阵的组合来降低参数量
            self.wq_a = Linear(self.dim, self.q_lora_rank)  # 第一个变换
            self.q_norm = RMSNorm(self.q_lora_rank)  # 归一化层
            self.wq_b = ColumnParallelLinear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim
            )  # 第二个变换

        # Key-Value变换矩阵的初始化
        self.wkv_a = Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim
        )  # KV的第一个变换
        self.kv_norm = RMSNorm(self.kv_lora_rank)  # KV的归一化层
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )  # KV的第二个变换

        # 输出投影矩阵
        self.wo = RowParallelLinear(
            self.n_heads * self.v_head_dim, self.dim
        )  # 多头注意力输出的合并变换

        # 注意力计算的缩放因子
        self.softmax_scale = self.qk_head_dim**-0.5  # 用于缩放注意力分数

        # 对于长序列的特殊处理
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = (
                self.softmax_sclae * mscale * mscale
            )  # 根据序列长度调整缩放因子

        # 缓存初始化，用于加速推理
        if attn_impl == "naive":  # 朴素实现方式
            # 分别存储K和V的缓存
            self.register_buffer(
                "k_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_heads,
                    self.qk_head_dim,
                ),
                persistent=False,
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_heads,
                    self.v_head_dim,
                ),
                persistent=False,
            )
        else:  # 优化实现方式
            # 合并存储KV和位置编码的缓存
            # register_buffer 用于注册模型参数，但不会被优化器更新
            self.register_buffer(
                "kv_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
                persistent=False,
            )
            self.register_buffer(
                "pe_cache",
                torch.zeros(
                    args.max_batch_size, args.max_seq_len, self.qk_rope_head_rank
                ),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.size()  # x是输入，bsz是batch size，seqlen是序列长度
        end_pos = start_pos + seqlen  # 计算序列的结束位置
        if self.q_lora_rank == 0:  # 如果不使用LoRA
            q = self.wq(x)  # 直接计算Q,这里为什么有w？
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(
                bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = (
                torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
                * self.softmax_scale
            )
        else:
            wkv_b = (
                self.wkv_b.weight
                if self.wkv_b.scale is None
                else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            )
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
