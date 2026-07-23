"""
MoE（Mixture of Experts）详细教学演示（带逐行中文注释）
=========================================================

本文件从零实现了一个极简版 MoE 层，涵盖以下核心组件：
  1. MoEConfig          — 超参数配置
  2. FeedForwardExpert  — 单个 Expert（两层 MLP）
  3. TopKRouter         — 路由门控（Gating / Router）
  4. MoELayer           — 完整的稀疏 MoE 层（Routing → Dispatch → Compute → Combine）
  5. ToyMoEModel        — 将 MoE 嵌入简单网络的示例
  6. 训练循环 + 路由统计可视化

MoE（混合专家）核心思想
======================
  传统 FFN（前馈网络）对所有 token 使用**同一套参数**处理。
  MoE 则将 FFN 拆成多个"专家网络"（Expert），由**路由门控**（Router）决定
  每个 token 激活哪些 Expert。

  这带来了两个关键优势：
    1. 模型总参数量可以很大，但每次前向只激活少量 Expert（稀疏激活），
       计算量不会随参数量线性增长。
    2. 不同 Expert 可以"各司其职"，学到不同的数据模式。

  著名的 MoE 大模型：
    - Switch Transformer (Google, 2021) — top-1 routing
    - Mixtral 8x7B (Mistral, 2024)     — top-2 routing, 8 experts
    - DeepSeek-MoE 系列                — 细粒度路由

本 Demo 的特点
==============
  - 所有超参数故意设得很小，CPU 即可秒级运行
  - 用 toy 回归任务模拟"不同 token 需要不同 Expert"的场景
  - 重点展示路由决策过程、负载均衡损失、Expert 分工等核心概念
"""

import math  # 未直接使用，但保留以备后续可能的数学运算
from dataclasses import dataclass  # 数据类装饰器，简化配置类定义
from typing import Dict, Tuple  # 类型提示

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    """
    MoE 教学示例的超参数配置。

    所有超参数都故意设得很小（而非工业级的大模型），
    目的是让代码在 CPU 上也能快速跑通，便于读者理解 MoE 的结构和路由机制。
    """

    # ---------- 特征维度 ----------
    input_dim: int = 16
    # 每个 token 的输入特征维度。
    # 通常由上一层（如 Transformer 的 Attention 输出）决定。
    hidden_dim: int = 32
    # Expert 内部前馈网络的隐藏层维度。
    # 对应两层 MLP 中的中间层大小。
    output_dim: int = 16
    # Expert 的输出特征维度。为了方便演示，这里设成与 input_dim 相同，
    # 这样可以直接使用残差连接。

    # ---------- MoE 核心参数 ----------
    num_experts: int = 4
    # Expert 的数量。可以理解成 4 个不同专长的小网络。
    # 工业界：Mixtral 8x7B 有 8 个 Expert，Switch Transformer 有 2048 个。
    top_k: int = 2
    # 每个 token 最多路由给多少个 Expert（稀疏激活的关键参数）。
    # - top_k = 1：每个 token 只激活 1 个 Expert（Switch Transformer 的方案）
    # - top_k = 2：每个 token 激活 2 个 Expert（Mixtral 8x7B 的方案）
    # top_k 越小，计算效率越高；top_k 越大，专家集成效果越好，但计算量也越大。

    dropout: float = 0.1
    # Expert MLP 中使用的 Dropout 概率。防止过拟合。

    # ---------- 训练相关 ----------
    batch_size: int = 3    # 每次训练的样本数
    seq_len: int = 5       # 每个样本包含多少个 token（序列长度）
    train_steps: int = 20  # 训练步数（仅用于演示收敛趋势）
    learning_rate: float = 1e-3  # Adam 优化器学习率
    aux_loss_coef: float = 0.01  # 负载均衡辅助损失的权重系数
    seed: int = 7          # 随机种子，保证结果可复现


class FeedForwardExpert(nn.Module):
    """
    单个专家网络（Expert）。

    从结构上看，它就是一个两层 MLP（多层感知机）：
        Linear(input_dim → hidden_dim) → GELU → Dropout → Linear(hidden_dim → output_dim)

    这个结构和 Transformer FFN（前馈网络）的子层完全一致。

    关键理解：
    - 在传统 Transformer 中，所有 token 共享同一组 FFN 参数。
    - 在 MoE 中，有多个这样的 FFN（称为 Expert），
      每个 Expert 有自己的独立参数，可以学习不同的数据模式。
    - 例如：在翻译任务中，Expert A 可能擅长处理名词短语，
      Expert B 可能擅长处理动词短语。

    在真实的大模型（如 Mixtral 8x7B）中，Expert 的参数量与完整 FFN 相同，
    因此 8 个 Expert 的总参数量是 8 倍，但由于稀疏激活，
    每个 token 只经过 top_k（如 2）个 Expert，计算量只增加了约 2 倍。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        # 两层 MLP：先升维（增加表达能力），再降维（恢复输出维度）
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 升维：input_dim → hidden_dim（通常 4x）
            nn.GELU(),                          # 非线性激活函数，比 ReLU 更平滑
            nn.Dropout(dropout),                # 随机丢弃部分神经元的输出，防止过拟合
            nn.Linear(hidden_dim, output_dim),  # 降维：hidden_dim → output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [num_tokens_assigned_to_this_expert, input_dim]
               — 注意：不是所有 token 都会进入这个 Expert，
                 只有被 Router 分配过来的 token 才会出现这里。
        返回:
            out: [num_tokens_assigned_to_this_expert, output_dim]
        """
        # 直接过 MLP：两层线性变换中间夹 GELU 激活
        return self.net(x)


class TopKRouter(nn.Module):
    """
    Top-K 路由门控（Router / Gate）。

    Router 是 MoE 的"大脑"，负责决定每个 token 应该送给哪些 Expert。

    工作流程（三步）：
      1. 打分：用线性层将 token 特征映射成 num_experts 维的 logits
      2. 概率化：softmax 得到每 Expert 的激活概率
      3. 稀疏截断：只保留 top-k 个概率最大的 Expert，其余丢弃

    这就是 MoE 中"稀疏激活"（Sparse Activation）的核心：
      虽然总共有 num_experts 个 Expert，但每个 token 只激活其中的 top_k 个。
      例如 Mixtral 8x7B：num_experts=8, top_k=2 → 每个 token 只激活 25% 的 Expert。

    为什么要稀疏激活？
      - 计算效率：总参数量可以大，但每次前向的计算量 ≈ top_k / num_experts
      - 模型容量：参数量大意味着模型能记住更多知识
      - 专家分工：不同 Expert 可以专门化处理不同类型的数据
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k 不能大于 num_experts")

        self.num_experts = num_experts  # 专家总数
        self.top_k = top_k              # 每个 token 激活的专家数

        # 门控线性层：将 token 特征映射为 num_experts 维的"偏好分数"
        # 这就是 Router 唯一可学习的参数！
        # Router 通过训练学会：什么样的 token 应该激活哪个 Expert
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x: [num_tokens, input_dim] — 展平后的所有 token 表示
        返回:
            router_probs:  [num_tokens, num_experts]
                每个 token 在所有 Expert 上的完整概率分布（用于辅助损失计算）
            topk_probs:    [num_tokens, top_k]
                每个 token 被选中的 top-k Expert 的**重新归一化**权重
            topk_indices:  [num_tokens, top_k]
                每个 token 被选中的 Expert 编号（用于后续 dispatch）
        """
        # ---- Step 1: 计算路由分数（logits） ----
        # gate(x) 输出形状 [num_tokens, num_experts]
        # 每个元素表示 Router 对"该 token 适合该 Expert"的偏好程度
        router_logits = self.gate(x)  # 未归一化的原始分数

        # ---- Step 2: Softmax 得到概率分布 ----
        # 在 num_experts 维度上做 softmax，使得 Expert 概率之和为 1
        # router_probs[i, j] = exp(logits[i, j]) / sum(exp(logits[i, :]))
        router_probs = F.softmax(router_logits, dim=-1)

        # ---- Step 3: Top-K 截断（稀疏化的关键步骤） ----
        # torch.topk 返回：
        #   - values: 最大的 k 个概率值
        #   - indices: 对应的 Expert 编号
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # ---- Step 4: 重新归一化 ----
        # 截断后，只保留了 top-k 个概率，它们的和 < 1。
        # 我们需要对 top-k 的权重重新归一化，使得它们的和为 1。
        # 这样才能保证加权求和时输出值的量级正确。
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        return router_probs, topk_probs, topk_indices


class MoELayer(nn.Module):
    """
    完整的稀疏 MoE（Mixture of Experts）层。

    这是整个文件的核心模块。数据流可以分为 5 个步骤：

       输入 [batch, seq, dim]
            │
            ▼
     ┌─────────────────────┐
     │  Step 1: Router     │ ← TopKRouter：决定每个 token 去哪些 Expert
     │  (门控路由)          │ 输出：topk_probs, topk_indices
     └─────────────────────┘
            │
            ▼
     ┌─────────────────────┐
     │  Step 2: Dispatch   │ ← 按 Expert 编号收集 token
     │  (分发)              │ 每个 Expert 只接收分配给它的 token 子集
     └─────────────────────┘
            │
            ▼
     ┌─────────────────────┐
     │  Step 3: Compute    │ ← 每个 Expert 独立处理自己的 token
     │  (专家计算)          │ FeedForwardExpert(input) → output
     └─────────────────────┘
            │
            ▼
     ┌─────────────────────┐
     │  Step 4: Combine    │ ← 按 Router 权重加权合并多个 Expert 的输出
     │  (合并)              │ output = Σ(weight_i * expert_i(input))
     └─────────────────────┘
            │
            ▼
       输出 [batch, seq, dim] + aux_loss + stats

    与传统 FFN 的关键区别：
    - 传统 FFN：所有 token 经同一组参数处理
    - MoE：不同 token 激活不同 Expert，每个 Expert 有自己的独立参数
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config

        # ---- 路由门控（Router） ----
        # Router 是一个可学习的线性层，决定每个 token 的 Expert 分配
        self.router = TopKRouter(config.input_dim, config.num_experts, config.top_k)

        # ---- 专家网络列表（Experts） ----
        # nn.ModuleList 是 PyTorch 专门用于存放多个子模块的容器
        # 确保每个 Expert 的参数都被正确注册和追踪
        self.experts = nn.ModuleList(
            [
                FeedForwardExpert(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.output_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_experts)  # 创建 num_experts 个 Expert
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        MoE 层的前向传播。

        参数:
            x: [batch_size, seq_len, input_dim] — 输入 token 表示
        返回:
            combined_output: [batch_size, seq_len, output_dim] — MoE 处理后的输出
            aux_loss: 标量 — 负载均衡辅助损失（鼓励 Router 均匀分配）
            stats: Dict — 路由统计信息（用于教学和调试）
        """
        batch_size, seq_len, input_dim = x.shape
        num_tokens = batch_size * seq_len  # batch 中所有 token 的总数

        # ---- Step 1: 展平（方便路由操作） ----
        # 将 [batch, seq, dim] → [num_tokens, dim]
        # 路由是对每个 token 独立进行的，不需要区分它属于哪个样本
        flat_x = x.reshape(num_tokens, input_dim)

        # ---- Step 2: 路由计算 ----
        # router_probs:  [num_tokens, num_experts] — 完整的概率分布
        # topk_probs:    [num_tokens, top_k]       — top-k 的归一化权重
        # topk_indices:  [num_tokens, top_k]       — top-k 的 Expert 编号
        router_probs, topk_probs, topk_indices = self.router(flat_x)

        # ---- Step 3: 初始化输出缓冲区 ----
        # 每个 Expert 的贡献会累加到这个张量中
        # 形状 [num_tokens, output_dim]
        combined_output = torch.zeros(num_tokens, self.config.output_dim, device=x.device, dtype=x.dtype)

        # ---- Step 4: 构造 Dispatch Mask（分发掩码） ----
        # dispatch_mask: [num_tokens, top_k, num_experts]
        # 对于每个 token 的每个 top-k 槽位，one-hot 表示选中了哪个 Expert
        # 例如：topk_indices[i, 0] = 2 → dispatch_mask[i, 0, 2] = 1
        dispatch_mask = F.one_hot(topk_indices, num_classes=self.config.num_experts).to(x.dtype)

        # ---- Step 5: 逐个 Expert 处理 ----
        for expert_id, expert in enumerate(self.experts):
            # expert_mask: [num_tokens, top_k]
            # 值为 1 的位置表示"该 token 的该槽位选中了当前 Expert"
            expert_mask = dispatch_mask[:, :, expert_id]

            # 找出所有被分配给当前 Expert 的 token 索引和对应槽位
            # token_indices: [num_assigned] — 被分配的 token 在 flat_x 中的位置
            # slot_indices:  [num_assigned] — 对应 top-k 的哪个槽位
            token_indices, slot_indices = torch.nonzero(expert_mask, as_tuple=True)

            if token_indices.numel() == 0:
                # 没有 token 分配给当前 Expert，跳过
                continue

            # ---- Step 5a: 取出这些 token 的输入 ----
            # expert_input: [num_assigned, input_dim]
            expert_input = flat_x[token_indices]

            # ---- Step 5b: 当前 Expert 前向计算 ----
            # expert_output: [num_assigned, output_dim]
            expert_output = expert(expert_input)

            # ---- Step 5c: 加权累加到输出 ----
            # 同一个 token 可能被分配给多个 Expert（top_k > 1），
            # 因此需要按 Router 权重加权求和
            # expert_weight: [num_assigned, 1]
            expert_weight = topk_probs[token_indices, slot_indices].unsqueeze(-1)
            combined_output[token_indices] += expert_output * expert_weight

        # ---- Step 6: 恢复为原始形状 ----
        # 将 [num_tokens, output_dim] → [batch_size, seq_len, output_dim]
        combined_output = combined_output.view(batch_size, seq_len, self.config.output_dim)

        # 计算负载均衡辅助损失和路由统计信息
        aux_loss = self._load_balance_loss(router_probs, topk_indices)
        stats = self._collect_routing_stats(router_probs, topk_indices, topk_probs)

        return combined_output, aux_loss, stats

    def _load_balance_loss(self, router_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡辅助损失（Load Balancing Auxiliary Loss）。

        为什么需要这个损失？
        -------------------
        MoE 有一个经典的"崩溃"问题：如果 Router 自学习，
        它可能会倾向于把所有 token 都送给同一两个"强" Expert，
        导致其他 Expert 永远得不到训练 → 模型实际容量退化。

        解决思路：
          在总损失中加入一个辅助项，惩罚"分配不均衡"的情况。
          这个辅助项鼓励 Router 尽量均匀地将 token 分配给所有 Expert。

        本实现使用两个指标（参考 Switch Transformer 论文）：
          1. **importance**（重要性）：router_probs.mean(dim=0)
             — 所有 token 对每个 Expert 的平均概率，反映 Router 的"偏好"
          2. **load**（负载）：实际被选中的频率
             — 每个 Expert 实际处理了多少 token

        当 token 在所有 Expert 上均匀分布时，损失最小。
        系数 aux_loss_coef 控制辅助损失的强度（太小没用，太大会干扰主任务）。

        参考：Switch Transformer (Fedus et al., 2021) 中的负载均衡损失。
        """
        num_experts = self.config.num_experts

        # ---- 指标 1: Importance（平均概率分布） ----
        # router_probs: [num_tokens, num_experts]
        # importance:   [num_experts] — 每个 Expert 的"平均被偏好程度"
        importance = router_probs.mean(dim=0)

        # ---- 指标 2: Load（实际分配次数） ----
        # topk_indices: [num_tokens, top_k]
        # 先展平为 [num_tokens * top_k]，每个元素是 Expert 编号
        flat_selected = topk_indices.reshape(-1)
        # one-hot 后求平均 → [num_experts]，每个 Expert 被选中的平均频率
        load = F.one_hot(flat_selected, num_classes=num_experts).float().mean(dim=0)

        # ---- 理想目标：均匀分布 ----
        # 完美均衡时，每个 Expert 应该被选中的概率 = 1 / num_experts
        target = torch.full_like(importance, 1.0 / num_experts)

        # 两个损失都用 MSE（均方误差）衡量与均匀分布的差距
        importance_loss = F.mse_loss(importance, target)  # Router 偏好的均匀度
        load_loss = F.mse_loss(load, target)              # 实际分配均匀度

        # 返回两者之和作为辅助损失
        return importance_loss + load_loss

    def _collect_routing_stats(
        self,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        收集路由统计信息，便于教学观察和调试。

        统计项：
        - mean_router_prob_per_expert：每个 Expert 的平均路由概率
          （Router 的"偏好"分布）
        - tokens_per_expert：每个 Expert 实际处理的 token 数量
          （实际分配情况）
        - mean_topk_weight：top-k 槽位上权重的平均值
          （路由"置信度"：高值表示 Router 非常确信这个分配）
        """

        # 统计每个 Expert 实际被分配的 token 数
        flat_selected = topk_indices.reshape(-1)
        tokens_per_expert = F.one_hot(flat_selected, num_classes=self.config.num_experts).sum(dim=0)

        stats = {
            "mean_router_prob_per_expert": router_probs.mean(dim=0).detach().cpu(),
            "tokens_per_expert": tokens_per_expert.detach().cpu(),
            "mean_topk_weight": topk_probs.mean(dim=0).detach().cpu(),
        }
        return stats


class ToyMoEModel(nn.Module):
    """
    一个极简模型，用来演示"MoE 层如何嵌入到普通网络中"。

    结构：
        input → LayerNorm → MoE → Linear + Residual → output

    说明：
    - LayerNorm 在 MoE 之前，确保输入分布稳定（Pre-LN 风格）
    - 残差连接允许梯度直接流过，缓解深层梯度消失
    - 最后的 Linear 是对 MoE 输出做进一步变换

    在真实大模型中，MoE 层通常替换的是 Transformer 的 FFN（前馈网络）子层。
    典型的 MoE Transformer 结构：
        Attention → LayerNorm → MoE-FFN → Residual → LayerNorm → ...
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.input_dim)           # LayerNorm：稳定训练
        self.moe = MoELayer(config)                           # MoE 层（核心）
        self.output_proj = nn.Linear(config.output_dim, config.output_dim)  # 输出投影

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # 保存残差（Pre-LN 风格：先 Norm，再 MoE，再加残差）
        residual = x

        # MoE 前向：先 LayerNorm，再进 MoE 层
        moe_out, aux_loss, stats = self.moe(self.norm(x))

        # 残差连接 + 输出投影
        # 注意：这里 output_dim == input_dim，所以可以直接相加
        # 如果维度不同，需要先用 Linear 投影对齐
        out = self.output_proj(moe_out) + residual

        return out, aux_loss, stats


def build_toy_regression_batch(config: MoEConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造玩具回归任务的 batch。

    任务设计：
    - 输入 x 是随机噪声
    - 目标 target 根据输入特征均值的正负，采用不同的变换函数
        - 均值 > 0：target = x * 1.5 + 0.2  （正相关模式）
        - 均值 ≤ 0：target = x * (-0.8) - 0.1 （负相关模式）

    为什么这样设计？
    - 这个任务模拟了"不同 token 需要不同处理"的场景
    - 正均值 token 和负均值 token 的最优变换不同
    - 如果模型只有一个 FFN，学会一种模式可能会干扰另一种
    - 而 MoE 可以让 Expert A 专攻正均值模式，Expert B 专攻负均值模式
    - 这样在训练后，我们可以看到 Expert 的"专业化"现象

    注：这个任务非常简化，目的是让读者快速看到路由机制在起作用。
    """

    # 生成随机输入 [batch_size, seq_len, input_dim]
    x = torch.randn(config.batch_size, config.seq_len, config.input_dim, device=device)

    # 根据输入特征的均值，分段生成目标
    # mean_feature: [batch_size, seq_len, 1] — 每个 token 的特征均值
    mean_feature = x.mean(dim=-1, keepdim=True)

    # 正均值 → 线性放大 (1.5x) + 正向偏移 (+0.2)
    # 负均值 → 线性缩小并反向 (-0.8x) + 负向偏移 (-0.1)
    target = torch.where(mean_feature > 0, x * 1.5 + 0.2, x * -0.8 - 0.1)

    return x, target


def demo_forward_pass(config: MoEConfig, device: torch.device) -> None:
    """
    演示 MoE 的一次完整前向传播，并打印关键张量的形状和路由统计信息。

    目的：
    1. 展示输入/输出维度变换
    2. 观察未训练时 Router 的初始分配（通常是均匀的，因为线性层刚初始化）
    3. 验证负载均衡损失的计算
    """

    print("=" * 80)
    print("1. MoE 单次前向传播演示（未训练时）")
    print("=" * 80)

    model = ToyMoEModel(config).to(device)
    x, _ = build_toy_regression_batch(config, device)
    out, aux_loss, stats = model(x)

    print(f"输入张量形状: {tuple(x.shape)}")
    print(f"  └─ [batch_size={config.batch_size}, seq_len={config.seq_len}, input_dim={config.input_dim}]")
    print(f"输出张量形状: {tuple(out.shape)}")
    print(f"  └─ [batch_size={config.batch_size}, seq_len={config.seq_len}, output_dim={config.output_dim}]")
    print(f"总 token 数: {config.batch_size * config.seq_len}（所有样本的 token 合并路由）")
    print()
    print(f"辅助负载均衡损失: {aux_loss.item():.6f}")
    print("  └─ 越小表示分配越均匀，初始时通常很小（线性层初始化均匀）")
    print()
    print("每个 Expert 平均获得的 Router 概率（完整分布）:")
    for i, prob in enumerate(stats["mean_router_prob_per_expert"]):
        print(f"  Expert {i}: {prob:.4f}")
    print("  └─ 表示 Router 平均有多少「偏好」分配给每个 Expert")
    print()
    print("每个 Expert 实际接收的 token 数:")
    for i, cnt in enumerate(stats["tokens_per_expert"]):
        print(f"  Expert {i}: {int(cnt.item())} 个 token")
    print(f"  └─ 总 token 应等于 {config.batch_size * config.seq_len} × top_k({config.top_k}) = {config.batch_size * config.seq_len * config.top_k}")
    print()
    print("top-k 槽位上的平均权重:")
    for i, w in enumerate(stats["mean_topk_weight"]):
        print(f"  槽位 {i}: {w:.4f}")
    print("  └─ 槽位 0 是第 1 偏好的 Expert，槽位 1 是第 2 偏好")


def demo_training_loop(config: MoEConfig, device: torch.device) -> None:
    """
    演示极简的 MoE 训练循环。

    训练目标：
    - 主损失：MSE 回归损失 — 模型预测与目标之间的差距
    - 辅助损失：负载均衡损失 — 鼓励 Router 将 token 均匀分配给所有 Expert
    - 总损失：main_loss + aux_loss_coef * aux_loss

    训练中会观察：
    1. main_loss 逐渐下降（模型在学习映射函数）
    2. aux_loss 的变化（Router 的分配策略在调整）
    3. tokens_per_expert 的分布（Expert 是否形成了"分工"）

    MoE 训练的特殊之处：
    - Router 和 Expert 是**联合训练**的
    - Router 学习"什么 token 该去什么 Expert"
    - Expert 学习"在自己的专长领域做到最好"
    """

    print("\n" + "=" * 80)
    print("2. MoE toy training demo")
    print("=" * 80)

    model = ToyMoEModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"{'Step':>5s} | {'Main Loss':>10s} | {'Aux Loss':>9s} | {'Token 分配 (各 Expert)':>30s}")
    print("-" * 70)

    for step in range(1, config.train_steps + 1):
        # ---- Step 1: 构造 toy batch ----
        x, target = build_toy_regression_batch(config, device)

        # ---- Step 2: 前向传播 ----
        pred, aux_loss, stats = model(x)

        # ---- Step 3: 计算损失 ----
        # 主任务：MSE（均方误差），适合回归任务
        main_loss = F.mse_loss(pred, target)

        # 总损失 = 主损失 + 辅助损失 × 系数
        # aux_loss_coef 通常很小（0.01 ~ 0.001），因为辅助损失只是"辅助"
        total_loss = main_loss + config.aux_loss_coef * aux_loss

        # ---- Step 4: 反向传播 + 参数更新 ----
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---- Step 5: 打印训练信息 ----
        if step == 1 or step % 5 == 0 or step == config.train_steps:
            token_dist = stats["tokens_per_expert"].tolist()
            print(
                f"{step:5d} | {main_loss.item():10.6f} | {aux_loss.item():9.6f} | {token_dist}"
            )

    # ---- 训练后分析：查看 Expert 专业化程度 ----
    print("\n" + "-" * 70)
    print("训练后路由分析（Expert 是否形成了分工？）")
    print("-" * 70)

    with torch.no_grad():
        x, _ = build_toy_regression_batch(config, device)
        _, _, stats = model(x)

    print("\n每个 Expert 的平均 Router 概率（归一化偏好）:")
    for i, prob in enumerate(stats["mean_router_prob_per_expert"]):
        bar = "█" * int(prob * 50) + "░" * (50 - int(prob * 50))
        print(f"  Expert {i}: {prob:.3f} {bar}")

    print("\n每个 Expert 实际处理的 token 数:")
    for i, cnt in enumerate(stats["tokens_per_expert"]):
        n = int(cnt.item())
        bar = "█" * n + "░" * (max(0, config.batch_size * config.seq_len - n))
        print(f"  Expert {i}: {n:3d} 个 token {bar}")

    print("\n📊 分析：")
    print("   - 如果各 Expert 的 token 数接近，说明负载均衡有效")
    print("   - 如果不同 Expert 的分配出现明显差异，说明 Expert 正在形成分工")
    print("   - 经过充分训练后，不同 Expert 会对不同特征的 token 产生偏好")


def main() -> None:
    """
    主函数：运行完整的 MoE 教学演示。

    流程：
    1. 设置配置和随机种子
    2. 演示单次前向传播（查看路由初始状态）
    3. 演示极简训练循环（观察 Expert 分工形成过程）
    4. 打印关键概念回顾 + 进阶指引
    """

    config = MoEConfig()
    torch.manual_seed(config.seed)  # 固定随机种子，确保每次运行结果一致

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print("\nMoE 配置:", config)
    print("-" * 50)
    print(f"  Expert 总数: {config.num_experts}")
    print(f"  每个 token 激活: {config.top_k} 个 Expert")
    print(f"  稀疏激活比例: {config.top_k / config.num_experts * 100:.0f}%")
    print(f"  总参数量 ≈ 单 Expert × {config.num_experts}")
    print(f"  前向计算量 ≈ 单 Expert × {config.top_k}")
    print("-" * 50)

    demo_forward_pass(config, device)
    demo_training_loop(config, device)

    print("\n" + "=" * 80)
    print("3. 关键概念回顾")
    print("=" * 80)
    print("📌 MoE 的核心思想：稀疏激活（Sparse Activation）")
    print("   不是把网络简单加宽，而是让不同 token 稀疏地激活不同 Expert。")
    print("   总参数量可以很大，但单次前向的计算量 ≈ (top_k / num_experts)。")
    print()
    print("📌 MoE 的四个关键步骤：")
    print("   1. Router（路由）: 决定每个 token 去哪些 Expert")
    print("   2. Dispatch（分发）: 按 Expert 收集对应的 token")
    print("   3. Compute（计算）: 每个 Expert 独立处理自己的 token")
    print("   4. Combine（合并）: 按 Router 权重加权求和")
    print()
    print("📌 负载均衡（Load Balancing）的重要性：")
    print("   没有辅助损失时，Router 可能会「崩溃」——")
    print("   所有 token 都涌向少数 Expert，其他 Expert 被浪费。")
    print("   辅助损失通过惩罚不均衡分配来解决这个问题。")
    print()
    print("📌 MoE 的真实应用：")
    print("   - Mixtral 8x7B (Mistral): 8 Expert, top-2, 46.7B 总参/12.9B 活跃参")
    print("   - Switch Transformer (Google): 2048 Expert, top-1, 1.6T 总参")
    print("   - DeepSeek-V2 / V3: 细粒度 MoE, 多级路由")
    print("   - Qwen2-MoE / GPT-4 (传闻): 混合专家架构")
    print()
    print("📌 本 Demo 的局限：")
    print("   - 去掉了 Batch 和 Sequence 维度，所有 token 统一路由（真实系统中 token 独立路由）")
    print("   - 没有处理 capacity（每个 Expert 的 token 容量上限）")
    print("   - 任务过于简单，Expert 专业化程度有限")
    print("   - 没有实现 Token Dropping（超载时丢弃 token 的策略）")
    print()
    print("🔥 推荐进阶阅读：")
    print("   1. MoE 综述: https://arxiv.org/abs/2402.07870")
    print("   2. Switch Transformer: https://arxiv.org/abs/2101.03961")
    print("   3. Mixtral 8x7B: https://arxiv.org/abs/2401.04088")
    print("   4. Outrageously Large Neural Networks (稀疏门控 MoE): https://arxiv.org/abs/1701.06538")


if __name__ == "__main__":
    main()
