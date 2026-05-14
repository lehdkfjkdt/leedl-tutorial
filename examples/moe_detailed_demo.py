import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    """
    MoE 示例的超参数配置。

    这里把所有关键超参数集中管理，便于教学时统一查看和修改。
    """

    input_dim: int = 16
    # 每个 token 的输入特征维度。
    hidden_dim: int = 32
    # Expert 内部前馈网络的隐藏层维度。
    output_dim: int = 16
    # 每个 token 输出的特征维度；为了方便演示，这里设成与输入相同。
    num_experts: int = 4
    # Expert 的数量。可以把它理解成 4 个不同专长的小网络。
    top_k: int = 2
    # 每个 token 最多路由给多少个 Expert。常见选择是 top-1 或 top-2。
    dropout: float = 0.1
    # Expert MLP 中使用的 dropout。
    batch_size: int = 3
    # toy demo 里的 batch size。
    seq_len: int = 5
    # toy demo 中每个样本包含多少个 token。
    train_steps: int = 20
    # 演示训练循环的步数。
    learning_rate: float = 1e-3
    # 演示训练时使用的学习率。
    aux_loss_coef: float = 0.01
    # 负载均衡辅助损失的权重。
    seed: int = 7
    # 随机种子，保证示例结果可复现。


class FeedForwardExpert(nn.Module):
    """
    单个 Expert。

    从结构上看，它就是一个两层 MLP：
    Linear -> GELU -> Dropout -> Linear

    在真实大模型里，不同 Expert 会学到不同的数据模式；
    在本示例里，我们重点看“如何分配 token 给不同 Expert”。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [num_tokens_assigned_to_this_expert, input_dim]

        返回:
            out: [num_tokens_assigned_to_this_expert, output_dim]
        """

        return self.net(x)


class TopKRouter(nn.Module):
    """
    Router 负责决定“每个 token 应该送给哪些 Expert”。

    做法很直接：
    1. 用一个线性层把 token 映射成 num_experts 维的打分。
    2. 对所有 Expert 的分数做 softmax，得到概率分布。
    3. 只保留 top-k 个概率最大的 Expert，其他位置置零。

    这就是 MoE 里最关键的“稀疏激活”思想：
    虽然总共有很多 Expert，但每个 token 实际只走很少几条路径。
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k 不能大于 num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x: [num_tokens, input_dim]

        返回:
            router_probs:
                [num_tokens, num_experts]
                对所有 Expert 的完整 softmax 概率，用于分析和辅助损失。
            topk_probs:
                [num_tokens, top_k]
                每个 token 被选中的 top-k Expert 的归一化权重。
            topk_indices:
                [num_tokens, top_k]
                每个 token 被选中的 Expert 编号。
        """

        # logits 表示 router 对每个 token、每个 expert 的原始偏好分数。
        router_logits = self.gate(x)

        # softmax 后得到完整的概率分布，所有 expert 的概率和为 1。
        router_probs = F.softmax(router_logits, dim=-1)

        # 从完整分布里取出 top-k 个 expert。
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # top-k 截断后，剩余概率和不再等于 1，因此需要重新归一化。
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return router_probs, topk_probs, topk_indices


class MoELayer(nn.Module):
    """
    一个完整的 Mixture of Experts 层。

    数据流可以概括成：
    1. 输入 token 先经过 Router，决定去哪些 Expert。
    2. 按 expert id 收集对应 token。
    3. 每个 Expert 独立处理自己的 token 子集。
    4. 再按照 router 权重，把多个 Expert 的结果加权合并回原顺序。

    这和普通 MLP 最大的不同在于：
    普通 MLP 对所有 token 使用同一套参数；
    MoE 会让不同 token 激活不同的专家网络。
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.router = TopKRouter(config.input_dim, config.num_experts, config.top_k)
        self.experts = nn.ModuleList(
            [
                FeedForwardExpert(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.output_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        参数:
            x: [batch_size, seq_len, input_dim]

        返回:
            combined_output: [batch_size, seq_len, output_dim]
            aux_loss: 标量，负载均衡辅助损失
            stats: 记录路由统计信息，便于教学和调试
        """

        batch_size, seq_len, input_dim = x.shape
        num_tokens = batch_size * seq_len

        # 为了简化路由实现，先把 [batch, seq, dim] 展平成 [num_tokens, dim]。
        flat_x = x.reshape(num_tokens, input_dim)

        # router_probs 是完整分布；topk_probs / topk_indices 才是实际执行的稀疏路由结果。
        router_probs, topk_probs, topk_indices = self.router(flat_x)

        # 初始化最终输出。后续每个 expert 的贡献都会累加到这里。
        combined_output = torch.zeros(num_tokens, self.config.output_dim, device=x.device, dtype=x.dtype)

        # 构造一个 one-hot mask，表示某个 token 是否被派给某个 expert。
        # dispatch_mask: [num_tokens, top_k, num_experts]
        dispatch_mask = F.one_hot(topk_indices, num_classes=self.config.num_experts).to(x.dtype)

        # 对每个 expert 单独处理。
        for expert_id, expert in enumerate(self.experts):
            # expert_mask: [num_tokens, top_k]
            # 其中为 1 的位置表示“这个 token 的某个路由槽位选中了当前 expert”。
            expert_mask = dispatch_mask[:, :, expert_id]

            # 找出所有被分给当前 expert 的 (token_id, slot_id)。
            token_indices, slot_indices = torch.nonzero(expert_mask, as_tuple=True)

            if token_indices.numel() == 0:
                # 没有 token 分配给当前 expert，就跳过它。
                continue

            # 取出这些 token 的输入，交给当前 expert 计算。
            expert_input = flat_x[token_indices]
            expert_output = expert(expert_input)

            # 同一个 token 可能被送到多个 expert。
            # 因此我们要按对应槽位的 router 权重进行加权求和。
            expert_weight = topk_probs[token_indices, slot_indices].unsqueeze(-1)
            combined_output[token_indices] += expert_output * expert_weight

        # reshape 回原来的 [batch, seq, dim] 结构，方便接到后续网络。
        combined_output = combined_output.view(batch_size, seq_len, self.config.output_dim)

        aux_loss = self._load_balance_loss(router_probs, topk_indices)
        stats = self._collect_routing_stats(router_probs, topk_indices, topk_probs)
        return combined_output, aux_loss, stats

    def _load_balance_loss(self, router_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        计算一个简化版负载均衡损失。

        目标：
        1. 不希望所有 token 都挤到少数 Expert。
        2. 也不希望 Router 的概率分布长期极端偏向个别 Expert。

        这里使用两个量：
        - importance: 平均概率质量，表示 router 从“信心”上偏爱谁。
        - load: 实际被选中的频率，表示 token 从“数量”上流向谁。

        当两者更均匀时，loss 会更小。
        """

        num_tokens = router_probs.size(0)

        # importance: [num_experts]
        importance = router_probs.mean(dim=0)

        # 统计每个 expert 实际被选中的次数。
        # 先拉平成 [num_tokens * top_k]，再做 one-hot 统计。
        flat_selected = topk_indices.reshape(-1)
        load = F.one_hot(flat_selected, num_classes=self.config.num_experts).float().mean(dim=0)

        # 均匀分布是最理想的目标。
        target = torch.full_like(importance, 1.0 / self.config.num_experts)

        importance_loss = F.mse_loss(importance, target)
        load_loss = F.mse_loss(load, target)
        return importance_loss + load_loss

    def _collect_routing_stats(
        self,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        收集一些对教学很有帮助的统计量。
        """

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
    一个很小的模型，用来演示“MoE 层如何嵌入到普通网络里”。

    结构：
    input -> layernorm -> MoE -> residual -> linear
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.input_dim)
        self.moe = MoELayer(config)
        self.output_proj = nn.Linear(config.output_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        residual = x
        moe_out, aux_loss, stats = self.moe(self.norm(x))

        # 因为 output_dim 和 input_dim 相同，这里可以直接做残差连接。
        out = self.output_proj(moe_out) + residual
        return out, aux_loss, stats


def build_toy_regression_batch(config: MoEConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造一个简单的 toy 回归任务。

    我们让模型学习一个“分段函数”：
    - 当 token 特征均值较大时，更像 expert A 需要处理的模式
    - 当 token 特征均值较小时，更像 expert B 需要处理的模式

    虽然这个任务很人工，但很适合演示：
    不同 token 的确可能需要不同的子网络来处理。
    """

    x = torch.randn(config.batch_size, config.seq_len, config.input_dim, device=device)

    # 根据输入均值生成不同风格的目标，让路由机制有机会学会“分工”。
    mean_feature = x.mean(dim=-1, keepdim=True)
    target = torch.where(mean_feature > 0, x * 1.5 + 0.2, x * -0.8 - 0.1)
    return x, target


def demo_forward_pass(config: MoEConfig, device: torch.device) -> None:
    """
    演示一次前向传播，并打印关键张量信息。
    """

    print("=" * 80)
    print("1. MoE 单次前向传播演示")
    print("=" * 80)

    model = ToyMoEModel(config).to(device)
    x, _ = build_toy_regression_batch(config, device)
    out, aux_loss, stats = model(x)

    print(f"输入张量形状: {tuple(x.shape)}")
    print(f"输出张量形状: {tuple(out.shape)}")
    print(f"辅助负载均衡损失: {aux_loss.item():.6f}")
    print("每个 Expert 平均获得的 router 概率:")
    print(stats["mean_router_prob_per_expert"])
    print("每个 Expert 实际接收的 token 数:")
    print(stats["tokens_per_expert"])
    print("top-k 槽位上的平均权重:")
    print(stats["mean_topk_weight"])


def demo_training_loop(config: MoEConfig, device: torch.device) -> None:
    """
    演示一个极简训练循环。

    目标不是追求很好的效果，而是让读者看到：
    1. 主任务损失怎么和 MoE 的辅助损失组合。
    2. 路由统计会如何随训练发生变化。
    """

    print("\n" + "=" * 80)
    print("2. MoE toy training demo")
    print("=" * 80)

    model = ToyMoEModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for step in range(1, config.train_steps + 1):
        x, target = build_toy_regression_batch(config, device)
        pred, aux_loss, stats = model(x)

        # 主任务使用 MSE 回归损失。
        main_loss = F.mse_loss(pred, target)

        # 总损失 = 主任务损失 + 辅助负载均衡损失。
        total_loss = main_loss + config.aux_loss_coef * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % 5 == 0 or step == config.train_steps:
            print(
                f"step={step:02d} | "
                f"main_loss={main_loss.item():.6f} | "
                f"aux_loss={aux_loss.item():.6f} | "
                f"expert_tokens={stats['tokens_per_expert'].tolist()}"
            )


def main() -> None:
    config = MoEConfig()
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print("MoE 配置:", config)

    demo_forward_pass(config, device)
    demo_training_loop(config, device)

    print("\n" + "=" * 80)
    print("3. 关键概念回顾")
    print("=" * 80)
    print("MoE 的核心不是把网络简单加宽，而是让不同 token 稀疏地激活不同 Expert。")
    print("这样做的好处是：总参数量可以很大，但单次前向真正参与计算的参数量不会线性增长。")
    print("实际大模型还会加入 capacity 限制、token dropping、更复杂的负载均衡策略等机制。")


if __name__ == "__main__":
    main()
