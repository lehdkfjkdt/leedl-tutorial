"""
RLHF 训练大模型教学演示（带详细中文注释）
=========================================

本文件实现一个"能在 CPU 上跑通"的极简版大语言模型训练闭环，覆盖：
  1. Tokenizer 与 toy 指令数据
  2. Decoder-Only Transformer（大模型主干的最小原型）
  3. 语言模型预训练（Next Token Prediction）
  4. SFT（Supervised Fine-Tuning，监督微调）
  5. 奖励模型（Reward Model）
  6. PPO 风格的强化学习对齐（RLHF）

为什么这是"教学版"而不是真实工业训练脚本？
  - 真实大模型训练往往需要数十亿到数万亿 token、海量 GPU、分布式并行和复杂工程栈。
  - 这里的目标不是追求性能，而是让读者看懂：一个大模型从预训练到对齐的大致流程到底怎么串起来。
  - 因此我们把所有超参数都缩小到 CPU 也能秒级运行的规模。

如果只记一句话，可以把本 Demo 理解成：
  先用自回归语言建模学会"说话"，再用 SFT 学会"按指令回答"，
  然后训练奖励模型区分"更好"与"更差"的回答，最后用 PPO 让策略模型朝高奖励方向更新。
"""

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RLHFConfig:
    """
    教学版 RLHF 配置。

    所有维度都故意设得很小，方便在 CPU 上快速跑通。
        真实工业训练时，这里的很多数字会扩大几个数量级：
            - vocab_size 可能是 3 万到 20 万
            - hidden_size 可能是 2048 / 4096 / 8192
            - num_layers 可能是 24 / 48 / 80+
            - max_seq_len 可能是 4K / 8K / 32K 甚至更长
    """

        # ---------- 模型结构 ----------
    vocab_size: int = 33
    max_seq_len: int = 16
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_hidden_size: int = 128
    dropout: float = 0.1

        # ---------- 特殊 token ----------
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

        # ---------- 训练步数 ----------
    pretrain_steps: int = 80
    sft_steps: int = 80
    reward_steps: int = 120
    ppo_steps: int = 60

        # ---------- batch 大小 ----------
    pretrain_batch_size: int = 4
    sft_batch_size: int = 4
    reward_batch_size: int = 4
    ppo_batch_size: int = 4

        # ---------- 学习率 ----------
    learning_rate: float = 3e-3
    reward_learning_rate: float = 2e-3
    ppo_learning_rate: float = 1e-3

        # ---------- PPO 相关超参数 ----------
    ppo_clip_range: float = 0.2
    kl_coef: float = 0.05
    value_coef: float = 0.5
    max_new_tokens: int = 4
    temperature: float = 1.0

    seed: int = 1234


class ToyTokenizer:
    """
    一个极小的空格分词器。

    为了让整个 demo 自包含，我们手写一个固定词表，而不是依赖 BPE / SentencePiece。
    真正的大模型会用更复杂的子词分词器，但这里不影响理解训练流程。
    """

    def __init__(self) -> None:
        # 词表故意设计成非常小，并且只覆盖 demo 里的教学样例。
        # 真实 tokenizer 通常不会把 "hello"、"friend" 这种整词全手写进列表，
        # 而是通过 BPE / SentencePiece 自动学习子词单元。
        vocab = [
            "[PAD]",
            "[BOS]",
            "[EOS]",
            "user",
            "assistant",
            "greet",
            "math",
            "two",
            "plus",
            "explain",
            "transformer",
            "code",
            "python",
            "loop",
            "safe",
            "advice",
            "stress",
            "hello",
            "friend",
            "four",
            "five",
            "attention",
            "model",
            "for",
            "item",
            "in",
            "items",
            "breathe",
            "slowly",
            "ignore",
            "random",
            "answer",
            "calm",
        ]
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        # 用最简单的空格切分。
        # 因为词表固定，所以输入文本必须只包含 vocab 中出现过的 token。
        return [self.vocab[token] for token in text.split()]

    def decode(self, token_ids: Sequence[int]) -> str:
        # 解码时忽略特殊 token，避免打印时出现 [BOS] / [EOS] 等控制符号。
        tokens = []
        for idx in token_ids:
            token = self.id_to_token[idx]
            if token in {"[PAD]", "[BOS]", "[EOS]"}:
                continue
            tokens.append(token)
        return " ".join(tokens)


class CausalSelfAttention(nn.Module):
    """
    标准多头因果自注意力。

    与 BERT 的双向注意力不同，这里使用 causal mask，
    确保每个位置只能看到它左边和自己，不能偷看未来 token。
    这正是 GPT 类大语言模型的核心约束。
    """

    def __init__(self, config: RLHFConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]
        # 多头注意力本质上是把总隐藏维拆成多个子空间并行处理。
        batch_size, seq_len, hidden_size = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # _split_heads 的逆操作：把多个头重新拼回 hidden_size。
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Q / K / V 都来自当前序列，所以这是 self-attention。
        q = self._split_heads(self.q_proj(hidden_states))
        k = self._split_heads(self.k_proj(hidden_states))
        v = self._split_heads(self.v_proj(hidden_states))

        # 缩放点积注意力：score 越大，说明 query 与 key 越匹配。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attention_mask 同时包含：
        # 1. causal mask：禁止看未来
        # 2. padding mask：忽略补齐位置
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        # 用注意力权重对 V 做加权求和，得到每个位置聚合后的上下文表示。
        context = torch.matmul(probs, v)
        return self.out_proj(self._merge_heads(context))


class DecoderBlock(nn.Module):
    """
    Decoder-Only Transformer 的单层 Block。

        结构是典型的：
            LayerNorm -> Causal Self-Attention -> Residual
            LayerNorm -> FFN -> Residual

        这是 GPT 类模型最常见的基本积木块。
    """

    def __init__(self, config: RLHFConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(config.ffn_hidden_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 先做注意力子层，再做 FFN 子层；每个子层外面都包一层残差连接。
        attn_out = self.attn(self.attn_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + self.attn_dropout(attn_out)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class TinyDecoderBackbone(nn.Module):
    """
    一个极简版 Decoder-Only Transformer 主干。

        可以把它理解成：
            token id -> token embedding + position embedding -> 多层 decoder block -> hidden states

        这里只实现最核心的前向结构，不包含 KV Cache、RoPE、GQA、FlashAttention 等工业优化。
    """

    def __init__(self, config: RLHFConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # 下三角矩阵表示：当前位置只能关注它自己以及左边的位置。
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # padding 位置为 0，后续在 attention score 上会被屏蔽掉。
        padding_mask = (input_ids != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)
        return causal_mask & padding_mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # 所有样本共享同一套位置编号 0,1,2,...,seq_len-1。
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Decoder-only 模型的输入表示 = token embedding + position embedding。
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.dropout(hidden_states)
        attention_mask = self.build_attention_mask(input_ids)

        # 逐层提炼上下文化表示。
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return self.final_norm(hidden_states)


class TinyCausalLM(nn.Module):
    """
    语言模型头：根据每个位置的隐藏状态预测下一个 token。

        训练时常见的做法是：
            输入 x_0, x_1, ..., x_{t-1}
            预测 x_1, x_2, ..., x_t

        也就是把输入和标签错开一位，这就是 next-token prediction。
    """

    def __init__(self, config: RLHFConfig) -> None:
        super().__init__()
        self.backbone = TinyDecoderBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.backbone(input_ids)
        logits = self.lm_head(hidden_states)
        return logits, hidden_states


class TinyRewardModel(nn.Module):
    """
    奖励模型：给整个 prompt-response 序列打一个分数。

    真实系统里常见做法是：
      - 先从 SFT / LM 主干初始化
      - 在人类偏好数据上做 pairwise ranking
    这里也沿用这一路线。
    """

    def __init__(self, config: RLHFConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = TinyDecoderBackbone(config)
        self.reward_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(input_ids)
        # 这里取最后一个非 padding token 的隐藏状态做序列级打分。
        # 可以理解为：用整段 prompt-response 的最终语义表示来估计回答质量。
        last_hidden = gather_last_nonpad_hidden(hidden_states, input_ids, self.config.pad_token_id)
        return self.reward_head(last_hidden).squeeze(-1)


class TinyPolicyValueModel(nn.Module):
    """
    PPO 阶段使用的策略 + 价值网络。

    - lm_head 负责给出 token 概率分布（策略）
    - value_head 负责估计当前序列的价值（baseline）
    """

    def __init__(self, config: RLHFConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = TinyDecoderBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.backbone(input_ids)
        logits = self.lm_head(hidden_states)
        # PPO 中 value head 不是预测下一个 token，而是估计当前整条序列的期望回报。
        last_hidden = gather_last_nonpad_hidden(hidden_states, input_ids, self.config.pad_token_id)
        values = self.value_head(last_hidden).squeeze(-1)
        return logits, values, hidden_states


def gather_last_nonpad_hidden(hidden_states: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    取出每个样本最后一个非 padding token 对应的隐藏状态。

    为什么不是直接取最后一列？
    因为 batch 对齐后，最后一列常常是 [PAD]，它不代表真实文本内容。
    """

    lengths = (input_ids != pad_token_id).sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
    return hidden_states[batch_indices, lengths]


def set_seed(seed: int) -> None:
    # 固定随机种子，保证每次运行采样顺序和初始化更可复现。
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_sequences(sequences: List[List[int]], pad_value: int, device: torch.device) -> torch.Tensor:
    # 将不同长度的序列补齐成同一长度，方便拼成 batch tensor。
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long, device=device)


def build_demo_datasets() -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    返回三个数据集：
      1. 预训练文本
      2. SFT 指令-回答对
      3. 奖励模型偏好三元组 (prompt, chosen, rejected)
    """

    # 预训练语料：只让模型先学到最基本的 token 共现规律和句式模式。
    pretrain_texts = [
        "user greet assistant hello friend",
        "user math two plus two assistant four",
        "user explain transformer assistant attention model",
        "user code python loop assistant for item in items",
        "user safe advice stress assistant breathe slowly calm",
        "assistant hello friend",
        "assistant attention model",
        "assistant for item in items",
        "assistant breathe slowly calm",
        "assistant four",
    ]

    # SFT 数据：告诉模型“遇到某个指令时，理想回答长什么样”。
    sft_pairs = [
        ("user greet", "assistant hello friend"),
        ("user math two plus two", "assistant four"),
        ("user explain transformer", "assistant attention model"),
        ("user code python loop", "assistant for item in items"),
        ("user safe advice stress", "assistant breathe slowly calm"),
    ]

    # 偏好数据：同一个 prompt 下，chosen 比 rejected 更符合人类偏好。
    preference_triples = [
        ("user greet", "assistant hello friend", "assistant random answer"),
        ("user math two plus two", "assistant four", "assistant five"),
        ("user explain transformer", "assistant attention model", "assistant random answer"),
        ("user code python loop", "assistant for item in items", "assistant random answer"),
        ("user safe advice stress", "assistant breathe slowly calm", "assistant ignore stress"),
    ]
    return pretrain_texts, sft_pairs, preference_triples


def make_pretrain_batch(
    texts: Sequence[str],
    tokenizer: ToyTokenizer,
    config: RLHFConfig,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造预训练 batch。

    返回：
      - input_ids:  除去最后一个 token 的序列
      - labels:     除去第一个 token 的序列

    这正好形成“输入预测下一个 token”的监督信号。
    """
    sequences = []
    for _ in range(batch_size):
        text = random.choice(texts)
        # 在句首加 [BOS]，句尾加 [EOS]，模拟真实语言模型的边界标记。
        seq = [config.bos_token_id] + tokenizer.encode(text) + [config.eos_token_id]
        if len(seq) > config.max_seq_len:
            # 超长就截断，并强制最后一位仍是 [EOS]，避免句尾信息丢失。
            seq = seq[: config.max_seq_len]
            seq[-1] = config.eos_token_id
        sequences.append(seq)

    batch = pad_sequences(sequences, config.pad_token_id, device)
    # 例如：
    # 原序列   [BOS, user, greet, assistant, hello, EOS]
    # 输入     [BOS, user, greet, assistant, hello]
    # 标签     [user, greet, assistant, hello, EOS]
    return batch[:, :-1], batch[:, 1:]


def make_sft_batch(
    sft_pairs: Sequence[Tuple[str, str]],
    tokenizer: ToyTokenizer,
    config: RLHFConfig,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造 SFT batch。

    与预训练的关键区别：
    - 预训练通常所有位置都参与 next-token loss。
    - SFT 经常只让“回答部分”参与损失，prompt 部分只作为条件输入。
    """
    inputs = []
    labels = []

    for _ in range(batch_size):
        prompt, response = random.choice(sft_pairs)
        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(response)
        full_seq = [config.bos_token_id] + prompt_ids + response_ids + [config.eos_token_id]
        full_seq = full_seq[: config.max_seq_len]
        if full_seq[-1] != config.eos_token_id:
            full_seq[-1] = config.eos_token_id

        input_ids = full_seq[:-1]
        target_ids = full_seq[1:]

        # 只训练模型生成 response，不让 prompt 部分参与损失。
        # response_start 表示 response 在 full_seq 中从哪里开始。
        response_start = 1 + len(prompt_ids)
        masked_labels = []
        for idx, token_id in enumerate(target_ids):
            if idx >= response_start - 1:
                # response 对应的位置保留真实标签，参与交叉熵损失。
                masked_labels.append(token_id)
            else:
                # prompt 对应的位置填 -100，交叉熵会自动忽略。
                masked_labels.append(-100)

        inputs.append(input_ids)
        labels.append(masked_labels)

    return pad_sequences(inputs, config.pad_token_id, device), pad_sequences(labels, -100, device)


def make_reward_batch(
    triples: Sequence[Tuple[str, str, str]],
    tokenizer: ToyTokenizer,
    config: RLHFConfig,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造奖励模型训练 batch。

    每个样本会生成两条序列：
      - chosen:   更好的回答
      - rejected: 更差的回答

    奖励模型的目标不是“生成答案”，而是“给答案排序”。
    """
    chosen_sequences = []
    rejected_sequences = []

    for _ in range(batch_size):
        prompt, chosen, rejected = random.choice(triples)
        chosen_seq = [config.bos_token_id] + tokenizer.encode(prompt) + tokenizer.encode(chosen) + [config.eos_token_id]
        rejected_seq = [config.bos_token_id] + tokenizer.encode(prompt) + tokenizer.encode(rejected) + [config.eos_token_id]

        chosen_seq = chosen_seq[: config.max_seq_len]
        rejected_seq = rejected_seq[: config.max_seq_len]
        chosen_seq[-1] = config.eos_token_id
        rejected_seq[-1] = config.eos_token_id

        chosen_sequences.append(chosen_seq)
        rejected_sequences.append(rejected_seq)

    return (
        pad_sequences(chosen_sequences, config.pad_token_id, device),
        pad_sequences(rejected_sequences, config.pad_token_id, device),
    )


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    # ignore_index=-100 的作用非常关键：
    # 它允许我们灵活屏蔽某些位置，比如 SFT 中 prompt 段不计损失。
    return F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)


def select_response_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    取出 labels 指定 token 的对数概率，并在有效位置上求和。

    这一步在 RLHF 里很常见，因为我们经常需要知道：
    “策略模型到底给当前回答分配了多大概率？”

    返回的是每个样本一整个 response 的总 log-prob，
    而不是单个 token 的概率。
    """

    log_probs = F.log_softmax(logits, dim=-1)
    valid_mask = labels != -100
    # gather 需要合法索引，所以先把无效位置临时改成 0；
    # 这些位置后面会乘 valid_mask，不会真的参与结果。
    safe_labels = labels.masked_fill(~valid_mask, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * valid_mask).sum(dim=-1)


def build_sequence_for_scoring(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    config: RLHFConfig,
) -> List[int]:
    # 这里故意不自动补 [EOS]，因为打分阶段希望直接沿用“生成出来的 response 原样”。
    seq = [config.bos_token_id] + list(prompt_ids) + list(response_ids)
    seq = seq[: config.max_seq_len]
    return seq


def compute_sequence_logprob(
    model: TinyPolicyValueModel,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    config: RLHFConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算生成 response 的总对数概率以及当前状态价值。

    这里将整段 response 的 log-prob 相加，视作一个 sequence-level action。
    这样虽然比 token-level PPO 更简化，但足以解释 RLHF 的核心机制。
    """

    # 这里把“一整段回答”视为 sequence-level action。
    # 这比真实 token-level PPO 更简化，但概念上更容易看懂。
    full_seq = build_sequence_for_scoring(prompt_ids, response_ids, config)
    if len(full_seq) < 2:
        raise ValueError("序列长度至少为 2")

    # 同样把完整序列拆成“输入”和“下一个 token 标签”。
    input_ids = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_seq[1:]], dtype=torch.long, device=device)
    logits, values, _ = model(input_ids)

    response_start = 1 + len(prompt_ids)
    masked_labels = torch.full_like(target_ids, fill_value=-100)
    start_idx = max(response_start - 1, 0)
    # 只统计 response 段的 log-prob，不把 prompt 段算进策略动作概率里。
    masked_labels[:, start_idx:] = target_ids[:, start_idx:]

    seq_logprob = select_response_logprobs(logits, masked_labels)
    return seq_logprob.squeeze(0), values.squeeze(0)


def compute_reference_logprob(
    model: TinyCausalLM,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    config: RLHFConfig,
    device: torch.device,
) -> torch.Tensor:
    # 参考模型通常冻结不更新，它的作用是提供 KL 约束基线，
    # 防止策略模型为了追求奖励而偏离原始语言分布太远。
    full_seq = build_sequence_for_scoring(prompt_ids, response_ids, config)
    input_ids = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_seq[1:]], dtype=torch.long, device=device)
    logits, _ = model(input_ids)

    response_start = 1 + len(prompt_ids)
    masked_labels = torch.full_like(target_ids, fill_value=-100)
    start_idx = max(response_start - 1, 0)
    masked_labels[:, start_idx:] = target_ids[:, start_idx:]

    return select_response_logprobs(logits, masked_labels).squeeze(0)


def score_response(
    reward_model: TinyRewardModel,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    config: RLHFConfig,
    device: torch.device,
) -> torch.Tensor:
    # 奖励模型看到的是“prompt + response”的整段序列，
    # 然后输出一个标量分数，表示这条回答总体有多好。
    seq = [config.bos_token_id] + list(prompt_ids) + list(response_ids) + [config.eos_token_id]
    seq = seq[: config.max_seq_len]
    seq[-1] = config.eos_token_id
    batch = torch.tensor([seq], dtype=torch.long, device=device)
    return reward_model(batch).squeeze(0)


def generate_response(
    model: TinyCausalLM,
    prompt_ids: Sequence[int],
    config: RLHFConfig,
    device: torch.device,
    greedy: bool = False,
) -> List[int]:
    """
    自回归生成若干个新 token。

    每一轮都把当前已经生成的 token 重新送回模型，
    让模型继续预测下一个 token，这就是 autoregressive generation。

    greedy=True  时，总是取最大概率 token。
    greedy=False 时，按概率采样，更适合 PPO rollout 阶段探索。
    """

    model.eval()
    generated = [config.bos_token_id] + list(prompt_ids)
    response_tokens: List[int] = []

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(input_ids)
            # 只取最后一个位置的 logits，因为它对应“下一个 token”的预测分布。
            next_token_logits = logits[0, -1] / config.temperature
            probs = F.softmax(next_token_logits, dim=-1)
            if greedy:
                next_token = probs.argmax().item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            response_tokens.append(next_token)
            if next_token == config.eos_token_id:
                break

    if not response_tokens or response_tokens[-1] != config.eos_token_id:
        response_tokens.append(config.eos_token_id)
    return response_tokens


def copy_lm_weights_to_policy(policy_model: TinyPolicyValueModel, lm_model: TinyCausalLM) -> None:
    # PPO 通常不是从随机策略开始，而是从已经 SFT 好的模型继续优化。
    policy_model.backbone.load_state_dict(copy.deepcopy(lm_model.backbone.state_dict()))
    policy_model.lm_head.load_state_dict(copy.deepcopy(lm_model.lm_head.state_dict()))


def copy_lm_weights_to_reward(reward_model: TinyRewardModel, lm_model: TinyCausalLM) -> None:
    # 奖励模型一般也会继承语言模型主干，再额外接一个 reward head。
    reward_model.backbone.load_state_dict(copy.deepcopy(lm_model.backbone.state_dict()))


def evaluate_generations(
    title: str,
    model: TinyCausalLM,
    tokenizer: ToyTokenizer,
    prompts: Sequence[str],
    config: RLHFConfig,
    device: torch.device,
) -> None:
    # 用 greedy 解码做一个可视化检查，直观看模型在不同训练阶段输出是否改善。
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for prompt in prompts:
        response_ids = generate_response(model, tokenizer.encode(prompt), config, device, greedy=True)
        print(f"Prompt   : {prompt}")
        print(f"Response : {tokenizer.decode(response_ids)}")
        print("-" * 80)


def evaluate_policy_generations(
    title: str,
    model: TinyPolicyValueModel,
    tokenizer: ToyTokenizer,
    prompts: Sequence[str],
    config: RLHFConfig,
    device: torch.device,
) -> None:
    # policy_model 不直接带 generate 接口，这里临时包装成 TinyCausalLM 来复用生成逻辑。
    lm_wrapper = TinyCausalLM(config).to(device)
    lm_wrapper.backbone.load_state_dict(copy.deepcopy(model.backbone.state_dict()))
    lm_wrapper.lm_head.load_state_dict(copy.deepcopy(model.lm_head.state_dict()))
    evaluate_generations(title, lm_wrapper, tokenizer, prompts, config, device)


def train_pretraining_stage(
    model: TinyCausalLM,
    texts: Sequence[str],
    tokenizer: ToyTokenizer,
    config: RLHFConfig,
    device: torch.device,
) -> None:
    """
    第一阶段：预训练。

    目的不是让模型“听懂指令”，而是先学会语言的基本统计规律：
    哪些 token 容易一起出现，句子通常如何延续。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("\n" + "=" * 80)
    print("1. 预训练阶段（Language Modeling）")
    print("=" * 80)

    model.train()
    for step in range(1, config.pretrain_steps + 1):
        input_ids, labels = make_pretrain_batch(
            texts=texts,
            tokenizer=tokenizer,
            config=config,
            batch_size=config.pretrain_batch_size,
            device=device,
        )
        logits, _ = model(input_ids)
        loss = causal_lm_loss(logits, labels, config.vocab_size)

        # 标准的梯度下降三步：清梯度 -> 反传 -> 更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0 or step == config.pretrain_steps:
            print(f"step={step:03d} | pretrain_loss={loss.item():.6f}")


def train_sft_stage(
    model: TinyCausalLM,
    sft_pairs: Sequence[Tuple[str, str]],
    tokenizer: ToyTokenizer,
    config: RLHFConfig,
    device: torch.device,
) -> None:
    """
    第二阶段：SFT。

    这一步把模型从“泛化语言续写器”校正成“按 prompt 回答问题的助手”。
    常见现象是：SFT 后模型输出会更稳定、更贴近标注答案风格。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("\n" + "=" * 80)
    print("2. SFT 阶段（监督微调）")
    print("=" * 80)

    model.train()
    for step in range(1, config.sft_steps + 1):
        input_ids, labels = make_sft_batch(
            sft_pairs=sft_pairs,
            tokenizer=tokenizer,
            config=config,
            batch_size=config.sft_batch_size,
            device=device,
        )
        logits, _ = model(input_ids)
        loss = causal_lm_loss(logits, labels, config.vocab_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0 or step == config.sft_steps:
            print(f"step={step:03d} | sft_loss={loss.item():.6f}")


def train_reward_model(
    reward_model: TinyRewardModel,
    triples: Sequence[Tuple[str, str, str]],
    tokenizer: ToyTokenizer,
    config: RLHFConfig,
    device: torch.device,
) -> None:
    """
    第三阶段：奖励模型训练。

    训练目标不是生成 token，而是让 chosen 的分数高于 rejected。
    当 reward model 学好后，它就能充当 RLHF 里的“自动裁判”。
    """
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=config.reward_learning_rate)

    print("\n" + "=" * 80)
    print("3. 奖励模型阶段（Pairwise Preference Modeling）")
    print("=" * 80)

    reward_model.train()
    for step in range(1, config.reward_steps + 1):
        chosen_batch, rejected_batch = make_reward_batch(
            triples=triples,
            tokenizer=tokenizer,
            config=config,
            batch_size=config.reward_batch_size,
            device=device,
        )
        chosen_scores = reward_model(chosen_batch)
        rejected_scores = reward_model(rejected_batch)

        # 如果 chosen_score 比 rejected_score 大很多，loss 会变小；
        # 反之，说明奖励模型排序还不对，需要继续训练。
        loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 30 == 0 or step == config.reward_steps:
            margin = (chosen_scores - rejected_scores).mean().item()
            print(f"step={step:03d} | reward_loss={loss.item():.6f} | avg_margin={margin:.6f}")


def train_ppo_stage(
    policy_model: TinyPolicyValueModel,
    reference_model: TinyCausalLM,
    reward_model: TinyRewardModel,
    prompts: Sequence[str],
    tokenizer: ToyTokenizer,
    config: RLHFConfig,
    device: torch.device,
) -> None:
    """
    第四阶段：PPO 风格 RLHF。

    这部分是整个 demo 最关键的地方。完整逻辑是：
      1. 当前策略模型先对 prompt 采样出回答（rollout）
      2. 奖励模型给回答打分
      3. 参考模型提供 KL 约束
      4. policy/value 网络按 PPO 目标更新

    真实工业系统经常是 token-level PPO，这里简化成 sequence-level PPO，
    这样更容易把训练主线讲清楚。
    """
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.ppo_learning_rate)

    print("\n" + "=" * 80)
    print("4. PPO 阶段（RLHF 对齐）")
    print("=" * 80)

    reward_model.eval()
    reference_model.eval()

    for step in range(1, config.ppo_steps + 1):
        # rollouts 用来缓存当前策略采样得到的一批“经验”。
        rollouts = []
        policy_model.eval()

        for _ in range(config.ppo_batch_size):
            prompt = random.choice(prompts)
            prompt_ids = tokenizer.encode(prompt)

            # 这里临时拷一份只带 lm_head 的 policy，用于做纯生成。
            # 这样写虽然不如工业代码高效，但结构更直观。
            policy_for_generation = TinyCausalLM(config).to(device)
            policy_for_generation.backbone.load_state_dict(copy.deepcopy(policy_model.backbone.state_dict()))
            policy_for_generation.lm_head.load_state_dict(copy.deepcopy(policy_model.lm_head.state_dict()))

            # 1) 当前策略采样回答
            response_ids = generate_response(policy_for_generation, prompt_ids, config, device, greedy=False)
            # 2) 记录旧策略对这条回答的 log-prob，以及旧 value 估计
            old_logprob, old_value = compute_sequence_logprob(policy_model, prompt_ids, response_ids, config, device)
            # 3) 计算参考模型 log-prob，用于构造 KL 惩罚
            ref_logprob = compute_reference_logprob(reference_model, prompt_ids, response_ids, config, device)
            # 4) 奖励模型对整条回答打分
            rm_reward = score_response(reward_model, prompt_ids, response_ids, config, device)

            # 5) 总奖励 = 奖励模型分数 - KL 惩罚
            # 如果策略过度偏离 reference，哪怕 reward model 给分高，也会被拉回来。
            total_reward = rm_reward - config.kl_coef * (old_logprob.detach() - ref_logprob.detach())

            rollouts.append(
                {
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "old_logprob": old_logprob.detach(),
                    "old_value": old_value.detach(),
                    "reward": total_reward.detach(),
                }
            )

        # advantage = 实际回报 - value baseline
        # 它表示：“这条回答比 value head 原来预期的更好还是更差？”
        advantages = torch.stack([item["reward"] - item["old_value"] for item in rollouts])
        # 标准化 advantage，能让训练更稳定，避免尺度抖动太大。
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        policy_model.train()
        policy_losses = []
        value_losses = []
        rewards = []

        for idx, item in enumerate(rollouts):
            current_logprob, current_value = compute_sequence_logprob(
                policy_model,
                item["prompt_ids"],
                item["response_ids"],
                config,
                device,
            )

            # ratio = 新策略概率 / 旧策略概率
            # 它衡量当前更新后，模型对这条旧回答的偏好变化了多少。
            ratio = torch.exp(current_logprob - item["old_logprob"])
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - config.ppo_clip_range, 1.0 + config.ppo_clip_range) * advantages[idx]
            # PPO 的关键就在 clip：限制每次策略更新不要走太大步。
            policy_loss = -torch.min(unclipped, clipped)
            # value head 则学习回归总奖励，充当 baseline。
            value_loss = F.mse_loss(current_value, item["reward"])

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            rewards.append(item["reward"])

        # 总损失 = 策略损失 + value 损失 × 系数
        total_loss = torch.stack(policy_losses).mean() + config.value_coef * torch.stack(value_losses).mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % 15 == 0 or step == config.ppo_steps:
            print(
                f"step={step:03d} | "
                f"policy_loss={torch.stack(policy_losses).mean().item():.6f} | "
                f"value_loss={torch.stack(value_losses).mean().item():.6f} | "
                f"avg_reward={torch.stack(rewards).mean().item():.6f}"
            )


def print_pipeline_summary() -> None:
    # 最后再把读者从代码细节里拉出来，看一遍完整生产流程的大图景。
    print("\n" + "=" * 80)
    print("5. 大模型完整流程回顾")
    print("=" * 80)
    print("1. 数据准备：清洗语料、分词、构建指令数据、人类偏好数据。")
    print("2. 预训练：让模型学会 next-token prediction，得到基础语言能力。")
    print("3. SFT：把模型从‘会续写文本’变成‘会按指令回答’。")
    print("4. Reward Model：学习什么样的回答更符合人类偏好。")
    print("5. RLHF / PPO：在 KL 约束下，让策略朝高奖励方向更新。")
    print("6. 评测与安全：离线 benchmark、人工评审、红队测试、拒答策略。")
    print("7. 部署与数据飞轮：上线收集反馈，再继续做 SFT / 偏好优化。")
    print()
    print("📌 真实工业系统还会补充：")
    print("   - 分布式并行：DP / TP / PP / ZeRO / FSDP")
    print("   - 高效训练：FlashAttention、混合精度、梯度检查点")
    print("   - 更大规模对齐：DPO、ORPO、GRPO、RLAIF")
    print("   - 在线评测与安全治理：内容审核、工具调用监控、回滚策略")


def main() -> None:
    """
    主函数：串起整个教学流程。

    执行顺序：
      0. 看训练前模型会胡乱输出什么
      1. 做预训练
      2. 做 SFT
      3. 训练奖励模型
      4. 做 PPO 对齐
      5. 再观察输出变化，并回顾完整大模型流程
    """
    config = RLHFConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ToyTokenizer()
    pretrain_texts, sft_pairs, preference_triples = build_demo_datasets()
    prompts = [prompt for prompt, _ in sft_pairs]

    print("使用设备:", device)
    print("\nRLHF 配置:", config)
    print("\n数据集规模:")
    print(f"  预训练语料条数: {len(pretrain_texts)}")
    print(f"  SFT 样本条数: {len(sft_pairs)}")
    print(f"  偏好比较条数: {len(preference_triples)}")

    # base_model 表示“只经过预训练”的基座语言模型。
    base_model = TinyCausalLM(config).to(device)
    evaluate_generations("0. 训练前的模型输出", base_model, tokenizer, prompts, config, device)

    train_pretraining_stage(base_model, pretrain_texts, tokenizer, config, device)
    evaluate_generations("1. 预训练后的模型输出", base_model, tokenizer, prompts, config, device)

    # SFT 从预训练权重出发，而不是从随机初始化重新训练。
    sft_model = TinyCausalLM(config).to(device)
    sft_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    train_sft_stage(sft_model, sft_pairs, tokenizer, config, device)
    evaluate_generations("2. SFT 后的模型输出", sft_model, tokenizer, prompts, config, device)

    # 奖励模型从 SFT 主干初始化，这也是 RLHF 常见做法。
    reward_model = TinyRewardModel(config).to(device)
    copy_lm_weights_to_reward(reward_model, sft_model)
    train_reward_model(reward_model, preference_triples, tokenizer, config, device)

    # reference_model 是 PPO 中的冻结参考策略，用来做 KL 约束。
    reference_model = TinyCausalLM(config).to(device)
    reference_model.load_state_dict(copy.deepcopy(sft_model.state_dict()))
    for param in reference_model.parameters():
        param.requires_grad = False

    # policy_model 则是在 SFT 基础上继续做强化学习对齐的策略网络。
    policy_model = TinyPolicyValueModel(config).to(device)
    copy_lm_weights_to_policy(policy_model, sft_model)
    train_ppo_stage(policy_model, reference_model, reward_model, prompts, tokenizer, config, device)
    evaluate_policy_generations("3. PPO 对齐后的模型输出", policy_model, tokenizer, prompts, config, device)

    print_pipeline_summary()

    print("\n🔥 推荐进阶阅读：")
    print("   1. InstructGPT: https://arxiv.org/abs/2203.02155")
    print("   2. PPO: https://arxiv.org/abs/1707.06347")
    print("   3. DPO: https://arxiv.org/abs/2305.18290")
    print("   4. TRL: https://github.com/huggingface/trl")
    print("   5. Megatron-LM: https://github.com/NVIDIA/Megatron-LM")


if __name__ == "__main__":
    main()