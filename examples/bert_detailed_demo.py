"""
BERT 详细教学演示（带逐行中文注释）
=====================================

本文件从零实现了一个极简版 BERT（Bidirectional Encoder Representations from Transformers），
涵盖以下核心组件：
  1. BertConfig        — 超参数配置
  2. BertEmbeddings    — 输入表示（Token + Position + Segment 三种 Embedding 求和）
  3. MultiHeadSelfAttention — 多头双向自注意力
  4. BertLayer         — 单层 Transformer Encoder Block
  5. BertEncoder       — 多层堆叠
  6. BertPooler        — [CLS] 句级表示
  7. BertForPretrainingDemo — MLM + NSP 双头预训练模型
  8. ToyTokenizer      — 玩具级词表（空格分词）
  9. 训练循环演示

说明：
  有些同学会把 "diffusion" 误写成 "defussion"。
  diffusion 一般指"扩散模型"，是一类生成模型，核心思想是：
  先逐步给数据加噪声，再训练模型一步步去噪，最终生成新样本。
  它常见于图像生成、视频生成等任务。
  而这个文件讲的是 BERT，它属于 Transformer 编码器模型，
  主要用于文本表示学习、理解任务，以及 MLM / NSP 这类预训练目标。
  所以如果你问这里的 "defussion" 是什么，通常是在问 "diffusion"；
  但它不是这个 BERT demo 的核心概念。
"""

import math  # 数学函数，用于 sqrt(self.head_dim) 进行注意力分数缩放
from dataclasses import dataclass  # 数据类装饰器，简化配置类的定义
from typing import Dict, List, Tuple  # 类型提示

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块（Linear, Embedding, Dropout 等）
import torch.nn.functional as F  # 函数式 API（softmax, cross_entropy 等）


@dataclass
class BertConfig:
    """
    教学版 BERT 超参数配置。

    所有超参数都故意设得很小（而非工业级的大模型），
    目的是让代码在 CPU 上也能快速跑通，便于读者理解结构。
    """

    # ---------- 词表与嵌入相关 ----------
    vocab_size: int = 32
    # 词表大小。为了方便演示，我们手写一个很小的词表，仅包含 20 多个词。
    max_position_embeddings: int = 32
    # 最大序列长度（即最多支持多少个 token 位置）。位置编码表的大小由此决定。
    type_vocab_size: int = 2
    # token type（segment）的类别数。
    # BERT 用 0 表示句子 A，1 表示句子 B，方便区分两个句子的边界。

    # ---------- 模型主干维度 ----------
    hidden_size: int = 64
    # BERT 的隐藏层维度。所有 token 的表示都映射到这个维度。
    # 工业版（BERT-base）是 768，这里用 64 加速演示。
    num_heads: int = 4
    # 多头自注意力中的头数。hidden_size 必须能被 num_heads 整除。
    # 每个头的维度 = hidden_size / num_heads = 16。
    intermediate_size: int = 128
    # FFN（前馈网络）中间层的维度。通常为 hidden_size 的 2~4 倍。
    num_layers: int = 2
    # Transformer Encoder 堆叠的层数。BERT-base 是 12 层，这里用 2 层。
    dropout: float = 0.1
    # Dropout 概率。在注意力权重和 FFN 输出后使用，防止过拟合。

    # ---------- 特殊 Token ID ----------
    pad_token_id: int = 0  # [PAD]：填充符，用于对齐 batch 中不同长度的句子
    cls_token_id: int = 1  # [CLS]：分类符，放在句首，其表示用于句级分类任务
    sep_token_id: int = 2  # [SEP]：分隔符，放在句子末尾，也用于区分句对
    mask_token_id: int = 3  # [MASK]：掩码符，MLM 任务中将部分 token 替换为此

    # ---------- 训练相关 ----------
    batch_size: int = 2      # 每次训练的样本数
    learning_rate: float = 1e-3  # Adam 优化器学习率
    train_steps: int = 30    # 训练步数（仅用于演示收敛趋势）
    seed: int = 42           # 随机种子，保证结果可复现


class BertEmbeddings(nn.Module):
    """
    BERT 的输入表示层。

    BERT 的输入并不只是词向量，而是由三部分 Embedding **逐元素相加**得到：
      1. Token Embedding（词嵌入）    — 每个 token 本身的语义表示
      2. Position Embedding（位置嵌入）— 告诉模型 token 在句子中的位置
      3. Token Type Embedding（段嵌入）— 区分当前 token 属于句子 A 还是句子 B

    为什么要用三种 Embedding 相加？
    - Transformer 本身没有序列顺序的概念（不像 RNN 逐步读取），
      所以必须额外注入位置信息。
    - 对于句对任务（如 NSP、推理），需要告诉模型句子边界。
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # 词嵌入表：将 vocab_size 个 token 映射到 hidden_size 维向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # 位置嵌入表：最多支持 max_position_embeddings 个位置
        # 每个位置学到一个独立的向量，与 token 内容无关
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 段（token type）嵌入表：只有 2 行（segment 0 / segment 1）
        # 注意：这不是指"词性"或"类型"，而是句子 A / B 的区分
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Layer Normalization：稳定训练，加速收敛
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        # Dropout：随机丢弃部分神经元的输出，防止过拟合
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        参数:
            input_ids:     [batch_size, seq_len] — token 在词表中的索引
            token_type_ids:[batch_size, seq_len] — 每个 token 属于 0（句子A）还是 1（句子B）
        返回:
            embeddings:    [batch_size, seq_len, hidden_size] — 三种 Embedding 求和后的结果
        """
        batch_size, seq_len = input_ids.shape  # 获取 batch 大小和序列长度

        # 构造位置 ID：从 0 到 seq_len-1，形状为 [1, seq_len]
        # unsqueeze(0) 在第 0 维加一维，方便后续广播到整个 batch
        # 注意：每个样本使用相同的位置序列（0, 1, 2, ...）
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # ---- 分别查找三种 Embedding ----
        word_embed = self.word_embeddings(input_ids)          # [batch, seq_len, hidden]
        pos_embed = self.position_embeddings(position_ids)    # [1, seq_len, hidden]，广播到 batch
        type_embed = self.token_type_embeddings(token_type_ids)  # [batch, seq_len, hidden]

        # 逐元素相加：这是 BERT 输入表示的核心公式
        # embedding = token + position + segment
        embeddings = word_embed + pos_embed + type_embed

        # LayerNorm + Dropout（Pre-norm 风格，详见 BertLayer）
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class MultiHeadSelfAttention(nn.Module):
    """
    多头双向自注意力机制（Multi-Head Self-Attention）。

    BERT 使用**双向（bidirectional）**自注意力，与自回归语言模型（如 GPT）不同：
    - GPT 使用 Causal Mask（因果掩码），每个 token 只能看到它自己和左侧的 token。
    - BERT **不做**因果掩码，每个 token 可以同时看到左右两侧的所有 token，
      这正是 BERT 中 "Bidirectional"（双向）的含义。

    数学公式（Scaled Dot-Product Attention）：
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    多头机制：将 hidden_size 切分成 num_heads 个子空间，
    每个头独立做注意力，最后拼接起来。这使模型能从不同角度关注序列。
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # 检查维度是否整除：hidden_size 必须能被 num_heads 整除
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")

        self.hidden_size = config.hidden_size  # 总体隐藏维度，如 64
        self.num_heads = config.num_heads      # 注意力头数，如 4
        self.head_dim = config.hidden_size // config.num_heads  # 每个头的维度，如 16

        # 三个线性投影层：将 hidden_size 映射到 hidden_size
        # 注意：这里是一次性算出所有头的 Q / K / V，后续再 split 成多个头
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Query 投影
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Key 投影
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Value 投影

        # 输出投影层：将多头拼接后的结果映射回 hidden_size
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Attention Dropout：在注意力权重（softmax 之后）上做 dropout
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 [batch, seq_len, hidden_size] 重排为 [batch, num_heads, seq_len, head_dim]。

        步骤：
        1. view:  [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads, head_dim]
        2. transpose: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        最后两维分别对应"序列位置"和"每个头的维度"。
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # 交换 seq_len 和 num_heads 维度

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        _split_heads 的逆操作。
        将 [batch, num_heads, seq_len, head_dim] 恢复为 [batch, seq_len, hidden_size]。
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        return x.view(batch_size, seq_len, num_heads * head_dim)  # 拼接所有头

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            hidden_states:  [batch_size, seq_len, hidden_size]  — 上一层的输出
            attention_mask: [batch_size, 1, 1, seq_len]         — padding mask
        返回:
            context:          [batch_size, seq_len, hidden_size]  — 注意力加权后的输出
            attention_probs:  [batch_size, num_heads, seq_len, seq_len] — 注意力权重（可用于可视化）
        """
        # ---- Step 1: 线性投影 + 多头拆分 ----
        # Q / K / V 都来自同一个 hidden_states（这就是"自注意力"中"自"的含义）
        q = self._split_heads(self.q_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        k = self._split_heads(self.k_proj(hidden_states))  # 同上
        v = self._split_heads(self.v_proj(hidden_states))  # 同上

        # ---- Step 2: Scaled Dot-Product Attention ----
        # scores = Q * K^T，形状 [batch, num_heads, seq_len, seq_len]
        # scores[i, h, j, k] 表示第 i 个样本、第 h 个头中，位置 j 对位置 k 的注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 除以 sqrt(head_dim) 做缩放：防止内积过大导致 softmax 梯度消失

        # ---- Step 3: 应用 Attention Mask ----
        # attention_mask 中，有效 token 位置为 1，padding 位置为 0
        # 将 padding 位置的分数设为极小值，softmax 后这些位置的概率 ≈ 0
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)

        # ---- Step 4: Softmax + Dropout ----
        attention_probs = F.softmax(scores, dim=-1)  # 在最后一个维度（key 侧）做 softmax
        attention_probs = self.dropout(attention_probs)  # 防止注意力过度依赖某些位置

        # ---- Step 5: 加权求和 ----
        context = torch.matmul(attention_probs, v)  # [batch, num_heads, seq_len, head_dim]

        # ---- Step 6: 合并多头 + 输出投影 ----
        context = self._merge_heads(context)  # [batch, seq_len, hidden_size]
        context = self.out_proj(context)      # 线性变换，整合多头信息

        return context, attention_probs


class BertLayer(nn.Module):
    """
    单层 Transformer Encoder Block。

    结构顺序（Pre-LayerNorm 变体，即先 Norm 后计算，有助于训练稳定）：
        hidden_states
            │
            ├──> LayerNorm ──> MultiHeadSelfAttention ──> Dropout ──> + ──> LayerNorm ──> FFN ──> + ──> 输出
            │                                                                             │
            └───────────────────────── 残差连接 ───────────────────────────────────────────┘

    每个子层（Attention / FFN）周围都有**残差连接（Residual Connection）**：
        output = LayerNorm(x + sublayer(x))
    残差连接帮助梯度直接流过深层网络，缓解梯度消失问题。
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # ---- 子层 1: 多头自注意力 ----
        self.attention = MultiHeadSelfAttention(config)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.attn_norm = nn.LayerNorm(config.hidden_size)

        # ---- 子层 2: 前馈神经网络（FFN） ----
        # 结构: Linear -> GELU -> Linear -> Dropout
        # GELU (Gaussian Error Linear Unit) 是 BERT 原始论文中使用的激活函数，
        # 相比 ReLU，它在负数区域有平滑的梯度，训练更稳定
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),  # 升维: hidden -> 4 * hidden
            nn.GELU(),                                                # 非线性激活
            nn.Linear(config.intermediate_size, config.hidden_size),  # 降维: 4 * hidden -> hidden
            nn.Dropout(config.dropout),                               # 防止过拟合
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            hidden_states:  [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
        返回:
            hidden_states:  [batch_size, seq_len, hidden_size]  — 经过该层处理后的表示
            attn_probs:     [batch_size, num_heads, seq_len, seq_len] — 注意力权重
        """
        # ---- 残差子层 1: 自注意力 ----
        attn_out, attn_probs = self.attention(hidden_states, attention_mask)
        hidden_states = self.attn_norm(hidden_states + self.attn_dropout(attn_out))

        # ---- 残差子层 2: 前馈网络 ----
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ffn_out)

        return hidden_states, attn_probs


class BertEncoder(nn.Module):
    """
    多层 Transformer Encoder（堆叠多个 BertLayer）。

    BERT 的深度来自于堆叠多层 Encoder Block（BERT-base 有 12 层）。
    每一层都对序列做了"自注意力 + FFN"变换，使表示越来越"上下文感知"。

    浅层（靠近输入）学到更多词法和句法特征，
    深层（靠近输出）学到更多语义和篇章特征。
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # nn.ModuleList 是 PyTorch 专门用来存放多个模块的容器
        # 它确保每个子模块的参数量被正确注册和追踪
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        参数:
            hidden_states:  [batch_size, seq_len, hidden_size]  — Embedding 层的输出
            attention_mask: [batch_size, 1, 1, seq_len]         — padding mask
        返回:
            hidden_states:  [batch_size, seq_len, hidden_size]  — 最后一层的输出
            all_attention_probs: List of [batch, num_heads, seq_len, seq_len]  — 每层的注意力权重
        """
        all_attention_probs = []  # 收集所有层的注意力权重，供可视化或分析使用
        for i, layer in enumerate(self.layers):
            hidden_states, attn_probs = layer(hidden_states, attention_mask)
            all_attention_probs.append(attn_probs)
        return hidden_states, all_attention_probs


class BertPooler(nn.Module):
    """
    句子级表示提取器（[CLS] Pooler）。

    BERT 通常将 [CLS] token（序列第一个 token）的输出表示用于句子级分类任务。
    这是因为 [CLS] 在自注意力过程中能够"聚合"整个序列的信息。

    Pooler 的计算：
        pooled = tanh(Linear(sequence_output[:, 0]))

    注意：
    - 某些现代变体（如 RoBERTa）发现去掉 pooler 效果更好，直接取 [CLS] 向量即可。
    - 这里保留 pooler 是为了忠实复现原始 BERT 论文的设计。
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 线性变换
        self.activation = nn.Tanh()  # tanh 激活函数，将输出压缩到 (-1, 1) 范围

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        参数:
            hidden_states: [batch_size, seq_len, hidden_size] — Encoder 最后一层的输出
        返回:
            pooled_output: [batch_size, hidden_size] — [CLS] 的句级表示
        """
        cls_state = hidden_states[:, 0]  # 取第一个位置（[CLS]）的向量
        return self.activation(self.dense(cls_state))


class BertModel(nn.Module):
    """
    BERT 主干网络（不带预训练任务的 Head）。

    前向流程：
        input_ids ──┐
        token_type  ──┤──> BertEmbeddings ──> BertEncoder ──> sequence_output ──> BertPooler ──> pooled_output
        attention   ──┘

    输出两类表示：
      1. sequence_output ([batch, seq_len, hidden])：每个 token 的上下文化表示
         — 可用于 token 级任务（如序列标注、QA）
      2. pooled_output ([batch, hidden])：[CLS] 的句子级表示
         — 可用于句级任务（如分类、相似度计算）
      3. all_attention_probs：所有层的注意力权重，可用于分析模型关注了什么
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(config)  # 输入表示层
        self.encoder = BertEncoder(config)         # 多层 Transformer Encoder
        self.pooler = BertPooler(config)           # [CLS] 池化层

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        embedding_output = self.embeddings(input_ids, token_type_ids)     # [batch, seq_len, hidden]
        sequence_output, all_attention_probs = self.encoder(embedding_output, attention_mask)  # [batch, seq_len, hidden]
        pooled_output = self.pooler(sequence_output)  # [batch, hidden]
        return sequence_output, pooled_output, all_attention_probs


class BertForPretrainingDemo(nn.Module):
    """
    BERT 预训练模型（教学版）。

    在 BERT 主干之上添加两个预训练任务的 Head：
    1. **MLM 头**（Masked Language Model）：
       输入中被 [MASK] 替换的位置，预测原始 token 是什么。
       这迫使模型利用上下文来理解被遮住位置的语义。

    2. **NSP 头**（Next Sentence Prediction）：
       判断句子 B 是否是句子 A 的下一句。
       这里我们不严格复刻论文数据构造，只做一个直观示例。

    在完整 BERT 中，这两个任务联合训练，总损失 = MLM 损失 + NSP 损失。

    注意：后续研究（如 RoBERTa）发现 NSP 并非必需，但它是原始 BERT 的标准组件。
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # BERT 主干网络（共享参数）
        self.bert = BertModel(config)

        # ---- MLM 头 ----
        # 先用一个变换层（Linear + GELU + LayerNorm）提升表示质量
        self.mlm_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),  # 线性变换
            nn.GELU(),                                          # 非线性激活
            nn.LayerNorm(config.hidden_size),                   # LayerNorm 稳定训练
        )
        # 然后映射回词表大小，得到每个位置在词表上的概率分布
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size)

        # ---- NSP 头 ----
        # 基于 [CLS] 的 pooled_output，二分类：是否是下一句
        self.nsp_classifier = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        参数:
            input_ids:      [batch_size, seq_len] — token 索引序列
            token_type_ids: [batch_size, seq_len] — 段类型（0 或 1）
            attention_mask: [batch_size, 1, 1, seq_len] — padding mask
        返回:
            mlm_logits:          [batch_size, seq_len, vocab_size] — 每个位置在词表上的 logits
            nsp_logits:          [batch_size, 2] — 句对分类 logits
            all_attention_probs: 每层的注意力权重
        """
        # ---- BERT 主干前向 ----
        sequence_output, pooled_output, all_attention_probs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # ---- MLM 预测 ----
        mlm_hidden = self.mlm_transform(sequence_output)  # [batch, seq_len, hidden]
        mlm_logits = self.mlm_decoder(mlm_hidden)         # [batch, seq_len, vocab_size]

        # ---- NSP 预测 ----
        nsp_logits = self.nsp_classifier(pooled_output)   # [batch, 2]

        return mlm_logits, nsp_logits, all_attention_probs


class ToyTokenizer:
    """
    玩具级分词器（Toy Tokenizer）。

    为了让 demo 完全自包含，这里手写一个极小词表。
    包含 5 个特殊 token（[PAD], [CLS], [SEP], [MASK], [UNK]）和 16 个常见英文单词。

    真实场景一般会使用 WordPiece（BERT 原生）或 BPE（GPT/BART 等）等子词分词器：
    - WordPiece 会将 "playing" 拆成 "play" + "##ing"
    - BPE 会将 "lowest" 拆成 "low" + "est"
    本示例只想突出 BERT 模型结构，因此用空格分词即可。
    """

    def __init__(self) -> None:
        # 词表：token -> id 的映射
        # 注意：真实词表通常有几万个 token，这里仅包含演示需要的几个基本词
        self.vocab = {
            # 特殊 Token（必须与 BertConfig 中的 ID 保持一致）
            "[PAD]": 0,   # 填充符，用于对齐 batch 中不同长度的序列
            "[CLS]": 1,   # 分类符，放在每个句子的最前面
            "[SEP]": 2,   # 分隔符，放在句子末尾或句对之间
            "[MASK]": 3,  # 掩码符，MLM 任务中替换被遮住的词
            # 普通词汇（仅用于构造简单的演示句子）
            "i": 4, "love": 5, "nlp": 6, "you": 7,           # 第 1 组
            "like": 8, "deep": 9, "learning": 10, "we": 11,  # 第 2 组
            "study": 12, "bert": 13, "moe": 14, "is": 15,    # 第 3 组
            "fun": 16, "powerful": 17, "this": 18,           # 第 4 组
            "model": 19, "great": 20,                         # 第 5 组
        }
        # 反向映射：id -> token，方便解码时查看预测结果
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def encode_sentence(self, text: str) -> List[int]:
        """
        将句子转化为 token id 列表。

        这里用最简单的空格分词（str.split()），
        而不是 WordPiece / BPE 等子词分词。
        如果遇到词表中不存在的词，会抛出 KeyError。
        """
        return [self.vocab[token] for token in text.split()]

    def decode_id(self, idx: int) -> str:
        """
        将单个 token id 解码回对应的字符串。
        用于打印预测结果，方便人类阅读。
        """
        return self.id_to_token[idx]


def build_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    构造 BERT 的 Padding Mask（填充掩码）。

    在自注意力计算中，我们需要告诉模型哪些位置是有效的 token，
    哪些位置是填充的 [PAD]（不应参与注意力计算）。

    返回形状: [batch_size, 1, 1, seq_len]
    其中有效位置为 1，padding 位置为 0。
    维度中的两个 1 分别对应 num_heads 和 seq_len(query)，方便广播。

    例如:
        input_ids = [[1, 4, 6, 2, 0]]  # 0 是 [PAD]
        attention_mask = [[[[1, 1, 1, 1, 0]]]]  # 最后一个位置被屏蔽
    """
    # (input_ids != pad_token_id) 得到布尔张量，形状 [batch, seq_len]
    # unsqueeze(1) 扩展出 num_heads 维度: [batch, 1, seq_len]
    # unsqueeze(2) 扩展出 query 的 seq_len 维度: [batch, 1, 1, seq_len]
    return (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)


def build_toy_pretraining_batch(
    tokenizer: ToyTokenizer,
    config: BertConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    构造一个极小的玩具级预训练 batch，用于演示 MLM + NSP。

    BERT 标准句对格式:
        [CLS] token_a1 token_a2 ... [SEP] token_b1 token_b2 ... [SEP]
         ^      句 子  A               ^      句 子  B              ^
         |                               |                         |
     分类标记                         分隔符                    分隔符

    MLM 数据构造逻辑：
    1. 按标准格式拼好完整的输入序列。
    2. 手工选一个词替换为 [MASK]。
    3. 记录被遮住的原始 token id 作为 MLM 标签。
    4. 没被 mask 的位置，标签设为 -100（PyTorch 交叉熵会忽略 -100 的位置）。

    NSP 数据构造逻辑：
    每对句子有一个二元标签：1 表示"是下一句"，0 表示"不是"。
    """

    # 定义两个句子对及其 NSP 标签
    # 格式: (句子A, 句子B, NSP标签)
    # 标签 1 = 匹配（是下一句），标签 0 = 不匹配（随机拼接）
    sentence_pairs = [
        ("i love nlp", "bert is powerful", 1),       # 正例：两句话语义相关
        ("we study bert", "deep learning is fun", 0), # 负例：两句话无关
    ]

    input_id_list = []       # 存储每个样本的 input_ids
    token_type_id_list = []  # 存储每个样本的 segment ids
    mlm_label_list = []      # 存储每个样本的 MLM 标签
    nsp_labels = []          # 存储每个样本的 NSP 标签

    for sent_a, sent_b, pair_label in sentence_pairs[: config.batch_size]:
        # ---- Step 1: 分词 ----
        ids_a = tokenizer.encode_sentence(sent_a)  # 句子 A 的 token id 列表
        ids_b = tokenizer.encode_sentence(sent_b)  # 句子 B 的 token id 列表

        # ---- Step 2: 拼成 BERT 标准格式 ----
        # [CLS] + A + [SEP] + B + [SEP]
        input_ids = [config.cls_token_id] + ids_a + [config.sep_token_id] + ids_b + [config.sep_token_id]

        # ---- Step 3: 构造 token_type_ids (segment ids) ----
        # [CLS] 和句子 A 的所有 token 都标记为 0
        # 句子 B 和 [SEP] 都标记为 1
        # 形状: [CLS(0)] + A的每个词(0) + [SEP(0)] + B的每个词(1) + [SEP(1)]
        token_type_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)

        # ---- Step 4: 初始化 MLM 标签，默认全部忽略 (-100) ----
        mlm_labels = [-100] * len(input_ids)

        # ---- Step 5: 选择要 mask 的位置 ----
        # 这里手工指定要 mask 的词，便于读者清楚看到任务定义
        # 第一个样本（pair_label=1）：遮住 "nlp"（位置 3: [CLS] i love nlp ... → 下标 3）
        # 第二个样本（pair_label=0）：遮住 "learning"（位置在句子A长度+3处）
        if pair_label == 1:
            mask_position = 3      # 遮住 "nlp"
        else:
            mask_position = len(ids_a) + 3  # 遮住 "learning"

        # 记录被 mask 前的原始 token id 作为标签
        mlm_labels[mask_position] = input_ids[mask_position]
        # 将输入中的该 token 替换为 [MASK]
        input_ids[mask_position] = config.mask_token_id

        # 收集到列表中
        input_id_list.append(input_ids)
        token_type_id_list.append(token_type_ids)
        mlm_label_list.append(mlm_labels)
        nsp_labels.append(pair_label)

    # ---- Step 6: Padding（对齐 batch 中不同的序列长度） ----
    max_len = max(len(x) for x in input_id_list)  # batch 中最长序列的长度

    def pad_to_max_len(items: List[List[int]], pad_value: int) -> torch.Tensor:
        """将 List[List[int]] 补齐到相同长度，转为 Tensor"""
        padded = [item + [pad_value] * (max_len - len(item)) for item in items]
        return torch.tensor(padded, dtype=torch.long, device=device)

    batch = {
        "input_ids": pad_to_max_len(input_id_list, config.pad_token_id),       # 用 [PAD]=0 补齐
        "token_type_ids": pad_to_max_len(token_type_id_list, 0),               # padding 位用 0 补齐
        "mlm_labels": pad_to_max_len(mlm_label_list, -100),                    # 忽略的标签保持 -100
        "nsp_labels": torch.tensor(nsp_labels, dtype=torch.long, device=device), # [batch]
    }
    # 根据 input_ids 自动计算 padding mask
    batch["attention_mask"] = build_attention_mask(batch["input_ids"], config.pad_token_id)
    return batch


def print_batch_example(batch: Dict[str, torch.Tensor], tokenizer: ToyTokenizer) -> None:
    """
    打印 toy batch 的内容，帮助读者直观理解输入输出格式。

    会显示：
    - input_ids 及对应的实际 token
    - token_type_ids（哪个词属于句子 A / B）
    - MLM 标签（哪个位置被 mask，原始 token 是什么）
    - NSP 标签（句对是否匹配）
    """

    print("=" * 80)
    print("1. BERT 输入样例")
    print("=" * 80)
    input_ids = batch["input_ids"][0].tolist()
    mlm_labels = batch["mlm_labels"][0].tolist()
    # 将 token id 解码为可读的文字（不在词表中的显示为 <id>）
    tokens = [tokenizer.decode_id(idx) if idx in tokenizer.id_to_token else f"<{idx}>" for idx in input_ids]

    print("样本 0 的 input_ids:", input_ids)
    print("样本 0 的 token 序列:", tokens)
    print("样本 0 的 token_type_ids:", batch["token_type_ids"][0].tolist())
    print("样本 0 的 mlm_labels:", mlm_labels)
    print("（-100 表示忽略，非 -100 的位置就是被 mask 的原始 token id）")
    print("样本 0 的 nsp_label:", batch["nsp_labels"][0].item())


def demo_forward_pass(config: BertConfig, device: torch.device) -> None:
    """
    演示一次完整前向传播，展示输入、输出格式和各组件的工作结果。

    主要目的：
    1. 展示 BERT 网络各层输出的形状（shape）
    2. 检查 MLM 头在没有训练时能否猜对 masked token
    3. 让读者直观看到 attention 分数的形状
    """

    tokenizer = ToyTokenizer()
    batch = build_toy_pretraining_batch(tokenizer, config, device)
    print_batch_example(batch, tokenizer)

    model = BertForPretrainingDemo(config).to(device)
    mlm_logits, nsp_logits, all_attention_probs = model(
        input_ids=batch["input_ids"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"],
    )

    print("\n" + "=" * 80)
    print("2. BERT 单次前向传播（未训练时的输出）")
    print("=" * 80)
    print(f"MLM logits 形状: {tuple(mlm_logits.shape)}")
    print(f"  └─ [batch_size, seq_len, vocab_size]：每个位置在词表上的得分")
    print(f"NSP logits 形状: {tuple(nsp_logits.shape)}")
    print(f"  └─ [batch_size, 2]：两个类别的得分（匹配 / 不匹配）")
    print(f"Encoder 层数: {len(all_attention_probs)}")
    print(f"第 1 层 attention 形状: {tuple(all_attention_probs[0].shape)}")
    print(f"  └─ [batch_size, num_heads, seq_len, seq_len]：注意力权重矩阵")

    # ---- 检查 MLM 预测是否正确 ----
    mask_positions = batch["mlm_labels"] != -100  # 找出被 mask 的位置
    masked_logits = mlm_logits[mask_positions]    # 取这些位置的 logits
    masked_labels = batch["mlm_labels"][mask_positions]  # 真实 token id
    pred_ids = masked_logits.argmax(dim=-1)       # logits 最大的 id 作为预测结果

    print("被 mask 位置的真实 token id:", masked_labels.tolist())
    print("被 mask 位置的预测 token id:", pred_ids.tolist())
    print("（未训练时预测往往是错的，这很正常！训练后会逐渐收敛。）")


def demo_training_loop(config: BertConfig, device: torch.device) -> None:
    """
    演示极简的预训练风格训练循环。

    BERT 预训练的损失函数 = MLM 损失 + NSP 损失
      - MLM 损失：预测被 mask 位置的原始 token，使用交叉熵损失
      - NSP 损失：判断句对是否匹配，使用二分类交叉熵损失

    注意：
    - 这里每步都重新生成 batch（而不是用固定的数据集），
      是为了让训练过程更自包含，但对模型收敛不利。
    - 真正的预训练需要海量数据和长时间训练。
    - 本 demo 仅演示代码流程，30 步后损失会下降但远未收敛。
    """

    tokenizer = ToyTokenizer()
    model = BertForPretrainingDemo(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("\n" + "=" * 80)
    print("3. BERT toy training demo")
    print("=" * 80)

    for step in range(1, config.train_steps + 1):
        # ---- Step 1: 构造一个 toy batch（MLM + NSP） ----
        batch = build_toy_pretraining_batch(tokenizer, config, device)

        # ---- Step 2: 前向传播 ----
        mlm_logits, nsp_logits, _ = model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )

        # ---- Step 3: 计算损失 ----
        # MLM 损失：交叉熵，忽略标签为 -100 的位置
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, config.vocab_size),  # [batch*seq_len, vocab]
            batch["mlm_labels"].view(-1),             # [batch*seq_len]
            ignore_index=-100,  # 非 mask 位置的标签是 -100，自动忽略
        )
        # NSP 损失：二分类交叉熵
        nsp_loss = F.cross_entropy(nsp_logits, batch["nsp_labels"])
        # 总损失 = 两个损失之和（注意：两个损失的量级需要平衡）
        total_loss = mlm_loss + nsp_loss

        # ---- Step 4: 反向传播 + 参数更新 ----
        optimizer.zero_grad()  # 清除上一轮的梯度（不清除梯度会累积）
        total_loss.backward()  # 反向传播，计算梯度
        optimizer.step()       # 用梯度更新模型参数

        # ---- Step 5: 每 5 步打印一次损失 ----
        if step == 1 or step % 5 == 0 or step == config.train_steps:
            print(
                f"step={step:02d} | "
                f"mlm_loss={mlm_loss.item():.6f} | "
                f"nsp_loss={nsp_loss.item():.6f} | "
                f"total_loss={total_loss.item():.6f}"
            )

    # 训练结束后，重新做一次预测，看看 MLM 是否更准确了
    print("\n训练后再次检查 MLM 预测：")
    with torch.no_grad():  # 推理时不需要计算梯度
        batch = build_toy_pretraining_batch(tokenizer, config, device)
        mlm_logits, _, _ = model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )
    mask_positions = batch["mlm_labels"] != -100
    masked_labels = batch["mlm_labels"][mask_positions]
    pred_ids = mlm_logits[mask_positions].argmax(dim=-1)
    for true_id, pred_id in zip(masked_labels.tolist(), pred_ids.tolist()):
        true_word = tokenizer.decode_id(true_id)
        pred_word = tokenizer.decode_id(pred_id)
        print(f"  真实词: {true_word} | 预测词: {pred_word}")


def main() -> None:
    """
    主函数：运行完整的 BERT 教学演示。

    流程：
    1. 设置配置和随机种子
    2. 演示一次前向传播（非训练状态）
    3. 演示一个极简训练循环（30 步）
    4. 打印关键概念回顾
    """

    config = BertConfig()
    torch.manual_seed(config.seed)  # 固定随机种子，确保每次运行结果一致

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print("\nBERT 配置:", config)

    demo_forward_pass(config, device)
    demo_training_loop(config, device)

    print("\n" + "=" * 80)
    print("4. 关键概念回顾")
    print("=" * 80)
    print("📌 BERT 的核心是双向 Transformer Encoder，它能同时利用左右上下文。")
    print("   与单向模型（如 GPT）不同，BERT 没有 causal mask，可以看到全部 tokens。")
    print()
    print("📌 输入表示由 token / position / token type 三种 embedding 相加得到。")
    print("   Token Embedding：词的语义。")
    print("   Position Embedding：词在句子中的位置。")
    print("   Token Type Embedding：区分句子 A 和句子 B。")
    print()
    print("📌 经典预训练任务包括 MLM 和 NSP：")
    print("   - MLM（Masked Language Model）：遮住部分词，让模型预测。")
    print("   - NSP（Next Sentence Prediction）：判断两句话是否相邻。")
    print("   现代变体（如 RoBERTa）中常见只保留 MLM 或换成其他目标（如 SOP）。")
    print()
    print("📌 本 demo 的局限：")
    print("   - 词表极小，仅用于演示结构。")
    print("   - 数据量极小（仅 2 个样本），模型无法真正学到语言知识。")
    print("   - 真实 BERT 需要在海量语料上训练数天甚至数周。")
    print()
    print("📌 常见混淆澄清：")
    print('   你看到的 "defussion" 大概率是 "diffusion"（扩散模型）的误拼。')
    print('   Diffusion 指一类生成模型：通过「加噪声 → 学去噪」来生成数据，常见于图像生成。')
    print("   它和这个文件里的 BERT 不是同一种模型方向，请勿混淆。")
    print()
    print("🔥 推荐进阶阅读：")
    print("   1. BERT 原论文: https://arxiv.org/abs/1810.04805")
    print("   2. The Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html")
    print("   3. HuggingFace Transformers 库: https://github.com/huggingface/transformers")


if __name__ == "__main__":
    main()
