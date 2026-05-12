import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """完整 Transformer 的超参数配置。"""

    src_vocab_size: int = 1000
    tgt_vocab_size: int = 1200
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 256
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2


def make_src_padding_mask(src_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    构造 Encoder 端的 padding mask。

    参数:
        src_tokens: [batch_size, src_len]
        pad_id: padding token 的 id

    返回:
        mask: [batch_size, 1, 1, src_len]
        True 表示这个位置可以被注意到，False 表示这个位置是 padding，需要被屏蔽。
    """

    return (src_tokens != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_padding_mask(tgt_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    构造 Decoder 端的 padding mask。

    参数:
        tgt_tokens: [batch_size, tgt_len]
        pad_id: padding token 的 id

    返回:
        mask: [batch_size, 1, 1, tgt_len]
    """

    return (tgt_tokens != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(tgt_len: int, device: torch.device) -> torch.Tensor:
    """
    构造 Decoder 自回归使用的 causal mask。

    参数:
        tgt_len: 目标序列长度

    返回:
        mask: [1, 1, tgt_len, tgt_len]

    含义:
        第 t 个位置只能看见 0..t 的位置，不能看未来。
    """

    return torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    同时结合 padding mask 和 causal mask。

    参数:
        tgt_tokens: [batch_size, tgt_len]

    返回:
        mask: [batch_size, 1, tgt_len, tgt_len]
    """

    tgt_padding_mask = make_tgt_padding_mask(tgt_tokens, pad_id)
    causal_mask = make_causal_mask(tgt_tokens.size(1), tgt_tokens.device)
    return tgt_padding_mask & causal_mask


class PositionalEncoding(nn.Module):
    """
    正弦位置编码。

    输入:
        x: [batch_size, seq_len, d_model]

    输出:
        out: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model]，方便和 batch 维广播相加
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding，并按论文做 sqrt(d_model) 缩放。

    输入:
        token_ids: [batch_size, seq_len]

    输出:
        embeddings: [batch_size, seq_len, d_model]
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * self.scale


class MultiHeadAttention(nn.Module):
    """
    多头注意力。

    输入:
        query: [batch_size, query_len, d_model]
        key:   [batch_size, key_len, d_model]
        value: [batch_size, key_len, d_model]
        mask:  [batch_size, 1, query_len, key_len] 或可广播到这个形状

    输出:
        out: [batch_size, query_len, d_model]
        attn_weights: [batch_size, num_heads, query_len, key_len]
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把最后一维拆成多头。

        输入:
            x: [batch_size, seq_len, d_model]

        输出:
            out: [batch_size, num_heads, seq_len, head_dim]
        """

        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把多头重新拼回 d_model。

        输入:
            x: [batch_size, num_heads, seq_len, head_dim]

        输出:
            out: [batch_size, seq_len, d_model]
        """

        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入张量维度:
        # query: [batch_size, query_len, d_model]
        # key:   [batch_size, key_len, d_model]
        # value: [batch_size, key_len, d_model]
        # mask:  [batch_size, 1, query_len, key_len] 或可广播到这个形状

        # 线性映射后再拆成多头:
        # q: [batch_size, num_heads, query_len, head_dim]
        q = self._split_heads(self.q_proj(query))
        # k: [batch_size, num_heads, key_len, head_dim]
        k = self._split_heads(self.k_proj(key))
        # v: [batch_size, num_heads, key_len, head_dim]
        v = self._split_heads(self.v_proj(value))

        # 注意力分数: [batch_size, num_heads, query_len, key_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask 会广播到 scores 的形状: [batch_size, num_heads, query_len, key_len]
            scores = scores.masked_fill(~mask, float("-inf"))

        # attn_weights: [batch_size, num_heads, query_len, key_len]
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # context: [batch_size, num_heads, query_len, head_dim]
        context = torch.matmul(attn_weights, v)
        # 合并多头后 out: [batch_size, query_len, d_model]
        out = self._merge_heads(context)
        # 输出线性层后 out: [batch_size, query_len, d_model]
        out = self.out_proj(out)
        return out, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络。

    输入:
        x: [batch_size, seq_len, d_model]

    输出:
        out: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """
    单层 Encoder Block。

    结构:
        Self-Attention -> Add & Norm -> FFN -> Add & Norm

    输入:
        x: [batch_size, src_len, d_model]
        src_mask: [batch_size, 1, 1, src_len]

    输出:
        out: [batch_size, src_len, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # 子层 1: Self-Attention
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 子层 2: FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """
    单层 Decoder Block。

    结构:
        Masked Self-Attention -> Add & Norm
        Cross-Attention        -> Add & Norm
        FFN                    -> Add & Norm

    输入:
        x: [batch_size, tgt_len, d_model]
        memory: [batch_size, src_len, d_model]
        tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        src_mask: [batch_size, 1, 1, src_len]

    输出:
        out: [batch_size, tgt_len, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 输入张量维度:
        # x:        [batch_size, tgt_len, d_model]
        # memory:   [batch_size, src_len, d_model]
        # tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        # src_mask: [batch_size, 1, 1, src_len]

        # 子层 1: Decoder 自注意力。这里会用 causal mask，防止看未来。
        # self_attn_out: [batch_size, tgt_len, d_model]
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        # 残差相加和 LayerNorm 后，x 形状保持不变: [batch_size, tgt_len, d_model]
        x = self.norm1(x + self.dropout(self_attn_out))

        # 子层 2: Cross-Attention。query 来自 decoder，key/value 来自 encoder。
        # query=x:   [batch_size, tgt_len, d_model]
        # key=value: [batch_size, src_len, d_model]
        # cross_attn_out: [batch_size, tgt_len, d_model]
        cross_attn_out, _ = self.cross_attn(x, memory, memory, src_mask)
        # 残差相加和 LayerNorm 后，x 形状保持不变: [batch_size, tgt_len, d_model]
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 子层 3: FFN
        # ffn_out: [batch_size, tgt_len, d_model]
        ffn_out = self.ffn(x)
        # 输出 x: [batch_size, tgt_len, d_model]
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    """
    完整 Encoder。

    输入:
        src_tokens: [batch_size, src_len]
        src_mask: [batch_size, 1, 1, src_len]

    输出:
        memory: [batch_size, src_len, d_model]
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(config.src_vocab_size, config.d_model)
        self.position_encoding = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
                for _ in range(config.num_encoder_layers)
            ]
        )

    def forward(self, src_tokens: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embedding(src_tokens)
        x = self.position_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """
    完整 Decoder。

    输入:
        tgt_tokens: [batch_size, tgt_len]
        memory: [batch_size, src_len, d_model]
        tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        src_mask: [batch_size, 1, 1, src_len]

    输出:
        decoder_hidden: [batch_size, tgt_len, d_model]
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(config.tgt_vocab_size, config.d_model)
        self.position_encoding = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embedding(tgt_tokens)
        x = self.position_encoding(x)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x


class Transformer(nn.Module):
    """
    完整的 Encoder-Decoder Transformer。

    forward 输入:
        src_tokens: [batch_size, src_len]
        tgt_tokens: [batch_size, tgt_len]

    forward 输出:
        logits: [batch_size, tgt_len, tgt_vocab_size]

    说明:
        这里的 tgt_tokens 应当是“右移后的 decoder 输入”。
        如果真实目标序列是 [BOS, y1, y2, y3, EOS]，
        那么训练时通常会把 [BOS, y1, y2, y3] 送进来，
        让模型预测 [y1, y2, y3, EOS]。
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        # config 保存整个模型的超参数，比如词表大小、d_model、层数、pad_id 等。
        self.config = config

        # encoder 负责读取源序列 src_tokens，输出 memory。
        # encoder 输入:  src_tokens -> [batch_size, src_len]
        # encoder 输出: memory     -> [batch_size, src_len, d_model]
        self.encoder = Encoder(config)

        # decoder 负责读取目标前缀 tgt_tokens，并结合 encoder 的 memory 生成隐藏状态。
        # decoder 输入:  tgt_tokens      -> [batch_size, tgt_len]
        #               memory          -> [batch_size, src_len, d_model]
        # decoder 输出: decoder_hidden  -> [batch_size, tgt_len, d_model]
        self.decoder = Decoder(config)

        # output_projection 把 decoder 的隐藏状态映射到目标词表大小。
        # 输入:  [batch_size, tgt_len, d_model]
        # 输出: [batch_size, tgt_len, tgt_vocab_size]
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size)

    def encode(self, src_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # src_tokens: [batch_size, src_len]

        # src_mask 用来屏蔽源序列中的 PAD 位置。
        # src_mask: [batch_size, 1, 1, src_len]
        src_mask = make_src_padding_mask(src_tokens, self.config.pad_id)

        # encoder 读取源序列，输出每个源位置的上下文表示。
        # memory: [batch_size, src_len, d_model]
        memory = self.encoder(src_tokens, src_mask)

        # 返回两个东西:
        # 1. memory: 后面给 decoder 的 cross-attention 使用
        # 2. src_mask: 后面给 decoder 的 cross-attention 屏蔽源端 PAD 使用
        return memory, src_mask

    def decode(self, tgt_tokens: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # tgt_tokens: [batch_size, tgt_len]
        # memory:     [batch_size, src_len, d_model]
        # src_mask:   [batch_size, 1, 1, src_len]

        # tgt_mask 同时包含:
        # 1. target padding mask
        # 2. causal mask（防止 decoder 看未来）
        # tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        tgt_mask = make_tgt_mask(tgt_tokens, self.config.pad_id)

        # decoder 结合目标前缀和 encoder memory，输出目标侧隐藏状态。
        # decoder_hidden: [batch_size, tgt_len, d_model]
        decoder_hidden = self.decoder(tgt_tokens, memory, tgt_mask, src_mask)

        # 线性映射到目标词表，得到每个位置对所有词的分类分数。
        # logits: [batch_size, tgt_len, tgt_vocab_size]
        return self.output_projection(decoder_hidden)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        # src_tokens: [batch_size, src_len]
        # tgt_tokens: [batch_size, tgt_len]

        # 第一步: 先编码源序列。
        # memory:   [batch_size, src_len, d_model]
        # src_mask: [batch_size, 1, 1, src_len]
        memory, src_mask = self.encode(src_tokens)

        # 第二步: 再解码目标前缀，得到最终词表 logits。
        # logits: [batch_size, tgt_len, tgt_vocab_size]
        return self.decode(tgt_tokens, memory, src_mask)


@torch.no_grad()
def greedy_decode(model: Transformer, src_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """
    最简单的贪心解码示例。

    输入:
        src_tokens: [batch_size, src_len]

    输出:
        generated: [batch_size, <= max_new_tokens + 1]
    """

    # 推理阶段关闭 dropout 等训练专用行为。
    model.eval()

    # 先只对源序列做一次编码。
    # memory:   [batch_size, src_len, d_model]
    # src_mask: [batch_size, 1, 1, src_len]
    memory, src_mask = model.encode(src_tokens)

    # batch_size: 标量，表示一批里有多少条样本。
    batch_size = src_tokens.size(0)

    # 生成序列的初始状态只包含 BOS。
    # generated: [batch_size, 1]
    generated = torch.full(
        (batch_size, 1),
        fill_value=model.config.bos_id,
        dtype=torch.long,
        device=src_tokens.device,
    )

    for _ in range(max_new_tokens):
        # 把“当前已经生成的前缀”送入 decoder。
        # generated: [batch_size, cur_len]
        # logits:    [batch_size, cur_len, tgt_vocab_size]
        logits = model.decode(generated, memory, src_mask)

        # 只取最后一个位置的预测结果。
        # logits[:, -1, :]: [batch_size, tgt_vocab_size]
        # next_token:       [batch_size, 1]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # 把新预测出的 token 接到 generated 后面。
        # 拼接后 generated: [batch_size, cur_len + 1]
        generated = torch.cat([generated, next_token], dim=1)

        # 如果这一轮所有样本都生成了 EOS，就提前结束解码。
        # next_token.squeeze(1): [batch_size]
        if torch.all(next_token.squeeze(1) == model.config.eos_id):
            break

    return generated


def run_unit_test() -> None:
    """
    一个最小可运行测试：
    1. 构造假的源序列和目标序列
    2. 跑完整 forward
    3. 打印输入输出维度
    4. 计算一个训练时常见的交叉熵损失
    5. 跑一次贪心解码
    """

    torch.manual_seed(7)

    config = TransformerConfig(
        src_vocab_size=100,
        tgt_vocab_size=120,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.1,
        max_len=32,
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )

    model = Transformer(config)

    # 形状: [batch_size=2, src_len=7]
    src_tokens = torch.tensor(
        [
            [1, 11, 12, 13, 2, 0, 0],
            [1, 21, 22, 23, 24, 2, 0],
        ],
        dtype=torch.long,
    )

    # 这里假设完整目标序列是 [BOS, y1, y2, y3, EOS, PAD]
    # 形状: [batch_size=2, tgt_len=6]
    tgt_full = torch.tensor(
        [
            [1, 31, 32, 33, 2, 0],
            [1, 41, 42, 2, 0, 0],
        ],
        dtype=torch.long,
    )

    # decoder 输入: 去掉最后一个 token，形状 [2, 5]
    decoder_input = tgt_full[:, :-1]
    # 训练目标: 去掉第一个 token，形状 [2, 5]
    decoder_target = tgt_full[:, 1:]

    logits = model(src_tokens, decoder_input)

    print("=" * 80)
    print("完整 Transformer 前向测试")
    print("=" * 80)
    print(f"src_tokens.shape      = {tuple(src_tokens.shape)}")
    print(f"decoder_input.shape   = {tuple(decoder_input.shape)}")
    print(f"decoder_target.shape  = {tuple(decoder_target.shape)}")
    print(f"logits.shape          = {tuple(logits.shape)}")
    print("说明: logits 的最后一维是目标词表大小 tgt_vocab_size")

    loss = F.cross_entropy(
        logits.reshape(-1, config.tgt_vocab_size),
        decoder_target.reshape(-1),
        ignore_index=config.pad_id,
    )
    print(f"cross_entropy_loss    = {loss.item():.6f}")

    generated = greedy_decode(model, src_tokens[:1], max_new_tokens=6)
    print(f"generated.shape       = {tuple(generated.shape)}")
    print(f"generated tokens      = {generated.tolist()}")
    print("=" * 80)


def run_training_demo() -> None:
    """
    演示一个最小可读的训练流程。

    这里故意使用和 run_unit_test 一样的假数据，方便把注意力放在“训练代码长什么样”上：
    1. 构造模型和优化器
    2. 准备 src_tokens、decoder_input、decoder_target
    3. 前向计算 logits
    4. 计算交叉熵损失
    5. 反向传播并更新参数

    注意:
        这里只是教学示例，数据是手写的假样本。
        模型参数是随机初始化的，因此 loss 下降趋势和生成结果都没有真实任务语义。
    """

    # 固定随机种子，确保每次运行时初始参数和示例输出更稳定，方便教学观察。
    torch.manual_seed(7)

    config = TransformerConfig(
        src_vocab_size=100,
        tgt_vocab_size=120,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.1,
        max_len=32,
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )

    # model: 完整的 Encoder-Decoder Transformer
    # optimizer: 负责根据梯度更新模型参数
    model = Transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 源序列，形状: [batch_size=2, src_len=7]
    # 每一行代表一条输入样本，0 表示 PAD。
    src_tokens = torch.tensor(
        [
            [1, 11, 12, 13, 2, 0, 0],
            [1, 21, 22, 23, 24, 2, 0],
        ],
        dtype=torch.long,
    )

    # 完整目标序列，形状: [batch_size=2, tgt_len=6]
    # 这里假设格式是 [BOS, y1, y2, ..., EOS, PAD]
    tgt_full = torch.tensor(
        [
            [1, 31, 32, 33, 2, 0],
            [1, 41, 42, 2, 0, 0],
        ],
        dtype=torch.long,
    )

    # Teacher Forcing 训练时：
    # 1. decoder_input 是“右移后的目标前缀”，真正送入 decoder。
    # 2. decoder_target 是“下一个位置的真实答案”，用于和 logits 对齐计算 loss。
    # 3. 例如 [BOS, 31, 32, 33, EOS, PAD] 会被切成:
    #    decoder_input  = [BOS, 31, 32, 33, EOS]
    #    decoder_target = [31, 32, 33, EOS, PAD]
    # decoder_input:  [batch_size=2, tgt_len=5]
    # decoder_target: [batch_size=2, tgt_len=5]
    decoder_input = tgt_full[:, :-1]
    decoder_target = tgt_full[:, 1:]

    print("=" * 80)
    print("Transformer 训练示例")
    print("=" * 80)

    # 这里做 3 步优化，目的是把标准训练循环写完整。
    for step in range(1, 4):
        # 切到训练模式。像 Dropout 这类层在 train/eval 下行为不同。
        model.train()

        # 清空上一步残留梯度。
        optimizer.zero_grad()

        # 前向传播:
        # src_tokens:    [batch_size=2, src_len=7]
        # decoder_input: [batch_size=2, tgt_len=5]
        # logits:        [batch_size=2, tgt_len=5, tgt_vocab_size=120]
        # 含义: 对 decoder 的每个位置，都输出一个长度为 tgt_vocab_size 的分类分数向量。
        logits = model(src_tokens, decoder_input)

        # 交叉熵要求输入形状是 [N, C]，目标形状是 [N]。
        # 因此这里把前两维 batch_size 和 tgt_len 合并:
        # logits.reshape(-1, config.tgt_vocab_size): [2 * 5, 120] = [10, 120]
        # decoder_target.reshape(-1):                [2 * 5]      = [10]
        # ignore_index=config.pad_id 表示 PAD 位置不参与 loss 计算。
        loss = F.cross_entropy(
            logits.reshape(-1, config.tgt_vocab_size),
            decoder_target.reshape(-1),
            ignore_index=config.pad_id,
        )

        # 反向传播: 根据 loss 计算每个可训练参数的梯度。
        loss.backward()

        # 参数更新: optimizer 使用当前梯度更新模型参数。
        optimizer.step()

        # predicted_tokens: [batch_size=2, tgt_len=5]
        # 这里只是把每个位置 logits 最大的 token id 取出来，方便观察当前模型预测。
        predicted_tokens = logits.argmax(dim=-1)
        print(f"step {step} loss        = {loss.item():.6f}")
        print(f"step {step} logits.shape = {tuple(logits.shape)}")
        print(f"step {step} pred tokens  = {predicted_tokens.tolist()}")

    print("=" * 80)


def run_inference_demo() -> None:
    """
    演示一个最小可读的推理流程。

    推理时和训练最大的区别是：
    1. 不再把完整目标序列喂给 decoder
    2. 只给 src_tokens
    3. decoder 从 BOS 开始，逐步生成下一个 token

    这里直接调用 greedy_decode，它内部会：
    - 先 encode(src_tokens)
    - 再从 [BOS] 开始一步一步生成
    - 每一步取当前最后一个位置上概率最大的 token
    """

    torch.manual_seed(7)

    config = TransformerConfig(
        src_vocab_size=100,
        tgt_vocab_size=120,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.1,
        max_len=32,
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )

    model = Transformer(config)

    # 推理时只需要源序列，不需要完整目标答案。
    # 这里放 2 条源序列，形状: [batch_size=2, src_len=7]
    src_tokens = torch.tensor(
        [
            [1, 11, 12, 13, 2, 0, 0],
            [1, 21, 22, 23, 24, 2, 0],
        ],
        dtype=torch.long,
    )

    generated = greedy_decode(model, src_tokens, max_new_tokens=6)

    print("=" * 80)
    print("Transformer 推理示例")
    print("=" * 80)
    print(f"src_tokens.shape      = {tuple(src_tokens.shape)}")
    print(f"generated.shape       = {tuple(generated.shape)}")
    print(f"generated tokens      = {generated.tolist()}")
    print("说明: 这里是随机初始化模型的贪心解码结果，只用来演示推理流程。")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_test()
    run_training_demo()
    run_inference_demo()