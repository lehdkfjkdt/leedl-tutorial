import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BertConfig:
    """
    一个教学版 BERT 的超参数配置。

    这个示例不是为了复刻完整工业实现，
    而是帮助理解 BERT 的关键组件如何拼起来。
    """

    vocab_size: int = 32
    # toy 词表大小。为了方便演示，我们手写一个很小的词表。
    max_position_embeddings: int = 32
    # 最长支持多少个 token。
    type_vocab_size: int = 2
    # segment embedding 的种类数。句子 A 用 0，句子 B 用 1。
    hidden_size: int = 64
    # BERT 主干隐藏维度。
    num_heads: int = 4
    # 多头注意力头数。
    intermediate_size: int = 128
    # FFN 中间层维度。
    num_layers: int = 2
    # Encoder 堆叠层数。
    dropout: float = 0.1
    # dropout 概率。
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    batch_size: int = 2
    learning_rate: float = 1e-3
    train_steps: int = 30
    seed: int = 42


class BertEmbeddings(nn.Module):
    """
    BERT 的输入表示由三部分相加得到：
    1. token embedding：词本身的语义
    2. position embedding：位置信息
    3. token type embedding：句子 A / 句子 B 的区分
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # position_ids: [1, seq_len]，再通过广播扩展到整个 batch。
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        word_embed = self.word_embeddings(input_ids)
        pos_embed = self.position_embeddings(position_ids)
        type_embed = self.token_type_embeddings(token_type_ids)

        embeddings = word_embed + pos_embed + type_embed
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class MultiHeadSelfAttention(nn.Module):
    """
    BERT 使用的是双向自注意力。

    与自回归语言模型不同，它不会做 causal mask，
    因为 BERT 在预训练时允许同时看到左右文。
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]

        返回:
            context: [batch_size, seq_len, hidden_size]
            attention_probs: [batch_size, num_heads, seq_len, seq_len]
        """

        q = self._split_heads(self.q_proj(hidden_states))
        k = self._split_heads(self.k_proj(hidden_states))
        v = self._split_heads(self.v_proj(hidden_states))

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # attention_mask 中有效位置为 1，padding 位置为 0。
        # 因此对无效位置加一个极小值，让 softmax 后趋近于 0。
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)

        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = self._merge_heads(context)
        context = self.out_proj(context)
        return context, attention_probs


class BertLayer(nn.Module):
    """
    单层 Transformer Encoder Block。

    结构顺序：
    self-attention -> add & norm -> FFN -> add & norm
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.attn_norm = nn.LayerNorm(config.hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_probs = self.attention(hidden_states, attention_mask)
        hidden_states = self.attn_norm(hidden_states + self.attn_dropout(attn_out))

        ffn_out = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ffn_out)
        return hidden_states, attn_probs


class BertEncoder(nn.Module):
    """
    堆叠多个 BertLayer。
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_attention_probs = []
        for layer in self.layers:
            hidden_states, attn_probs = layer(hidden_states, attention_mask)
            all_attention_probs.append(attn_probs)
        return hidden_states, all_attention_probs


class BertPooler(nn.Module):
    """
    BERT 常使用 [CLS] 位置的表示做句级任务。

    pooler 的做法是：
    取第一个 token 的 hidden state -> 线性层 -> tanh
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_state = hidden_states[:, 0]
        return self.activation(self.dense(cls_state))


class BertModel(nn.Module):
    """
    教学版 BERT 主干。

    输出两类表示：
    1. sequence_output: 每个 token 的上下文化表示
    2. pooled_output: [CLS] 的句级表示
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output, all_attention_probs = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output, all_attention_probs


class BertForPretrainingDemo(nn.Module):
    """
    一个用于教学的 BERT 预训练示例模型。

    包含两个头：
    1. MLM 头：预测被 [MASK] 遮住的位置原本是什么词
    2. NSP 风格句对分类头：判断句子 B 是否是句子 A 的“匹配句”
       这里我们不严格复刻论文数据构造，只做一个直观示例
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.bert = BertModel(config)

        self.mlm_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.nsp_classifier = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        sequence_output, pooled_output, all_attention_probs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        mlm_hidden = self.mlm_transform(sequence_output)
        mlm_logits = self.mlm_decoder(mlm_hidden)
        nsp_logits = self.nsp_classifier(pooled_output)
        return mlm_logits, nsp_logits, all_attention_probs


class ToyTokenizer:
    """
    为了让 demo 完全自包含，这里手写一个极小词表。

    真实场景一般会使用 WordPiece / BPE 等子词分词器；
    本示例只想突出 BERT 模型结构，因此用空格分词即可。
    """

    def __init__(self) -> None:
        self.vocab = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[MASK]": 3,
            "i": 4,
            "love": 5,
            "nlp": 6,
            "you": 7,
            "like": 8,
            "deep": 9,
            "learning": 10,
            "we": 11,
            "study": 12,
            "bert": 13,
            "moe": 14,
            "is": 15,
            "fun": 16,
            "powerful": 17,
            "this": 18,
            "model": 19,
            "great": 20,
        }
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def encode_sentence(self, text: str) -> List[int]:
        return [self.vocab[token] for token in text.split()]

    def decode_id(self, idx: int) -> str:
        return self.id_to_token[idx]


def build_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    构造 BERT 的 padding mask。

    返回形状:
        [batch_size, 1, 1, seq_len]
    """

    return (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)


def build_toy_pretraining_batch(
    tokenizer: ToyTokenizer,
    config: BertConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    构造一个极小的 toy 预训练 batch。

    每个样本都由两句话组成：
    [CLS] sentence_a [SEP] sentence_b [SEP]

    为了演示 MLM，我们会手工挑一个位置替换成 [MASK]，
    并记录原始标签；没被 mask 的位置标签设为 -100，
    这样交叉熵会自动忽略它们。
    """

    sentence_pairs = [
        ("i love nlp", "bert is powerful", 1),
        ("we study bert", "deep learning is fun", 0),
    ]

    input_id_list = []
    token_type_id_list = []
    mlm_label_list = []
    nsp_labels = []

    for sent_a, sent_b, pair_label in sentence_pairs[: config.batch_size]:
        ids_a = tokenizer.encode_sentence(sent_a)
        ids_b = tokenizer.encode_sentence(sent_b)

        # 拼出 BERT 标准句对格式。
        input_ids = [config.cls_token_id] + ids_a + [config.sep_token_id] + ids_b + [config.sep_token_id]

        # [CLS] A [SEP] 都记为 segment 0；B [SEP] 记为 segment 1。
        token_type_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)

        # 初始化 MLM 标签，默认全部忽略。
        mlm_labels = [-100] * len(input_ids)

        # 这里手工选择一个 token 做 mask，便于读者清楚看见任务定义。
        # 第一个样本遮住 "nlp"，第二个样本遮住 "learning"。
        if pair_label == 1:
            mask_position = 3
        else:
            mask_position = len(ids_a) + 3

        mlm_labels[mask_position] = input_ids[mask_position]
        input_ids[mask_position] = config.mask_token_id

        input_id_list.append(input_ids)
        token_type_id_list.append(token_type_ids)
        mlm_label_list.append(mlm_labels)
        nsp_labels.append(pair_label)

    max_len = max(len(x) for x in input_id_list)

    def pad_to_max_len(items: List[List[int]], pad_value: int) -> torch.Tensor:
        padded = [item + [pad_value] * (max_len - len(item)) for item in items]
        return torch.tensor(padded, dtype=torch.long, device=device)

    batch = {
        "input_ids": pad_to_max_len(input_id_list, config.pad_token_id),
        "token_type_ids": pad_to_max_len(token_type_id_list, 0),
        "mlm_labels": pad_to_max_len(mlm_label_list, -100),
        "nsp_labels": torch.tensor(nsp_labels, dtype=torch.long, device=device),
    }
    batch["attention_mask"] = build_attention_mask(batch["input_ids"], config.pad_token_id)
    return batch


def print_batch_example(batch: Dict[str, torch.Tensor], tokenizer: ToyTokenizer) -> None:
    """
    打印 toy batch，帮助读者把 token id 和任务标签对应起来。
    """

    print("=" * 80)
    print("1. BERT 输入样例")
    print("=" * 80)
    input_ids = batch["input_ids"][0].tolist()
    mlm_labels = batch["mlm_labels"][0].tolist()
    tokens = [tokenizer.decode_id(idx) if idx in tokenizer.id_to_token else f"<{idx}>" for idx in input_ids]

    print("样本 0 的 input_ids:", input_ids)
    print("样本 0 的 token 序列:", tokens)
    print("样本 0 的 token_type_ids:", batch["token_type_ids"][0].tolist())
    print("样本 0 的 mlm_labels:", mlm_labels)
    print("样本 0 的 nsp_label:", batch["nsp_labels"][0].item())


def demo_forward_pass(config: BertConfig, device: torch.device) -> None:
    """
    演示一次完整前向传播。
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
    print("2. BERT 单次前向传播")
    print("=" * 80)
    print(f"MLM logits 形状: {tuple(mlm_logits.shape)}")
    print(f"NSP logits 形状: {tuple(nsp_logits.shape)}")
    print(f"Encoder 层数: {len(all_attention_probs)}")
    print(f"第 1 层 attention 形状: {tuple(all_attention_probs[0].shape)}")

    mask_positions = batch["mlm_labels"] != -100
    masked_logits = mlm_logits[mask_positions]
    masked_labels = batch["mlm_labels"][mask_positions]
    pred_ids = masked_logits.argmax(dim=-1)

    print("被 mask 位置的真实 token id:", masked_labels.tolist())
    print("被 mask 位置的预测 token id:", pred_ids.tolist())


def demo_training_loop(config: BertConfig, device: torch.device) -> None:
    """
    演示一个极简的预训练风格训练循环。

    总损失 = MLM loss + NSP loss
    """

    tokenizer = ToyTokenizer()
    model = BertForPretrainingDemo(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("\n" + "=" * 80)
    print("3. BERT toy training demo")
    print("=" * 80)

    for step in range(1, config.train_steps + 1):
        batch = build_toy_pretraining_batch(tokenizer, config, device)

        mlm_logits, nsp_logits, _ = model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, config.vocab_size),
            batch["mlm_labels"].view(-1),
            ignore_index=-100,
        )
        nsp_loss = F.cross_entropy(nsp_logits, batch["nsp_labels"])
        total_loss = mlm_loss + nsp_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % 5 == 0 or step == config.train_steps:
            print(
                f"step={step:02d} | "
                f"mlm_loss={mlm_loss.item():.6f} | "
                f"nsp_loss={nsp_loss.item():.6f} | "
                f"total_loss={total_loss.item():.6f}"
            )


def main() -> None:
    config = BertConfig()
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print("BERT 配置:", config)

    demo_forward_pass(config, device)
    demo_training_loop(config, device)

    print("\n" + "=" * 80)
    print("4. 关键概念回顾")
    print("=" * 80)
    print("BERT 的核心是双向 Transformer Encoder，它能同时利用左右上下文。")
    print("输入表示由 token / position / token type 三种 embedding 相加得到。")
    print("经典预训练任务包括 MLM 和 NSP；现代变体中也常见只保留 MLM 或换成其他目标。")


if __name__ == "__main__":
    main()
