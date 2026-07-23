# Embedding（嵌入）详解：从词到向量的完整指南

> **主题**：Embedding——把离散 token 变成连续向量，让 Transformer 能"理解"语义
> **用途**：理解 Transformer 中所有 token（词、图像 patch、动作模式）如何变成模型可计算的形式
> **关联文档**：[Transformer 论文深度解读](./transformer_attention_guide.md) · [Positional Encoding 详解](./positional_encoding_guide.md) · [Attention 机制深度解读](./attention_guide.md)

> 📎 **扩展阅读**：本文也作为 [π₀-FAST 论文导读](../paper-review/Robots-VLA/FAST/fast_paper_guide.md) 的配套文档——VLA 中 embedding 的应用场景见 §四。

---

## 读之前先搞清楚

### Embedding 在做什么？

Embedding 回答一个根本问题：**"计算机只认数字，'猫'这个字怎么变成数字？"**

最粗暴的方式是给每个字一个编号——"猫"=537，"狗"=538。但这暗示"猫"和"狗"的关系跟"猫"和"冰箱"的关系一样（都是差 1）——显然不对。"猫"和"狗"应该更相似。

Embedding 的解决方案：不给每个词一个数字，而是**给每个词一个向量**（一串数字，比如 4096 个）。向量之间的距离代表语义相似度——"猫"的向量和"狗"的向量方向相近，"猫"和"冰箱"的方向差别大。

> **小白理解**：编号像给每个人一个学号——1 号和 2 号除了编号相邻没有任何关系。Embedding 像给每个人一张"性格卡片"——温柔度、幽默感、运动能力……几十个维度的评分。**性格卡片相近的人是真"像"，而不只是编号挨着。**

### 阅读本文需要的前置知识

| 概念 | 简单解释 |
|:---|:---|
| **Token（词元）** | Transformer 处理的最小单位——"playing" → ["play", "ing"]，每个是一个 token |
| **词汇表（Vocabulary）** | 模型能识别的所有 token 的集合——GPT-2 词表大小 50257 |
| **向量** | 一串有序数字，如 [0.3, -0.7, 1.2, ...]——代表一个点在多维空间中的位置 |
| **d_model** | Transformer 的隐藏维度——embedding、attention、FFN 都在这个维度下操作 |
| **GPT-2** | OpenAI 2019 年的纯文本生成模型——最简洁的 Transformer 范例 |
| **ViT（Vision Transformer）** | 把图像切成 16×16 的小块 patch，每块当"一个 token"喂给 Transformer |

---

## 一、Embedding 的直觉理解

### 1.1 从 one-hot 到 embedding——为什么要"嵌入"

计算机处理文字时，第一步总是把 token 变成数字。最直接的方式是 **one-hot 编码**：

```
假设词表只有 5 个词: ["我", "爱", "你", "猫", "狗"]

one-hot 表示:
  "我" → [1, 0, 0, 0, 0]     ← 第 0 位是 1，其余是 0
  "猫" → [0, 0, 0, 1, 0]     ← 第 3 位是 1，其余是 0
  "狗" → [0, 0, 0, 0, 1]     ← 第 4 位是 1，其余是 0
```

one-hot 有两个致命问题：

**问题 1：维度爆炸**。真实词表不是 5 个词，是 5 万~25 万个 token。每条文本可能几千个 token——存储和计算都受不了。

**问题 2：没有语义信息**。one-hot 里"猫"和"狗"的距离 = "猫"和"我"的距离 = 都一样远（都是 √2）。模型无法利用"猫和狗是相似的"这个信息。

Embedding 的解法：**用一个小维度（如 4096）的稠密向量（每个位置都是非零实数）代替巨大的稀疏 one-hot 向量**。

```
one-hot:  [0, 0, 0, 1, 0]          ← 5 维，只有 1 个非零值
          ↓ 乘以 Embedding 矩阵 E (5 × 3)
embedding: [0.12, -0.85, 0.34]     ← 3 维，所有位置都有值

训练后:
  "猫" → [ 0.12, -0.85,  0.34]
  "狗" → [ 0.15, -0.80,  0.30]   ← 和"猫"很接近！
  "我" → [ 0.92,  0.10, -0.45]   ← 和"猫"完全不同
```

> **小白理解**：one-hot 像名单——只告诉你"这是第 537 号学生"，不告诉你他是谁。Embedding 像学生档案——几十个维度的性格、能力、偏好评分。**两份档案相似的学生，是真有可能成为朋友的；两个学号挨着的人，可能毫无关系。**

---

## 二、Embedding 的工作原理

### 2.1 Embedding 矩阵——一张可学习的查找表

Embedding 的数学本质非常简单：**一个矩阵乘法——或者说，一次查表**。

```
Embedding 矩阵 E
  形状: (vocab_size, d_model)

  以 GPT-2 Small 为例:
  vocab_size = 50257, d_model = 768

  E = [                    ← 50257 行
    e_0,                   ← token ID=0 的 embedding，768 个数
    e_1,                   ← token ID=1 的 embedding
    e_2,                   ← token ID=2 的 embedding
    ...
    e_50256,              ← token ID=50256 的 embedding
  ]

输入: token_ids = [464, 380, 290]  ← "The cat sat" 的 GPT-2 token ID
输出: embeddings = [e_464, e_380, e_290]  ← shape: (3, 768)
```

**PyTorch 中就是一行代码**：

```python
# 伪代码
token_ids = torch.tensor([464, 380, 290])   # "The cat sat"
E = nn.Embedding(50257, 768)                 # 词表大小 × 嵌入维度
embeddings = E(token_ids)                    # shape: (3, 768)
# 等价于: embeddings = [E[464], E[380], E[290]]
```

**E 是怎么学出来的**：E 是随机初始化的（每个元素 ~N(0, 0.02²)），然后在训练中和模型其他参数一起通过反向传播更新。训练的目标是：让语义相近的 token 的 embedding 也相近。

#### ❓ 常见疑问：Embedding 矩阵是"查表"还是"计算"？

是查表，不是计算。token ID 464 不是一个"输入值"——它纯粹是行号。Embedding 层的操作就是 `E[464]`——取出第 464 行。这和全连接层（`y = x·W + b`）完全不同——全连接是做矩阵乘法，Embedding 不做乘法，只做索引。

> **小白理解**：Embedding 层像一个字典——你给它一个页码（token ID），它直接翻到那一页给你看内容（向量）。**它不会"计算"这一页的内容，而是训练时一点一点修改字典里每一页的字迹——让相似的词所在的页面内容也相似。**

### 2.2 嵌入维度 d_model 的选择

d_model 是一个关键超参数——它决定了每个 token 用多少个数字表示：

| 模型 | 词表大小 | d_model | 说明 |
|:---|:---:|:---:|:---|
| GPT-2 Small | 50257 | 768 | 1.24 亿参数，经典入门 |
| GPT-2 Medium | 50257 | 1024 | 3.55 亿参数 |
| GPT-3 | 50257 | 12288 | 1750 亿参数 |
| LLaMA-7B | 32000 | 4096 | 开源代表 |
| BERT-base | 30522 | 768 | 双向编码器 |
| ViT-B/16 | N/A（patch） | 768 | 图像 Transformer |

**为什么越大的模型 d_model 越大**：d_model 决定了模型内部"信息带宽"——每个 token 能携带多少信息。复杂任务（如理解长文档、生成代码）需要更大的 d_model。但更大的 d_model 意味着更多计算（attention 的 Q·Kᵀ 矩阵从 768² 变成 4096²）。

> **小白理解**：d_model 像快递盒的大小。d_model=768（GPT-2 Small）是"鞋盒"——装一个词的信息够用。d_model=4096（LLaMA-7B）是"搬家纸箱"——能装更多信息，但搬运（计算）也更费力。**模型越大→需要理解的上下文越复杂→每个 token 需要携带的信息越多→d_model 越大。**

> ❓ **常见疑问：GPT-2 词表 50257 个词，为什么不是整数？**
>
> 50257 = 50000（BPE 基础词表）+ 256（字节值）+ 1（EOS token）。BPE 训练时设了一个 50000 的目标大小，合并轮次到了自动停止。
>
> **任何不是"漂亮的整数"的词表大小都是因为 tokenizer 的训练过程自然收敛到了那里，而不是人为指定的。**

---

## 三、Embedding 的输入构造：Token Emb + Positional Encoding = Transformer 输入

### 3.1 完整流程

Token Embedding 只是 Transformer 输入的一半。还有另一半——位置编码：

```python
# Transformer 的输入构造（两步）
token_ids = [464, 380, 290]  # "The cat sat"

# Step 1: Token Embedding（内容信息——"这个词是什么"）
token_emb = embedding_matrix[token_ids]   # (3, 768)

# Step 2: Positional Encoding（位置信息——"这个词在第几个位置"）
pos_emb = positional_encoding([0, 1, 2])  # (3, 768)

# Step 3: 相加 → 喂给 Transformer
transformer_input = token_emb + pos_emb   # (3, 768)
```

> 详见：[Positional Encoding 详解](./positional_encoding_guide.md)

#### 示意图：GPT-2 处理 "The cat sat on the mat"

```
"The cat sat on the mat"
   ↓ GPT-2 Tokenizer
[464, 3797, 3332, 319, 262, 6603]
   ↓ Token Embedding (768D)
┌─────────────────────────────────────────────────┐
│ e_464  = [-0.12,  0.34,  0.08, ..., -0.21]      │  ← "The" 的语义
│ e_3797 = [ 0.45, -0.11,  0.67, ...,  0.03]      │  ← " cat" 的语义
│ e_3332 = [ 0.22,  0.18, -0.34, ...,  0.41]      │  ← " sat" 的语义
│ e_319  = [ 0.09, -0.52,  0.13, ..., -0.18]      │  ← " on" 的语义
│ e_262  = [-0.31,  0.27,  0.49, ...,  0.12]      │  ← " the" 的语义
│ e_6603 = [ 0.15,  0.44, -0.08, ..., -0.35]      │  ← " mat" 的语义
│              ↓ + Positional Encoding              │
│ [e_464+p0, e_3797+p1, ..., e_6603+p5]            │  ← 6×768 矩阵
└─────────────────────────────────────────────────┘
   ↓ 喂入 12 层 Transformer Decoder
```

### 3.2 Token Embedding 之外的 Embedding 类型

| 类型 | 作用 | 谁在用 |
|:---|:---|:---|
| **Token Embedding** | 把 token ID 变成向量——"这个词是什么" | 所有 Transformer（本文主角） |
| **Positional Encoding** | 告诉模型 token 在序列中的位置——"第几个" | 所有 Transformer（见 [PE 详解](./positional_encoding_guide.md)） |
| **Segment Embedding** | 区分不同句子——"这句话属于 A 还是 B" | BERT（NSP 任务） |
| **Type Embedding** | 区分不同模态——"这条是文本还是图像" | 多模态模型（VLA、PaliGemma） |

> **小白理解**：Token Embedding 是"名字"，Positional Encoding 是"座位号"，Segment Embedding 是"班级"，Type Embedding 是"数据类型"。**GPT 只用名字+座位号；BERT 用名字+座位号+班级；多模态模型用名字+座位号+数据类型。**

---

## 四、多模态模型中的 Embedding——不止是文本

Transformer 最初为文本设计，但很快被推广到图像、语音、机器人动作。核心挑战是：**怎么把非文本数据也变成 d_model 维的向量？**

#### 总览：三种模态的 Embedding 策略

```
┌──────────────────────────────────────────────────────────────────┐
│  ① 纯文本（GPT、LLaMA、BERT）——标准 Embedding                     │
│                                                                  │
│  "The cat sat"                                                   │
│      ↓ Tokenizer（BPE/WordPiece）                               │
│  [464, 3797, 3332]               ← 整数 ID                       │
│      ↓ nn.Embedding(vocab, d_model)                             │
│  (3, d_model) 向量                                               │
│                                                                  │
│  ② 图像（ViT）——Patch Embedding                                  │
│                                                                  │
│  224×224 RGB 图像                                                │
│      ↓ 切成 16×16 的 patch → 14×14=196 个 patch                  │
│      ↓ 每个 patch 展平: 16×16×3 = 768 个数                       │
│      ↓ Linear 投影: 768 → d_model（如 768）                      │
│  (196, d_model) 向量    ← 196个patch，每个是一个"图像词"         │
│                                                                  │
│  ③ 多模态（VLA / PaliGemma）——联合 Embedding                     │
│                                                                  │
│  文本 + 图像 两种 token 混合输入同一 Transformer                  │
│  文本: Tokenizer → Embedding 矩阵 → (N_txt, d_model)             │
│  图像: ViT/ SigLIP → Linear 投影 → (N_img, d_model)             │
│  → 拼接 → (N_txt + N_img, d_model) → 同一空间，注意力自由交互     │
└──────────────────────────────────────────────────────────────────┘
```

**关键原则**：不管什么模态，最终都映射到**同一个 d_model 维空间**。

- GPT-2 中 "cat" 和 "dog" 的 embedding 在同一个 768 维空间
- ViT 中 "左上角的 patch" 和 "右下角的 patch" 在同一个 768 维空间
- 多模态模型（VLA）中 "fold"（文本）和 "衣袖纹理 patch"（图像）在同一个 4096 维空间

> **小白理解**：Embedding 像一个"通用翻译器"——不管输入是中文（文本）、图片（视觉），都翻译成同一种"内部语言"（d_model 维向量）。Transformer 的注意力机制不关心向量从哪来——它们在同一空间里自由交互。

### ❓ 深入理解：图像 patch 没有词汇表——Embedding 是怎么工作的？

文本有明确的"词汇表"（猫=537），但图像没有。ViT 的解法是**不查表，直接线性投影**：

```python
# 文本: 查表
text_emb = embedding_matrix[token_id]  # token_id 决定查哪一行

# 图像: 线性投影——每个 patch 的像素值就是它的"身份"
patch = image_patch.flatten()          # (768,) — 16×16×3 的像素值
img_emb = Linear(768, d_model)(patch)  # (d_model,) — 投影到目标空间
```

图像 patch 没有 "token ID"——每个 patch 的像素值本身就是它的"身份"。Linear 投影层把像素信息映射到 d_model 维空间。

> 📎 **VLA 扩展**：机器人动作 token（如 FAST tokenizer 生成的 BPE token）介于文本和图像之间——有 token ID（靠 BPE 合并产生），但语义是在特定任务数据上训练出来的。详见 [π₀-FAST 论文导读](../paper-review/Robots-VLA/FAST/fast_paper_guide.md)。

---

## 五、Embedding 矩阵的训练与初始化

### 5.1 初始化策略

Embedding 矩阵不是全零初始化的——那会导致所有 token 的 embedding 一开始完全一样：

```python
# 标准初始化（GPT-2、BERT 等都用这个）
E = torch.randn(vocab_size, d_model) * 0.02  # 标准差 0.02 的小随机数
```

### 5.2 预训练 vs 微调时 Embedding 的处理

| 场景 | Embedding 矩阵状态 | 说明 |
|:---|:---|:---|
| **从头训练（如 GPT-2 预训练）** | 随机初始化 → 完整学 | 几 TB 数据，几万步梯度更新 |
| **微调（如 GPT-2 做分类）** | 预训练权重 → 继续微调 | 已有语义结构，只需适应新领域 |
| **扩展词表（如加中文 token）** | 旧 token → 保留；新 token → 随机初始化 | 新行从零学，旧行继续调 |

### 5.3 Embedding 的梯度流

```
训练时一个 batch:

  loss → 梯度反传 → Embedding 矩阵 E（50257 × 768）
                      ↓
  第 464 行（"The" 的 embedding）→ 被用到，收到梯度 → 更新
  第 3797 行（"cat" 的 embedding）→ 被用到，收到梯度 → 更新
  第 0 行（PAD token）            → 未被用到，无梯度 → 保持不变
  第 50000 行（罕见 token）       → 未被用到，无梯度 → 保持不变
```

> **小白理解**：Embedding 矩阵像一个 Excel 表，50257 行。训练时，当前 batch 里出现的 token 对应行会收到"修改建议"（梯度），没出现的行原封不动。**高频词（"the"、"a"）被更新几十万次→学得很准；低频词（"obfuscate"）可能只被更新几十次→学得粗糙。**

---

## 六、🔢 具体数字例子：GPT-2 Small 处理 "The cat sat"

**条件设定**：GPT-2 Small（vocab=50257, d_model=768），输入 "The cat sat"。

```
Step 0: Tokenization

  "The cat sat"
    → GPT-2 BPE Tokenizer
    → token IDs: [464, 3797, 3332]
    → 文字对应: 464="The", 3797=" cat", 3332=" sat"
    注: GPT-2 BPE 会在词首加空格，所以 "cat" 的 token 是 " cat"

Step 1: Token Embedding 查找

  E.shape = (50257, 768)

  token_464  → E[464]   = [-0.12,  0.34,  0.08, ..., -0.21]  (768 个数)
  token_3797 → E[3797]  = [ 0.45, -0.11,  0.67, ...,  0.03]  (768 个数)
  token_3332 → E[3332]  = [ 0.22,  0.18, -0.34, ...,  0.41]  (768 个数)

  token_emb.shape = (3, 768)

Step 2: Positional Encoding 相加

  pos_0 = [ 0.00,  1.00,  0.00,  1.00, ...]  (768 个数)
  pos_1 = [ 0.84,  0.54,  0.10,  0.99, ...]
  pos_2 = [ 0.91, -0.42,  0.20,  0.98, ...]

  input[0] = E[464] + pos_0
  input[1] = E[3797] + pos_1
  input[2] = E[3332] + pos_2

Step 3: 喂入 Transformer

  input.shape = (3, 768)  → 12 层 Decoder → 输出 logits (3, 50257)
```

> **关键理解**：3 个 token → 3 个 768 维向量。每个向量同时携带了"这个词是什么"（token embedding）和"在第几个位置"（positional encoding）。**Transformer 的每一层 attention 都在这个 768 维空间里计算 token 之间的交互——靠向量之间的内积大小判断谁和谁相关。**

---

## 七、Word2Vec 与 Transformer Embedding 的关系

很多初学者把 Word2Vec 和 Transformer Embedding 搞混——它们是同一个概念的不同阶段：

| | Word2Vec（2013） | Transformer Embedding（2017+） |
|:---|:---|:---|
| **学习方式** | 单独训练（只看词共现） | 和整个模型联合训练（end-to-end） |
| **目标** | 词义本身（静态） | 词在当前上下文中的角色（动态） |
| **同一词不同上下文** | 同一个 embedding（"bank" 永远是同一个向量） | 同一个 token embedding，但 attention 后产生不同上下文表示 |
| **现今用途** | 推荐系统、语义搜索 | LLM 的输入层 |

> **小白理解**：Word2Vec 像给每个词拍一张"标准照"——无论什么场合，都是这张照片。Transformer Embedding 像给每个词一个"初始服装"——然后 attention 给它化妆、换装，最终的样子取决于周围的词。"bank" 在 "river bank" 和 "bank account" 里初始 embedding 是一样的，但经过 12 层 attention 后变成了完全不同的向量。

---

## 八、一句话总结

> **Embedding 是 token 和 Transformer 之间的"翻译层"——把人类可读的离散符号翻译成 d_model 维的连续向量，让语义相近的东西在向量空间中也相近，为后续的注意力计算提供统一的"语言"。**

---

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer，§3.4 Embeddings and Softmax
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2，§2.1 Input Representation
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — ViT，§3.1 Vision Transformer
- [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) — Jay Alammar 的 Embedding 可视化讲解
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar 的 GPT-2 完整可视化（含 Embedding 部分）
