# Positional Encoding（位置编码）详解：让 Transformer 知道"先后顺序"

> **主题**：Positional Encoding——自注意力是"无序"的，位置编码补上"顺序"信息
> **用途**：理解 Transformer 中 token 序列的顺序信息如何注入模型
> **关联文档**：[Transformer 论文深度解读](./transformer_attention_guide.md) · [Embedding 详解](./embedding_guide.md) · [Attention 机制深度解读](./attention_guide.md)

> 📎 **扩展阅读**：本文也作为 [π₀-FAST 论文导读](../paper-review/Robots-VLA/FAST/fast_paper_guide.md) 的配套文档——VLA 中位置编码的应用场景见 §四。

---

## 读之前先搞清楚

### Positional Encoding 在做什么？

Transformer 的自注意力机制有一个"天生缺陷"：**它对输入的顺序不敏感**。如果你把句子 "我 爱 你" 的 token 顺序完全打乱成 "你 爱 我"，自注意力算出来的结果在数学上是等价的（不考虑 mask）——因为注意力只看 token 之间的内容相似度，不关注先后。

但"我打你"和"你打我"的意思完全不同。**位置编码就是用来告诉模型"谁在前面、谁在后面"的。**

> **小白理解**：自注意力像一个"全体讨论会"——每个人同时发言，谁都能听到谁。但如果不知道座次（谁坐哪），就分不清"张三说的"和"李四说的"。位置编码给每个人座位上贴了编号——"1 号位"、"2 号位"——这样讨论内容就和发言顺序绑定了。

### 阅读本文需要的前置知识

| 概念 | 简单解释 |
|:---|:---|
| **Self-Attention（自注意力）** | Transformer 的核心操作——每个 token 看所有其他 token，加权聚合信息 |
| **Token Embedding** | 把 token ID 变成连续向量的过程（见 [Embedding 详解](./embedding_guide.md)） |
| **d_model** | Transformer 的隐藏维度——GPT-2 Small=768，LLaMA-7B=4096 |
| **正弦函数 sin/cos** | 周期性波动的函数——原始 Transformer 用它生成位置编码 |
| **RoPE（旋转位置编码）** | 现代 LLM 最主流的位置编码方式——通过旋转向量来编码位置 |
| **因果注意（Causal Attention）** | Decoder-only 模型的核心——每个 token 只能看自己和前面的 token |

---

## 一、为什么需要位置编码

### 1.1 一个具体的反例——没有位置编码会怎样

```python
# 自注意力（简化版）
Q = K = V = [emb("我"), emb("打"), emb("你")]  # 三个 token 的 embedding

# 自注意力: 每个 Q 和所有 K 做点积，得到注意力权重
scores = Q @ K.T  # (3, 3) 矩阵——token 之间的相似度

# 如果把输入顺序打乱
Q_shuffled = [emb("你"), emb("打"), emb("我")]
scores_shuffled = Q_shuffled @ Q_shuffled.T

# scores 和 scores_shuffled 的值完全一样，只是行列重新排列
# → 模型无法区分"我打你"和"你打我"！
```

这就是为什么需要位置编码：**在 embedding 上叠加位置信息，让相同内容不同位置的 token 产生不同的表示**。

### 1.2 位置编码的通用形式

```
最终输入 = Token Embedding + Positional Encoding

示意图（以 d_model=4 的微型 Transformer 为例）:

位置 0, token "我":
  token_emb: [ 0.12, -0.85,  0.34,  0.67]
  pos_enc:   [+0.00, +1.00, +0.00, +1.00]   ← 位置 0 的编码
  最终输入:  [ 0.12,  0.15,  0.34,  1.67]   ← 逐元素相加

位置 1, token "爱":
  token_emb: [ 0.92,  0.10, -0.45,  0.23]
  pos_enc:   [+0.84, +0.54, +0.10, +0.99]   ← 位置 1 的编码（不同！）
  最终输入:  [ 1.76,  0.64, -0.35,  1.22]

位置 2, token "你":
  token_emb: [ 0.33, -0.21,  0.78, -0.15]
  pos_enc:   [+0.91, -0.42, +0.20, +0.98]   ← 位置 2 的编码（又不同！）
  最终输入:  [ 1.24, -0.63,  0.98,  0.83]
```

> **小白理解**：token embedding 是"词本身的含义"，positional encoding 是"这个词在第几个位置"。两者相加，就像给每个人的名牌上加了一个座位号——"张三是 3 号位的张三"。**同样的"张三"坐在 1 号位和 3 号位，得到的名牌不同——模型就能区分了。**

---

## 二、四种主流位置编码方法

#### 总览

```
┌──────────────────────────────────────────────────────────────┐
│                    位置编码进化树                              │
│                                                              │
│  ① Sinusoidal（正弦波）                                       │
│     原始 Transformer, 2017                                    │
│     · 固定公式生成，不需要训练                                  │
│     · 优点：可以外推到训练时没见过的长度                        │
│     · 用于: BERT, 原始 Transformer                            │
│         ↓                                                     │
│  ② Learned（可学习）                                          │
│     GPT-1/2, 2018                                             │
│     · 位置编码也是参数，训练中学出来                            │
│     · 优点：灵活、效果好（在训练长度内）                        │
│     · 缺点：不能外推到更长序列                                  │
│         ↓                                                     │
│  ③ RoPE（旋转位置编码）⭐ 现代主流                             │
│     LLaMA, Qwen, DeepSeek, 2023                               │
│     · 通过旋转 embedding 向量来编码位置                         │
│     · 优点：相对位置天然编码、可外推、效果好                    │
│     · 用于: LLaMA 全系列, Mistral, Qwen, DeepSeek, PaliGemma │
│         ↓                                                     │
│  ④ ALiBi（注意力偏置）                                        │
│     BLOOM, 2022                                               │
│     · 不在 embedding 上做，直接在 attention score 上加偏置      │
│     · 优点：极简、可外推                                       │
│     · 缺点：效果略逊于 RoPE                                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 三、方法详解

### 3.1 ① Sinusoidal（正弦波位置编码）

原始 Transformer（"Attention Is All You Need", 2017）的方案。用一个固定的数学公式生成位置编码，**不参与训练**。

**公式**：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**直觉解释**：

- `pos` = token 在序列中的位置（0, 1, 2, ...）
- `i` = 维度索引（0, 1, 2, ..., d_model/2 - 1）
- `10000^{2i/d_model}` = 频率——不同维度用不同频率的正弦波
  - 低维度（i 小）→ 高频 → 相邻位置变化大 → 捕捉局部位置差异
  - 高维度（i 大）→ 低频 → 相邻位置变化小 → 捕捉全局位置趋势

#### 🔢 具体数字例子：4 个位置 × 4 维的 Sinusoidal PE

**条件设定**：d_model=4，序列长度 4（最简单的演示）。

```
d_model=4 → 2 个维度对(i=0, i=1)
i=0: 频率 = 10000^{0/4} = 1          → 波长 = 2π ≈ 6.28 个位置
i=1: 频率 = 10000^{2/4} = 100        → 波长 = 2π/100 ≈ 0.063 个位置

计算 PE(0):
  维度 0 (sin): sin(0 / 1^0)    = sin(0)     = 0.00
  维度 1 (cos): cos(0 / 1^0)    = cos(0)     = 1.00
  维度 2 (sin): sin(0 / 100^1)  = sin(0)     = 0.00
  维度 3 (cos): cos(0 / 100^1)  = cos(0)     = 1.00
  → PE(0) = [0.00, 1.00, 0.00, 1.00]

计算 PE(1):
  维度 0 (sin): sin(1 / 1)      = sin(1)     = 0.84
  维度 1 (cos): cos(1 / 1)      = cos(1)     = 0.54
  维度 2 (sin): sin(1 / 100)    = sin(0.01)  ≈ 0.01
  维度 3 (cos): cos(1 / 100)    = cos(0.01)  ≈ 1.00
  → PE(1) = [0.84, 0.54, 0.01, 1.00]

最终 4×4 的 PE 矩阵:

  位置 pos=0: [0.00, 1.00, 0.00, 1.00]
  位置 pos=1: [0.84, 0.54, 0.01, 1.00]
  位置 pos=2: [0.91,-0.42, 0.02, 1.00]
  位置 pos=3: [0.14,-0.99, 0.03, 1.00]
```

> **关键理解**：维度 0-1（高频）在相邻位置间变化剧烈（0.00→0.84→0.91→0.14），维度 2-3（低频）几乎不变（~0.00→0.01→0.02→0.03）。**高频维度告诉模型"前一个还是后一个"，低频维度告诉模型"大概在序列的哪个区域"。**

### 3.2 ② Learned（可学习位置编码）

GPT-1/GPT-2 的方案：把位置编码当作普通参数，训练时一起学。

```python
# 伪代码
pos_embedding = nn.Embedding(max_seq_len, d_model)  # (2048, 4096)
# 初始随机，梯度更新

x = token_embedding + pos_embedding  # 和 token embedding 一样的加法
```

**优点**：灵活——模型自己学会最适合当前数据的位置表示。
**缺点**：`max_seq_len` 写死了——训练时设 2048，推理时就不能处理 3000 个 token 的序列（因为没有位置 2049~2999 的 embedding）。

### 3.3 ③ RoPE（旋转位置编码）⭐

RoPE（Rotary Position Embedding）是 2023 年后几乎所有新 LLM（LLaMA, Mistral, Qwen, DeepSeek, **PaliGemma**）的选择。

**核心思想**：不是"加"位置信息到 embedding 上，而是**"旋转"embedding 向量**——旋转的角度取决于位置。

**直觉理解**（二维简化）：

```
没有 RoPE: 两个相同的 token "猫" 在不同位置
  位置 0 的 "猫": → 向量 (0.7, 0.3)
  位置 5 的 "猫": → 向量 (0.7, 0.3)  ← 完全一样！

有 RoPE: 按位置旋转
  位置 0 的 "猫": → 向量 (0.70, 0.30)  ← 旋转 0°
  位置 5 的 "猫": → 向量 (0.50, 0.58)  ← 旋转 θ×5
  两个向量不同了——但内积仍保留语义相似度
```

**RoPE 的数学**（二维情况）：

```
设 token 的 embedding 向量在维度 (2i, 2i+1) 上的分量为 (x, y)
位置 pos 的旋转角度 θ_pos = pos / 10000^{2i/d_model}

RoPE 变换:
  x' = x·cos(θ_pos) - y·sin(θ_pos)
  y' = x·sin(θ_pos) + y·cos(θ_pos)

这就是标准的二维旋转矩阵!
  [x']   [cosθ  -sinθ] [x]
  [y'] = [sinθ   cosθ] [y]
```

**RoPE 最关键的性质——相对位置天然编码**：

```
两个 token 在位置 m 和 n 的 RoPE 后的内积:
  ⟨RoPE(q_m), RoPE(k_n)⟩ = ⟨q_m, k_n⟩ · cos((m-n)·θ)

内积只依赖相对位置 (m-n)，不依赖绝对位置 m 和 n！
→ 模型学的是"两个 token 相距多远"而不是"在第几个位置"
→ 这就是 RoPE 能外推的核心原因
```

> **小白理解**：Sinusoidal 和 Learned 位置编码是"绝对地址"——"你在第 5 号位"。RoPE 是"相对关系"——"你和你前面的 token 差 3 步"。**绝对地址换一条街就失效了（短序列→长序列），但相对关系在任何长度的街道上都有效。**

#### ❓ 深入理解：RoPE 的数学细节——为什么旋转能编码位置？

RoPE 对 Q 和 K 向量的每一对维度 (2i, 2i+1) 施加一个旋转角度 θ_pos × i，θ_pos = pos / 10000^{2i/d_model}：

```python
# RoPE 对 Q 向量的实现（对 K 同样操作）
def rope(q, pos):
    # q: (seq_len, d_model)
    for i in range(d_model // 2):
        theta = pos / (10000 ** (2*i / d_model))
        cos, sin = cos(theta), sin(theta)
        # 对维度对 (2i, 2i+1) 施加二维旋转
        q[:, 2*i]   = q[:, 2*i] * cos - q[:, 2*i+1] * sin
        q[:, 2*i+1] = q[:, 2*i] * sin + q[:, 2*i+1] * cos  # 注意：原始 q 值
    return q
```

**关键参数**：
- 旋转基频 θ_base = 10000.0（LLaMA/Gemma 标准值；Qwen 用 1000000 获得更好外推）
- 只在 Q 和 K 上施加 RoPE，V 不旋转——因为 V 是"信息载体"，不需要编码位置
- 不同模型（LLaMA, Mistral, Qwen, DeepSeek, Gemma）的实现几乎一致，区别仅在 θ_base 和是否对某些层跳过 RoPE

**为什么有效**：旋转后，两个 token 的 Q·K 内积变为 ⟨q_m, k_n⟩·cos((m-n)·θ)——内积只依赖相对位置差 (m-n)。**模型不需要记住"第 537 号位置"，只需要学习"前后关系"——这正是自然语言需要的。**

> **小白理解**：RoPE 像一个"旋转编码器"——你给每个 token 的向量做一次微小的旋转，旋转角度正比于它的位置。位置 0 不转，位置 1 转 1°，位置 100 转 100°。**两个 token 的"方向差"体现了它们的"位置差"——在同一条直线上的 token 方向差小（位置近），在不同角度上的 token 方向差大（位置远）。**

### 3.4 ④ ALiBi（注意力线性偏置）

不在 embedding 上加东西，直接在 attention score 上减去一个随距离增大的偏置：

```
Attention(Q, K, V) = softmax(QKᵀ/√d - m·|i-j|) · V
                                         ↑
                                    位置越远，偏置越大 → 注意力越低
```

- `m` 是每个注意力头不同的斜率（如 2^{-8/heads}）
- 不需要任何位置编码参数

---

## 四、位置编码在不同模型中的应用

### 4.1 GPT 系列的自回归位置编码

GPT-1/GPT-2 使用 **Learned Positional Encoding**——每个绝对位置（0, 1, 2, ..., 1023）有自己独立的可学习向量。训练时和 token embedding 一起更新。

```
GPT-2 输入构造:
  token "The" at pos 0 → E["The"] + pos_emb[0]
  token "cat" at pos 1 → E["cat"] + pos_emb[1]
  token "sat" at pos 2 → E["sat"] + pos_emb[2]
```

**局限性**：GPT-2 最大长度 1024——如果推理时输入 2000 个 token，位置 1024~1999 没有对应的 pos_emb → 无法处理。

### 4.2 BERT 的 Sinusoidal 位置编码

BERT 使用原始 Transformer 的 **Sinusoidal PE**——固定公式生成，不参与训练。

**为什么 BERT 不需要可学习 PE**：BERT 是双向编码器（所有 token 互相看），位置信息更多用于"构造输入表示"而不是"约束生成顺序"。Sinusoidal 的数学性质（任意两个位置的 PE 内积只依赖位置差）对双向注意力已经足够。

### 4.3 ViT 中的位置编码

ViT 把图像切成 196 个 patch，每个 patch 需要知道自己的空间位置（在图像左上角还是在右下角）。ViT 使用 **Learned PE**——196 个可学习向量，每个代表一个网格位置。

```
224×224 图像 → 14×14 patch 网格
  patch (0,0)  → pos_emb[0]   ← 左上角
  patch (0,1)  → pos_emb[1]   ← 第一行第二列
  ...
  patch (13,13) → pos_emb[195] ← 右下角
```

> **小白理解**：GPT 的位置编码是"一维"的——只记录"第几个"。ViT 的位置编码是"二维"的——要记录"第几行第几列"。**语言是线性的（一个词接一个词），图像是平面的（上下左右都有邻居关系）。**

### 4.4 多模态 VLA 中的连续位置编号

多模态 VLA（如 π₀-FAST / PaliGemma）将图像 token、文本 token、动作 token 拼接成一个长序列，**使用连续的位置编号**：

```
[图像tokens(256) | 文本tokens(45) | action tokens(6)]
   位置 0~255        位置 256~300        位置 301~306
```

所有 307 个 token 按位置 0~306 连续编号，不区分来源。图像的第 255 个 patch 之后紧跟着文本的第一个 token——跨模态的顺序关系被位置编码自然捕捉。

> 📎 详见 [π₀-FAST 论文导读](../paper-review/Robots-VLA/FAST/fast_paper_guide.md)

#### ❓ 深入理解：现代 LLM 用的是哪种位置编码？

2023 年后的主流 LLM（LLaMA、Mistral、Qwen、DeepSeek、Gemma/PaliGemma）**全部使用 RoPE**。

| 模型 | 位置编码 | 说明 |
|:---|:---|:---|
| GPT-1/2 | Learned | 可学习，不可外推 |
| BERT | Sinusoidal | 固定公式 |
| GPT-3/4 | 未公开（推测 RoPE 或 Learned） | — |
| **LLaMA 1/2/3** | **RoPE** | θ=10000 |
| **Mistral** | **RoPE** | θ=10000（+ 滑动窗口 attention） |
| **Qwen 1/2** | **RoPE** | θ=1000000（更大基频，更好外推） |
| **DeepSeek** | **RoPE** | θ=10000 |
| **Gemma/PaliGemma** | **RoPE** | θ=10000 |

RoPE 成为主流的原因：
1. **相对位置天然编码**——内积只依赖位置差
2. **可外推**——训练 2048 长度，推理可扩展到 8192+
3. **不增加参数**——不像 Learned PE 需要额外参数表
4. **实现简单**——就是对 Q 和 K 做一次旋转

---

## 五、四种方法对比总表

| 方法 | 是否可学习 | 外推能力 | 相对位置 | 代表模型 |
|:---|:---:|:---:|:---:|:---|
| **Sinusoidal** | ❌ 固定公式 | ✅ 好 | ❌ | BERT, 原始 Transformer |
| **Learned** | ✅ | ❌ 差 | ❌ | GPT-1/2, ViT |
| **RoPE** | ❌ 固定公式 | ✅ 优秀 | ✅ | LLaMA, Mistral, Qwen, DeepSeek, Gemma |
| **ALiBi** | ❌ 固定参数 | ✅ 好 | ✅ | BLOOM |

> **小白理解**：Sinusoidal 像"门牌号"——固定公式算出，不训练；Learned 像"学号"——训练时排出来的；RoPE 像"相对距离"——不关心绝对位置，只关心"谁在前面几个位"；ALiBi 像"距离惩罚"——离得越远越不关注。**

---

## 六、🔢 具体数字例子：RoPE 如何让模型区分"我打你"和"你打我"

**条件设定**：d_model=4 的微型 Transformer，两个句子用相同的 token embedding 但不同顺序。

```
句子 A: "我 打 你"  (位置 0, 1, 2)
句子 B: "你 打 我"  (位置 0, 1, 2)

Token Embedding（不含位置）:
  我: [0.12, -0.85, 0.34, 0.67]
  打: [0.92,  0.10, -0.45, 0.23]
  你: [0.33, -0.21, 0.78, -0.15]

加上 RoPE（以 θ_pos = pos 简化计算）:

句子 A "我 打 你":
  位置 0 "我": [0.12, -0.85, 0.34, 0.67] + RoPE(pos=0)
            ≈ [0.12, -0.85, 0.34, 0.67]  (旋转 0)
  位置 1 "打": [0.92,  0.10, -0.45, 0.23] + RoPE(pos=1)
            ≈ [0.49, 0.77, -0.39, 0.32]  (旋转后)
  位置 2 "你": [0.33, -0.21, 0.78, -0.15] + RoPE(pos=2)
            ≈ [-0.47, -0.05, 0.72, 0.35]  (旋转后)

句子 B "你 打 我":
  位置 0 "你": [0.33, -0.21, 0.78, -0.15] + RoPE(pos=0)
            ≈ [0.33, -0.21, 0.78, -0.15]
  位置 1 "打": [0.92,  0.10, -0.45, 0.23] + RoPE(pos=1)
            ≈ [0.49, 0.77, -0.39, 0.32]
  位置 2 "我": [0.12, -0.85, 0.34, 0.67] + RoPE(pos=2)
            ≈ [-0.90, 0.26, 0.28, 0.73]

关键对比:
  句子 A 中 "打" 的 RoPE 后向量: [0.49, 0.77, -0.39, 0.32]
  句子 B 中 "打" 的 RoPE 后向量: [0.49, 0.77, -0.39, 0.32]  ← 一样！因为位置相同

  但句子 A 中位置 2 是 "你": [-0.47, -0.05, 0.72, 0.35]
  句子 B 中位置 2 是 "我": [-0.90,  0.26, 0.28, 0.73]  ← 不同！

→ 两个句子的最终表示序列完全不同 → 模型能区分！
```

> **关键理解**："打"这个词在两句中都在位置 1，所以 RoPE 后一致。但"我"和"你"交换了位置 → 携带了不同的位置旋转 → 两个句子的完整表示不同。**这揭示了位置编码的本质：不是改变每个词本身的意思，而是让相同词在不同位置产生不同的"身份标识"。**

---

## 七、一句话总结

> **位置编码是 Transformer 的"时序感知器"——自注意力只看内容不看顺序，位置编码补上了"第几个"这个维度，让模型不仅能理解"说了什么"，还能理解"按什么顺序说的"。RoPE（旋转位置编码）是现代 LLM 的主流选择——通过旋转向量而非简单相加来编码位置，天然支持相对位置感知和长序列外推。**

---

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer，§3.5 Positional Encoding（Sinusoidal）
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 原始论文
- [Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409) — ALiBi 论文
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — ViT，§3.1（图像位置编码）
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar 的位置编码可视化
