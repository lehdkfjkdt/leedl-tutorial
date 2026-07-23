# Attention 机制深度解读：从传统注意力到 Flash Attention

> 本文档系统地梳理 Attention 机制从起源到现代的完整演进。与 [Transformer 论文深度解读](./transformer_attention_guide.md) 形成"机制原理 → 架构应用"的递进关系——本文聚焦"Attention 本身怎么算、为什么这样算"，Transformer 文档聚焦"怎么把 Attention 搭成完整模型"。

---

## 读之前先搞清楚

### Attention 到底解决了什么问题？

在 Attention 出现之前，神经网络处理序列信息只有两条路：

| 方案 | 做法 | 问题 |
|:---|:---|:---|
| **全连接** | 把所有输入拼成一个长向量，一次性喂给网络 | 输入长度必须固定，无法处理变长序列 |
| **RNN / LSTM** | 维护一个隐藏状态，逐时间步更新 | 所有历史信息压缩到单个向量，长程遗忘严重 |

两条路的核心缺陷是同一个：**信息必须被"压缩"**——要么压缩成固定长度向量，要么逐时间步挤压到隐藏状态里。

Attention 的答案是：**不压缩，而是"按需检索"**。需要哪个位置的信息，就直接去那个位置取，权重由"相关性"决定。

> **小白理解**：全连接像把整本书复印成一张纸（信息丢失），RNN 像逐字读但只记得最近几页。Attention 则像给每个词配了一个"呼叫器"——任何时候想找某个词的信息，按一下呼叫器就能直接联系到它，不需要从第一页开始翻。

### 读懂本文需要的前置知识

| 概念 | 简单解释 |
|:---|:---|
| **点积（Dot Product）** | 两个向量对应位置相乘再求和，衡量两个向量的"方向相似度" |
| **Softmax** | 把一组任意实数变成概率分布（0~1 之间，和为 1） |
| **张量（Tensor）** | 多维数组的统称——标量是 0 维，向量是 1 维，矩阵是 2 维 |
| **线性投影** | 用一个可学习的矩阵 $W$ 乘以输入：$y = xW$——改变向量的维度和语义 |
| **Seq2Seq** | 输入序列→输出序列的任务框架，Encoder 编码输入，Decoder 生成输出 |
| **GPU 内存层级** | HBM（高带宽显存，大但慢）vs SRAM（片上缓存，小但快）——Flash Attention 的关键优化点 |

---

## 一、传统 Attention（Bahdanau 2015）：让 Decoder 学会"回头看"

### 1.1 动机：Seq2Seq 的信息瓶颈

2014 年，Sutskever 等人提出 Seq2Seq 框架用于机器翻译。工作流程是：
1. Encoder（RNN）把整个源句"吞"进去，输出一个固定长度的**上下文向量** $c$
2. Decoder（另一个 RNN）以 $c$ 为初始状态，逐词生成翻译

```
源句: "The cat sat on the mat"
       ↓ Encoder RNN
上下文向量 c ∈ R^512    ← 整句话的语义压缩到 512 个数！
       ↓ Decoder RNN
目标句: "Le chat s'est assis sur le tapis"
```

问题在哪？**整句话的信息被硬塞进一个 512 维向量。** 句子越长，信息丢失越严重——就像一个 1000 字的文章让你用一句话总结，然后再从这句话还原全文。

> **小白理解**：这就像把一本 300 页的小说手抄到一张便利贴上，再从便利贴还原整本小说。50 页的短篇勉强能行，300 页的长篇注定丢失大量细节。

### 1.2 Bahdanau Attention 的核心思路

Bahdanau et al.(2015) 的改进是：**不要只给 Decoder 一个 $c$，而是让 Decoder 在每个生成步都能"回看" Encoder 的所有隐藏状态，并自动聚焦到与当前生成最相关的部分。**

```
源句: "The  cat  sat  on  the  mat"
       ↓    ↓    ↓    ↓    ↓    ↓
      h₁   h₂   h₃   h₄   h₅   h₆    ← 保留所有 Encoder 状态
       \    |    /    \    |    /
        加权求和（权重由"相关性"决定）
              ↓
          上下文向量 c_t（每个生成步都不同！）
              ↓
          生成目标词 y_t
```

### 1.3 数学定义（Additive Attention）

Bahdanau 使用的打分函数是"加性注意力"（Additive Attention），与 Transformer 的点积 Attention 不同：

**Step 1：计算注意力分数**

对 Decoder 第 $t$ 步的隐藏状态 $\mathbf{s}_t$ 和 Encoder 每个位置 $i$ 的隐藏状态 $\mathbf{h}_i$，用一个小型前馈网络打分：

$$e_{t,i} = \mathbf{v}^\top \tanh(W_a \mathbf{s}_t + U_a \mathbf{h}_i)$$

其中 $W_a, U_a$ 是可学习的权重矩阵，$\mathbf{v}$ 是可学习的向量。本质上就是用一个小神经网络学"$\mathbf{s}_t$ 和 $\mathbf{h}_i$ 有多相关"。

**Step 2：Softmax 归一化**

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{n} \exp(e_{t,j})}$$

得到注意力权重分布 $\boldsymbol{\alpha}_t = [\alpha_{t,1}, \ldots, \alpha_{t,n}]$，$\sum_i \alpha_{t,i} = 1$。

**Step 3：加权求和得到上下文向量**

$$\mathbf{c}_t = \sum_{i=1}^{n} \alpha_{t,i} \mathbf{h}_i$$

Decoder 再用 $\mathbf{c}_t$ 和上一时刻状态 $\mathbf{s}_{t-1}$ 一起生成当前词。

### 1.4 打分函数的三种形式

Attention 的核心操作是计算 Query 和 Key 的"匹配分数"。历史上出现过三种主流打分函数：

| 名称 | 公式 | 特点 | 使用场景 |
|:---|:---|:---|:---|
| **Additive（加性）** | $\mathbf{v}^\top \tanh(W\mathbf{q} + U\mathbf{k})$ | 非线性，表达能力更强 | Bahdanau (2015)，Q/K 维度不同时适用 |
| **Dot-Product（点积）** | $\mathbf{q}^\top \mathbf{k}$ | 无参数，速度快，但要求 Q/K 同维度 | 早期工作，小维度场景 |
| **Scaled Dot-Product（缩放点积）** | $\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}$ | 点积 + 方差归一化，Transformer 采用 | 大维度场景（$d_k \ge 64$） |

> **小白理解**：Additive 像请了一个专业评委来打分（有可学习参数，要训练），Scaled Dot-Product 像直接比大小（纯数学计算，无额外参数）。小维度时直接比大小就够了，但维度大了"比分"会虚高，所以要除以 $\sqrt{d_k}$ 压一压——就像篮球比赛比分通常 100+，乒乓球才 11 分，不同量级的比分需要不同的归一化方式。

#### 🔢 具体数字例子：Additive Attention 打分

**条件设定**：Decoder 状态 $\mathbf{s} = [1,\ 0]$（当前需要"主语"信息），Encoder 两个位置：

| 位置 | $\mathbf{h}_i$ | 词 |
|:---|:---|:---|
| $i=1$ | $[3,\ 0]$ | "cat"（主语） |
| $i=2$ | $[0,\ 2]$ | "mat"（宾语） |

设简化参数 $W_a = U_a = I$，$\mathbf{v} = [1,\ 1]$：

$$\begin{aligned} e_1 &= [1,1]^\top \tanh([1,0] + [3,0]) = [1,1]^\top \tanh([4,0]) \\ &\approx [1,1]^\top [0.999, 0] = \mathbf{0.999} \\ e_2 &= [1,1]^\top \tanh([1,0] + [0,2]) = [1,1]^\top \tanh([1,2]) \\ &\approx [1,1]^\top [0.762, 0.964] = \mathbf{1.726} \end{aligned}$$

softmax 后：$\alpha_1 \approx 0.33$，$\alpha_2 \approx 0.67$。

> **关键理解**：这个例子有点反直觉——"mat"的分数反而比"cat"高。原因是我们设了 $\mathbf{v}=[1,1]$ 即两维等权重，而 "mat" 的第二维比较大。**实际训练中 $\mathbf{v}$ 会自动学到"哪个维度更能反映相关性"**——如果任务需要找主语，$\mathbf{v}$ 的第一维权重会被训练得更大。

---

## 二、Self-Attention（自注意力）：序列自己注意自己

### 2.1 从"Decoder 查 Encoder"到"序列内部互查"

传统 Attention 是 **跨序列** 的：Decoder（目标语言）查询 Encoder（源语言）。

Self-Attention 把查询范围从"跨序列"扩展到"**序列内部**"——每个词去查同一个序列中的所有词（包括自己）。这就是 Transformer 的核心创新。

```
传统 Attention（跨序列）:
  Decoder  ──查询──→  Encoder
  (目标语言)          (源语言)

Self-Attention（序列内部）:
  Encoder  ──查询──→  Encoder（自己）
  位置 i              位置 j（包括 i）
```

### 2.2 Q / K / V 的深度理解

Self-Attention 把每个词向量 $\mathbf{x}_i$ 通过三组可学习的投影矩阵映射为三个角色：

$$\mathbf{q}_i = \mathbf{x}_i W_Q,\quad \mathbf{k}_i = \mathbf{x}_i W_K,\quad \mathbf{v}_i = \mathbf{x}_i W_V$$

| 角色 | 全称 | 职责 | 类比 |
|:---|:---|:---|:---|
| **Q（Query）** | 查询向量 | "我想找什么信息？" | 搜索引擎输入框里的关键词 |
| **K（Key）** | 键向量 | "我身上有什么特征可以被匹配？" | 数据库中每篇文章的索引标签 |
| **V（Value）** | 值向量 | "我被关注后，实际传递什么内容？" | 数据库中被索引行存储的实际内容 |

> **小白理解（自动驾驶版）**：ego（自车）正在直行。它的 Query 是"我现在最担心什么方向的风险？"（比如纵向碰撞）。前车的 Key 是"我是一辆正在减速的车，纵向坐标在 ego 正前方"，左车的 Key 是"我是一辆正在靠近的车，横向坐标在 ego 左侧"。Query 与 Key 匹配后，Value 传递实质性信息——前车的 Value 可能是"[减速 5 km/h, 距离 12m]"，左车的 Value 可能是"[横向速度 1.2 m/s, 距离 1.5m]"。

### 2.3 Scaled Dot-Product Attention 完整推导

给定输入序列 $X \in \mathbb{R}^{n \times d_{\text{model}}}$：

**Step 1：线性投影**

$$Q = XW_Q \in \mathbb{R}^{n \times d_k},\quad K = XW_K \in \mathbb{R}^{n \times d_k},\quad V = XW_V \in \mathbb{R}^{n \times d_v}$$

**Step 2：计算注意力分数矩阵**

$$\text{Scores} = QK^\top \in \mathbb{R}^{n \times n}$$

$\text{Scores}_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$——第 $i$ 个词的 Query 与第 $j$ 个词的 Key 的点积。**矩阵乘法的视角**：$QK^\top$ 是一次性算出所有 $n \times n$ 对词之间的相似度。

**Step 3：缩放**

$$\text{ScaledScores} = \frac{QK^\top}{\sqrt{d_k}}$$

当 $d_k$ 较大时，点积的方差 ≈ $d_k$。假设 $\mathbf{q}$ 和 $\mathbf{k}$ 的每个分量独立且均值为 0、方差为 1：

$$\text{Var}(\mathbf{q} \cdot \mathbf{k}) = \sum_{j=1}^{d_k} \text{Var}(q_j \cdot k_j) = d_k$$

除以 $\sqrt{d_k}$ 将方差归一化到 1。这不是可选的——不缩放，梯度直接消失。

**Step 4：Softmax + 加权融合**

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

```mermaid
graph TD
    X[输入 X ∈ ℝ^{n×d}] --> Q[Q = X·W_Q]
    X --> K[K = X·W_K]
    X --> V[V = X·W_V]
    Q --> matmul1[Q·K^T]
    K --> matmul1
    matmul1 --> scale[÷ √d_k]
    scale --> softmax[Softmax]
    softmax --> matmul2[weights · V]
    V --> matmul2
    matmul2 --> out[输出 ∈ ℝ^{n×d_v}]
```

### 2.4 Self-Attention 的信息路径：$O(1)$ vs $O(n)$

这是 Self-Attention 相对于 RNN 最根本的优势。考虑序列中的两个位置 $i$ 和 $j$：

| | RNN / LSTM | Self-Attention |
|:---|:---:|:---:|
| **最短信息路径** | $\|i-j\|$ 步（逐时间步传递） | 1 步（一次点积直达） |
| **梯度路径** | 反向传播经过 $\|i-j\|$ 个时间步 | 反向传播经过 1 层 |
| **长程依赖能力** | 指数衰减 | 理论上无损 |

> **小白理解**：RNN 中 "cat" 要给 "mat" 传消息，必须经过中间的 "sat → on → the"，每一步都可能丢失信息。Self-Attention 中 "cat" 和 "mat" 在同一个矩阵乘法中面对面——不需要传话的中间人。

### 2.5 Causal Mask（因果掩码）

Decoder 的 Self-Attention 必须阻止位置 $i$ 看到位置 $j > i$（未来）。实现方式：在 Softmax 前将上三角区域置为 $-\infty$：

$$\text{MaskedScores}_{ij} = \begin{cases} \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} & j \leq i \\ -\infty & j > i \end{cases}$$

Softmax 后，$-\infty$ → 0 权重：

$$\text{Attention Mask} = \begin{bmatrix}
w_{11} & 0 & 0 & \cdots & 0 \\
w_{21} & w_{22} & 0 & \cdots & 0 \\
w_{31} & w_{32} & w_{33} & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
w_{n1} & w_{n2} & w_{n3} & \cdots & w_{nn}
\end{bmatrix}$$

#### 🔢 具体数字例子：Causal Mask 在推理时的效果

**条件**：正在生成第 3 个词。Decoder 输入 = [已生成的词1, 词2, 当前要预测的词3位置（置为起始符）]。

分数矩阵（缩放后）：

$$\text{Scores} = \begin{bmatrix}
2.1 & -1.3 & 0.8 \\
0.5 & 3.2 & -0.7 \\
1.4 & 0.9 & 2.6
\end{bmatrix}$$

施加 Mask（上三角 → $-\infty$）：

$$\text{MaskedScores} = \begin{bmatrix}
2.1 & -\infty & -\infty \\
0.5 & 3.2 & -\infty \\
1.4 & 0.9 & 2.6
\end{bmatrix}$$

Softmax 后每行独立归一化：

$$\text{Weights} \approx \begin{bmatrix}
1.00 & 0 & 0 \\
0.06 & 0.94 & 0 \\
0.17 & 0.11 & 0.72
\end{bmatrix}$$

> **关键理解**：位置 3 只能 attend 到位置 1、2、3（权重 0.17, 0.11, 0.72），位置 4+ 的信息完全不参与。**这确保了自回归生成——训练时看到的和推理时生成的条件完全一致。**

---

## 三、Multi-Head Attention：多个视角同时看

### 3.1 为什么单头不够？

单头 Attention 每次只能学习一种"相关性模式"。但语言和场景中的关系是多维的：

- 语法依存：动词 ↔ 主语（"is" 关联 "The agreement"）
- 语义相关：同义词、反义词
- 指代关系：代词 → 先行词（"it" → "The cat"）
- 局部共现：形容词 ↔ 名词（"beautiful" ↔ "flower"）

> **小白理解（自动驾驶版）**：开车时需要同时关注——前车减速（纵向风险）、左车靠近（横向风险）、红绿灯状态（交规约束）、车道线位置（可行驶区域）。一个头只能聚焦一种关系，8 个头 = 8 个专项观察员各盯一路，最后综合判断。

### 3.2 数学定义

Multi-Head Attention 并行运行 $h$ 个独立 Attention，每个头有自己的投影矩阵：

$$\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)$$

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

关键参数：$d_k = d_v = d_{\text{model}} / h = 512 / 8 = 64$。每个头在 64 维子空间中独立运行，拼接后回到 512 维。

### 3.3 矩阵等价形式：拼接 vs 并行 reshape

Multi-Head 常被误解为"串行跑 8 次 Attention 再拼起来"。实际上，**它等价于一次矩阵乘法 + reshape**：

```
实现方式 A（概念上的）:
  for i in 1..h:
    head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)
  output = concat(head_1,...,head_h) @ W^O

实现方式 B（实际代码中的，等价且高效）:
  Q_all = Q @ W_Q_concat    # W_Q_concat = [W_1^Q | ... | W_h^Q]，shape (512, 512)
  Q_heads = reshape(Q_all, (batch, h, seq_len, d_k))
  # 一次 matmul 算所有头
  out = scaled_dot_product_attention(Q_heads, K_heads, V_heads)
  out = reshape(out, (batch, seq_len, d_model)) @ W^O
```

**两种方式数学上完全等价**。方式 B 通过把 $h$ 个 $W_i^Q$ 沿列方向拼接成一个大矩阵 $W_Q \in \mathbb{R}^{512 \times 512}$，一次矩阵乘法同时完成所有头的投影，再用 `reshape` 拆分并行计算。

#### 🔢 具体数字例子：Multi-Head 的维度拆分

**条件**：$d_{\text{model}} = 8$（简化），$h = 2$，$d_k = 4$。

```
输入 ego 向量: [1, 0, 3, 0, 0, 2, 1, 1]
                    │              │
         ┌──────────┘       ┌──────┘
         ▼                  ▼
   head_1 输入:        head_2 输入:
   [1, 0, 3, 0]        [0, 2, 1, 1]
         │                  │
     Attention           Attention
   (纵向风险视角)       (横向风险视角)
         │                  │
         ▼                  ▼
   [0.9,0.1,0.5,0.0]   [0.1,0.8,0.2,0.7]
         │                  │
         └────────┬─────────┘
                  ▼
        Concat: [0.9,0.1,0.5,0.0, 0.1,0.8,0.2,0.7]
                  │
                  ▼  W^O (8×8)
        输出: [0.5,0.4,0.35,0.3,0.2,0.5,0.3,0.4]
```

> **关键理解**：head_1 输出的第一维 0.9 远大于 0.1，说明它重点关注纵向风险；head_2 输出的第二维 0.8 说明它重点关注横向风险。**每个头天然学到不同模式，因为 $W_i^Q, W_i^K, W_i^V$ 初始化不同、梯度路径不同——训练会自动将不同模式"分配"给不同头。**

#### ❓ 常见疑问：头数是不是越多越好？

> 不是。论文消融实验：$h=1$（单头）BLEU 低 0.9，$h=32$ 反而比 $h=8$ 低 0.2。

原因分析：
- 头数翻倍 → 每头维度减半（$d_k = d_{\text{model}}/h$），每个头能表示的"子空间"变小
- $h=32$ 时每头只有 $512/32 = 16$ 维，表达能力严重受限
- **$h=8$ 是论文实验中的最优平衡点**——足够多视角，每个视角有足够的表示空间

---

## 四、Cross-Attention：序列间的信息桥梁

### 4.1 与 Self-Attention 的核心区别

| | Self-Attention | Cross-Attention |
|:---|:---|:---|
| **Q 来源** | 当前序列自身 | Decoder（目标序列） |
| **K, V 来源** | 当前序列自身 | Encoder（源序列） |
| **作用** | 捕捉序列内部依赖 | 跨序列信息检索 |
| **注意力矩阵形状** | $n \times n$（方阵） | $m \times n$（矩形，$m$=目标长度，$n$=源长度） |
| **是否有 Causal Mask** | Decoder Self-Attn 有 | 无（目标词可以看所有源词） |

### 4.2 数学定义

Cross-Attention 使用与 Self-Attention 完全相同的 Scaled Dot-Product Attention 公式，唯一的区别是 Q、K、V 的来源：

$$Q = X_{\text{decoder}} W_Q,\quad K = X_{\text{encoder}} W_K,\quad V = X_{\text{encoder}} W_V$$

> **小白理解**：Self-Attention 是"自己照镜子"——每个词看同一句话里的其他词。Cross-Attention 是"看参考书"——Decoder 生成翻译时，去源句编码中查找最相关的信息。Decoder 问"我现在需要一个主语"，源句的 "cat" 回答"我是主语，我的编码是 [0.9, 0.4]，拿去吧。"

### 4.3 张量形状追踪

```
Encoder 输出: Z_enc ∈ (batch, n_src, d_model)     # n_src = 源句长度
Decoder 状态: Z_dec ∈ (batch, n_tgt, d_model)     # n_tgt = 目标句长度

Q = Z_dec @ W_Q  ∈ (batch, n_tgt, d_k)
K = Z_enc @ W_K  ∈ (batch, n_src, d_k)
V = Z_enc @ W_V  ∈ (batch, n_src, d_v)

Scores = Q @ K^T ∈ (batch, n_tgt, n_src)   ← 矩形！不是方阵
Weights = softmax(Scores / √d_k) ∈ (batch, n_tgt, n_src)
Output  = Weights @ V ∈ (batch, n_tgt, d_v)
```

---

## 五、现代高效 Attention 变体

标准 Self-Attention 的内存和计算复杂度为 $O(n^2 d)$。当序列长度 $n$ 超过几千时，显存和速度都变得不可接受。以下是社区最重要的改进方案。

### 5.1 Flash Attention（2022）：不改数学，只优化 IO

**核心洞察**：Attention 的瓶颈不在"计算量"，而在"显存读写"。标准 Attention 的步骤是：
1. 从 HBM（高带宽显存）读取 Q、K → 计算 $S = QK^\top$ → 把 $S \in \mathbb{R}^{n \times n}$ 写回 HBM
2. 从 HBM 读取 $S$ → 计算 Softmax → 写回 HBM
3. 从 HBM 读取 Softmax 结果和 V → 计算输出 → 写回 HBM

每一步都把巨大的中间矩阵（$n \times n$）在 HBM 和计算单元之间搬来搬去。Flash Attention 的解决思路是：

- **分块计算（Tiling）**：把 Q、K、V 切成小块，每次只加载一个小块到 SRAM（片上缓存），在 SRAM 内部完成 Softmax 的所有计算
- **在线 Softmax**：用数值稳定的增量算法，在分块场景下正确计算 Softmax 归一化因子
- **不存储中间矩阵**：$S$ 和 Softmax 后的权重矩阵从不写回 HBM

```
标准 Attention：
  HBM → 读取 Q,K → 算 S → 写回 S → 读取 S → 算 Softmax → 写回 → 读取 → 算输出
  ⚠️ 中间 n×n 矩阵在 HBM 和计算单元间反复搬运

Flash Attention：
  HBM → 读取 Q,K,V 的一个小块 → 在 SRAM 内完成全部计算 → 直接写回输出块
  ✅ 中间矩阵从不离开 SRAM
```

> **小白理解**：标准 Attention 像在图书馆（HBM）和书桌（计算单元）之间反复搬书——每算一步就要把整本 $n \times n$ 的"大书"搬来搬去。Flash Attention 像只从图书馆搬当前需要的那几页（分块），在书桌上完成全部批注（计算），直接交作业——大书从未离开书架。**结果是：数学结果完全一致，速度提升 4-8 倍，显存节省 10-20 倍。**

### 5.2 Multi-Query Attention（MQA, 2019）

**思想**：所有 Attention 头**共享同一组 K 和 V**，只有 Q 是每头独立的。

$$\text{head}_i = \text{Attention}(QW_i^Q,\ KW^K,\ VW^V)$$

- 显存节省：K/V 的缓存量减少 $h$ 倍（7-8×）
- 代价：表达能力略有下降
- 代表模型：PaLM

### 5.3 Grouped-Query Attention（GQA, 2023）

**思想**：MQA 和 Multi-Head 的折中——将 $h$ 个头分为 $g$ 组，组内共享 K/V。

$$\text{head}_{i} = \text{Attention}(QW_i^Q,\ KW_{\lfloor i/g \rfloor}^K,\ VW_{\lfloor i/g \rfloor}^V)$$

- $g=1$ 时退化为 MQA（所有头共享 K/V），$g=h$ 时退化为标准 Multi-Head
- 实际常用 $g=8$（$h=32$ 时），接近 Multi-Head 的质量 + 接近 MQA 的速度
- 代表模型：LLaMA 2/3、Mistral

### 5.4 稀疏 / 线性 Attention 简介

| 方案 | 核心思路 | 复杂度 | 代表工作 |
|:---|:---|:---:|:---|
| **局部 Attention** | 每个位置只看窗口内的邻居 | $O(n \cdot w)$ | Longformer |
| **低秩近似** | 用低秩矩阵近似 Attention 矩阵 | $O(n)$ | Linformer |
| **核方法** | 将 Softmax 分解为核函数，改变计算顺序 | $O(n)$ | Performer |
| **状态空间模型** | 完全不用 Attention，用结构化状态空间替代 | $O(n)$ | Mamba, Mamba-2 |

> **小白理解**：$O(n^2)$ 的 Attention 是所有大模型的"阿喀琉斯之踵"。Flash Attention 是"在不改数学的前提下优化工程实现"，它不能降低理论复杂度但能大幅降低实际开销。GQA 是"在模型设计层面节省 KV Cache"。Mamba 等 SSM 则走得更远——完全抛弃 Attention，用另一种数学机制实现类似的"选择性记忆"效果。

---

## 六、核心公式速查

### Attention 通用形式

$$\text{Attention}(\mathbf{q}, K, V) = \sum_{i=1}^{n} \alpha_i \mathbf{v}_i, \quad \alpha_i = \frac{\exp(\text{score}(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(\text{score}(\mathbf{q}, \mathbf{k}_j))}$$

### 打分函数

| 类型 | 公式 |
|:---|:---|
| Dot-Product | $\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$ |
| Scaled Dot-Product | $\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}$ |
| Additive (Bahdanau) | $\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{v}^\top \tanh(W\mathbf{q} + U\mathbf{k})$ |

### Scaled Dot-Product Attention（矩阵形式）

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 复杂度对比

| 机制 | 计算复杂度 | 显存复杂度 |
|:---|:---:|:---:|
| Standard Self-Attention | $O(n^2 d)$ | $O(n^2)$ |
| Flash Attention | $O(n^2 d)$（同） | $O(n)$（显著降低） |
| MQA (Multi-Query) | $O(n^2 d)$ | $O(n d)$ (KV Cache 减少 $h$×) |
| Sparse Attention (Longformer) | $O(n \cdot w \cdot d)$ | $O(n \cdot w)$ |

---

## 七、PyTorch 实现对比

### 标准 Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def standard_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    返回: output, attention_weights
    """
    d_k = Q.size(-1)
    # (batch, heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights
```

### PyTorch 2.0+ 内置 Flash Attention

```python
# PyTorch 2.0+ 内置，自动使用 Flash Attention (当条件满足时)
output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=True)
# 内部自动选择最优实现: Flash Attention > Memory Efficient > 标准实现
```

### Flash Attention 伪代码（示意核心逻辑）

```python
def flash_attention(Q, K, V, block_size=64):
    """分块计算，避免物化 n×n 注意力矩阵"""
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)
    L = torch.zeros(B, H, N, 1)     # 分母累加器
    M = torch.full((B, H, N, 1), -float('inf'))  # 最大值追踪器

    # 外层循环：遍历 K/V 的块
    for start_k in range(0, N, block_size):
        end_k = min(start_k + block_size, N)
        Kj = K[:, :, start_k:end_k, :]
        Vj = V[:, :, start_k:end_k, :]

        # 内层循环：遍历 Q 的块
        for start_q in range(0, N, block_size):
            end_q = min(start_q + block_size, N)
            Qi = Q[:, :, start_q:end_q, :]

            # 块内计算，结果存在 SRAM 中
            S = torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(d)
            # 在线 Softmax 更新...
            O[:, :, start_q:end_q, :] += ...

    return O
```

---

## 参考资料

- Bahdanau et al.(2015): [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Vaswani et al.(2017): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Dao et al.(2022): [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Shazeer (2019): [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150)（MQA）
- Ainslie et al.(2023): [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
