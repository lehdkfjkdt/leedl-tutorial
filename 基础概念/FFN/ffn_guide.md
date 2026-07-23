# FFN（Feed-Forward Network）完全解读：Transformer 的"独立思考"层

> **核心论文**：Attention is All You Need（Vaswani et al., NeurIPS 2017）——Transformer Block = Attention + FFN
> **概念类型**：Transformer 架构的基础组件，与 Self-Attention 并列构成每个 Transformer Block
> **关键变体**：ReLU-FFN（原版）/ GELU-FFN（BERT/GPT）/ SwiGLU-FFN（LLaMA/现代 LLM）/ MoE-FFN（Mixtral/DriveMoE）

---

## 读之前先搞清楚

### FFN 要解决什么问题？

Transformer 的 Self-Attention 有一个天生缺陷：**它本质上是"线性组合"操作。**

```
Self-Attention 在做什么：
  output_i = Σⱼ w_ij · V_j
  
  其中 V_j 是第 j 个 token 的 Value 向量
       w_ij 是 attention 权重（通过 softmax 归一化）

  这意味着：output_i 只是所有输入 Value 向量的加权平均——
  无论 attention 权重怎么调，output 永远在 V 向量张成的子空间内！
```

这意味着 Self-Attention 缺少非线性变换能力——它可以把信息"混合"但不能"变形"。而深度学习的大部分能力恰恰来自**非线性**。

**FFN 的作用**：在 Attention 混合信息之后，对每个 token 独立施加非线性变换，让模型对信息进行"深度加工"。

```
Transformer Block 的核心公式：

  x = x + Attention(LayerNorm(x))    ← ① Attention：混合不同 token 的信息
  x = x + FFN(LayerNorm(x))          ← ② FFN：对每个 token 独立做非线性变换

  类比：Attention = 开会讨论（大家交流信息）
       FFN = 会后各自思考（每个人独立消化吸收）
```

> **小白理解**：Attention 像一次团队头脑风暴——所有人把想法说出来，互相参考。FFN 像会后每个人回到工位，自己闷头思考——把会上听到的信息消化、变形、整合成自己的理解。没有头脑风暴（Attention），信息不流通；没有独立思考（FFN），想法不深入。两样缺一不可。

### 读懂 FFN 需要的前置知识

| 概念 | 简单解释 |
|:---|:---|
| **Token** | Transformer 输入序列中的每个位置向量（一个词 / 一个 patch / 一个 state） |
| **Self-Attention** | Q·K^T 算 token 间相似度 → softmax → 加权 V，输出是 V 的线性组合 |
| **非线性激活函数** | ReLU(x)=max(0,x)、GELU：给网络"弯曲"能力，不是简单直线映射 |
| **Position-wise** | 对序列中每个位置用同一个 FFN，但不同位置独立计算（不像 Attention 那样跨位置交互） |
| **Skip Connection（残差）** | 输出 = 输入 + 变换(输入)，防止深层网络退化 |
| **Layer Normalization** | 沿特征维度（而非 batch 维度）做标准化，稳定训练 |

---

## 一、FFN 的核心结构

### 第 1 步：最经典的 FFN —— Transformer 原版

```
原版 FFN（Vaswani et al., 2017）：

  FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂

  其中：
    x ∈ R^d              ← 输入 token 向量（d=512 或 768）
    W₁ ∈ R^{d_ff × d}    ← 升维矩阵（d_ff 通常是 4×d，如 512→2048）
    W₂ ∈ R^{d × d_ff}    ← 降维矩阵（2048→512）
    b₁ ∈ R^{d_ff}        ← 偏置
    b₂ ∈ R^d             ← 偏置

  流程：d → (线性升维) → 4d → ReLU → (线性降维) → d
        ↑                  ↑                ↑
       输入              中间表示          输出（与输入同维度）
```

**为什么是"升维再降维"（d → 4d → d）？**

```
直觉：升维 → 在更高维空间中做非线性变换 → 降维回到原位

d=2（二维平面）的例子：
  在二维平面上，一条直线把两个类分开 → 可能分不开（线性不可分）
  升到三维：把平面"折"成立体 → 在三维中很容易用一个曲面分开
  降回二维：曲面投影回平面 → 分类边界变成曲线 → 两个类分开了！

  这就是"升维→非线性→降维"的魔力——
  高维空间中简单的非线性变换，降维后就变成了低维空间中的复杂变换。
```

### 第 2 步：不同激活函数的 FFN 变体

```
原版 ReLU-FFN：
  FFN(x) = W₂ · max(0, W₁·x)         ← ReLU：负值直接截断为 0
  
  优点：最简单，计算最快
  缺点：x<0 时梯度为 0 → "死神经元"问题


GELU-FFN（BERT / GPT-2 / GPT-3）：
  FFN(x) = W₂ · GELU(W₁·x)
  
  GELU(x) = x · Φ(x)  ≈  x · sigmoid(1.702·x)
  其中 Φ(x) 是标准高斯分布的 CDF

  优点：比 ReLU 平滑，处处可导 → 训练更稳定
  缺点：计算稍慢于 ReLU


SwiGLU-FFN（LLaMA / PaLM / 现代 LLM 标配）：
  FFN(x) = (SiLU(W_g · x) ⊙ W_up · x) · W_down
  
  拆解：
    gate = SiLU(W_g · x)    ← SiLU(x) = x · sigmoid(x)，也叫 Swish
    up   = W_up · x          ← 线性升维
    gated = gate ⊙ up        ← 逐元素乘法（门控！）
    output = W_down · gated  ← 线性降维

  参数：W_g, W_up ∈ R^{d_ff×d}, W_down ∈ R^{d×d_ff}
        ↑ 三个矩阵而非两个，参数量是 ReLU-FFN 的 1.5 倍

  优点：门控机制让模型自己"决定"哪些信息通过、哪些抑制
        → 比 ReLU 和 GELU 表达能力更强
```

```
三种激活函数的形状对比：

  ReLU:              GELU:                SiLU (Swish):
    │                  │                    │
    │    ╱             │    ╱               │    ╱
    │   ╱              │   ╱                │   ╱
    │  ╱               │  ╱                 │  ╱
    │ ╱                │ ╱                  │ ╱
    │╱                 │╱                   │╱
  ──┼────           ───┼────             ──┼──────
    │                  │                    │╲
    │ (x<0=0)          │ (x<0 略有负值)      │ ╲ (x<0 有负的"小尾巴")
```

> **小白理解**：ReLU 像电灯开关——要么全亮要么全灭。GELU 像调光器——可以逐渐变亮变暗。SwiGLU 像智能调光器——加了一个"门"（gate），不只能调亮度，还能决定"这盏灯要不要开"。门控机制让网络对信息的控制更精细——重要的放大（gate≈1），不重要的关掉（gate≈0）。

---

## 二、FFN 在 Transformer Block 中的位置

```
Transformer Block 的完整结构：

        输入 x（N 个 token，每个 d 维）
         │
    ┌────┴────┐
    │ Pre-Norm│  ← LayerNorm(x)
    └────┬────┘
         │
    ┌────┴────┐
    │Multi-Head│ ← Attention：token 之间交互
    │Attention │   Q=W_Q·x, K=W_K·x, V=W_V·x
    └────┬────┘   output = concat(heads)·W_O
         │
    ┌────┴────┐
    │  + 残差 │  ← x = x + Attention_output
    └────┬────┘
         │
    ┌────┴────┐
    │ Pre-Norm│  ← LayerNorm(x)
    └────┬────┘
         │
    ┌────┴────┐
    │   FFN   │  ← 每个 token 独立处理
    │ W₁→σ→W₂ │    不跨 token 交互！
    └────┬────┘
         │
    ┌────┴────┐
    │  + 残差 │  ← x = x + FFN_output
    └────┬────┘
         │
        输出 x'
```

**关键理解**：Attention 是**跨 token** 操作（token 之间交互），FFN 是**逐 token** 操作（每个 token 独立）。两者配合 = 既交流又思考。

---

## 三、FFN 为什么不可或缺？

### 实验证据：去掉 FFN 会怎样？

```
完整的 Transformer Block：
  x → Attention → + → FFN → + → output
  ✅ 既有 token 交互，又有逐 token 非线性

只保留 Attention（去掉 FFN）：
  x → Attention → + → output
  ❌ 多层 Attention 叠加仍然是线性的（在 V 子空间内）
  ❌ 表达能力大幅下降

只保留 FFN（去掉 Attention）：
  x → FFN → + → output
  ❌ 每个 token 独立处理 → 信息不流通
  ❌ 相当于每个 token 各自为战，没有上下文
```

### 为什么 FFN 参数量远大于 Attention？

```
典型 Transformer Block 的参数分配（d=768, d_ff=3072, n_heads=12）：

  Attention 参数：
    W_Q, W_K, W_V: 3 × (768×768) = 1.77M
    W_O (输出投影): 768×768 = 0.59M
    Attention 合计 ≈ 2.36M

  FFN 参数（ReLU 版）：
    W₁: 768×3072 = 2.36M
    W₂: 3072×768 = 2.36M
    FFN 合计 ≈ 4.72M

  FFN 参数量 ≈ Attention 的 2 倍！

原因：
  Attention 的"工作"是路由信息（决定 token A 应该关注 token B 多少）
  → 路由逻辑不需要太多参数

  FFN 的"工作"是知识存储 + 非线性变换
  → 需要大量参数来"记住"世界知识
  → d_ff = 4×d 不是随便选的——研究证明这是一个 sweet spot
     （太小：知识容量不够；太大：过拟合+推理慢）
```

> **小白理解**：Attention 像图书管理员——知道哪本书在哪个架子上（路由信息），本身不需要背下所有书的内容。FFN 像书的内容——大量参数存储了"世界知识"（语法规则、事实、推理模式）。**Transformer 的知识主要存储在 FFN 的权重中**，而非 Attention 中。这也是为什么 MoE 只替换 FFN 而不替换 Attention——替换 FFN 就是替换"知识库"。

---

## 四、FFN 与 MoE 的关系

MoE（Mixture of Experts）本质就是**把单个 FFN 替换成多个 Expert FFN + Router**：

```
标准 FFN：
  x → FFN(x) → output          一个专家处理所有 token

MoE FFN：
  x → Router(x) → 选择 Expert₂, Expert₅
    → 0.7×Expert₂(x) + 0.3×Expert₅(x) → output
  
  多个专家，每个专注于不同类型的 token
  但每个 token 只激活其中几个 → 参数大了，计算量没怎么涨
```

**所有专家都是 FFN**——MoE 的"Expert"就是标准 FFN layer。Router 决定"这个 token 该交给哪个 FFN 专家处理"。

> 详见 `MoE/moe_guide.md`

---

## 五、FFN 的各变体对比

| 变体 | 公式 | 参数量 | 优点 | 典型使用 |
|:---|:---|:---|:---|:---|
| **ReLU-FFN** | W₂·ReLU(W₁·x) | 2·d·d_ff | 最简单，最快 | 原始 Transformer |
| **GELU-FFN** | W₂·GELU(W₁·x) | 2·d·d_ff | 平滑，训练稳定 | BERT, GPT-2/3 |
| **SwiGLU-FFN** | (SiLU(W_g·x)⊙W_up·x)·W_down | 3·d·d_ff | 门控，表达力最强 | LLaMA, PaLM, Qwen |
| **MoE-FFN** | Σ g_i·FFN_i(x) | N·2·d·d_ff（总） | 大容量+小计算 | Mixtral, DriveMoE |
| **Identity-FFN** | x（什么都不做） | 0 | 最快的推理 | 某些层的"跳过"策略 |

---

## 六、一句话总结

> **FFN 是 Transformer Block 的"独立思考"层——在 Attention 跨 token 混合信息后，对每个 token 独立施加 d→4d→d 的升维→非线性→降维变换。Transformer 的大部分"知识"存储在 FFN 的权重中（FFN 参数 ≈ Attention 的 2 倍），现代 LLM 用 SwiGLU 门控机制替代 ReLU 进一步提升表达能力，而 MoE 则把 FFN 扩展成多个专家实现参数高效扩展。**

---

## 参考资料

- **Attention is All You Need（Vaswani et al., NeurIPS 2017）**：https://arxiv.org/abs/1706.03762 —— Transformer 原版，ReLU-FFN
- **GELU（Hendrycks & Gimpel, 2016）**：https://arxiv.org/abs/1606.08415 —— GELU 激活函数
- **SwiGLU（Shazeer, 2020）**：https://arxiv.org/abs/2002.05202 —— SwiGLU 门控 FFN
- **MoE 概念指南**：见 `MoE/moe_guide.md`
