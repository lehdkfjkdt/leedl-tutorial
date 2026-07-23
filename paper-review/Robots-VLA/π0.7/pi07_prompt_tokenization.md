# π0.7 Prompt 中的三种 Token 编码方式详解

> 配套 [π0.7 论文深度解读](./pi07_paper_guide.md) 第二节第 1 步，详解三种 prompt token 的编码机制。

---

## 一、离散 Token —— Metadata 和 Control Modality 如何变成模型能理解的输入

### 先说普通情况：文本 token 化

语言模型（LLM/VLM）接收的所有输入最终都是 token。一段文本如 "fold the shirt" 会被切分为若干 token ID：

```
"fold the shirt" → tokenizer → [4532, 278, 8912] → Embedding Lookup → 3个向量 ∈ ℝ^d
```

每个 token ID 对应词表中的一个位置，通过查表得到一个 d 维向量。

### 再说当前情况：非文本信息如何 token 化

> ⚠️ 以下 token 化机制基于 VLM 通用做法推断，论文未公开具体 token ID 和 embedding 细节。

π0.7 的 metadata（quality / speed / success）和 control modality（joint / end-effector）**不是文字**，不能直接用文本 tokenizer。做法是：

```
metadata 的例子——quality ∈ {high, medium, low}：

  在词表中预留 3 个特殊 token（通用 VLM 做法）：
    <quality_high>   → 特殊 token A
    <quality_medium> → 特殊 token B
    <quality_low>    → 特殊 token C

  推理时用户选 quality=high：
    输入序列中直接插入 token A
    → Embedding Lookup → 一个 d 维向量
    → 这个向量和 image token、text token 拼在一起 → 送入 VLA Backbone

  ★ 关键：这些特殊 token 的 embedding 是**随机初始化 + 可训练**的
    不是固定的——训练中梯度反传会教会模型：
      token A（quality_high）→ 关联"精准、慢速、小心"的动作模式
      token C（quality_low） → 关联"粗糙、快速"的动作模式
```

**metadata 的完整取值空间**：

| 字段 | 取值 | 预留 token 数 | 说明 |
|:---|:---|:---|:---|
| quality | high / medium / low | 3 | 质量等级 |
| speed | fast / normal / slow | 3 | 速度等级 |
| success | 1 / 0 / 0.5 | 3 | 成功/失败/部分成功 |

**control modality 只有 2 种取值**——所以叫"离散二分类 token"：

```
joint_control        → 特殊 token（二选一）
end_effector_control → 特殊 token（二选一）
```

### 最后说为什么：为什么不用连续值？

速度标签写成 "speed=1.5" 这种连续值不是更精确吗？为什么选离散的三档？

1. **token 化的天然限制**：VLM 的输入是离散 token，不支持浮点数直接输入。要用连续值必须走 state token 的投影路线（MLP → d 维），但这样就和 text token 不在同一个语义空间了
2. **离散 = 显式 conditioning**：三个独立 token 对应三个独立的 embedding 向量，模型能学会"quality_high 的 embedding 应该把动作推向精准方向"——这是可解释的
3. **训练数据天然离散**：人工标注 quality 时就是打高/中/低三档，不会精确到 1.5

> **小白理解**：离散 token 就像遥控器上的"运动模式/标准模式/节能模式"按钮——三个档位，每档背后对应一套完整的参数调整策略。如果换成旋钮（连续值），模型反而不知道该学到什么——0.73 和 0.74 对应的行为差异太小，训练信号不够强。

---

## 二、MEM Encoder → 时序编码 Token —— 历史帧怎么压缩成一个 token

### 先说普通情况：单帧图像编码

一张 224×224 的 RGB 图送进 ViT，输出 256 个 patch token。每个 token 携带图像中一个局部区域的信息。这是 VLM 处理单张图片的标准方式。

### 再说当前情况：K 帧历史怎么高效编码

> ⚠️ MEM Encoder 具体架构论文未公开。以下流程中 ① 每帧过 ViT 是确定的部分，② 时序聚合的内部机制是通用推理，具体实现可能与实际不同。

π0.7 需要把过去 K 帧（如 K=5）的历史画面作为 conditioning 输入。如果直接送 K 张图进 ViT，token 数会膨胀到 K × 256，训练和推理成本翻 K 倍。

论文描述的流程：

```
Observation Memory 编码流程：

  过去 K 帧（如 K=5，每帧 224×224）：
    Frame_{t-4}, Frame_{t-3}, Frame_{t-2}, Frame_{t-1}, Frame_t
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │ ① 每帧独立过 ViT                              │
  │    Frame_{t-4} → ViT → 256 tokens, each ℝ^d  │
  │    Frame_{t-3} → ViT → 256 tokens            │
  │    ...                                        │
  │    Frame_t     → ViT → 256 tokens            │
  │                                               │
  │    总共：K × 256 个 token × d 维               │
  └─────────────────────┬───────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────┐
  │ ② "MEM-style video history encoder"         │
  │   （论文仅给名称，未公开架构）                 │
  │        │                                      │
  │        ▼                                      │
  │   256 个 memory tokens（与单帧 image token     │
  │   数量相同，每个融合了 K 帧的时序信息）         │
  └─────────────────────────────────────────────┘
```

#### MEM Encoder 的作用与推测机制

论文没有公开 MEM Encoder 的具体架构细节，但明确了它的**输入输出和目的**：

- **输入**：过去 K 帧的观测画面（如 K=5，每帧 224×224）
- **输出**：256 个 memory tokens（和单帧的 image tokens 数量相同）
- **目的**：让模型知道"之前发生了什么"——单帧画面只能看到"抽屉现在开着"，加上历史帧才知道"抽屉是被打开的还是被关上的"

核心约束是：**输出 token 数不能膨胀**。如果 K 帧直接拼成 5×256 个 token，VLA Backbone 的 Self-Attention 计算量会翻 5 倍。

> ⚠️ 以下三步机制是基于"MEM-style"类方法的通用推理，非论文确认的实现细节。

**步骤 A：空间池化——每帧压缩成少量 query token**

```
Frame_{t-4} 的 256 个 patch token:
  [p₁, p₂, ..., p₂₅₆]   每个 ∈ ℝ^d

  → 可学习的 query 向量 [q₁...q_m] 做 Cross-Attention（m ≪ 256，如 m=16）
  → 得到 m 个 query token，每个已经融合了整帧的空间信息

  结果：每帧从 256 个空间 patch token → m 个帧级 token
  4 帧（t-4 到 t-1）+ 当前帧 = 5m 个 token
```

为什么不能按 patch 网格直接对齐？因为相机在机器人身上随运动移动——Frame_{t-4} 的 patch(14,14) 可能对着微波炉，Frame_t 的 patch(14,14) 可能对着旁边的台面，两个 patch 拍的不是同一物理位置。所以先压缩成帧级表示再融合才合理。

**步骤 B：时序自注意力——帧级 token 之间互相关注**

```
所有帧级 token 拼成序列（共 5m 个，每个 ∈ ℝ^d）
      │
      ▼
  给每个 token 加时间位置编码（知道"我是 0.08s 前的"）
      │
      ▼
  Multi-Head Self-Attention（所有 token 互相看）
      │
      ▼
  FFN + 残差
      │
      ▼
  输出：还是 5m 个 token，但每个已经融合了跨时间信息
```

注意力权重由模型自己学习——最近的帧通常权重最高（对当前决策影响最大），较早的帧虽然权重低但提供了运动方向的信息（抽屉从关→开 vs 从开→关）。

**步骤 C：投影回原始维度——对齐 image token 数量**

```
5m 个时序融合 token → 线性投影 → 256 个 memory tokens
```

这样 memory tokens 数量与单帧 image tokens 数量相同（256 个），可以直接并排送入 VLA Backbone。

> ⚠️ 以下数字例子旨在帮助理解时序编码的**目的**，具体数值是示意性的。

**核心直觉**：假设画面右下角有一个抽屉在 5 帧内从关闭变到打开：

| 帧 | 画面右下角的内容 | 学到的权重（示意） |
|:---|:---|:---:|
| Frame_{t-4} | 抽屉完全关闭 | 0.05 |
| Frame_{t-3} | 抽屉微开 1cm | 0.08 |
| Frame_{t-2} | 抽屉半开 5cm | 0.17 |
| Frame_{t-1} | 抽屉大半开 8cm | 0.25 |
| **Frame_t** | **抽屉完全打开** | **0.45** |

> **关键理解**：无论 MEM Encoder 内部怎么聚合，它的输出必须编码"这个区域在 5 帧内从关闭变成了打开"这一时序变化。如果只给当前帧（Frame_t），模型只能看到"抽屉开着"，无法判断它是正在被打开还是正在被关上——**方向信息只能从多帧对比中得到。**
| Frame_{t-2} | 抽屉半开 5cm |
| Frame_{t-1} | 抽屉大半开 8cm |
| Frame_t | 抽屉完全打开 |

> **关键理解**：无论 MEM Encoder 内部怎么聚合，它的输出必须编码"这个区域在 5 帧内从关闭变成了打开"这一时序变化。如果只给当前帧（Frame_t），模型只能看到"抽屉开着"，无法判断它是正在被打开还是正在被关上——**方向信息只能从多帧对比中得到。**

### 最后说为什么：为什么需要 MEM Encoder

1. **不增加 token 数**：memory token 数量与单帧 image token 数量相同，不增加 VLA Backbone 的 Self-Attention 计算量（O(N²)）
2. **时序信息融入**：单帧画面是静止快照，多帧才能捕捉运动方向和状态变化

> **小白理解**：MEM Encoder 就像一个"视频摘要器"——5 秒的画面压缩成和 1 张截图一样大小的信息包，但这个包里不仅说了"现在抽屉开着"，还说了"抽屉是从关闭变成打开的"。**只看一张图，你不知道抽屉正在被打开还是被关上；看了 5 帧的摘要，你就知道了。**

---

## 三、三种 Token 在 VLA Backbone 中的交互

三种 token 在输入序列中并排送入 VLA Backbone，通过 Self-Attention 互相"看见"对方：

```
完整输入序列（拼接后）：

  [img_1, ..., img_256 | subgoal_1, ..., subgoal_256 | lang_1, ..., lang_n |
   <quality_high> | <speed_fast> | <success_1> |
   mem_1, ..., mem_256 | <joint_control>]

  → VLA Backbone（~5B Transformer）
  → Self-Attention：每个 token 都能 attend 到所有其他 token
  → 形成统一的跨模态表示
  → Action Expert 输出 action chunk
```

**关键**：metadata token 虽然只有 3-5 个，但因为 Self-Attention 的全局性，它们在**每一层**都能影响所有 image token 和 memory token 的表示——相当于给整个场景理解过程"调参"。

> **小白理解**：三种 token 的关系就像做菜——image token 是"食材长什么样"，memory token 是"刚才切到哪了"，metadata token 是"今天要做精致料理还是快餐"。三种信息在大厨（VLA Backbone）脑子里同时处理，互相影响——做快餐时可能跳过精细切配步骤（metadata 影响 action），看到食材不新鲜时可能降低 quality 预期（image 影响 metadata 解读）。

---

## 参考资料

- [π0.7 论文深度解读](./pi07_paper_guide.md)
- π0.7 论文 PDF：https://www.pi.website/download/pi07.pdf
