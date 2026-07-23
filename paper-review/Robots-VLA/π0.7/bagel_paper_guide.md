# BAGEL 论文导读：用统一 decoder-only 架构 + 万亿级多模态交错数据预训练，实现理解与生成的涌现统一

> **论文标题**：Emerging Properties in Unified Multimodal Pretraining
> **会议/期刊**：arXiv 2025.05（37 页，17 张图）
> **作者**：Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, Haoqi Fan
> **机构**：ByteDance Seed / 深圳先进技术研究院 / Monash University / HKUST / UC Santa Cruz
> **arXiv**：https://arxiv.org/abs/2505.14683
> **代码**：https://bagel-ai.org（开源权重+代码）

---

## 读之前先搞清楚

### 这篇论文要解决什么问题？

多模态 AI 有两个方向各自发展得很好：**理解**（看图问答、OCR、物体检测）和**生成**（文生图、图像编辑、视频生成）。但把它们放进**同一个模型**里，让一个模型既能读懂图片又能生成图片——这件事一直很难。

已有的统一模型（如 Emu、Chameleon）主要用图文配对数据训练，缺乏视频、网页等更丰富的交错多模态数据。数据单一 → 能力割裂 → 理解和生成互相拖后腿。

BAGEL 的目标：**用万亿级的多模态交错数据（文本+图片+视频+网页），在一个 decoder-only 的统一架构上预训练，让模型既能在理解任务上超越专用模型，又能在生成任务上媲美生成模型，还能涌现出之前没教过的新能力。**

> **小白理解**：之前的模型要么是"读者"（只会看图回答问题），要么是"画家"（只会画图）。BAGEL 是一个"既能读又能画"的通才——给它一张图它能分析，给它一句话它能画出来。秘诀是它看过的东西比别人多得多——不仅是图文配对，还有带图文的网页、视频、交错数据，总量万亿级。

### 读懂这篇论文需要的前置知识

| 概念 | 简单解释 |
|:---|:---|
| **Decoder-only 架构** | 只用 Transformer 的 Decoder 部分（类似 GPT），逐 token 生成输出 |
| **Mixture-of-Transformers（MoT）** | 两个独立的 Transformer Expert 协同工作，而非一个统一模型 |
| **Unified Multimodal Model** | 一个模型同时支持多模态理解（看图回答）和生成（文生图） |
| **Interleaved Data（交错数据）** | 文本和图片交替出现的数据，如带图的网页、视频帧+字幕 |
| **Flow Matching** | 从纯噪声到目标图像的直线路径去噪方法，5-10 步即可 |
| **VAE（Variational Autoencoder）** | 把图像压缩为 latent code 再生成回去，降低计算量 |
| **Catastrophic Forgetting** | 用新任务数据训练后，模型忘记了之前学到的能力 |

---

## 一、BAGEL 的解决思路

核心 idea：**用一个 decoder-only 的 MoT 架构，在万亿 token 的多模态交错数据上从头预训练——两个 Transformer Expert 分别处理"理解"（离散 token）和"生成"（连续 latent），所有 token 共享 Self-Attention，让理解和生成在同一个表示空间中自然融合。**

```
传统方法：
  理解模型（LLaVA, Qwen-VL）← 图文配对数据 → 只会"看"
  生成模型（SD3, Flux）      ← 图文配对数据 → 只会"画"
  两者互不相干，无法在一个模型里同时做好

BAGEL 的方法：
  万亿级多模态交错数据（文本+图片+视频+网页）
          │
          ▼
  ┌─────────────────────────────────────────┐
  │  Decoder-only MoT（14B 参数）            │
  │                                          │
  │  ┌──────────────┐  ┌──────────────┐      │
  │  │Understanding │  │ Generation   │      │
  │  │Expert        │  │ Expert       │      │
  │  │(离散 token)   │  │ (连续 latent) │      │
  │  └──────┬───────┘  └──────┬───────┘      │
  │         └────────┬────────┘              │
  │                  ▼                       │
  │    共享 Multi-Head Self-Attention        │
  │                  │                       │
  │     ┌────────────┴────────────┐          │
  │     ▼                         ▼          │
  │  理解输出                  生成输出       │
  │  (文本回答)              (图像/视频帧)    │
  └─────────────────────────────────────────┘
```

> **小白理解**：传统方法是两个专家各干各的——读者只看书，画家只画画。BAGEL 让读者和画家坐在同一张桌子上，看同一份资料（万亿级多模态数据），互相交流（共享 Self-Attention）。结果读者的画画水平居然提高了，画家的阅读水平也提高了——这就是"涌现"。

---

## 二、核心设计：MoT 架构 + 多模态交错预训练

![图2：BAGEL 架构总览（原论文 Figure 2）](./bagel_paper_guide_figures/fig2_architecture.png)

*图2：BAGEL 使用两个 Transformer Expert 分别处理理解（离散 token）和生成（连续 latent）信息，所有 token 通过共享的 Multi-Head Self-Attention 交互。This is the architecture that π0.7's World Model is initialized from.*

### 第 1 步：双 Expert 架构 —— "读"和"画"分开处理

BAGEL 不是一个大模型硬扛所有事，而是两个 Expert 协作：

| | Understanding Expert | Generation Expert |
|:---|:---|:---|
| **处理什么** | 离散 token（文本、图像 token） | 连续 latent（VAE 压缩的图像特征） |
| **输入来源** | Text tokenizer + ViT 编码的图像 patch | VAE 编码的图像 latent |
| **输出** | 下一个文本/图像 token（交叉熵 loss） | 去噪后的图像 latent（MSE loss） |
| **参数量** | ~7B（LLM backbone + ViT 400M） | ~7B（generation backbone） |
| **总参数量** | — | **14B**（MoT，两个 Expert 合计） |

**为什么用两个 Expert 而不是一个**：离散 token 和连续 latent 在数值分布上差异巨大——前者是整数 ID，后者是浮点向量。硬塞进同一个 FFN 里会互相干扰。分 Expert 让每个 Expert 的 FFN 权重专注于自己那类数据。

### 第 2 步：共享 Self-Attention —— "读"和"画"互相看见

虽然 FFN 分开了，但 Self-Attention 是**共享的**。理解 Expert 的 token 和生成 Expert 的 token 在每一层都互相做 Attention：

```
BAGEL 第 l 层的操作：

  [理解 Expert 的离散 token] + [生成 Expert 的连续 latent]
              │
              ▼
  共享 Multi-Head Self-Attention（所有 token 互相看）
              │
      ┌───────┴───────┐
      ▼               ▼
  理解 FFN        生成 FFN
  (离散 token →    (连续 latent →
   离散 token)     连续 latent)
      │               │
      └───────┬───────┘
              ▼
        下一层（重复）
```

**为什么共享 Attention**：理解 Expert 看到生成 Expert 的 latent 后，能"感知"到图像的空间结构；生成 Expert 看到理解 Expert 的文本 token 后，能"理解"用户的文字指令。这就是"理解和生成在一个空间里融合"的关键。

### 第 3 步：万亿级多模态交错预训练

BAGEL 的训练数据不是"图文配对"那种简单格式，而是**交错数据**——文本、图片、视频帧交替出现，模拟互联网的真实形态：

| 数据类型 | 规模 | 格式 |
|:---|:---|:---|
| 图文交错网页数据 | 万亿 token 级 | 文本段落 + 嵌入图片交替 |
| 视频交错数据 | 百亿帧级 | 视频帧 + 字幕/语音文本交替 |
| 纯文本数据 | 补充 | 保持语言能力 |
| 图文配对数据 | 补充 | 标准图文理解/生成任务 |

**数据配比**：生成数据 : 理解数据 ≈ 4:1。论文消融实验证明这个比例最优——生成需要更多数据（因为是连续空间），理解相对数据高效。

> **小白理解**：普通模型的"教材"是课本（图文配对），BAGEL 的"教材"是整个互联网——带图的新闻、带字幕的视频、图文交错的百科页面。课本教出来的只会做课后题，互联网教出来的啥都会。**4:1 的生成:理解比例就像"实践课:理论课"——画画需要更多动手练习，阅读做题相对省时间。**

---

## 三、训练流程

> **总览**：多模态交错数据 → 统一 token 化 → Decoder-only MoT 预训练 → 涌现能力

BAGEL 是**从头预训练**的，不是从某个 LLM/VLM fine-tune。训练分三个阶段：

1. **第一阶段（图文配对）**：先用标准图文数据训练基础能力
2. **第二阶段（加入交错数据）**：逐步混入网页交错数据和视频数据，提升复杂推理
3. **第三阶段（高质量精调）**：用高质量数据精调，强化指令遵循和生成质量

训练配置：14B 参数，万亿 token，数百/数千 GPU，数周训练时间。Loss = CE（理解） + MSE（生成），联合优化。

---

## 四、核心创新点

| 创新点 | 具体内容 | 为什么重要 |
|:---|:---|:---|
| **统一 decoder-only MoT** | 双 Expert + 共享 Self-Attention | 理解和生成在同一个表示空间自然融合 |
| **万亿级交错数据预训练** | 文本+图片+视频+网页交替 | 首次展示交错数据对统一模型的涌现能力作用 |
| **涌现能力** | free-form 图像编辑、未来帧预测、3D 操作、世界导航 | 这些能力没有被显式教过 |
| **14B 开源** | 权重/代码/数据配方全公开 | 社区可复现的统一多模态 baseline |

---

## 五、BAGEL 在 π0.7 中的角色

π0.7 的 World Model 从 BAGEL 的权重初始化，沿用 BAGEL 的训练配方：

- BAGEL 学会了"文字描述 → 预测画面变化"——这正是 subgoal image 生成需要的
- BAGEL 的 ViT（语义）+ VAE（细节）双编码器让 World Model 能同时理解场景语义和保留纹理细节
- π0.7 的 World Model 相当于一个小型 BAGEL，专门针对机器人数据 fine-tune

> 📖 详细对比见 [BAGEL 在 π0.7 中的角色详解](./pi07_bagel_intro.md)

---

## 六、局限性

| 局限性 | 为什么是局限 |
|:---|:---|
| **生成质量不及专用模型** | 14B 统一模型在纯文生图任务上不及同规模的专用生成模型（如 Flux） |
| **复杂文字渲染仍困难** | 图片中的长文本、复杂排版仍会出错 |
| **推理成本高** | 14B 全量推理，部署门槛高 |
| **训练成本巨大** | 万亿 token 预训练需要数百 GPU 数周 |

---

## 七、一句话总结

> **BAGEL 用 decoder-only MoT 架构 + 万亿级多模态交错数据预训练，首次在一个 14B 开源模型中实现了理解与生成的统一涌现——既能看图问答，又能图像编辑预测未来帧——这也是 π0.7 World Model 的基石。**

---

## 参考资料

- BAGEL 论文：https://arxiv.org/abs/2505.14683
- BAGEL 项目页：https://bagel-ai.org
- [π0.7 论文深度解读](./pi07_paper_guide.md)
- [BAGEL 在 π0.7 中的角色详解](./pi07_bagel_intro.md)
