# BAGEL 图像生成模型简介

> 配套 [π0.7 论文深度解读](./pi07_paper_guide.md) 第二节第 2 步，解释 World Model 的基础模型 BAGEL。

---

## BAGEL 是什么

BAGEL（[105] Chaorui Deng et al., RSS 2025）是一个 **14B 参数的 Mixture-of-Transformers 图像生成模型**，带 400M 参数的视觉编码器，经过 web-scale 预训练。π0.7 的 World Model 从 BAGEL 初始化，并沿用其训练配方。

> ⚠️ 以下解释基于 π0.7 论文对 BAGEL 的引用描述，非 BAGEL 原论文的完整解读。

---

## 先说普通情况：图像生成模型怎么工作

图像生成模型（如 Stable Diffusion、DALL-E）的核心任务：**给定文字描述，生成一张对应的图片**。

```
"a cat sitting on a table" → 图像生成模型 → 一张猫坐在桌子上的图
```

主流方法用 diffusion——从纯噪声出发，逐步去噪，每一步都看着文字条件来修正方向，最终得到一张清晰的图。

---

## 再说当前情况：BAGEL 独特在哪

π0.7 论文引用 BAGEL 时强调了几个关键特性：

**Mixture-of-Transformers 架构**：BAGEL 不是一个单一模型，而是组合了多个 Transformer：
- 视觉编码器（400M ViT）理解输入图像
- VAE 编码器捕获细粒度图像细节
- 7B LLM backbone 处理语言条件
- 7B generation backbone 生成图像

这和 π0.7 自己的 MoT 架构思路一脉相承——专业化分工。

**图像生成的输入不是纯文字**：BAGEL 接受：
- ViT 编码的当前观测（语义理解）
- VAE 编码的当前观测（细节纹理）
- VAE 编码的目标图像（带噪声，作为去噪起点）

三份输入同时送入，同时考虑"当前场景长什么样"和"目标画面应该长什么样"。

---

## BAGEL 在 π0.7 中的角色

π0.7 的 World Model **从 BAGEL 的权重初始化**，训练配方也基本沿用：

| | BAGEL（原版） | π0.7 World Model（适配后） |
|:---|:---|:---|
| **基础模型** | 14B MoT | 从 BAGEL 初始化，但 π0.7 描述为 "lightweight" |
| **输入** | 图像 + 文字 → 生成图像 | 当前观测（3 相机）+ 子任务指令 → 段末目标画面 |
| **视觉编码** | ViT（语义）+ VAE（细节） | 同，沿用 BAGEL 的双编码器 |
| **训练数据** | web-scale 图文数据 | robot demo 段末帧 + 人类自我视频 + 开源图像/视频数据 |
| **训练目标** | 图文匹配 → 图像生成 | 当前画面 + 指令 → 段末真实画面（ot_end） |

论文说 World Model "largely uses the same training recipe"——意味着训练流程（loss 函数、优化器、去噪步数等）和 BAGEL 基本一致，但换成了机器人数据。

**论文明确的训练数据组成**（§VI-C）：

| 数据来源 | 作用 |
|:---|:---|
| robot demo 子集 | 学习"当前画面+指令 → 做完后画面"的核心映射 |
| egocentric human video（高质量语言分段标注） | 补充人类视角的操作变化，增强泛化 |
| 开源图像编辑数据集 | 保持 BAGEL 原版的图像合成/编辑语义能力 |
| 开源视频数据集 | 学习时序变化模式，提高 subgoal 的时序一致性 |

论文特别强调：**语言标注质量（尤其是时序分段质量）对 subgoal 质量有很大影响。**

---

### 🔢 具体数字例子：一次 World Model 推理的计算量

**条件设定**：当前观测 3 相机视图，需预测 ∆=4 秒后的 subgoal image。

| 步骤 | 处理内容 | 分辨率/维度 | 说明 |
|:---|:---|:---|:---|
| ViT 编码（语义） | 3 张当前观测 | 448×336（每张） | patch=14，理解"物体是什么、在哪" |
| VAE 编码（细节） | 3 张当前观测 | 512×384（每张） | patch=16，保留纹理细节 |
| 子任务指令 | "pick up the sweet potato" | 文本 token 序列 | 告诉模型"这一步的目标是什么" |
| 目标预测 | 段末 3 张画面 | — | 作为去噪目标（训练时有 ground truth，推理时需预测） |
| **推理频率** | **每 ∆=4 秒重新生成** | — | 论文沿用 SuSIE [93] 的间隔设置 |

> 分辨率差异原因：ViT patch=14，VAE patch=16，为保持整数个 patch，分辨率分别取 448×336 和 512×384。

> **关键理解**：World Model 不是每帧都跑——每 4 秒才生成一次 subgoal。在这 4 秒内，VLA 拿着同一个 subgoal 持续输出 action。**这和 ChatGPT 的 continuous batching 思路类似——让"慢"的生成模型低频运行，让"快"的动作模型高频运行。**

---

## 最后说为什么：为什么选 BAGEL

1. **web-scale 预训练**：BAGEL 在互联网规模的图文数据上预训练过，已经学会了"文字描述→画面变化"的映射——这对 π0.7 来说就是天然的"子任务指令→做完后的画面"映射基础
2. **双编码器设计**：ViT 负责理解物体是什么（语义），VAE 负责保留纹理细节——这对 subgoal image 的质量至关重要，因为机器人需要看清物体精确位置
3. **MoT 架构兼容**：BAGEL 的 Mixture-of-Transformers 思想和 π0.7 的 VLA Backbone 一脉相承，初始化权重更容易适配

> **小白理解**：BAGEL 就像一个已经学会"根据菜谱描述画出成品菜"的画师。π0.7 把他请来，告诉他"不用画菜了，画'这步操作做完后机器人应该是什么姿势'"。因为画师已经有 web-scale 的绘画功底（预训练），只需要针对机器人场景稍作调整（fine-tune），就能画出质量不错的 subgoal image。

#### ❓ 常见疑问：BAGEL 原版 14B，为什么 π0.7 叫它 "lightweight"？

> π0.7 论文中 World Model 被描述为 "lightweight"，但 BAGEL 原版是 14B 的大家伙。可能的原因是：
>
> 1. **只用了 BAGEL 的一部分**：π0.7 初始化 World Model 时可能只取了 BAGEL 的图像生成 decoder 部分（7B generation backbone），而非完整的 14B MoT
> 2. **相对轻重**：相比于 VLA Backbone 的 ~5B + Action Expert 860M，World Model 即使有 7B 也是"相对轻量"的——因为它只在推理时每 4 秒运行一次
> 3. **论文未明确**：具体参数量论文未公开，"lightweight"的判断标准可能是相对于整体系统的计算占比
>
> **结论**：论文原话为 "lightweight world model"，但具体参数量未公开。以上为合理推测，非论文确认的细节。

---

## 参考资料

- BAGEL 原论文：[105] Chaorui Deng et al., RSS 2025 → 📖 [BAGEL 论文深度解读](./bagel_paper_guide.md)
- π0.7 论文 §VI-C: https://www.pi.website/download/pi07.pdf
- [π0.7 论文深度解读](./pi07_paper_guide.md)
