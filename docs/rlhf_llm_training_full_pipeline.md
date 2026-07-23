# 强化学习训练大模型 Demo 与完整流程

这篇文档对应示例脚本 [examples/rlhf_llm_detailed_demo.py](../examples/rlhf_llm_detailed_demo.py)。

目标不是复刻真实工业大模型的算力规模，而是把一条最重要的训练主线讲清楚：

1. 先做语言模型预训练。
2. 再做指令监督微调（SFT）。
3. 然后训练奖励模型（Reward Model）。
4. 最后用强化学习做对齐（RLHF，这里用 PPO 演示）。

如果你之前一直听说“大模型训练很复杂”，但不清楚各阶段到底怎么衔接，可以把这篇文档当成一张总地图。

## 1. 先厘清一个常见误区

“强化学习训练大模型”并不等于：

- 从零开始只靠强化学习把一个语言模型训出来。

真实流程通常是：

- 绝大多数语言能力来自海量语料上的自监督预训练。
- 指令跟随能力主要来自 SFT。
- 强化学习或偏好优化，主要负责把模型从“会回答”进一步推向“更符合人类偏好、更稳、更安全、更有用”。

也就是说，RLHF 通常不是第一阶段，而是后对齐阶段。

## 2. 一条完整的大模型训练流水线

可以把大模型系统拆成下面 8 步。

### 2.1 数据建设

这是整个系统的地基，通常包括四类数据：

1. 预训练语料：网页、书籍、代码、论文、论坛问答等。
2. SFT 数据：prompt-response 指令样本。
3. 偏好数据：同一个 prompt 下，哪条回答更好。
4. 评测与安全数据：benchmarks、红队样本、风险样本。

核心工作包括：

- 去重
- 清洗脏数据
- 过滤低质量文本
- 敏感信息脱敏
- 多语言与领域分桶
- 采样配比设计

真实工程里，数据质量往往比“单纯堆更多 token”更关键。

### 2.2 分词器与词表

大模型不会直接看“字符串”，而是看 token 序列。

典型方案：

- BPE
- SentencePiece
- Unigram

分词器决定：

- 词表大小
- 序列长度
- 中文/英文/代码的切分粒度
- 训练吞吐和压缩率

本仓库的 demo 为了教学可读性，只用了最简单的空格分词。

### 2.3 基座模型预训练

这一阶段的目标是学会语言统计规律，也就是 next-token prediction：

$$
P(x_t \mid x_{<t})
$$

主流结构通常是 Decoder-Only Transformer，也就是 GPT / LLaMA / Qwen 这一类。

预训练阶段学到的能力包括：

- 基础语言流畅性
- 常识知识
- 模式补全能力
- 跨领域迁移能力
- 一定程度的推理雏形

真实训练里，这一步最吃算力。

### 2.4 指令监督微调（SFT）

预训练模型“会续写”，但不一定“会按要求回答”。

SFT 的作用是把模型从通用语言模型，拉到“助手”分布上。典型训练目标是：

- 输入：用户指令 prompt
- 输出：高质量参考回答 response

SFT 后模型会显著改善：

- 指令遵循
- 回复格式稳定性
- 多轮对话风格
- 拒答模板一致性

但 SFT 的上限取决于标签质量，而且它只能学到“示范答案”，很难直接表达“哪种回答更好”。

### 2.5 奖励模型（Reward Model）

这一步的输入通常不是单条答案，而是“偏好比较”：

- 同一 prompt
- 一个 chosen 回答
- 一个 rejected 回答

奖励模型学习一个标量分数：

$$
r(x, y)
$$

并希望满足：

$$
r(x, y_{chosen}) > r(x, y_{rejected})
$$

常见训练方式是 pairwise ranking loss，例如 Bradley-Terry 风格目标：

$$
\mathcal{L}_{RM} = -\log \sigma\left(r(x, y_w) - r(x, y_l)\right)
$$

这里的价值在于：

- SFT 只知道“示范答案是什么”
- 奖励模型开始显式学习“人类更偏好什么”

### 2.6 强化学习对齐（RLHF）

这一步最常见的经典组合是：

1. 策略模型 policy
2. 冻结的参考模型 reference
3. 奖励模型 reward model
4. PPO 或其他偏好优化算法

基本思路是：

- policy 生成回答
- reward model 给出奖励
- 同时约束 policy 不要偏离 reference 太远
- 用 PPO 更新 policy

典型的总奖励写法是：

$$
R = r_{rm}(x, y) - \beta \cdot \mathrm{KL}(\pi_{policy} \Vert \pi_{ref})
$$

其中 KL 约束的作用很关键：

- 如果没有它，策略可能为了刷奖励而迅速偏离原始语言分布
- 会导致胡言乱语、模式坍塌或 reward hacking

## 3. PPO 在这个场景里到底做了什么

在 RLHF 里，可以把“一次完整回答”视为一段 action sequence。

PPO 的直觉是：

- 如果某条回答拿到更高奖励，就提高它再次被采样到的概率。
- 但每次更新不能太猛，要用 clip 限制策略步长。

经典 PPO 目标中的关键比率是：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

然后通过 clipped objective 控制更新幅度。

对于语言模型，这里的 action 就是生成 token，state 则是当前上下文。

## 4. 为什么现在很多团队不只用 PPO

虽然 PPO 是 RLHF 经典路线，但工程上并不是唯一选择。现在常见的还有：

1. DPO：直接偏好优化，不显式训练 value function，训练更简单。
2. ORPO：把偏好优化直接融入 odds-ratio 风格目标。
3. GRPO：按组比较回答质量，减少对 value model 的依赖。
4. RLAIF：用 AI feedback 替代部分人工标注。

PPO 仍然重要，因为它最能帮助理解“奖励 + KL + 策略更新”的完整闭环。

## 5. 真实大模型工程里还缺哪些东西

教学 demo 只覆盖了算法主线，但真实系统还需要很多基础设施。

### 5.1 训练系统

- Data Parallel
- Tensor Parallel
- Pipeline Parallel
- ZeRO / FSDP
- 混合精度训练
- 梯度检查点
- FlashAttention

### 5.2 推理与服务系统

- KV Cache
- 连续批处理
- 张量并行推理
- vLLM / TensorRT-LLM
- 服务限流与降级

### 5.3 安全治理

- 敏感内容识别
- 拒答策略
- 工具调用权限控制
- 系统 prompt 防注入
- 上线后的红队与回归评测

### 5.4 数据飞轮

上线后通常还会继续：

1. 收集真实用户反馈。
2. 挖掘失败案例。
3. 补充 SFT 和偏好数据。
4. 重新训练或增量训练。
5. 回归测试后再次上线。

这才是一个持续演化的大模型系统。

## 6. 本仓库 Demo 对应这条流程的哪个位置

示例脚本 [examples/rlhf_llm_detailed_demo.py](../examples/rlhf_llm_detailed_demo.py) 按下面顺序实现：

1. 定义一个极简 Decoder-Only Transformer。
2. 用 toy 语料做语言模型预训练。
3. 用 toy 指令数据做 SFT。
4. 用 chosen/rejected 对训练奖励模型。
5. 用 PPO 做一个最小可运行版 RLHF 更新。

它故意简化了很多工业细节，比如：

- 没有大规模数据加载器
- 没有分布式训练
- 没有混合精度
- 没有长上下文
- 没有工具调用与 agent 训练
- 没有复杂安全分类器

但这并不影响它作为“理解主链路”的教学价值。

## 7. 应该如何使用这个 Demo

建议按下面顺序阅读：

1. 先看模型结构：理解 Decoder-Only Transformer 为什么适合生成。
2. 再看预训练：确认 next-token loss 是如何构造的。
3. 再看 SFT：注意 prompt 部分通常不计入损失。
4. 再看奖励模型：理解 chosen 和 rejected 怎么形成偏好监督。
5. 最后看 PPO：重点关注 reward、KL、old_logprob、ratio、clip 的关系。

如果你要把这个 demo 升级成更接近真实项目的版本，优先级建议是：

1. 把 toy tokenizer 换成真实分词器。
2. 把 toy 数据换成 JSONL 指令数据集。
3. 用 Hugging Face Transformers + TRL 重写训练循环。
4. 加入 LoRA / QLoRA，降低显存成本。
5. 再决定使用 PPO、DPO 还是 GRPO。

## 8. 一句话总结

大模型的“强化学习训练”本质上不是凭空创造语言能力，而是：

- 在预训练和 SFT 已经打好基础之后，
- 用奖励模型把“更好回答”的偏好显式化，
- 再用 RL 或偏好优化方法，把模型往更符合目标的方向推一段。

如果你已经看懂这条链路，就已经抓住了现代大模型对齐流程里最核心的一部分。