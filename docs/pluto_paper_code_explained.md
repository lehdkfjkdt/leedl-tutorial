# PLUTO 论文与官方代码详解

## 1. 论文信息

- 论文标题：PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving
- arXiv：<https://arxiv.org/abs/2404.14327>
- 项目主页：<https://jchengai.github.io/pluto/>
- 官方代码：<https://github.com/jchengai/pluto>
- 任务方向：自动驾驶中的 imitation learning-based planning

这篇论文的核心目标很明确：证明学习式规划器不一定只能作为规则系统的补充，它也有可能在闭环规划指标上真正超过强规则基线。PLUTO 的贡献不只是“模型更大”或者“训练更久”，而是围绕自动驾驶规划里的三个老问题，提出了一整套相互配合的方案：

1. 多模态驾驶行为学不好，尤其是横向行为不够灵活。
2. 纯 imitation learning 容易学到错误捷径，也很难直接把安全约束放进 vector-based planner 里联合训练。
3. 开环训练、闭环测试之间存在 distribution shift，模型还会出现 causal confusion，也就是“看起来学会了，其实只是模仿表象”。

PLUTO 对应地给出三把“刀”：

1. 纵向-横向解耦再组合的 query-based 规划架构。
2. 基于 differentiable interpolation 的高效辅助损失。
3. Contrastive Imitation Learning, 简称 CIL。

如果只记一句话，可以把 PLUTO 理解成：

> 它不是单纯让模型去回归一条未来轨迹，而是先系统地构造一批具有明确行为含义的候选规划，再用 imitation、prediction、auxiliary loss 和 contrastive loss 共同把这些候选学“对”，最后再在推理阶段结合规则安全检查选“稳”。

## 2. 论文解决的是什么问题

### 2.1 背景

自动驾驶规划大体可以分成两类思路：

1. 规则或优化方法。
2. 学习方法，尤其是 imitation learning。

规则方法通常安全性和可控性更好，但上限受人工设计约束。学习方法看起来更有潜力，因为可以直接从真实驾驶数据中学习人类行为，但现实问题是：很多学习式 planner 在闭环评测里并不强，甚至会输给强规则基线。论文里点名的强规则对手是 PDM-Closed，它在 nuPlan benchmark 上一度是非常强的基准。

PLUTO 的价值就在这里：它不是停留在 open-loop 轨迹误差更小，而是强调闭环 planning score 真正提升，并首次超过当时顶级规则规划器。

### 2.2 传统 imitation planner 的典型问题

论文把问题拆得很清楚。

#### 问题一：横向行为难学

很多 imitation planner 能学会跟车、减速、停车这类纵向行为，但对变道、绕障、无保护左转这些横向决策并不稳定。原因在于很多模型虽然输出“多模态轨迹”，但这些 mode 没有被设计成真正可控的行为组合，最后容易 mode collapse，或者只学出几条相似轨迹。

#### 问题二：安全约束难以直接进入 vector planner

在 planning 里，仅靠模仿专家轨迹不够。模型可能学会表面模式，但不知道越界、撞车为什么不该发生。过去有些方法会加入 collision loss、off-road loss，但常常依赖 raster representation 或 differentiable rasterizer。这样做在 vector-based 模型上不够自然，也不够高效。

#### 问题三：开环到闭环存在偏移

训练时模型看到的是专家分布，测试时看到的是自己执行后形成的新状态分布。这会造成 compounding error。另一个更隐蔽的问题是 causal confusion，比如模型减速不是因为看见红灯，而是因为训练数据里“减速时旁边总有慢车”，它学错了因果关系。

## 3. PLUTO 的整体方法框架

PLUTO 的整体流程可以概括为五段：

1. 构建向量化场景输入。
2. 用 Transformer encoder 融合场景上下文。
3. 用横向 query 和纵向 query 组合生成多模态规划候选。
4. 同时预测其他 agent 的未来轨迹，并在训练中加入 imitation、prediction、auxiliary、contrastive 四类损失。
5. 推理阶段对候选轨迹做 top-k 筛选、前向仿真和规则评估，选出最终轨迹。

这一套设计里最关键的是第三、四、五步。第三步决定“候选轨迹是不是够多样”；第四步决定“训练是否真的学到可闭环执行的行为”；第五步决定“线上选择是否更稳”。

### 3.1 一张先看懂全局的流程图

先不要急着看 loss 和代码，先把整条数据流抓住。PLUTO 的整体流可以画成下面这样：

```text
场景输入
├─ ego 当前状态
├─ 其他动态 agent 历史
├─ 静态障碍物
└─ 向量化地图 polyline

   │
   ▼
各子模块编码
├─ Ego State Dropout Encoder
├─ Agent History Encoder
├─ Static Object Encoder
└─ Map Polyline Encoder

   │
   ▼
场景 token 拼接 + 位置/语义编码

   │
   ▼
Transformer Scene Encoder
输出共享场景记忆 memory

   ├───────────────────────────────┐
   │                               │
   ▼                               ▼
Agent Predictor                    Planning Decoder
预测周围车未来                       横向 reference lines
               × 纵向 longitudinal modes
               生成多模态候选轨迹

   │                               │
   │                               ├─ trajectory regression
   │                               ├─ trajectory score
   │                               └─ ref-free trajectory head
   ▼
训练阶段
├─ imitation loss
├─ prediction loss
├─ auxiliary loss(off-road/collision)
└─ contrastive loss(CIL)

   │
   ▼
推理阶段
├─ top-k 候选筛选
├─ forward simulation
├─ 规则打分(progress / comfort / TTC / rule)
└─ 融合学习分数与规则分数，选最终轨迹
```

### 3.2 一条完整的数据流怎么走

如果按“张量从哪里来，往哪里去”来理解，PLUTO 的 forward 可以压缩成下面 8 步：

1. 先把原始结构化场景拆成 ego、动态 agent、静态障碍、地图 polyline 四类输入。
2. 四类输入分别编码成统一隐藏维度 $D$ 的 token embedding。
3. 把所有 token 拼成一个场景序列，叠加位置编码和语义类型编码。
4. 用 Transformer encoder 得到共享场景记忆 $M \in \mathbb{R}^{N_{tok} \times D}$。
5. 从地图里额外提取 reference lines，作为横向 query；再引入可学习 longitudinal queries，作为纵向 query。
6. 将横向 query 和纵向 query 做笛卡尔组合，形成二维规划 query 网格，并通过 factorized attention 与场景记忆交互。
7. 每个 query 输出一条未来轨迹 $\hat{\tau}$ 和一个分数 $\pi$，同时场景记忆还会输出其他 agent 的未来预测。
8. 训练时联合四类损失，推理时再用规则评估器对候选轨迹做二次选择。

### 3.3 维度符号先统一

为了后面不绕，先把常用符号统一一下。PLUTO 论文里有些维度是明确给出的，有些是实现里沿用统一 hidden size。下面这套记号最适合读论文和代码时对齐：

- $T_H$：历史长度，论文实验里是 20。
- $T_F$：未来长度，论文实验里是 80。
- $D$：统一隐藏维度，论文实验里是 128。
- $N_A$：动态 agent 数量。
- $N_S$：静态障碍物数量。
- $N_M$：地图 polyline 数量。
- $N_R$：reference line 数量，也就是横向 query 数量。
- $N_L$：longitudinal query 数量，论文里最优是 12。
- $N_{tok}$：进入场景 encoder 的总 token 数量，通常有

$$
N_{tok} = 1 + N_A + N_S + N_M
$$

这里最前面的 1 一般表示 ego token。

### 3.4 用一组具体数值把全链路代进去

如果你想对“维度大概有多大”有直觉，可以用一组典型配置来代入：

- 历史步数：$T_H = 20$
- 未来步数：$T_F = 80$
- hidden dim：$D = 128$
- longitudinal queries：$N_L = 12$
- covering circles：$N_c = 3$

假设某个场景里：

- 动态 agent 数量 $N_A = 32$
- 静态障碍物数量 $N_S = 20$
- 地图 polyline 数量 $N_M = 128$
- reference line 数量 $N_R = 6$

那么：

- 场景 encoder 输入 token 数量是 $N_{tok} = 1 + 32 + 20 + 128 = 181$
- 场景 memory 维度大致是 $181 \times 128$
- 规划 query 网格维度是 $6 \times 12 \times 128$
- 最终规划候选轨迹维度是 $6 \times 12 \times 80 \times 6$

也就是说，这个场景下模型会同时保留 72 条候选轨迹，每条轨迹有 80 个未来时刻，每个时刻输出 6 个量。

## 4. 输入表示与场景编码

论文使用的是 mid-to-mid 路线，不是直接吃原始相机图像，而是吃感知后的结构化特征。这样做的优点是：

1. 输入更贴近规划本身。
2. 能直接基于真实数据训练，不需要把重点放在 sim-to-real。
3. 模型复杂度可控，更利于分析规划模块本身。

PLUTO 的输入主要包括四类对象。

### 4.1 动态 agent 历史

每个 agent 在每个历史时刻的状态包含：

- 位置
- 朝向
- 速度
- box 尺寸
- 可见性标记

论文不是直接用绝对状态序列，而是把相邻时间步做差，得到更适合建模动态变化的向量表示。最后每个 agent 的历史特征张量大致是：

- $F_A \in \mathbb{R}^{N_A \times (T_H-1) \times 8}$

作者再用一个基于邻域注意力的序列编码器去压缩历史，输出 agent embedding。

把这一步展开，就是：

1. 原始单个 agent 的历史通常先按时间组织成 $T_H$ 个状态。
2. 相邻帧做差后，每个时间步保留 8 维动态变化特征。
3. 所有 agent 叠在一起后，输入是 $N_A \times (T_H-1) \times 8$。
4. 经过历史编码器后，每个 agent 被压成一个长度为 $D$ 的向量。

所以这个模块的输入输出可以写成：

- 输入：$F_A \in \mathbb{R}^{N_A \times (T_H-1) \times 8}$
- 输出：$E_A \in \mathbb{R}^{N_A \times D}$

在论文默认配置下，这一步常见就是：

- 输入：$N_A \times 19 \times 8$
- 输出：$N_A \times 128$

### 4.2 静态障碍物

论文特别强调 planning 和 forecasting 的差异。预测任务里，静态障碍经常被弱化；但规划任务里，锥桶、隔离带、停着的车都直接决定可行驶空间，所以必须单独编码。静态物体用两层 MLP 编码。

这个模块最重要的不是结构有多复杂，而是它把“不可动但会限制可行驶区域”的物体显式塞进场景 token 序列里。它的典型输入输出写法是：

- 输入：$F_S \in \mathbb{R}^{N_S \times d_S}$
- 输出：$E_S \in \mathbb{R}^{N_S \times D}$

这里 $d_S$ 是单个静态物体的原始属性维度，论文没有像动态 agent 那样把每一维完全展开写死，但从建模上理解，一般会包括位置、朝向、尺寸、类别等信息。最终重点是：每个静态物体都被映射成一个 $D$ 维 token。

### 4.3 自车当前状态

这里是 PLUTO 很关键的一个细节。作者没有把自车完整历史都直接喂给 planner，而是强调 imitation learning 容易从 ego history 中学 shortcut，比如简单做状态外推。为此，他们只保留自车当前状态，并引入 state dropout encoder，简称 SDE。这个设计在消融实验里有明显收益。

这一块的输入输出可以理解成：

- 输入：$F_E \in \mathbb{R}^{d_E}$
- 输出：$E_E \in \mathbb{R}^{1 \times D}$

也就是 ego 最终只占一个 token，而不是一整段历史 token。这个设计非常关键，因为它强行限制模型不能只靠“我刚刚朝哪开，下一步继续外推”这种捷径来做规划。

### 4.4 向量化地图

地图用 polyline 表示。每条 polyline 先统一采样点数，再构造点级特征，包括：

1. 当前点相对起点的位置。
2. 当前点相对前一点的位置。
3. 当前点相对左右边界点的位置。

左右边界信息非常重要，因为它等价于把“可驾驶区域形状”显式编码进来。论文用 PointNet-like 编码器把每条 polyline 变成一个 embedding。

如果把这一步写成张量形式，可以记为：

- 输入：$F_M \in \mathbb{R}^{N_M \times P \times d_M}$
- 输出：$E_M \in \mathbb{R}^{N_M \times D}$

这里：

- $N_M$ 是 polyline 条数。
- $P$ 是每条 polyline 统一采样后的点数。
- $d_M$ 是每个采样点的原始特征维度。

你可以把它理解成“每条车道线、多边形边界或道路元素，最终都被压成一个向量 token”，这样场景 encoder 就可以把地图和 agent 放在同一个注意力空间里处理。

### 4.5 全场景融合

把以下 embedding 拼在一起：

1. ego embedding
2. agent embedding
3. static obstacle embedding
4. map polyline embedding

再加上两类附加信息：

1. 基于 $(p, \theta)$ 的 Fourier positional embedding
2. 类型、速度限制、红绿灯等语义属性 embedding

最后送进若干层 Transformer encoder，得到场景级上下文表示。

这个编码器输出是整个 PLUTO 的共享语义底座。后面的预测头、规划头、CIL 投影头都建立在这个 encoder 表征上。

这一层如果完全按张量来写，就是：

1. 拼接前的各类 token：

- ego token：$E_E \in \mathbb{R}^{1 \times D}$
- agent tokens：$E_A \in \mathbb{R}^{N_A \times D}$
- static tokens：$E_S \in \mathbb{R}^{N_S \times D}$
- map tokens：$E_M \in \mathbb{R}^{N_M \times D}$

2. 拼接后：

$$
X \in \mathbb{R}^{N_{tok} \times D}, \quad N_{tok}=1+N_A+N_S+N_M
$$

3. 经过 4 层 Transformer encoder 后：

$$
M \in \mathbb{R}^{N_{tok} \times D}
$$

这一步最核心的意义是，后面所有头部模块都不再直接看原始输入，而是统一从共享场景记忆 $M$ 里取信息。

### 4.6 四个输入模块合在一起怎么看

如果你只想抓住“输入侧到底做了什么”，可以记下面这个最简版本：

```text
动态 agent 历史      [N_A, T_H-1, 8]   -> [N_A, D]
静态障碍物特征       [N_S, d_S]         -> [N_S, D]
ego 当前状态         [d_E]              -> [1, D]
地图 polyline 特征    [N_M, P, d_M]      -> [N_M, D]

拼接后场景序列        [N_tok, D]
Transformer 编码后    [N_tok, D]
```

## 5. 核心创新一：纵向-横向解耦的 query-based 规划

这是论文最核心的架构创新。

### 5.1 为什么普通多模态解码不够

很多 DETR 风格模型会直接用若干 learnable queries 去生成多条候选轨迹。但自动驾驶里的多模态不是任意的，它其实天然可以拆成两维：

1. 横向决策：走哪条 lane、是否变道、是否绕障。
2. 纵向决策：加速、减速、跟车、停车。

如果全部混在一起学，query 很容易没有明确行为语义，最终学到的 mode 既不稳定，也不够丰富。

### 5.2 横向 query：reference lines

PLUTO 先从自车附近车道图中搜索可能的参考线。具体做法是：

1. 找到一定半径内的 lane segment。
2. 从每个 segment 开始做拓扑搜索，连接 lane centerline。
3. 把得到的参考线路径截断并重采样成统一长度。
4. 用和 polyline map 相同思路编码为 embedding。

这些参考线本质上就是横向行为候选，也就是 lateral queries。

直观上理解：

- “留在当前车道”是一类 lateral query。
- “切到左边车道”是另一类 lateral query。
- “绕过障碍走外侧”又是一类 lateral query。

如果把 reference line encoder 也写成输入输出：

- 输入：$R \in \mathbb{R}^{N_R \times P_R \times d_R}$
- 输出：$E_R \in \mathbb{R}^{N_R \times D}$

这里：

- $N_R$ 是参考线条数。
- $P_R$ 是每条参考线采样后的点数。
- $d_R$ 是每个点的几何特征维度。

所以 lateral query 本质上不是一个抽象 id，而是“从真实地图拓扑里提出来的一组候选路线 embedding”。

### 5.3 纵向 query：learnable longitudinal modes

在横向 query 之外，作者再定义若干 learnable longitudinal queries，用来表达不同纵向行为，比如：

- 保守跟车
- 继续加速
- 更积极超车
- 提前刹停

论文实验里，纵向 query 的数量 $N_L = 12$ 最优。太少不够覆盖行为，太多又增加训练难度。

这一块的形状最简单：

- 输入：$N_L$ 个可学习参数向量
- 输出：$E_L \in \mathbb{R}^{N_L \times D}$

在默认配置下就是：

- $E_L \in \mathbb{R}^{12 \times 128}$

### 5.4 横向和纵向 query 组合

组合后形成二维 query 网格：

- $Q_0 \in \mathbb{R}^{N_R \times N_L \times D}$

这里：

- $N_R$ 是 reference line 数量
- $N_L$ 是 longitudinal query 数量

这个设计很巧妙。它把每条候选轨迹都解释成“某条横向路线 + 某种纵向行为”的组合，因此天然就有了更明确的可解释性。

这里你可以把每个格子理解成一个规划假设：

- 第 $r$ 条横向参考线
- 第 $l$ 个纵向驾驶模式

对应一个 query 向量 $Q_0[r,l,:]$。

如果代入论文默认维度：

- $Q_0 \in \mathbb{R}^{N_R \times 12 \times 128}$

假设某场景有 6 条 reference line，那么就是：

- $Q_0 \in \mathbb{R}^{6 \times 12 \times 128}$

也就是 72 个规划候选种子。

### 5.5 因子化 self-attention

如果直接在二维 query 网格上做全连接 self-attention，复杂度会是：

- $\mathcal{O}(N_R^2 N_L^2)$

太贵。于是论文采用 factorized attention：

1. 先沿 reference line 维度做 self-attention。
2. 再沿 longitudinal mode 维度做 self-attention。

复杂度下降为：

- $\mathcal{O}(N_R^2 N_L + N_R N_L^2)$

这一步既保留了全局交互，又避免了组合爆炸。

从张量角度看，这一步前后形状都不变，变化的是“信息交换方式”而不是张量大小：

- 输入：$Q \in \mathbb{R}^{N_R \times N_L \times D}$
- 沿 reference 维 attention 后：$Q' \in \mathbb{R}^{N_R \times N_L \times D}$
- 沿 longitudinal 维 attention 后：$Q'' \in \mathbb{R}^{N_R \times N_L \times D}$

然后它还会和场景 memory 做 cross attention：

- query：$Q'' \in \mathbb{R}^{N_R N_L \times D}$
- key/value：$M \in \mathbb{R}^{N_{tok} \times D}$
- 输出：$Q_{dec} \in \mathbb{R}^{N_R N_L \times D}$

再 reshape 回去，就是：

- $Q_{dec} \in \mathbb{R}^{N_R \times N_L \times D}$

### 5.6 轨迹解码输出什么

最终每个 query 解码出一条未来轨迹和一个分数。每个时间点输出六个通道：

1. $p_x$
2. $p_y$
3. $\cos\theta$
4. $\sin\theta$
5. $v_x$
6. $v_y$

也就是说，PLUTO 不是只回归平面坐标，而是同时表达朝向和速度，这样后续约束和后处理都更方便。

此外，作者还加了一个 reference-line free head，用于停车场等缺少可靠参考线的场景。这一项在消融实验里也带来收益。

把这一块完整写成模块输入输出，就是：

- 规划头输入：$Q_{dec} \in \mathbb{R}^{N_R \times N_L \times D}$
- 位置头输出：$Y_{loc} \in \mathbb{R}^{N_R \times N_L \times T_F \times 2}$
- 朝向头输出：$Y_{yaw} \in \mathbb{R}^{N_R \times N_L \times T_F \times 2}$
- 速度头输出：$Y_{vel} \in \mathbb{R}^{N_R \times N_L \times T_F \times 2}$

拼接后最终规划轨迹输出：

$$
\hat{\tau} \in \mathbb{R}^{N_R \times N_L \times T_F \times 6}
$$

轨迹分数输出：

$$
\pi \in \mathbb{R}^{N_R \times N_L}
$$

在默认实验配置下，未来长度 $T_F=80$，所以一个典型场景中候选轨迹张量是：

- $\hat{\tau} \in \mathbb{R}^{N_R \times 12 \times 80 \times 6}$

如果 $N_R=6$，那就是：

- $\hat{\tau} \in \mathbb{R}^{6 \times 12 \times 80 \times 6}$

这就是为什么说 PLUTO 不是“直接输出一条轨迹”，而是同时输出一整组有行为语义的候选轨迹。

### 5.7 规划模块完整地走一遍

如果把 planning decoder 的流程按顺序拆开，可以写成：

1. 从地图拓扑生成 $N_R$ 条 reference lines。
2. 每条 reference line 编码成一个 lateral query，得到 $E_R \in \mathbb{R}^{N_R \times D}$。
3. 构造 $N_L$ 个 longitudinal query，得到 $E_L \in \mathbb{R}^{N_L \times D}$。
4. 做横向和纵向的组合，得到二维 query 网格 $Q_0 \in \mathbb{R}^{N_R \times N_L \times D}$。
5. 在 query 网格内部先做沿横向维的注意力，再做沿纵向维的注意力。
6. 用场景 memory $M$ 对 query 网格做 cross attention，让每个候选都能看到周围 agent、障碍和地图上下文。
7. 对每个 query 解码出一条 $T_F$ 步的未来轨迹和一个分数。
8. 如果 reference line 不可靠，再走 ref-free head 提供额外候选。

这 8 步本质上就是 PLUTO 最核心的那句设计哲学：先把行为结构拆开，再组合成候选，而不是让模型直接在一个模糊的多模态空间里硬学。

## 6. 核心创新二：高效可微的辅助损失

这是论文第二个很强的点，也是工程上非常有价值的一点。

### 6.1 过去方法的问题

如果想对轨迹加入“别出界”“别撞障碍”这类 loss，一个直观思路是把轨迹 rasterize 到图像上，再和 cost map 做重叠计算。但这样有几个问题：

1. 需要 differentiable rasterizer。
2. 分辨率和时域长度会受显存和算力限制。
3. 对 vector-based planner 不够自然。

### 6.2 PLUTO 的思路

PLUTO 不把轨迹渲染成图，而是反过来：

1. 先把约束转换成可查询的 cost map，比如 drivable area 的 ESDF。
2. 用轨迹点去 cost map 上做连续坐标查询。
3. 用 bilinear interpolation 取出对应代价值。

因为 bilinear interpolation 本身可微，所以整条路径对 cost 的梯度可以直接反传。

### 6.3 covering circles

作者用多个 covering circles 近似车辆形状。每个轨迹点会对应若干圆心，这些圆心投到 cost map 上查询 signed distance。如果距离太小，小于圆半径加安全 margin，就施加惩罚。

损失形式可以概括为：

$$
\mathcal{L}_{aux} = \frac{1}{T_f} \sum_{t=1}^{T_f} \sum_{i=1}^{N_c} \max(0, R_c + \epsilon - d_i^t)
$$

其中：

- $T_f$ 是未来时间步数
- $N_c$ 是 covering circles 数量
- $R_c$ 是圆半径
- $\epsilon$ 是安全阈值
- $d_i^t$ 是插值得到的 signed distance

### 6.4 为什么这个设计重要

因为它让 vector planner 也能方便地引入：

1. off-road penalty
2. collision penalty
3. 其他基于 cost field 的软约束

而且可以 batch-wise 计算，适合现代深度学习框架。这不是一个只服务于 PLUTO 的技巧，放到其他轨迹生成模型里也很有迁移价值。

## 7. 核心创新三：Contrastive Imitation Learning

这是论文第三个主创新。

### 7.1 为什么 imitation learning 需要 contrastive 视角

模仿学习有个天然问题：模型只看到了“专家做了什么”，没看到“如果场景稍微变化，原动作会不会失效”。于是它容易从相关性里找答案，而不是从因果性里找答案。

PLUTO 的思路是：

1. 构造保持原真值仍然有效的 positive augmentations。
2. 构造让原真值失效的 negative augmentations。
3. 让模型在表征空间里靠近正样本、远离负样本。

这相当于告诉模型：

- 哪些扰动不该改变驾驶决策。
- 哪些场景变化会真正改变驾驶决策。

### 7.2 六种数据增强

论文提出六种增强，其中两类是正样本增强，四类是负样本增强。

正样本增强：

1. State Perturbation：轻微扰动自车当前状态，让模型学会 recovery。
2. Non-interactive Agents Dropout：删掉未来不与 ego 交互的 agent，减少 shortcut。

负样本增强：

1. Leading Agents Dropout：移除前车。
2. Leading Agent Insertion：在 ego 路径前方插入前车。
3. Interactive Agent Dropout：移除会和 ego 交互的 agent。
4. Traffic Light Inversion：在合适场景中翻转红绿灯状态。

这几种增强设计得非常有针对性。它们并不是 generic augmentation，而是围绕 driving causality 来构造的。

### 7.3 三元对比损失

每个原始样本 $x$，构造出：

- 正样本 $x^+$
- 负样本 $x^-$

共享 encoder 得到三个 hidden，再用投影头映射到对比空间，形成 $z, z^+, z^-$。然后用 triplet-like softmax contrastive loss：

$$
\mathcal{L}_c = -\log \frac{\exp(sim(z, z^+) / \sigma)}{\exp(sim(z, z^+) / \sigma) + \exp(sim(z, z^-) / \sigma)}
$$

其中 $sim$ 用归一化后的点积。

值得注意的是：

1. 原样本和正增强样本都继续使用 imitation supervision。
2. 负样本只参与 contrastive loss，因为原 GT 对它可能已经无效。

这个处理很合理，否则会把本来“被改坏”的场景也错误拉回原 GT。

## 8. 训练目标与总损失

PLUTO 训练时一共用了四类损失：

1. imitation loss
2. prediction loss
3. auxiliary loss
4. contrastive loss

总损失是：

$$
\mathcal{L} = w_1 \mathcal{L}_i + w_2 \mathcal{L}_p + w_3 \mathcal{L}_{aux} + w_4 \mathcal{L}_c
$$

论文里默认这几个权重都取 1。

### 8.1 imitation loss

这里有两部分：

1. trajectory regression loss
2. trajectory score classification loss

作者使用 teacher forcing。具体做法是：

1. 把 GT 轨迹终点投影到 reference line 上。
2. 找到横向最接近的 reference line。
3. 再根据终点沿 reference line 的投影位置，确定目标 longitudinal query。
4. 从而确定目标 query 对应的 supervision trajectory。

回归用 smooth L1，分类用 cross entropy。这个设计本质上是在给二维 query 网格中的某个格子打标签。

### 8.2 prediction loss

PLUTO 还要预测其他动态 agent 的单模态未来轨迹。这个任务有两个作用：

1. 作为 dense supervision，增强 encoder 表征。
2. 推理时给后处理中的 TTC 和碰撞过滤提供基础。

这个头的张量形式通常可以写成：

- 输入：场景 memory 中对应 agent token 的表示
- 输出：$\hat{Y}_{agent} \in \mathbb{R}^{N_A \times T_F \times 6}$

也就是每个周围 agent 都预测一条未来轨迹，每个时刻同样包含位置、朝向和速度相关量。默认配置下就是：

- $\hat{Y}_{agent} \in \mathbb{R}^{N_A \times 80 \times 6}$

### 8.3 auxiliary loss

主要包括 off-road 和 collision 两类安全约束，是让 planner 真正在闭环里更稳的重要来源。

### 8.4 contrastive loss

主要作用是增强对 causal structure 的辨别能力，以及提升对输入分布偏移的鲁棒性。

如果写成表征空间里的输入输出，大概是：

- encoder hidden：$h, h^+, h^- \in \mathbb{R}^{D}$
- projection 后：$z, z^+, z^- \in \mathbb{R}^{D_p}$

论文更关注对比关系本身，而不是投影维度具体取多少。真正重要的是：anchor、positive、negative 共用同一个场景 encoder，所以 contrastive loss 直接在共享语义空间上施加约束。

## 9. 推理与后处理

PLUTO 的推理不是“直接取概率最大的轨迹就完事”。它走的是 hybrid 路线。

### 9.1 推理步骤

1. 模型生成全部候选轨迹和分数。
2. 先根据学习分数保留 top-k。
3. 对 top-k 轨迹做闭环 forward simulation。
4. 用规则评估器打分，考虑 progress、comfort、traffic rules、TTC 等。
5. 将规则分数和学习分数加权融合：

$$
\pi = \pi_{rule} + \alpha \pi_0
$$

6. 选择总分最高轨迹作为最终输出。

把推理时各步的输入输出也写清楚，可以记成：

1. 模型原始输出：

- 候选轨迹：$\hat{\tau} \in \mathbb{R}^{N_R \times N_L \times T_F \times 6}$
- 候选分数：$\pi \in \mathbb{R}^{N_R \times N_L}$
- 周围车预测：$\hat{Y}_{agent} \in \mathbb{R}^{N_A \times T_F \times 6}$

2. 展平候选后：

- 候选列表大小：$N_R \times N_L$

3. top-k 之后：

- 保留 $K$ 条候选轨迹，张量变成 $K \times T_F \times 6$

4. forward simulation 之后：

- 每条候选得到一条更接近真实可执行结果的 rollout

5. 规则评估后：

- 学习分数：$\pi_0 \in \mathbb{R}^{K}$
- 规则分数：$\pi_{rule} \in \mathbb{R}^{K}$
- 融合分数：$\pi \in \mathbb{R}^{K}$

6. 最终输出：

- 最优自车轨迹：$\tau^* \in \mathbb{R}^{T_F \times 6}$

### 9.2 为什么要 forward simulation

因为神经网络直接输出的轨迹不一定完全符合底层动力学执行结果。论文里用 LQR tracker 和 kinematic bicycle model 对候选轨迹做 rollout，相当于提前模拟“真正执行之后会怎样”。这一步能减少 open-loop 和 actual execution 之间的差异。

### 9.3 为什么不是纯规则，也不是纯学习

这里体现了作者很务实的工程思路。

- 纯规则上限不够，横向行为不灵活。
- 纯学习又还没有强到可以完全放弃安全兜底。

所以 PLUTO 不是让 post-processing 去优化轨迹，而是只做 selection。也就是说，神经网络负责生成候选，规则系统负责兜底和偏好注入。

### 9.4 从输入到最终输出，给你一版最完整的维度总表

如果你要的是“整个模型每一层大概长什么样”，下面这张总表最直接：

| 模块 | 输入 | 输出 | 默认/典型维度 |
| --- | --- | --- | --- |
| Ego encoder | $[d_E]$ | $[1,D]$ | $[d_E] \to [1,128]$ |
| Agent history encoder | $[N_A,T_H-1,8]$ | $[N_A,D]$ | $[N_A,19,8] \to [N_A,128]$ |
| Static encoder | $[N_S,d_S]$ | $[N_S,D]$ | $[N_S,d_S] \to [N_S,128]$ |
| Map encoder | $[N_M,P,d_M]$ | $[N_M,D]$ | $[N_M,P,d_M] \to [N_M,128]$ |
| Scene encoder | $[N_{tok},D]$ | $[N_{tok},D]$ | $[N_{tok},128] \to [N_{tok},128]$ |
| Reference line encoder | $[N_R,P_R,d_R]$ | $[N_R,D]$ | $[N_R,P_R,d_R] \to [N_R,128]$ |
| Longitudinal queries | learnable params | $[N_L,D]$ | $12 \times 128$ |
| Query combination | $[N_R,D] + [N_L,D]$ | $[N_R,N_L,D]$ | $[N_R,12,128]$ |
| Planning decoder | $[N_R,N_L,D]$, $[N_{tok},D]$ | $[N_R,N_L,D]$ | 形状不变 |
| Trajectory heads | $[N_R,N_L,D]$ | $[N_R,N_L,T_F,6]$ | $[N_R,12,80,6]$ |
| Score head | $[N_R,N_L,D]$ | $[N_R,N_L]$ | $[N_R,12]$ |
| Agent predictor | agent memory | $[N_A,T_F,6]$ | $[N_A,80,6]$ |
| Final selector | top-k candidates | $[T_F,6]$ | $[80,6]$ |

这张表里最关键的两点是：

1. 场景 encoder 前后维度不变，变的是每个 token 的上下文语义。
2. 规划头不是输出单条轨迹，而是输出 $N_R \times N_L$ 条候选轨迹。

## 10. 实验设置与结果怎么看

### 10.1 数据与训练配置

论文基于 nuPlan。

- 历史长度：2 秒，对应 20 个历史 step
- 未来长度：8 秒，对应 80 个未来 step
- hidden dim：128
- encoder layers：4
- decoder layers：4
- longitudinal queries：12
- 覆盖圆数量：3
- 优化器：AdamW
- 学习率：前 3 个 epoch warmup 到 1e-3，再 cosine decay
- 训练总 epoch：25
- 4 张 RTX 3090

### 10.2 结果亮点

最重要的结果是：

1. 纯学习版本 Pluto† 已经超过当时纯学习强基线 PlanTF。
2. 加上 post-processing 的 Pluto 超过了 PDM-Closed。
3. 安全指标比如 collision、TTC、drivable area 的提升尤其明显。

这说明论文的进步不是单纯来自“更会模仿”，而是来自：

1. 候选行为空间更合理。
2. 安全约束进入训练。
3. 场景增强和对比学习提升了闭环鲁棒性。

### 10.3 消融实验值得重点看什么

论文的消融链条很干净，大致是：

1. Base
2. + SDE
3. + Auxiliary loss
4. + Reference free head
5. + CIL
6. + Post-processing

这个顺序其实就是作者心目中的系统搭建路径。你可以把它理解为：

- 先解决 ego shortcut。
- 再补安全约束。
- 再补特殊场景覆盖。
- 再补表征鲁棒性。
- 最后加上线选择器。

## 11. 这篇论文真正强在哪里

我认为 PLUTO 的强点不在单一模块，而在“每个模块都刚好对应一个真实短板”。

### 11.1 架构不是为了炫技，而是围绕行为分解

把横向和纵向显式解耦，是非常贴合驾驶任务结构的归纳偏置。相比直接增加 mode 数量，这种设计更稳，也更容易解释。

### 11.2 auxiliary loss 很有工程含金量

很多论文会说“我们也加了碰撞损失”，但真正可用的关键是：

1. 能否 batch 计算。
2. 能否不强依赖 raster。
3. 能否无缝塞进 vector model。

PLUTO 在这方面做得很完整。

### 11.3 CIL 不是泛泛地加 contrastive，而是贴着 driving causality 做增强

这里和很多“顺手加个 contrastive head”的论文不同。PLUTO 的正负样本构造和驾驶常识强绑定，所以不是形式主义，而是确实在帮助 planner 区分什么变化该影响决策，什么不该。

### 11.4 后处理很克制

它不去大改神经网络轨迹，而只做 selection。这种设计很适合工程落地，因为：

1. 容易加安全规则。
2. 容易解释。
3. 不会让下游轨迹优化器和主模型互相打架。

## 12. 这篇论文的局限性

论文最后也诚实承认了一些局限，我补充解释一下。

### 12.1 agent prediction 还是单模态

模型对其他车的预测只有单模态，这在很多复杂交互里并不充分。planning 本身是多模态的，但 surrounding agents 也同样多模态。未来如果把 joint multimodal prediction 做得更好，后处理中的碰撞过滤和 TTC 会更可靠。

### 12.2 post-processing 还是外置模块

虽然 hybrid 策略有效，但它也说明模型本体还没强到“生成即可靠”。如果全部候选都坏，selector 也救不了。未来更理想的方向可能是把安全评价更早地注入生成过程，而不只是最后筛选。

### 12.3 依赖 reference line 的场景先验

尽管加了 ref-free head，但整体设计还是明显依赖 reference lines。对于更开放、更非结构化、弱地图先验的环境，迁移难度会更高。

## 13. 官方代码怎么读

官方仓库是可以对照论文逐段看的。下面给出“论文模块 -> 代码位置 -> 职责”的对应关系。

### 13.1 入口与运行脚本

- `run_training.py`：训练入口。
- `run_simulation.py`：仿真入口。
- `script/run_pluto_planner.sh`：运行 planner demo 的脚本。

README 里给了三类命令：

1. 建 cache
2. 训练
3. 加载 checkpoint 跑 planner simulation

如果只想理解系统链路，先看 README 和仿真脚本就够了；如果想研究模型实现，再往 `src/models/pluto` 里走。

### 13.2 模型主入口

核心文件是：

- `src/models/pluto/pluto_model.py`

这个文件相当于总装厂，主要做几件事：

1. 创建 agent encoder、map encoder、static object encoder。
2. 创建场景 Transformer encoder blocks。
3. 创建 agent predictor。
4. 创建 planning decoder。
5. 可选创建 hidden projection 和 ref-free trajectory head。

它的 forward 逻辑基本对应论文第 III-B 和 III-C 节：

1. 编码 agent / map / static。
2. 拼接成统一 token 序列。
3. 加位置编码后送进 encoder。
4. 输出其他 agent 的 prediction。
5. 如果 reference line 可用，则调用 planning decoder 生成候选轨迹和概率。
6. 推理时再整理成最终输出格式。

### 13.3 场景编码器

相关文件包括：

- `src/models/pluto/modules/agent_encoder.py`
- `src/models/pluto/modules/map_encoder.py`
- `src/models/pluto/modules/static_objects_encoder.py`
- `src/models/pluto/layers/transformer.py`
- `src/models/pluto/layers/embedding.py`

其中值得特别注意的是：

1. `layers/embedding.py` 里的 `NATSequenceEncoder`，说明作者对 agent 历史序列不是简单 MLP，而是用了基于 Neighborhood Attention 的序列编码器。
2. `layers/transformer.py` 则是 encoder 和 decoder 的通用 Transformer layer 实现。

论文里提到的“邻域注意力 FPN 式历史编码”在代码里主要就是通过 `NATSequenceEncoder` 落地。

### 13.4 规划解码器

最关键的文件之一是：

- `src/models/pluto/modules/planning_decoder.py`

这个文件和论文第三个小节几乎一一对应。

它做的事情可以概括为：

1. 编码 reference line，得到 lateral query embedding。
2. 维护 learnable 的 mode embedding 和 mode positional embedding，对应 longitudinal queries。
3. 把 reference embedding 和 mode embedding 拼接后通过 `q_proj` 投到统一空间。
4. 经过多层 decoder block，分别进行：
   - reference-to-reference attention
   - mode-to-mode attention
   - cross attention 到 scene memory
5. 最后通过多个 head 输出：
   - `loc_head`
   - `yaw_head`
   - `vel_head`
   - `pi_head`

也就是说，论文中的 factorized lateral-longitudinal attention，在代码层面就是沿两个维度分开 reshape 后做两次 attention，再做 scene cross attention。

这是官方实现里最值得精读的部分。

### 13.5 其他 agent 预测头

对应文件：

- `src/models/pluto/modules/agent_predictor.py`

这个模块很直接：

1. 分别用 MLP 预测位置、朝向、速度。
2. 然后把它们拼起来形成其他 agent 的未来轨迹。

虽然结构不复杂，但它在系统里作用很大，因为后处理需要用 agent 预测来算 TTC 和碰撞过滤。

### 13.6 训练器与损失函数

对应文件：

- `src/models/pluto/pluto_trainer.py`

这是论文损失设计的直接落地点。建议重点看这几个函数：

1. `_compute_objectives`
2. `get_prediction_loss`
3. `get_planning_loss`
4. `_compute_contrastive_loss`

从仓库内容可以看出：

1. 训练目标里显式包含 planning loss、prediction loss、collision loss、contrastive loss、ref-free reg loss。
2. contrastive loss 是按 batch 中 anchor、positive、negative 三段 chunk 来算的。
3. 是否启用 CIL，通过配置项 `model.use_hidden_proj=true` 和 `custom_trainer.use_contrast_loss=true` 控制。

这一点和论文叙述一致：CIL 并不是另起一个模型，而是在原模型 encoder 表征上挂 projection head 做训练增强。

### 13.7 对比增强实现

对应文件：

- `src/data_augmentation/contrastive_scenario_generator.py`

这个文件几乎就是论文第 III-E 节的代码化版本。建议重点看：

1. `generate_positive_sample`
2. `generate_negative_sample`
3. `neg_leading_agent_insertion`
4. `_generate_agent_from_idx`
5. `safety_check`

从代码可以直接看出：

1. 正增强里会对 current state 采样随机偏移，并做安全检查。
2. 负增强中的 leading agent insertion 会从 mini-batch 中找速度相近的真实 agent 拷贝并插入到 ego 前方路径上。
3. 这个实现思路非常讲究 realism，不是简单伪造障碍，而是尽量从真实轨迹样本中拼接负例。

如果你想真正理解 CIL 的工程细节，这个文件比论文图更重要。

### 13.8 planner 推理与后处理

对应文件：

- `src/planners/pluto_planner.py`

这个文件负责把模型输出变成仿真里的实际 planner。主要逻辑包括：

1. 初始化模型和场景管理器。
2. 从 simulation 中构造 planner feature。
3. 调模型 forward 得到 candidate trajectories、probability、prediction、ref-free trajectory。
4. 做候选筛选。
5. 调 trajectory evaluator 结合规则打分。
6. 结合学习分数和规则分数选最终轨迹。
7. 如果有必要还会触发 emergency brake。

换句话说，如果 `pluto_model.py` 是“学”，`pluto_planner.py` 就是“用”。

### 13.9 你可以怎么按顺序读代码

如果你是第一次读这个仓库，我建议顺序是：

1. README
2. `src/models/pluto/pluto_model.py`
3. `src/models/pluto/modules/planning_decoder.py`
4. `src/models/pluto/pluto_trainer.py`
5. `src/data_augmentation/contrastive_scenario_generator.py`
6. `src/planners/pluto_planner.py`

按这个顺序读，基本能把“模型结构 -> 训练目标 -> 数据增强 -> 推理闭环”完整串起来。

## 14. 论文和代码如何一一对应

下面给一个压缩版映射表。

| 论文概念 | 代码位置 | 作用 |
| --- | --- | --- |
| Scene encoding | `src/models/pluto/pluto_model.py` + `modules/*encoder.py` | 编码 ego、agents、map、static objects |
| Agent history encoder | `src/models/pluto/layers/embedding.py` | NAT/FPN 风格时序编码 |
| Transformer scene fusion | `src/models/pluto/layers/transformer.py` | 融合全场景 token |
| Lateral queries | `modules/planning_decoder.py` 中的 reference line encoder | 提供横向候选路线 |
| Longitudinal queries | `modules/planning_decoder.py` 中的 `m_emb` / `m_pos` | 提供纵向行为模式 |
| Factorized attention | `modules/planning_decoder.py` 中 decoder block 的两阶段 attention | 分别建模 route 和 mode |
| Agent prediction | `modules/agent_predictor.py` | 输出其他车辆未来轨迹 |
| Imitation / prediction / collision / contrastive losses | `pluto_trainer.py` | 训练目标聚合 |
| CIL augmentation | `data_augmentation/contrastive_scenario_generator.py` | 正负样本生成 |
| Hybrid post-processing | `planners/pluto_planner.py` | top-k、仿真、规则评估、最终选择 |

## 15. 如果你想复现，应该先关注什么

如果你的目标是“跑起来”，先关注：

1. nuPlan 环境和数据准备。
2. feature cache。
3. checkpoint 下载。
4. planner simulation 脚本。

如果你的目标是“改模型”，先关注：

1. `pluto_model.py`
2. `planning_decoder.py`
3. `pluto_trainer.py`

如果你的目标是“复用思路到别的 planner”，最值得迁移的部分是：

1. 纵向-横向 query 分解。
2. differentiable interpolation auxiliary loss。
3. 围绕 driving causality 构造的 contrastive augmentation。

## 16. 一句话总结

PLUTO 这篇论文之所以强，不是因为它只在某一个部件上做了改进，而是因为它把“行为空间设计、可微安全约束、闭环鲁棒训练、规则兜底选择”这四件事连成了一个完整系统。

如果你从研究角度看，它最值得学的是：

1. 如何把任务结构变成 query 结构。
2. 如何让 vector planner 也能优雅地吃安全约束。
3. 如何设计真正有因果意义的 contrastive augmentations。

如果你从工程角度看，它最值得学的是：

1. 不盲信纯学习，也不退回纯规则。
2. 在模型输出和安全上线之间放一个轻量但有效的 selector。
3. 每个模块都服务于闭环指标，而不是只服务于 open-loop loss。

这也是为什么 PLUTO 在自动驾驶 planning 方向里，是一篇很值得精读论文和源码配合着看的工作。