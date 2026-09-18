# Actor-Critic 家族：从出动作到更新策略的完整流程

> 配套 demo：`examples/actor_critic_a2c_easy_demo.py`

这篇文档只讲一条主线：**Actor 根据状态出动作，环境给奖励，Critic 评价动作，再用评价结果更新网络**。A2C、PPO、A3C、DDPG、SAC 都沿用这条主线，只是改变了采样方式、评价目标或策略更新方式。

## 先把一轮训练串起来

### 为什么需要两个角色

只训练策略（REINFORCE）可以直接学会选动作，但每次都要等一整局结束才知道动作好坏，奖励噪声很大。只训练价值（DQN）在离散动作上方便，却很难直接处理方向盘角度、关节力矩这类连续动作。Actor-Critic 把两种思路放在一起：Actor 负责做决定，Critic 负责提供及时的评价。

这个评价不是简单地看眼前奖励，而是比较“执行动作后得到的奖励加上未来价值”和“原来对当前状态的估计”。因此 TD error 可以理解成一次反馈的“惊喜值”。

```text
s_t → Actor 采样 a_t → 环境返回 r_t、s_{t+1}、done
    → Critic 评价这一步 → 更新 Critic → 更新 Actor → 进入下一轮
```

Actor 表示策略 $\pi_\theta(a|s)$，Critic 可以表示状态价值 $V_\phi(s)$ 或动作价值 $Q_\phi(s,a)$。基础 Actor-Critic 用 TD error 作为反馈：

$$
\delta_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t)
$$

这里可以把它拆成两部分来理解：

$$
\underbrace{r_t+\gamma(1-d_t)V_\phi(s_{t+1})}_{\text{这一步之后的目标分数}}
-
\underbrace{V_\phi(s_t)}_{\text{Critic 原本对当前状态的预估}}
$$

其中：

- $r_t$ 是当前这一步拿到的即时奖励。
- $\gamma \in [0,1)$ 是折扣因子，表示“未来奖励现在值多少钱”。
- $d_t$ 是终止标记，终止时 $d_t=1$，此时不再使用下一状态价值 bootstrap。
- $V_\phi(s_{t+1})$ 是 Critic 对下一状态未来总回报的估计。

$\delta_t>0$ 表示结果超出预期，增加该动作的概率；$\delta_t<0$ 则降低概率。对应的目标为：

$$
y_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})
$$

$$
L_V=(V_\phi(s_t)-y_t)^2
$$

$$
L_\pi=-\log\pi_\theta(a_t|s_t)\,\delta_t
$$

这三条式子分别对应三件事：

- $y_t$ 是 Critic 想追上的 TD 目标。
- $L_V$ 让 Critic 的估计尽量接近这个目标。
- $L_\pi$ 让 Actor 在 $\delta_t>0$ 时增大当前动作概率，在 $\delta_t<0$ 时减小当前动作概率。

如果把一整条轨迹的真实回报写出来，就是：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots
$$

而 TD 方法的核心，就是不用等完整的 $G_t$ 都出现，先用 $V_\phi(s_{t+1})$ 来近似后半段，这样更新更快、方差更小。

`terminated` 才表示真正终止；`truncated`（时间上限）通常仍需 bootstrap。

### 贯穿示例：机器狗速度控制

设状态包含目标方向、与障碍物距离和当前速度，动作是连续向量
$a_t=[v_x,v_y,\omega_z]$。机器人执行动作后，靠近目标得到正奖励，碰撞得到负奖励，环境返回下一状态。无论使用 PPO 还是 SAC，前半段都完全相同：

```text
观测 [目标方向, 障碍距离, 当前速度]
  → Actor 输出速度动作 [vx, vy, wz]
  → 仿真器推进一步，返回 reward 和 next_obs
  → Critic 判断这次速度是否比预期好
  → 用评价更新网络，再读取 next_obs
```

差别只出现在“如何攒数据、如何算评价、如何更新”：PPO 使用当前并行 rollout 和 clip；SAC 把 transition 放进 Replay Buffer，用双 Critic 和熵项反复学习。

## 1. 基础 Actor-Critic：单步闭环

初始化随机 Actor、Critic 后，每一步都执行“采样→评价→更新”。下面是帮助理解数据流的伪代码，不是可直接运行的 Python；省略了张量形状、设备和优化器细节：

```text
动作分布 ← Actor(state)
action ← 从动作分布中采样
next_state, reward, terminated ← 环境执行(action)

value ← Critic(state)
next_value ← Critic(next_state)（只作为目标，不反传梯度）
target ← reward + gamma × (1 - terminated) × next_value
td_error ← target - value

Critic_loss ← (value - target)^2
Actor_loss ← -log(Actor 对 action 的概率) × td_error
用 Critic_loss 更新 Critic
用 Actor_loss 更新 Actor
```

先更新 Critic 使价值估计更准，再更新 Actor 提高好动作的概率。实现时常见的 `detach()` 表示把 Critic 的评价当作固定反馈，防止 Actor 的损失反过来改变 Critic。

## 2. A2C：用 advantage 批量、同步更新

A2C 把“动作好不好”改成“这个动作比当前状态的平均水平好多少”。这里的平均水平就是 Critic 的 $V(s_t)$：例如向右让机器人更接近目标，向左却远离目标；减去同一个基准线后，Actor 更容易比较两种动作，更新方差也更小。

把它想成教练带一组学员练习：学员先各自走一小段路，教练不在每一步都打断，而是等大家走完几步后统一点评。“这位学员这次表现比他平时水平好多少”就是 advantage。这样一次看一批样本，偶然的好坏不会马上决定策略，训练更平稳。

A2C（Advantage Actor-Critic）常用 $A_t=R_t-V(s_t)$ 表示相对优势。经典写法里可以用 $n$ 步 return，也可以像很多现代实现那样使用 GAE；下文采用的是工程里很常见的 GAE 版本。它先收集一批数据再更新，因此流程是连续的：

更细一点写，A2C 里常用的几组量分别是：

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)
$$

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)}=\delta_t+\gamma\lambda(1-d_t)A_{t+1}^{\mathrm{GAE}}
$$

把递推式继续展开，就得到更直观的形式：

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\delta_t
+\gamma\lambda(1-d_t)\delta_{t+1}
+(\gamma\lambda)^2(1-d_t)(1-d_{t+1})\delta_{t+2}
+\cdots
$$

最后再用 advantage 还原 return：

$$
R_t=A_t+V(s_t)
$$

这组公式的直觉是：

- $\delta_t$ 看的是“当前这一步有没有比预期更好”。
- $A_t$ 把后续几步的 TD 误差也折回来，得到更平滑的优势估计。
- $R_t$ 则是给 Critic 学习用的回报目标。

把这 5 步展开来看：

1. **先收集，不急着改参数**：每个环境按当前 Actor 行动，把沿途的状态、动作和奖励记下来。此时所有样本都来自同一个“旧版本”策略，方便公平比较。
2. **给最后一个状态估个分**：如果走了 $T$ 步还没到终点，就用 $V(s_T)$ 猜后面还能拿多少分；如果真的结束，后面价值就是 0。
3. **从后往前回传评价**：越靠后的动作越容易判断，先算最后一步的 TD 误差，再逐步向前累计，得到 $A_t$；随后再由 $R_t=A_t+V(s_t)$ 还原出给 Critic 用的 return。
4. **统一批改作业**：把 advantage 做标准化，避免某一批奖励特别大导致参数抖动，然后一次性更新 Actor 和 Critic。
5. **扔掉这批数据再来一轮**：Actor 已经变了，旧数据不再完全代表新策略，所以 A2C 通常不会反复使用它。

```text
N 个环境同步 rollout T 步
  → 保存 state/action/reward/done/log_prob/value
  → 用 V(s_T) bootstrap
  → 从后往前计算 GAE：A_t=δ_t+γλ(1-d_t)A_{t+1}
  → 标准化 advantage，得到 return=A+V
  → 更新 Actor + Critic，丢弃这批 on-policy 数据
```

下面的代码仍是伪代码，重点是变量之间的关系，而不是 Python 语法：

```text
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
dist = actor(states)
policy_loss = -(dist.log_prob(actions) * adv.detach()).mean()
value_loss = 0.5 * (critic(states) - returns.detach()).pow(2).mean()
loss = policy_loss + value_coef * value_loss - entropy_coef * dist.entropy().mean()
更新 Actor 和 Critic（反向传播 + 优化器）
```

`λ=0` 接近单步 TD，`λ=1` 接近完整回报，常用 `0.95`。A2C 的 rollout 通常只使用一次。

更精确地说：

- 当 $\lambda=0$ 时，$A_t$ 基本只看当前一步的 $\delta_t$，偏差可能更大，但方差较小。
- 当 $\lambda\to1$ 时，$A_t$ 会吸收更长时间范围的信息，更接近 Monte Carlo 回报，偏差更小，但方差更大。

所以 GAE 实际上在做一个偏差和方差之间的折中。

## 3. PPO：A2C 流程上加“安全带”

普通策略梯度可能因为一次偶然的“大惊喜”把动作概率从 0.2 直接推到 0.9，随后状态分布改变，Critic 也会失真。PPO 不禁止更新，只限制新旧策略的变化幅度。

可以把 PPO 想成教练给学员改动作：学员这次做得好，当然要鼓励，但不能因为一次发挥好就要求他以后百分之百照这个动作做。PPO 会先记住“练习当时这个动作有多大概率”，更新时比较新旧概率；差距太大就只按允许的最大幅度计算奖励。

A2C 的批量流程不变。PPO 采样时保存 `old_log_prob`，更新时比较新旧策略。

把 PPO 的训练流程直接写成“每一步对应一个公式”，可以写成：

1. 先固定旧策略采样，得到当前 rollout：

$$
\mathcal D=\{(s_t,a_t,r_t,s_{t+1},d_t,\log\pi_{old}(a_t|s_t),V_\phi(s_t))\}_{t=1}^T
$$

这里 $\mathcal D$ 表示当前这一轮收集到的数据；除了状态、动作、奖励外，还要保存旧策略概率和价值估计，供后续更新使用。

2. 再用 rollout 计算 advantage 和 return：

$$
\delta_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t)
$$

$$
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1},\qquad R_t=A_t+V_\phi(s_t)
$$

这一步先算 TD 误差 $\delta_t$，再递推得到 advantage $A_t$，最后还原出给 Critic 学习的回报目标 $R_t$。

如果这些符号看着太抽象，可以先只记住下面这张“小抄”：

- $r_t$：这一时刻环境真正给你的即时奖励。
- $V_\phi(s_t)$：Critic 原来估计“当前状态大概值多少分”。
- $\delta_t$：现实结果和 Critic 预估之间的差，也就是“这一小步的惊喜值”。
- $A_t$：这个动作最终到底比平均水平好多少。
- $R_t$：给 Critic 当监督答案的回报目标。

你完全可以把这一步读成：

```text
先看这一步实际拿了多少奖励
  → 再看 Critic 原来估得准不准
  → 两者的差变成 delta
  → 再把后面几步的影响也折回来，得到 advantage
  → 最后给 Critic 准备一个要去拟合的 return
```

举个单样本数字例子。假设某一步：

- 当前奖励 $r_t=1.0$
- 折扣因子 $\gamma=0.99$
- 终止标记 $d_t=0$
- Critic 估计当前状态价值 $V(s_t)=2.0$
- Critic 估计下一状态价值 $V(s_{t+1})=2.5$

那么

$$
\delta_t=1.0+0.99\times2.5-2.0=1.475
$$

这表示：**这一步的真实结果，比 Critic 原先预期的还要好 1.475 分。**

如果后面几步累计下来，最后算出的 $A_t$ 仍然是正的，比如 $A_t=1.2$，那 PPO 就会认为：这个动作值得鼓励。

3. 用当前策略重新评估旧动作的概率变化：

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$$

这个比值衡量的是：同一个历史动作 $a_t$，当前策略相对旧策略到底更想做，还是更不想做。

- $r_t(\theta)>1$ 表示当前策略提高了该动作概率。
- $r_t(\theta)<1$ 表示当前策略降低了该动作概率。
- $r_t(\theta)=1$ 表示该动作概率没有变化。

这里最容易混淆的是：上面的 $r_t(\theta)$ 和前面的奖励 $r_t$ 不是一回事。

- 前面的 $r_t$ 是 reward，表示环境给了多少分。
- 这里的 $r_t(\theta)$ 是 ratio，表示新旧策略概率的比值。

为了避免脑子打结，你可以强行把它读成：

$$
	ext{ratio} = \frac{\text{新策略对这个动作的概率}}{\text{旧策略对这个动作的概率}}
$$

如果旧策略对“向右”这个动作的概率是 0.20，新策略现在变成了 0.24，那么：

$$
r_t(\theta)=\frac{0.24}{0.20}=1.2
$$

意思就是：**新策略比旧策略更愿意做这个动作了，而且多愿意了 20%。**

$$
L_t^{CLIP}(\theta)=\min\Big(r_t(\theta)A_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\Big)
$$

这条式子的两项含义分别是：

- $r_t(\theta)A_t$：不加限制时，普通策略梯度希望优化的方向。
- $\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t$：人为给比例上限和下限后的保守版本。

取两者较小值，就是为了防止策略朝同一个方向走得太猛。

这条式子如果直接看不顺，可以拆成两种情况：

### 情况 1：这个动作是好动作，$A_t>0$

这时 PPO 本来想把这个动作概率调大。

假设：

- $A_t=2$
- 旧概率是 0.20
- 新概率想调到 0.30
- 那么 $r_t(\theta)=0.30/0.20=1.5$
- 设 $\epsilon=0.2$

不加限制时：

$$
r_t(\theta)A_t=1.5\times2=3.0
$$

但 clip 后，比例最多只能按 $1.2$ 算：

$$
\operatorname{clip}(1.5,0.8,1.2)\times2=1.2\times2=2.4
$$

于是 PPO 取较小值 2.4，而不是 3.0。意思就是：

**这个动作确实好，可以鼓励，但别一下子鼓励过头。**

### 情况 2：这个动作是坏动作，$A_t<0$

这时 PPO 本来想把这个动作概率调小。

假设：

- $A_t=-2$
- 旧概率是 0.20
- 新概率想降到 0.10
- 那么 $r_t(\theta)=0.10/0.20=0.5$

不加限制时：

$$
r_t(\theta)A_t=0.5\times(-2)=-1.0
$$

clip 后，下限只能按 $0.8$ 算：

$$
\operatorname{clip}(0.5,0.8,1.2)\times(-2)=0.8\times(-2)=-1.6
$$

PPO 仍然取两者较小值，也就是更保守地限制这次更新幅度。直觉上就是：

**这个动作该压，但也别一口气把它概率砍得太狠。**

4. 用 clip 后的代理目标更新 Actor：

训练时真正最大化的一般是整批样本的期望：

$$
\mathcal{L}^{CLIP}(\theta)=\mathbb{E}_t\left[L_t^{CLIP}(\theta)\right]
$$

5. 同时更新 Critic，并保留熵正则：

$$
L^{PPO}=L_{policy}+c_vL_{value}-c_e\,\mathbb E_t\left[\mathcal H\big(\pi_\theta(\cdot|s_t)\big)\right]
$$

其中

$$
L_{policy}=-\mathbb E_t\left[L_t^{CLIP}(\theta)\right],\qquad
L_{value}=\mathbb E_t\left[(V_\phi(s_t)-R_t)^2\right]
$$

这表示：

- $L_{policy}$ 负责更新策略。
- $L_{value}$ 负责让 Critic 逼近回报目标 $R_t$。
- 熵项 $\mathcal H$ 用来防止动作分布过早塌缩。
- $c_v$ 和 $c_e$ 是控制价值损失与熵奖励权重的超参数。

如果把这一堆损失再翻回人话，其实就是三个老师同时给分：

- 第一个老师看策略：好动作概率有没有往上提，坏动作概率有没有往下降，而且别改太猛。
- 第二个老师看价值：Critic 对 return 的估计准不准。
- 第三个老师看探索：动作分布是不是过早变得太死板。

所以 PPO 不是只做一件事，而是在同时平衡三件事：

```text
策略要变好
  → 但别变得太猛
  → Critic 还得估得更准
  → 同时保留一点探索
```

这样一来，PPO 的每一步都可以用公式读成：采样旧数据 $\rightarrow$ 算优势 $\rightarrow$ 比较新旧概率 $\rightarrow$ clip 更新策略 $\rightarrow$ 同步更新价值网络。

这条式子最好分正负 advantage 两种情况看：

- 如果 $A_t>0$，说明这个动作比预期好，希望把它概率变大，但最多放大到 $1+\epsilon$ 附近。
- 如果 $A_t<0$，说明这个动作偏差，希望把它概率变小，但最多缩小到 $1-\epsilon$ 附近。

因此 clip 的作用不是“让策略不学习”，而是“别一次学过头”。

$A_t>0$ 时鼓励动作但限制增幅，$A_t<0$ 时抑制动作但限制降幅。PPO 只能有限重复使用当前 rollout，不能长期回放任意旧数据。

因此 PPO 一轮训练可以这样读：先让机器人按旧策略采样，给每条经验贴上“原来动作概率”和“好坏分数”；然后用这批经验训练几遍，每一遍都检查新策略有没有偏离太远。训练完就把这批经验作废，重新让当前策略去环境中采样。

如果你把 PPO 想成真正会运行的训练循环，可以按下面这 6 句话理解：

1. **先用当前策略出去采样**：机器人真的去环境里走一段，记下状态、动作、奖励，以及“当时这个动作的旧概率”。
2. **先别急着改参数**：这一整批数据要先固定住，因为 PPO 的核心就是“拿新策略去对比旧策略”。
3. **回头给每一步打分**：用 GAE 算出每一步动作到底比预期好多少，也就是 advantage。
4. **开始反复复习这一批数据**：不是只学一遍，而是切成 minibatch 多学几个 epoch。
5. **每次更新时都问一句**：新策略对这个旧动作的概率，和旧策略相比变了多少？如果变太猛，就 clip 掉。
6. **这批数据用完就扔**：因为一旦策略变了，旧 rollout 就不再是“当前策略真实采到的数据”了。

把它浓缩成伪代码式的人话就是：

```text
旧策略先去采样一批数据
  → 回头算每一步 advantage
  → 用这批旧数据训练当前策略几遍
  → 训练时始终拿“当前概率 / 旧概率”做对照
  → 如果变化太大就踩刹车
  → 这批数据作废，再出去采下一批
```

PPO 最容易卡住的点只有一个：**它不是“边走边学”，而是“先完整收集一批旧数据，再拿这批旧数据小心地更新当前策略”。** 这就是它和最基础 Actor-Critic 的最大流程差别。

举个数字例子：旧策略在某状态下以 20% 概率向右，向右的 advantage 为正。普通更新可能把概率直接改到 80%；PPO 设定 $ε=0.2$ 后，允许的比例大约只到 1.2 倍，超过部分不再继续奖励。于是策略会变好，但不会被一次样本带偏。

## 4. A3C：把 A2C 的采样改成异步 worker

单个环境产生的连续轨迹高度相关，例如机器人会连续多步向同一个方向走。A3C 让不同 worker 在不同环境或随机种子下同时探索，牺牲一点参数同步性，换取更丰富的数据和更高的 CPU 采样速度。

它像开了很多间练习室：每间房里都有一个“分身教练”和一台机器人。每个分身练几步就把心得写回总教练，其他房间不用等它。总教练因此能更快收到各种场景的反馈，但某个分身拿到的教材可能已经旧了一点，这就是 stale gradient。

A3C 的主干和 A2C 很接近，但经典 A3C 更常见的是 $n$ 步 return，而不是必须使用 GAE。这里把它按更标准的 $n$ 步写法展开；真正的区别在于多个 worker 何时更新全局网络：

把 A3C 的局部更新直接写成“每一步对应一个公式”，可以写成：

1. 每个 worker 先从全局网络拷贝本地参数：

$$
{}\theta_i\leftarrow\theta,\qquad \phi_i\leftarrow\phi
$$

这里 $\theta,\phi$ 是全局 Actor 和 Critic 参数，$\theta_i,\phi_i$ 是第 $i$ 个 worker 的本地副本。

2. 本地 worker 独立和环境交互，得到一段长度为 $n$ 的轨迹：

$$
{}\tau_i=\{(s_t,a_t,r_t,\dots,s_{t+n})\}
$$

不同 worker 各自收集轨迹，这就是 A3C 降低样本相关性的来源。

3. 对这段轨迹计算 $n$ 步回报：

$$
R_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^nV(s_{t+n})
$$

如果在第 $t+n$ 步之前 episode 已经终止，那么最后一项 bootstrap 价值直接记为 0。

4. 用 $n$ 步回报计算 advantage：

$$
A_t=R_t^{(n)}-V(s_t)
$$

5. 构造本地策略损失和价值损失：

$$
L_\pi=-\log\pi_\theta(a_t|s_t)A_t
$$

$$
L_V=\big(V_\phi(s_t)-R_t^{(n)}\big)^2
$$

这几条式子的意思很直接：

- $R_t^{(n)}$ 先累计接下来 $n$ 步的真实奖励，再拼接一个末端状态价值。
- $A_t$ 判断当前动作相对状态基线 $V(s_t)$ 是否更好。
- $L_\pi$ 用 advantage 更新 Actor，$L_V$ 用 $n$ 步回报更新 Critic。

6. 本地算出梯度后，直接异步更新全局参数：

$$
{}\theta\leftarrow\theta-\eta\nabla_\theta L_\pi,\qquad
\phi\leftarrow\phi-\eta\nabla_\phi L_V
$$

这一步的关键就是不等其他 worker，同步障碍更少，但梯度可能稍微陈旧。

因此 A3C 的“公式主干”并没有脱离 Actor-Critic，只是把这些局部回报和梯度分散到多个 worker 上异步计算。

异步会产生稍旧的参数（stale gradient），换来更低的轨迹相关性和更高的 CPU 采样吞吐；A2C 则等所有环境采完再同步更新。

具体地说，Worker 1 可能正在学“遇到障碍向左绕”，Worker 2 正在学“开阔地加速”。谁先算完就先把梯度交给全局网络；下一个 worker 再读取更新后的参数继续走。这样不用等最慢的环境，但不同 worker 的经验可能不是基于完全相同的参数。

## 5. DDPG：Replay Buffer 加确定性连续动作

例如机械臂动作是实数角度而不是“左/右”。Actor 可以直接输出 0.35 的力矩，Critic 评价这个具体动作的长期价值。Replay Buffer 让经验被重复学习，目标网络慢速更新则避免学习目标剧烈漂移。

可以把 Replay Buffer 当成“训练录像库”：机器人每走一步就把“当时看到什么、做了什么、得到多少分、后来到了哪里”录下来。训练时随机抽旧录像复习，而不是只能依赖刚刚发生的那一段。由于 Actor 是确定性的，它像一个只给单一答案的控制器，所以还要在训练动作上故意加一点噪声，避免永远走同一条路。

连续动作无法枚举所有动作概率，DDPG 让 Actor 直接输出 $a=\mu_\theta(s)$，并维护在线/目标两套网络：

把 DDPG 的一轮更新直接写成“每一步对应一个公式”，可以写成：

1. 在线 Actor 输出动作，并叠加探索噪声：

$$
a_t=\mu_\theta(s_t)+\varepsilon_t
$$

这里 $\mu_\theta(s_t)$ 是确定性策略输出，$\varepsilon_t$ 是探索噪声；没有这项噪声，DDPG 几乎不会主动探索。

2. 把交互得到的 transition 放入 Replay Buffer：

$$
\mathcal D\leftarrow \mathcal D\cup\{(s_t,a_t,r_t,s_{t+1},d_t)\}
$$

这一步的作用是积累可重复使用的离策略样本。

3. 用 target Actor 计算下一状态动作：

$$
a_{t+1}'=\mu_{\theta'}(s_{t+1})
$$

这里用的是 target Actor，而不是当前在线 Actor，因为 target 网络更稳定。

4. 用 target Critic 构造 Bellman 目标：

$$
y_t=r_t+\gamma(1-d_t)Q_{\phi'}\big(s_{t+1},\mu_{\theta'}(s_{t+1})\big)
$$

5. 用均方 Bellman 误差更新 Critic：

$$
L_Q=\mathbb{E}_{(s_t,a_t,r_t,s_{t+1})\sim\mathcal{D}}\left[\big(Q_\phi(s_t,a_t)-y_t\big)^2\right]
$$

这个损失的意思是让 Critic 对 $(s_t,a_t)$ 的估计尽量逼近目标值 $y_t$。

6. 固定 Critic，更新 Actor：

$$
L_\pi=-\mathbb{E}_{s_t\sim\mathcal{D}}\left[Q_\phi\big(s_t,\mu_\theta(s_t)\big)\right]
$$

负号表示最小化损失等价于最大化 Q 值，也就是让 Actor 输出更高价值的动作。

这里的关键点是：

- $\mathcal{D}$ 表示 Replay Buffer 中的数据分布。
- 目标动作由 target Actor $\mu_{\theta'}$ 产生，而不是当前在线 Actor。
- 目标 Q 值由 target Critic $Q_{\phi'}$ 给出，因此目标更稳定。

7. 最后软更新目标网络：

$$
{}\theta'\leftarrow \tau\theta+(1-\tau)\theta',\qquad
\phi'\leftarrow \tau\phi+(1-\tau)\phi'
$$

这里的 $\tau$ 通常很小，因此 target 网络只会缓慢追随在线网络，避免训练目标剧烈摆动。

从优化角度看：

- $L_Q$ 让 Critic 学会满足 Bellman 方程，判断动作到底值多少分。
- $L_\pi$ 前面的负号表示，最小化损失等价于最大化 $Q_\phi(s,\mu_\theta(s))$，也就是让 Actor 输出更高价值的动作。

$\tau$ 往往很小，例如 $0.005$，表示目标网络只缓慢跟随在线网络。

Actor 本身不随机，训练动作必须加入高斯或 OU 噪声。DDPG 对 Critic 高估和噪声较敏感，实际项目常优先使用 TD3 或 SAC。

整轮更新的直觉是：先用目标网络估算“这一步之后最多还能拿多少分”，把它当成标准答案；Critic 对照标准答案改进，Actor 再根据 Critic 的方向微调动作。最后只把目标网络向在线网络靠近一点点，下一轮的标准答案就不会突然跳变。

如果把 DDPG 写成最容易记住的“代码主循环”，其实就是下面这 7 步：

1. **先让 Actor 出一个连续动作**：例如输出一个力矩或速度。
2. **为了探索，给动作加一点噪声**：否则 Actor 每次都出几乎一样的值，很难学到新东西。
3. **把这一步经验存进 Replay Buffer**：先存下来，不急着立刻用它更新。
4. **从 Buffer 随机抽一批旧经验**：注意这里抽到的往往不是刚才那一步，而是过去任意时刻的录像。
5. **先更新 Critic**：用 target 网络算标准答案 $y_t$，让 Critic 学会“这条经验到底值多少分”。
6. **再更新 Actor**：让 Actor 输出的动作朝“Critic 觉得更高分”的方向移动。
7. **最后慢慢更新 target 网络**：不要让标准答案跟着在线网络一起剧烈跳动。

把它翻成一句特别口语的话：

```text
先把经验攒进录像库
  → 随机抽旧录像做训练
  → 先让 Critic 学会打分
  → 再让 Actor 专门讨好 Critic
  → target 网络慢慢跟，不要一起乱跳
```

DDPG 最容易卡住的点有两个：

- **为什么先更新 Critic？** 因为 Actor 要靠 Critic 指方向，Critic 分都打不准，Actor 就会被带偏。
- **为什么要 target 网络？** 因为你在学的同时，标准答案也在变；如果不用 target 网络，训练目标会像“边考试边改答案”，很容易不稳定。

例如录像中记录“距离目标 2 米时输出 0.35 力矩，得到 +0.4 分”。训练时先问目标网络：到达下一状态后预计还能得 1.0 分，于是这条经验的标准答案约为 $0.4+\gamma\times1.0$。Critic 调整自己对这条经验的估计，Actor 则尝试把力矩往能让 Q 值更高的方向移动。

## 6. SAC：DDPG 流程上加入随机性和双 Critic

DDPG 找到一个看似不错的动作后可能停止探索。SAC 把“保持选择多样性”也计入价值，在证据不足时继续尝试其他动作；两个 Critic 取较小值，则能降低单个 Critic 过度乐观的风险。

想象两位教练分别给动作打分：SAC 不盲信分数更高的那位，而是取两人中较保守的分数，避免“自我感觉太好”。同时，Actor 不只追求高分，还会因为尝试不同动作得到一点“探索奖励”（熵）。如果策略变得过于死板，温度参数 $\alpha$ 会提高，提醒它多试试；探索太多时，$\alpha$ 会降低，让它更专注于高价值动作。

SAC 仍从 Replay Buffer 采样，但 Actor 输出随机分布，两个 Critic 取较小值，并把熵计入目标：

把 SAC 的一轮更新直接写成“每一步对应一个公式”，可以写成：

1. 先从随机策略中采样动作：

$$
a_t\sim\pi_\theta(\cdot|s_t)
$$

这里动作本身就来自分布采样，而不是像 DDPG 那样先输出确定性动作再额外加噪声。

2. 把采样得到的 transition 写入 Replay Buffer：

$$
\mathcal D\leftarrow \mathcal D\cup\{(s_t,a_t,r_t,s_{t+1},d_t)\}
$$

因此 SAC 也是 off-policy，可以反复利用旧数据。

3. 在下一状态重新采样动作，并计算其对数概率：

$$
a_{t+1}\sim\pi_\theta(\cdot|s_{t+1}),\qquad
\log\pi_\theta(a_{t+1}|s_{t+1})
$$

这个对数概率后面会直接进入目标函数，用来衡量探索收益。

4. 构造 soft Bellman 目标：

这条流程中最容易迷糊的是 `log_prob`：它表示“Actor 认为自己刚才这个动作有多常见”。动作越少见，熵奖励越大，SAC 就越愿意保留探索；动作虽然得分高但过于单一时，熵项会提醒 Actor 不要立刻把其他可能性全部删掉。等训练充分后，Q 值的作用逐渐占主导，策略才会稳定下来。

SAC 的 Critic 目标通常写成：

$$
y_t=r_t+\gamma(1-d_t)\left[\min_{j\in\{1,2\}}Q_{\phi'_j}(s_{t+1},a_{t+1})-\alpha\log\pi_\theta(a_{t+1}|s_{t+1})\right]
$$

其中 $a_{t+1}\sim\pi_\theta(\cdot|s_{t+1})$，也就是下一步动作是从当前随机策略里采样出来的。

这里多出来的 $-\alpha\log\pi_\theta(a_{t+1}|s_{t+1})$ 是 SAC 的核心。它表示：目标里不只看未来奖励，还把“保持一定随机性”的收益也算进去。因此 SAC 学到的不是普通 Q 值，而是带熵正则的软 Q 值。

把这种思想写成整体目标，就是：

$$
J(\pi)=\mathbb E\left[\sum_{t=0}^{\infty}\gamma^t\big(r(s_t,a_t)+\alpha\mathcal H(\pi(\cdot|s_t))\big)\right]
$$

它表示 SAC 最大化的是“累计奖励 + 累计熵收益”。

5. 分别更新两个 Critic：

$$
L_Q^{(j)}=\mathbb E_{(s_t,a_t,r_t,s_{t+1})\sim\mathcal D}\left[\big(Q_{\phi_j}(s_t,a_t)-y_t\big)^2\right],\qquad j\in\{1,2\}
$$

两个 Critic 都去拟合同一个软目标，但后面取较小值参与 Actor 更新，以抑制高估。

6. 更新随机 Actor：

Actor 的优化目标是：

$$
L_\pi=\mathbb E_{s_t\sim\mathcal D,\, a_t\sim\pi_\theta}\left[\alpha\log\pi_\theta(a_t|s_t)-\min_jQ_{\phi_j}(s_t,a_t)\right]
$$

这条式子可以拆成两股力量：

- $-\min_jQ_{\phi_j}(s_t,a_t)$ 鼓励选择高价值动作。
- $\alpha\log\pi_\theta(a_t|s_t)$ 鼓励保留一定随机性，避免策略太早塌缩成单一动作。

7. 如果启用自动温度调节，再最小化温度损失：

$$
L_\alpha=\mathbb E_{a_t\sim\pi_\theta}\left[-\alpha\big(\log\pi_\theta(a_t|s_t)+\mathcal H_{target}\big)\right]
$$

它的作用是把策略熵推向目标熵 $\mathcal H_{target}$：熵太低就增大 $\alpha$，熵太高就减小 $\alpha$。不少实现会把可学习参数改成 $\log\alpha$ 来优化，损失写法会有等价变形，但优化目标的含义不变。

8. 最后软更新两个 target Critic：

$$
\phi_1'\leftarrow \tau\phi_1+(1-\tau)\phi_1',\qquad
\phi_2'\leftarrow \tau\phi_2+(1-\tau)\phi_2'
$$

这一步保证用于构造目标值的 target Critic 变化更平滑。

把 SAC 的三组目标连起来看：

- Critic 通过 $y_t$ 学习软 Q 值。
- Actor 通过 $L_\pi$ 在“价值高”和“保持探索”之间找平衡。
- 温度参数通过 $L_\alpha$ 自动决定探索到底应该占多大权重。

高熵意味着保留更多探索；熵过低时自动增大 $\alpha$，熵过高时减小 $\alpha$，因此 SAC 通常比 DDPG 更稳。

所以 SAC 的一轮可以用一句话概括：从录像库抽样，先问两位 Critic“保守估计能得多少分”，再让随机 Actor 在“得分高”和“别太早固化”之间取平衡，最后慢慢同步目标 Critic。

如果你已经懂了 DDPG，再看 SAC 可以直接把它当成“DDPG 的三处升级版”：

1. **Actor 不再输出唯一动作，而是输出一个概率分布**。
2. **Critic 不再只用一个，而是同时用两个，取更保守的那个**。
3. **优化目标里不只追求高 Q 值，还额外奖励探索**。

把 SAC 的训练循环按程序顺序翻成人话，就是：

1. **先按随机策略和环境交互**：动作不是固定值，而是从策略分布里采样出来的。
2. **把经验存进 Replay Buffer**：这一点和 DDPG 一样，都是 off-policy。
3. **从 Buffer 抽一批数据**：拿旧录像反复学。
4. **先更新两个 Critic**：但构造目标时取两者中较小的 Q 值，防止其中一个过度乐观。
5. **再更新 Actor**：既想让 Critic 打高分，又不能让策略过早变得只会一种动作。
6. **如果开了自动温度调节，再更新 $\alpha$**：决定“探索这件事值多少钱”。
7. **最后软更新两个 target Critic**：让目标保持平滑。

可以把它压缩成下面这条记忆链：

```text
随机 Actor 先采样动作
  → 经验进 Buffer
  → 双 Critic 先学“保守打分”
  → Actor 再学“高分 + 保持随机”
  → alpha 自动调探索强度
```

SAC 最容易卡住的点是：**为什么损失里会有 $\alpha\log\pi$ 这一项？**

直觉上，它不是在“惩罚动作概率”本身，而是在说：

- 如果策略已经太死板，只会一种动作，那探索不够，应该把随机性拉回来一点。
- 如果策略已经很发散，到处乱试，那就让随机性降下来，把注意力更多放到高价值动作上。

所以 SAC 不是单纯追求“分最高”，而是在追求“分高，同时别太早把其他可能性全删掉”。

假设两个 Critic 对同一个动作分别打 0.8 分和 1.1 分，SAC 先采用较保守的 0.8 分；如果 Actor 这次动作概率很低，说明它是在尝试新动作，熵项会抵消一部分惩罚。训练初期因此敢于探索，训练后期才逐渐集中到真正可靠的动作上。

## 一张表定位差异

| 算法 | 主流程中的关键改动 | Replay Buffer | Actor 输出 |
|---|---|---|---|
| Actor-Critic | 单步 TD 闭环 | 否 | 随机策略 |
| A2C | 多环境同步 rollout + advantage | 否 | 随机策略 |
| PPO | 旧概率 + ratio/clip，多 epoch | 仅当前 rollout | 随机策略 |
| A3C | 多 worker 异步应用梯度 | 否 | 随机策略 |
| DDPG | 回放 + 确定性 Actor + Q 梯度 | 是 | 确定性连续动作 |
| SAC | 双 Critic + 随机 Actor + 熵正则 | 是 | 随机连续动作 |

### 读流程图时的词汇对照

- **rollout**：让当前策略真的去环境里走几步，并把经历记下来。
- **bootstrap**：后面还没走完时，不等待最终结果，而是用 Critic 对未来价值的估计先补一个数。
- **on-policy**：数据必须由“当前这版策略”采集，策略一更新，旧数据价值就会下降。
- **off-policy**：可以从录像库取过去的数据反复训练，DDPG 和 SAC 属于这一类。
- **entropy（熵）**：动作选择的分散程度；越高代表越愿意尝试不同动作。

## 看 demo 和调试：沿同一条数据流检查

依次打印 `state → action → reward → next_state`、`value/Q`、`td_error/advantage` 和 Actor/Critic loss。PPO 额外检查 `ratio`、`approx_kl`、`clip_fraction`；DDPG/SAC 检查 Q 值、动作饱和比例、Replay Buffer 利用率，SAC 再检查 `alpha` 和 entropy。

**记忆路线：Actor-Critic 是骨架；A2C 用 advantage 批量更新；PPO 限制更新幅度；A3C 并行异步采样；DDPG 用确定性动作和回放；SAC 在回放上加入双 Critic 与最大熵探索。**
