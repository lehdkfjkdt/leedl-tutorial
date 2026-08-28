# NavRL 源码级详细解析：把它看成机器人的“导航小脑”

> 本文基于本地仓库 `/home/ubuntu/workspace/NavRL` 的源码编写，重点解释**代码真实做了什么**，而不是只复述论文摘要。
>
> 论文：Zhefan Xu 等，*NavRL: Learning Safe Flight in Dynamic Environments*，IEEE Robotics and Automation Letters，2025。
>
> 论文与官方仓库：<https://arxiv.org/abs/2409.15634> · <https://github.com/Zhefan-Xu/NavRL>

## 1. 一句话理解 NavRL

NavRL 是一个**局部导航策略**：机器人每个控制周期读取“我现在的状态、目标在哪、LiDAR 看到了什么、附近动态障碍物怎么动”，神经网络输出一个期望速度；训练时用 PPO 让这个速度既朝向目标，又远离障碍、保持运动平滑和合理高度。

在完整部署中，链路不是“神经网络直接接管电机”，而是：

```text
传感器/状态
    -> 观测构造
    -> NavRL PPO 策略
    -> 期望世界系速度
    -> safety shield（ROS 中独立安全动作服务）
    -> 速度控制器/飞控
    -> 电机或机器人执行器
```

因此可以把它称为机器人的“导航小脑”：它负责局部、快速、反应式的运动决策；全局地图、任务规划、姿态稳定和最终急停仍属于其他模块。

## 2. 先建立源码地图

| 文件 | 作用 |
| --- | --- |
| `isaac-training/training/scripts/env.py` | Isaac Sim 导航环境：场景、LiDAR、动态障碍、观测、奖励和终止条件 |
| `isaac-training/training/scripts/ppo.py` | CNN/MLP 特征提取器、Beta Actor、Critic 和 PPO 更新 |
| `isaac-training/training/scripts/utils.py` | GAE、ValueNorm、坐标变换、动作分布和批处理工具 |
| `isaac-training/training/scripts/train.py` | Isaac Sim 启动、速度控制器、数据采集器和训练循环 |
| `quick-demos/agent.py` | 加载预训练权重并以确定性均值动作推理 |
| `quick-demos/ppo.py` | 快速演示使用的 PPO 网络定义 |
| `ros1/navigation_runner/scripts/navigation.py` | ROS1 真实/仿真导航、策略推理、安全服务调用和速度发布 |
| `ros1/navigation_runner/include/navigation_runner/safeAction.cpp` | ROS1 safety shield 的几何安全动作计算 |
| `ros2/navigation_runner/` | ROS2 对应的导航、感知和安全节点 |

训练、快速演示和 ROS 推理中的网络结构基本保持一致，但配置中的动作上限可能不同。例如 Isaac 训练配置使用 `2.0 m/s`，ROS1 脚本配置中可见 `1.0 m/s`；部署时应以实际使用的 YAML 为准。

## 3. 从环境角度看：NavRL 的 MDP

强化学习环境可以写成：

```text
观测 o_t  --策略 πθ--> 动作 a_t  --物理仿真--> 新状态 s_(t+1)
   ^                                      |
   |                                      v
   +----------- 下一次观测、奖励 r_t、done
```

在 `env.py` 中，Isaac 环境的生命周期大致是：

1. `_pre_sim_step()`：取策略动作并调用 `self.drone.apply_action(actions)`。
2. Isaac Sim 推进一步物理仿真。
3. `_post_sim_step()`：移动动态障碍物，更新 LiDAR。
4. `_compute_state_and_obs()`：构造观测、计算奖励和碰撞统计。
5. `_compute_reward_and_done()`：把奖励、`terminated` 和 `truncated` 写入 TorchRL 的 TensorDict。

这就是一个标准的“采样 rollout -> 计算优势 -> PPO 更新”闭环。

## 4. 观测到底是什么

环境在 `_set_specs()` 中声明了四类策略输入：

```python
observation = {
    "state":             (8,),
    "lidar":             (1, horizontal_beams, vertical_beams),
    "direction":         (1, 3),
    "dynamic_obstacle":  (1, 5, 10),
}
```

默认 `lidar_hres=10` 度，所以水平射线数是 `360 / 10 = 36`；垂直方向默认 4 条射线，LiDAR 输入为 `(1, 36, 4)`。动态障碍物特征最多保留 5 个，每个 10 维。

### 4.1 `state`：不是原始世界坐标

源码中 `drone_state` 由以下量拼接而成：

```text
目标方向单位向量（在目标坐标系）  3 维
目标水平距离                    1 维
目标高度差                      1 维
机器人速度（在目标坐标系）        3 维
--------------------------------------
总计                            8 维
```

代码对应：

```python
rpos = target_pos - drone_position
distance = ||rpos||
rpos_clipped = rpos / distance
rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d)
vel_g = vec_to_new_frame(world_velocity, target_dir_2d)
drone_state = cat([
    rpos_clipped_g,
    distance_2d,
    distance_z,
    vel_g,
])
```

这里的关键是 `vec_to_new_frame()`：策略不是直接学习固定世界坐标，而是在一个“x 轴朝向目标”的局部坐标系中学习。这样同一套策略可以处理目标在不同方向的情况，减少旋转等价场景带来的样本浪费。

### 4.2 `direction`：用于动作坐标变换

`direction` 保存目标方向的 3D 向量，主要用途不是再提供一份普通状态，而是让 `ppo.py` 把策略在目标局部坐标系中产生的速度转回世界坐标：

```python
actions = 2 * action_normalized * action_limit - action_limit
actions_world = vec_to_world(actions, observation["direction"])
```

### 4.3 `lidar`：距离越近，数值越大

LiDAR 的原始射线命中距离先被截断到 `lidar_range`，然后源码使用：

```python
lidar_scan = lidar_range - distance_to_hit
```

因此它不是通常的“距离越远数值越大”，而是：

```text
没有近处障碍：接近 0
障碍物很近：接近 lidar_range
```

这种编码直接服务于奖励函数：`lidar_scan` 越大，碰撞风险越高。

### 4.4 `dynamic_obstacle`：只取最近的 5 个

动态障碍处理并不是把所有障碍物全部送进网络，而是：

1. 计算每个障碍物相对机器人的二维距离。
2. 用 `torch.topk(..., largest=False)` 选最近的 5 个。
3. 超过 LiDAR 范围的障碍物用掩码排除，并将相关特征置零。
4. 把位置、距离、相对速度、尺寸类别等拼成 10 维特征。

每个动态障碍特征大致包含：

```text
目标坐标系中的相对位置单位向量      3
水平距离                           1
垂直距离                           1
目标坐标系中的障碍物速度             3
宽度类别                           1
高度类别                           1
                                   = 10
```

注意：这是一个固定大小的输入。障碍物多于 5 个时只保留最近者，障碍物少于 5 个时用零填充。这种设计计算稳定，但也意味着策略不会看到全部远处障碍物。

## 5. 网络结构：两个感知分支，一个决策头

源码中的 `PPO.__init__()` 把网络拆成三部分。

### 5.1 LiDAR CNN

```python
Conv2d(4,  kernel=[5, 3]) -> ELU
Conv2d(16, kernel=[5, 3], stride=[2, 1]) -> ELU
Conv2d(16, kernel=[5, 3], stride=[2, 2]) -> ELU
Flatten
LazyLinear(128) -> LayerNorm(128)
```

这里卷积输入的通道数是 1，代码用 `LazyConv2d` 延迟确定输入维度；输出被压成 128 维 `_cnn_feature`。

LiDAR 分支的作用是从局部射线图中提取：正前方是否有障碍、左右哪边更开阔、障碍物大致高度分布等空间模式。它不是显式建图，也不输出障碍物列表。

### 5.2 动态障碍 MLP

动态障碍输入 `(1, 5, 10)` 先展平，再经过：

```python
Flatten -> make_mlp([128, 64])
```

`make_mlp()` 每一层是 `LazyLinear -> LeakyReLU -> LayerNorm`，最终形成 64 维 `_dynamic_obstacle_feature`。

### 5.3 融合层

网络把：

```text
LiDAR 特征 128
机器人状态 8
动态障碍特征 64
----------------
拼接后       200
```

送入 `make_mlp([256, 256])`。所以最后的 `_feature` 是 256 维共享表示，Actor 和 Critic 都读取它。

这体现了一个重要的“共享小脑”结构：感知编码只做一次，策略头决定动作，价值头评估当前状态好不好。

## 6. Actor 如何输出速度

### 6.1 为什么使用 Beta 分布

Actor 不是直接输出一个确定的速度，而是输出每个动作维度的 Beta 分布参数：

```python
alpha = 1 + Softplus(alpha_layer(feature))
beta  = 1 + Softplus(beta_layer(feature))
```

Beta 分布的采样值天然位于 `(0, 1)`，适合表示归一化动作 `action_normalized`。训练时从这个分布随机采样，产生探索；评估或快速演示时取均值，动作更稳定。

### 6.2 从 `(0, 1)` 映射到速度范围

源码使用：

```python
actions = (2 * action_normalized * action_limit) - action_limit
```

也就是：

```text
0        -> -action_limit
0.5      -> 0
1        -> +action_limit
```

默认 Isaac 训练配置的 `action_limit=2.0`，策略动作是三维速度向量，每个分量约束在 `[-2, 2] m/s`。ROS 配置可能将上限改为 `1.0 m/s`，这是部署时必须核对的参数。

### 6.3 从目标坐标系变回世界坐标系

策略在“目标方向坐标系”中工作，源码随后调用 `vec_to_world()`：

```text
目标方向作为局部 x 轴
z 轴取世界竖直方向
y 轴由叉乘得到
局部速度 -> 世界速度
```

最终写入：

```python
tensordict["agents", "action"] = actions_world
```

所以 `agents.action` 是真正交给控制器/环境的世界坐标速度，不是 Beta 分布里的无量纲数值。

## 7. Critic 和 PPO 更新

Critic 是一个 `LazyLinear(1)`，输出状态价值 `V(s)`。PPO 的训练流程在 `ppo.py` 中很清楚：

### 7.1 数据采集

`train.py` 使用 `SyncDataCollector`：

```python
frames_per_batch = env.num_envs * training_frame_num
exploration_type = RANDOM
```

每个 batch 保存状态、动作、动作对数概率、奖励、终止标志和当时的状态价值。

### 7.2 GAE 计算优势

源码配置：

```text
gamma = 0.99
lambda = 0.95
```

时间差分误差为：

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)
$$

GAE 优势为：

$$
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}
$$

然后优势标准化，回报经过 `ValueNorm` 归一化。这样可以减少奖励尺度变化对 Critic 和 Actor 更新的影响。

### 7.3 PPO 截断目标

新旧策略概率比：

$$
r_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t))
$$

源码使用 `clip_ratio=0.1`：

$$
L^{CLIP}=\min(r_tA_t,\operatorname{clip}(r_t,0.9,1.1)A_t)
$$

Actor 最小化负的 clipped objective，并加入熵正则鼓励探索；Critic 使用 Huber loss，并对 value 更新做 clipping。梯度范数被限制在 5，避免一次更新过大。

### 7.4 训练超参数的实际含义

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `learning_rate` | `5e-4` | 特征提取器和 Actor 的 Adam 学习率 |
| `action_limit` | `2.0` | 每个速度分量的上限，单位 m/s |
| `training_frame_num` | `32` | 每个并行环境每批采集的时间帧数 |
| `training_epoch_num` | `4` | 同一批 rollout 重复优化次数 |
| `num_minibatches` | `16` | 一个 batch 切成多少小批次 |
| `entropy_loss_coefficient` | `1e-3` | 探索强度 |

## 8. 奖励函数：它究竟在鼓励什么

`env.py` 的奖励不是一个简单的“到目标 +1、碰撞 -100”，而是多个连续项的组合。

### 8.1 朝目标的速度奖励

```python
vel_direction = rpos / distance
reward_vel = (drone.vel_w[..., :3] * vel_direction).sum(-1)
```

这是速度在目标方向上的投影：朝目标飞得越快，奖励越大；横向或反向运动的贡献较小或为负。

### 8.2 静态障碍安全奖励

```python
reward_safety_static = log(lidar_range - lidar_scan).mean(...)
```

由于 `lidar_scan = lidar_range - distance`，所以这里实际等价于对“障碍物距离”取对数平均。离障碍越远，奖励越大；距离接近 0 时，奖励急剧变小。

### 8.3 动态障碍安全奖励

最近动态障碍物的估计距离也进入对数奖励：

```python
reward_safety_dynamic = log(closest_dyn_obs_distance_reward).mean(...)
```

这会让策略在动态障碍附近更谨慎，而不是只在发生碰撞时才收到信号。

### 8.4 平滑性惩罚

```python
penalty_smooth = ||current_velocity - previous_velocity||
```

最终乘以 `0.1` 从总奖励中扣除，抑制速度突然跳变。它对真实飞控很重要，因为高频抖动会放大执行器延迟和模型误差。

### 8.5 高度惩罚

机器人高度超出 `[height_min, height_max]` 附近的缓冲区时，按平方距离惩罚，权重为 `8.0`。这会避免策略为了绕过障碍无限上升或下降。

### 8.6 碰撞和终止

静态碰撞判断：LiDAR 中任意射线接近量程上限（源码把 `lidar_scan > lidar_range - 0.3` 视为碰撞）。动态碰撞则根据相对位置、障碍物宽度和高度判断。

终止条件：

```text
碰撞
高度低于 0.2 m
高度高于 4.0 m
```

时间步超过 `max_episode_length` 时是 `truncated`。到达目标（距离小于 `0.5 m`）会被记录为 `reach_goal`，但源码当前主要将其作为统计量，并没有在 `terminated` 中直接加入目标到达条件。这是阅读和修改代码时必须注意的实现细节。

另外，源码中碰撞额外惩罚 `-50` 被注释掉了：

```python
# self.reward[collision] -= 50.
```

因此训练中的安全性主要来自持续的距离奖励和碰撞终止，而不是一个显式的大额碰撞负奖励。如果你要复现实验或修改任务，是否打开该项会显著改变奖励分布，应单独做消融实验。

## 9. Isaac Sim 中“策略动作”与“执行动作”的区别

`train.py` 中有一个容易被忽略的变换：

```python
controller = LeePositionController(...)
vel_transform = VelController(controller, yaw_control=False)
transformed_env = TransformedEnv(env, Compose(vel_transform)).train()
```

含义是：

```text
PPO 输出：期望速度
VelController：把期望速度转换成姿态/推力等控制量
LeePositionController：执行无人机的低层位置/姿态控制
Isaac Sim：推进动力学
```

所以不能把 `ppo.py` 的三维动作误认为“三个电机转速”。动作是高层速度命令，低层控制器负责让无人机跟踪它。

## 10. ROS 部署时 safety shield 在哪里

ROS1 的 `navigation.py` 先调用策略：

```python
cmd_vel_world = self.get_action(...)
```

然后把策略速度、机器人当前速度、机器人尺寸、静态/动态障碍物信息等发送给：

```text
rl_navigation/get_safe_action
```

服务由 `safe_action_node.cpp` 和 `safeAction.cpp` 提供。返回的 `safe_action` 才会继续经过坐标系处理，最后发布到 MAVROS 或仿真速度 topic。

完整部署链路是：

```text
NavRL policy:        “我想这样飞”
safe_action service: “这个速度是否会撞？必要时改成安全速度”
速度限幅/急停逻辑:    “通信异常或紧急状态时停下”
最终 ROS publisher:  发送给飞控/仿真器
```

源码还启动了安全检查线程，并在最终发布前处理安全停止、局部速度限幅和坐标变换。这个 shield 是部署侧的重要防线，但不能替代真实系统中的独立急停、限速和失联保护。

## 11. 快速演示的执行路径

`quick-demos/simple-navigation.py` 展示了最小推理闭环：

1. 随机生成静态圆形障碍物。
2. 初始化机器人位置、速度和目标。
3. `Agent(device)` 创建网络并加载 `navrl_checkpoint.pt`。
4. 每一帧调用 `get_robot_state()` 构造 8 维状态。
5. `get_ray_cast()` 构造 `(1, 36, 4)` LiDAR 输入。
6. 动态障碍输入用全零 Tensor 表示“当前没有动态障碍”。
7. `agent.plan()` 在 `ExplorationType.MEAN` 下取策略均值动作。
8. 用 `robot_pos += velocity * DT` 更新二维位置并绘图。

这段 demo 很适合验证：权重能否加载、输入形状是否正确、策略是否会基本绕障。但它没有 Isaac Sim 的真实动力学、三维碰撞、ROS 通信延迟和独立安全服务，因此不能作为真实飞行安全验证。

## 12. 为什么 NavRL 可以作为“小脑”

从机器人系统分层看，NavRL 位于中间层：

```text
大脑/任务层：全局地图、任务目标、路径规划
        ↓ 目标点或目标方向
小脑/NavRL：局部感知、避障、速度决策
        ↓ 期望速度
脊髓/控制层：姿态控制、推力分配、执行器闭环
        ↓
电机/腿/轮
```

它的优势是反应快、输入紧凑、可以在大量仿真环境中并行训练；它的边界也很清楚：没有全局可达性保证，不负责长期路线规划，也不能单独承担飞控级安全责任。

## 13. 阅读和改代码的建议顺序

### 想理解推理

按顺序看：

```text
quick-demos/simple-navigation.py
    -> quick-demos/agent.py
    -> quick-demos/ppo.py
    -> quick-demos/utils.py
```

重点跟踪 `robot_state -> lidar -> PPO -> action -> position update`。

### 想理解训练

```text
isaac-training/training/scripts/train.py
    -> env.py::_compute_state_and_obs
    -> ppo.py::train
    -> utils.py::GAE
```

重点跟踪 `collector` 产生的 TensorDict 如何被 `policy.train(data)` 消费。

### 想接入真实机器人

```text
ros1/.../navigation.py
    -> get_action()
    -> get_safe_action()
    -> safeAction.cpp
    -> ROS velocity publisher
```

优先确认坐标系、单位（m/s）、LiDAR 量程、机器人尺寸、速度上限和安全服务是否使用同一套定义。

## 14. 复现时的现实限制

- 完整训练依赖 NVIDIA Isaac Sim `2023.1.0-hotfix.1`、CUDA、GPU 和仓库内的 Orbit/OmniDrones 组合；当前机器没有这些依赖时，不应直接运行 `setup.sh`。
- 快速 demo 只需要推理依赖和预训练权重，适合先验证网络链路。
- ROS1 示例面向 Ubuntu 20.04 + ROS Noetic；ROS2 示例面向 Ubuntu 22.04 + ROS2 Humble，系统版本不要混用。
- 预训练权重不是任意机器人通用模型。更换传感器分辨率、动作单位、机器人尺寸、动力学或坐标系后，至少需要重新校准输入和控制器，通常还需要重新训练。

## 15. 最终总结

NavRL 的核心不是“一个 PPO 文件”，而是一个完整的导航闭环：

1. `env.py` 将三维仿真和传感器压缩成目标坐标系状态、LiDAR 和最近动态障碍物。
2. `ppo.py` 用 CNN + MLP 融合感知，Beta Actor 输出有界随机速度，Critic 估计价值。
3. `env.py` 用目标速度、安全距离、平滑性和高度构造奖励，并在碰撞/越界时终止。
4. `train.py` 用 Isaac Sim 并行采样，GAE + PPO 反复更新策略。
5. ROS 部署中，策略速度还要经过独立 safety shield，最后才发送给飞控。

最准确的理解是：**NavRL 学的是“在局部观测下选择下一小步速度”的能力；安全盾牌和低层控制器共同把这个能力变成可执行、可约束的机器人运动。**
