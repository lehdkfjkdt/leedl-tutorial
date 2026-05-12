# 高频交易 Transformer 系统设计文档

## 1. 目标与范围

本文档定义一个用于高频交易场景的端到端 Transformer 系统。系统输入为过去 5 分钟的订单簿快照序列，采样频率为 100ms，目标同时覆盖两类监督任务：

1. 预测未来盘口状态
2. 预测由真实交易数据打标得到的业务标签

本文档聚焦以下内容：

1. 数据定义与张量组织方式
2. 特征工程与归一化方案
3. 模型结构设计
4. 多任务输出头与损失函数
5. 训练、验证、推理与部署建议

本文档不包含具体训练代码实现，但设计应可直接映射为 PyTorch 工程。

## 2. 业务问题定义

### 2.1 输入

输入为长度固定的历史窗口：

1. 时间跨度：5 分钟
2. 采样间隔：100ms
3. 时间步数量：3000

记历史窗口长度为 T，则：

- T = 3000

每个时间步为一个 snapshot。每个 snapshot 包含：

1. bid 侧 500 档价格
2. bid 侧 500 档数量
3. ask 侧 500 档价格
4. ask 侧 500 档数量
5. 最新成交价格
6. 最新成交数量
7. 可选的成交方向、成交笔数、时间间隔等增强特征

### 2.2 输出任务

系统支持两个主要任务。

#### 任务 A：未来盘口预测

至少支持以下监督目标：

1. 下一时刻 100ms 的盘口状态
2. 可扩展为多 horizon 联合预测，例如 100ms、500ms、1s

推荐不要直接回归所有绝对价格，而是将盘口预测拆成：

1. 价格中心变化，例如 mid price 变化、spread 变化
2. 深度结构变化，例如每档数量变化或局部深度变化
3. 可选的档位有效性或偏移结构

#### 任务 B：业务标签预测

标签来源于真实数据规则打标，可能包括：

1. 二分类信号
2. 多分类行为标签
3. 回归目标，例如未来收益、滑点、成交概率等

模型采用多任务学习，同时输出盘口预测与标签预测结果。

## 3. 数据表示与特征工程

### 3.1 原始 snapshot 的问题

原始 snapshot 维度很高，但并不建议直接把原始绝对价格和绝对数量平铺后送入模型，原因如下：

1. 绝对价格跨时间、跨品种不可比
2. 深度远端噪声较大
3. 不同量纲特征差异很大
4. 市场微结构更依赖相对位置和局部形状，而非绝对数值本身

### 3.2 推荐特征分组

每个 snapshot 建议拆成四类特征。

#### A. 盘口价格结构特征

建议使用相对价格，而不是绝对价格。

推荐特征：

1. mid price
2. spread
3. 每档相对 mid price 的价格偏移
4. 每档相对 best bid 或 best ask 的 tick 偏移

#### B. 盘口数量结构特征

推荐特征：

1. 每档挂单量
2. 对数变换后的挂单量
3. 累积深度
4. 局部 bid/ask imbalance
5. 深度斜率、深度差值

#### C. 最新成交与短窗成交特征

推荐特征：

1. 最新成交价相对 mid 的偏移
2. 最新成交量
3. aggressor side
4. 过去短窗内累计成交量
5. 过去短窗内成交笔数
6. 过去短窗内买卖主动成交不平衡

#### D. 时间与上下文特征

推荐特征：

1. 时间到开盘或收盘的位置
2. session 或交易阶段标识
3. 过去短窗收益率
4. 过去短窗波动率 proxy
5. order flow imbalance

### 3.3 单个 snapshot 的原始张量组织

若保留全量 500 档订单簿，可将单个 snapshot 组织为：

- book_price: [2, 500]
- book_size: [2, 500]
- trade_feat: [D_trade]
- global_feat: [D_global]

其中：

1. 第 1 维的 2 表示 bid 和 ask 两侧
2. 第 2 维的 500 表示 500 档位

若展开后平铺，单个 snapshot 的原始特征维约为：

- D_raw = 2 * 500 + 2 * 500 + D_trade + D_global = 2000 + D_trade + D_global

如果 D_trade 和 D_global 合计为 16 到 64，则单 snapshot 原始维度通常落在 2016 到 2064 以上。

## 4. token 化策略

### 4.1 不推荐的方案

不推荐以下方案直接作为第一版：

1. 每个 snapshot flatten 成一个超大原始向量后直接送入全局 Transformer
2. 直接把 3000 个 snapshot 的所有原始盘口数据拼接成长序列做全局 self-attention

主要问题是：

1. 忽略盘口的结构先验
2. 计算量大
3. 在线推理不友好

### 4.2 推荐 token 方案

推荐采用两层 token 化。

#### 第一层：snapshot 内结构编码

先对单个 snapshot 内的盘口结构做编码，不直接把它当一个离散 token。

推荐两种落地方式：

1. 每个 snapshot 编成一个全局 embedding
2. 每个 snapshot 编成少量结构化子 token

第一版建议先采用方案 1，降低工程复杂度。

#### 方案 1：每个 snapshot 输出一个 token

经过 snapshot encoder 后，单个 snapshot 输出：

- snapshot_embed: [D_model]

整个窗口变成：

- X: [B, T, D_model]

其中：

1. B 为 batch size
2. T = 3000
3. D_model 推荐为 128 或 256

#### 方案 2：每个 snapshot 输出多个 token

若希望更细致保留盘口结构，可让单个 snapshot 输出 K 个 token，例如 4 或 8 个：

- snapshot_tokens: [K, D_model]

整个窗口变成：

- X: [B, T, K, D_model]

进一步 reshape 为：

- X_flat: [B, T * K, D_model]

这个方案表达能力更强，但时间主干的计算量也更高。第一版不建议直接采用。

## 5. 模型总体结构

推荐系统采用以下四级结构：

1. Snapshot Encoder
2. Temporal Backbone
3. Multi-task Prediction Heads
4. Inference Buffer and Serving Layer

可写成：

raw snapshots
-> feature normalization
-> snapshot encoder
-> temporal transformer backbone
-> {book head, label head, optional auxiliary heads}

## 6. Snapshot Encoder 设计

### 6.1 目标

Snapshot Encoder 负责把单个时间点的高维盘口与成交特征，映射到固定宽度的表示向量。

输入：

- book_price: [B, 2, 500]
- book_size: [B, 2, 500]
- trade_feat: [B, D_trade]
- global_feat: [B, D_global]

输出：

- snapshot_embed: [B, D_model]

### 6.2 推荐结构

第一版推荐结构：

1. bid side encoder
2. ask side encoder
3. trade/global feature MLP
4. fusion MLP

#### 盘口侧编码器

对 bid 和 ask 分别做编码，建议使用 1D CNN 或小型 MLP。

可组织为：

- bid_input: [B, 500, C_book]
- ask_input: [B, 500, C_book]

其中 C_book 可取：

1. 价格偏移
2. 数量
3. 累积深度
4. 局部 imbalance

例如：

- bid_input: [B, 500, 4]
- ask_input: [B, 500, 4]

编码后得到：

- bid_repr: [B, D_side]
- ask_repr: [B, D_side]

#### 成交和全局特征编码

- trade_global_input: [B, D_trade + D_global]
- trade_global_repr: [B, D_aux]

#### 融合输出

拼接：

- fusion_input: [B, 2 * D_side + D_aux]

经过 MLP 输出：

- snapshot_embed: [B, D_model]

推荐超参数：

1. D_side = 128
2. D_aux = 64
3. D_model = 256

## 7. Temporal Backbone 设计

### 7.1 主要难点

时间长度 T = 3000，直接做全局 self-attention 成本较高。

普通 Transformer 编码复杂度约为：

- O(T^2 * D_model)

在 T = 3000 时，训练和推理成本都较高，尤其在在线场景中不够稳妥。

### 7.2 推荐方案：分层时间建模

第一版建议使用“下采样 + 中等长度 Transformer”的结构。

#### 第一步：时间下采样

使用 temporal convolution 或 strided 1D conv 沿时间维下采样。

输入：

- snapshot_seq: [B, T, D_model] = [B, 3000, 256]

若以 10 倍下采样为例：

- downsampled_seq: [B, 300, D_model]

若进一步下采样为 150：

- downsampled_seq: [B, 150, D_model]

#### 第二步：Transformer Encoder

对下采样后的时间序列使用 4 到 6 层 Transformer Encoder：

- temporal_hidden: [B, T_down, D_model]

例如：

- temporal_hidden: [B, 300, 256]

### 7.3 位置编码

在时间 backbone 中需要加入时间位置编码。

推荐：

1. learnable positional embedding
2. 或相对位置编码

若输入为：

- X: [B, 300, 256]

位置编码后仍为：

- X_pos: [B, 300, 256]

### 7.4 更强版本的后续升级

第二阶段可升级为：

1. 局部 attention
2. Latent query 压缩长序列
3. chunk-based hierarchical transformer
4. multi-scale temporal fusion

但第一版不建议直接上过于复杂的高效注意力结构，应先把端到端 baseline 跑通。

## 8. 输出头设计

模型输出分为两个主头和若干辅助头。

### 8.1 盘口预测头

推荐同时支持多 horizon。

设 horizon 数量为 H，例如：

1. 100ms
2. 500ms
3. 1s

则盘口预测头可以输出：

- book_pred: [B, H, 2, 500, F_book]

其中：

1. 第 1 维 H 为 horizon 数量
2. 第 2 维 2 表示 bid 和 ask
3. 第 3 维 500 表示盘口档位
4. F_book 表示预测的字段数

推荐第一版设置：

- F_book = 2

分别表示：

1. 价格变化或价格偏移
2. 数量变化

若只做单 horizon 100ms，则可简化为：

- book_pred: [B, 2, 500, 2]

### 8.2 标签预测头

标签头基于全窗口的全局表示。

推荐从 temporal backbone 中取以下表示之一：

1. 最后一个时间步 hidden state
2. 全序列 attention pooling
3. CLS token 表示

若为二分类：

- label_logits: [B, 1]

若为多分类：

- label_logits: [B, C]

若为回归：

- label_value: [B, D_out]

### 8.3 辅助头

建议加入辅助监督，提升训练稳定性：

1. mid price move 分类
2. short-horizon return 回归
3. order flow imbalance 预测

## 9. 损失函数设计

总损失定义为：

- L_total = lambda_book * L_book + lambda_label * L_label + lambda_aux * L_aux

### 9.1 盘口损失 L_book

盘口预测建议采用加权 Huber 或 Smooth L1。

原因：

1. 对异常波动更鲁棒
2. 比纯 MSE 更适合高噪声金融微结构数据

同时建议对不同档位做权重设计：

1. top 10 档权重最高
2. 10 到 50 档次之
3. 远端档位权重较低

理由是近端盘口对短期交易决策更关键。

### 9.2 标签损失 L_label

根据标签类型选择：

1. 二分类：BCEWithLogitsLoss
2. 多分类：CrossEntropyLoss
3. 回归：HuberLoss 或 MSELoss

### 9.3 辅助损失 L_aux

用于稳定学习和增强表示，建议权重低于主任务。

## 10. 训练数据组织

### 10.1 输入张量

推荐训练输入统一为：

- X_book_price: [B, T, 2, 500]
- X_book_size: [B, T, 2, 500]
- X_trade: [B, T, D_trade]
- X_global: [B, T, D_global]

若 snapshot encoder 内部先拼接，则也可以组织为：

- X_snapshot: [B, T, D_snapshot_raw]

其中 D_snapshot_raw 为单步原始特征维度。

### 10.2 标签张量

盘口监督：

- Y_book: [B, H, 2, 500, F_book]

业务标签：

- Y_label: [B] 或 [B, C] 或 [B, D_out]

### 10.3 数据切分原则

必须按时间切分 train、val、test，不能随机打乱。

推荐原则：

1. 按天或按周切分
2. 归一化统计只从训练集估计
3. 避免未来信息泄漏

## 11. 归一化与预处理

### 11.1 价格归一化

推荐：

1. 相对 mid price 的偏移
2. tick-based offset
3. 跨品种训练时做 instrument-aware normalization

### 11.2 数量归一化

推荐：

1. log(1 + size)
2. rolling z-score
3. 分品种分时段归一化

### 11.3 成交特征归一化

推荐：

1. 成交量对数化
2. 价格相对 mid 偏移
3. 时间间隔标准化

## 12. 推理系统设计

### 12.1 在线缓冲区

在线推理时维护长度固定的 rolling buffer：

- history buffer: [T, snapshot]

每来一个新 snapshot：

1. 推入最新 snapshot
2. 弹出最旧 snapshot
3. 更新模型输入窗口
4. 触发预测

### 12.2 在线延迟优化建议

第一版建议采用以下简单策略：

1. snapshot encoder 做逐步增量编码
2. temporal backbone 控制深度在 4 到 6 层
3. 避免使用过重的全局长序列 attention

后续可升级为：

1. 缓存 chunk 表示
2. 增量更新 temporal states
3. Latent query 压缩

## 13. 第一版 baseline 配置建议

推荐第一版模型参数如下：

1. Snapshot encoder 输出维度 D_model = 256
2. 时间下采样后长度 T_down = 300
3. Transformer encoder 层数 = 4
4. attention heads = 8
5. FFN hidden size = 1024
6. 多任务输出头：book head + label head + optional aux head

### 第一版输入输出维度示例

输入：

- X_book_price: [B, 3000, 2, 500]
- X_book_size: [B, 3000, 2, 500]
- X_trade: [B, 3000, D_trade]
- X_global: [B, 3000, D_global]

snapshot encoder 后：

- X_embed: [B, 3000, 256]

时间下采样后：

- X_down: [B, 300, 256]

temporal backbone 后：

- H_time: [B, 300, 256]

盘口头输出：

- Y_book_pred: [B, H, 2, 500, 2]

标签头输出：

- Y_label_pred: [B, C] 或 [B, 1]

## 14. 第二阶段升级路线

第一版 baseline 跑通后，建议按以下顺序升级：

1. 单 snapshot 输出多个结构化 token
2. 局部 attention 或 chunk attention
3. latent query 压缩长序列
4. factorized attention
5. 多尺度 horizon 联合学习
6. mixture 或 uncertainty 输出头

## 15. 风险与关键注意事项

本项目的主要风险不在于 Transformer 主干本身，而在于以下三点：

1. 目标定义不稳定：若直接回归全量绝对盘口，噪声可能过大
2. 数据泄漏：时间切分和归一化统计容易出错
3. 输入表示不合理：原始绝对价格和大维度平铺会降低建模效率

因此第一阶段必须优先保证：

1. 特征相对化
2. 目标可学且稳定
3. 时间主干计算量可控

## 16. 建议的后续实现顺序

建议按以下顺序推进工程：

1. 定义 snapshot 原始字段与目标字段
2. 写 dataset 和时间切窗逻辑
3. 写归一化模块
4. 实现第一版 snapshot encoder
5. 实现 temporal transformer backbone
6. 实现盘口 head 和标签 head
7. 实现多任务训练循环
8. 实现回测与线上推理缓冲逻辑

## 17. 总结

本设计文档给出的核心结论是：

1. 高频交易场景中，Transformer 不应直接吞下原始全量盘口平铺特征
2. 更合理的做法是先做单 snapshot 结构编码，再做时间建模
3. 第一版推荐“snapshot encoder + 时间下采样 + Transformer + 多任务头”结构
4. 盘口预测和业务标签预测可通过多任务学习统一训练
5. 第一版先跑通稳定 baseline，再逐步引入多 token snapshot、latent query 和更高效的 attention 机制
