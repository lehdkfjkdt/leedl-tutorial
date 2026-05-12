# HW05 Seq2seq 机器翻译（En→Zh-TW）总结

对应材料：
- 代码/示例 Notebook：[Homework/HW5_seq2seq/HW05_seq2seq.ipynb](Homework/HW5_seq2seq/HW05_seq2seq.ipynb)
- 作业说明 PDF：[Homework/HW5_seq2seq/HW05.pdf](Homework/HW5_seq2seq/HW05.pdf)

> 目标：把英文句子翻译成繁体中文。数据为 TED2020 英中平行语料，评测使用 BLEU。

---

## 1. 作业要求与整体路线（来自 PDF）

### 1.1 基线分层（Baseline Ladder）
PDF 把提升路径拆成四档，Notebook 的设计也与之对应：

1) **Simple baseline**：训练一个 **RNN seq2seq** 做翻译（跑通 sample code 即可）。

2) **Medium baseline**：
- 加入 **学习率调度器**（Noam/Inverse-Sqrt 风格）
- **训练更久**（PDF 举例把 `max_epoch` 从 15 提升到 30）

3) **Strong baseline**：
- **切换为 Transformer Encoder/Decoder**
- 调参（层数、hidden size、dropout、heads…），建议 encoder/decoder layers 从 1 增加到 4，并参考 *Attention is All You Need* 的 transformer-base 超参表。

4) **Boss baseline**：**Back-translation**
- 训练一个反向模型（Zh→En）
- 用反向模型翻译中文单语数据，得到“合成平行数据”
- 把合成数据混入正向模型（En→Zh）训练，再训练更久（PDF 提到正确的话新数据上约 30 epochs 可达标）

### 1.2 评测指标 BLEU（PDF）
- 使用 **Modified n-gram precision**（通常 $n=1..4$）
- 使用 **Brevity Penalty** 惩罚过短的翻译
- 最终 BLEU 是带 BP 的几何平均

Notebook 验证阶段用 `sacrebleu.corpus_bleu()` 计算分数（中文 tokenize 采用 `zh`）。

### 1.3 报告题（PDF）
PDF 的报告部分包含两题（与模型训练本身相对独立）：

- **Problem 1：可视化 Decoder 的 positional embedding 相似度矩阵**
  - 取 decoder 的位置向量表 $N\times D$，计算 $N\times N$ 的相似度（推荐 cosine similarity）
  - 可视化后解释现象（直觉：相邻位置更相似）
  - 提示代码：`pos_emb = model.decoder.embed_positions.weights.cpu().detach()`

- **Problem 2：梯度裁剪与梯度爆炸可视化**
  - 启用 `clip_grad_norm_` 并记录不同 step 的 grad norm
  - 圈出两处发生梯度爆炸的位置（如果存在）

### 1.4 提交与规则（PDF 摘要）
- JudgeBoi：每天提交次数上限、只允许 `.txt`、大小限制、可选择最多 2 份提交等
- 课程平台代码提交：打包 zip（只含代码与报告，不要上传模型和数据）
- 禁止：抄袭、手工改预测文件、分享代码/预测、使用额外数据或预训练模型、攻击服务器等

---

## 2. Notebook 的端到端工作流（代码主线）

Notebook 按“数据→预处理→子词→二进制化→建模→训练→验证→推理→提交→回译增强”的顺序组织。

### 2.1 数据下载与目录约定
- 原始数据目录：`./DATA/rawdata/ted2020/`
- 下载两份压缩包：训练/验证（`ted2020.tgz`）与测试（`test.tgz`）
- 统一重命名为：
  - `train_dev.raw.en / train_dev.raw.zh`
  - `test.raw.en / test.raw.zh`

测试集 `test.raw.zh` 是伪翻译（每行都是“。”），只能用于格式对齐，**不能用于 BLEU**。

### 2.2 数据清洗：规范化 + 过滤坏样本
核心函数：`clean_s()`、`clean_corpus()`。

- **中文**：全角转半角 `strQ2B`、去空格、统一引号、保留/切分常见标点（用空格隔开）
- **英文**：去括号内容、去连字符、标点分离
- 过滤策略：
  - 过短（`min_len`）
  - 过长（`max_len`）
  - 源/目标长度比例过大（`ratio`）

输出：`*.clean.en/zh`。

### 2.3 训练/验证拆分
- 使用 `valid_ratio=0.01` 随机打散索引后切分
- 生成：
  - `train.clean.en/zh`
  - `valid.clean.en/zh`

### 2.4 子词单元：SentencePiece（Subword Units）
目的：缓解 OOV（词表外词）问题。

- 训练：`SentencePieceTrainer.train(...)`
  - `vocab_size=8000`
  - `model_type='unigram'`（也可用 `'bpe'`）
  - 输入包含 train/valid 的 en+zh 四份文件
- 编码：`spm_model.encode(line, out_type=str)`
  - 把句子转成带 `▁` 前缀的子词序列，并以空格连接
- 输出文件：
  - `train.en / train.zh`
  - `valid.en / valid.zh`
  - `test.en / test.zh`

### 2.5 使用 fairseq 进行二进制化（preprocess）
调用 `fairseq_cli.preprocess` 生成 mmap 二进制数据：
- `train.en-zh.en.{bin,idx}`
- `train.en-zh.zh.{bin,idx}`
- 以及 `dict.*.txt`（使用 `--joined-dictionary` 共用词表）

之后训练/验证数据不再从纯文本读取，而是通过 fairseq 的 Task/Dataset 管理。

---

## 3. 训练框架：fairseq TranslationTask + 自定义模型

### 3.1 TranslationTask 的作用
Notebook 使用：
- `TranslationTask.setup_task(TranslationConfig(...))`
- `task.load_dataset('train'/'valid')`

带来的好处：
- 直接读取 `data-bin/` 下的二进制数据
- 内置 `source_dictionary/target_dictionary`
- 提供 batch iterator 与 beam search generator

### 3.2 Batch 的关键字段（非常重要）
fairseq 的一个 batch 是 dict：
- `net_input.src_tokens`：源句子 token ids（已 padding）
- `net_input.src_lengths`：padding 前长度
- `net_input.prev_output_tokens`：**右移后的目标序列**（decoder 输入）
- `target`：对齐的 decoder 监督目标

Notebook 解释了 fairseq 的右移方式：不是把 BOS 放最前，而是把 `eos` 移到最前，效果等价。

---

## 4. 模型部分：RNN Seq2seq（含 Attention）

Notebook 先实现 RNN 版本，再留出 TODO 切换 Transformer。

### 4.1 RNNEncoder（双向 GRU）
类：`RNNEncoder(FairseqEncoder)`
- Embedding：`embed_tokens(src_tokens)`
- 双向 GRU：`bidirectional=True`
- 输出：
  - `outputs`：$[S, B, 2H]$（每步输出，用于 attention）
  - `final_hiddens`：$[L, B, 2H]$（合并双向后的最终 hidden，供 decoder 初始化）
  - `encoder_padding_mask`：$[S, B]$（PAD 位置为 True，用于屏蔽）

为了适配 beam search，额外实现了 `reorder_encoder_out()`。

### 4.2 AttentionLayer（加性/点积风格的 attention）
类：`AttentionLayer(nn.Module)`
- 输入（decoder 当前步的表示）做线性投影成 $Q$
- 与 encoder outputs 做 batch matmul 得到打分并 softmax：

$$A = QK^T,\quad A' = \mathrm{softmax}(A)$$

- 用 $A'$ 对 encoder outputs 加权求和得到上下文向量，再与原输入 concat，经线性层+tanh 输出。
- 使用 `encoder_padding_mask` 把 PAD 的 score mask 成 $-\infty$。

### 4.3 RNNDecoder（单向 GRU + Attention + 增量推理）
类：`RNNDecoder(FairseqIncrementalDecoder)`
- 断言约束（保证维度可对齐）：
  - encoder/decoder 层数相同
  - decoder hidden dim = 2 × encoder hidden dim（因为 encoder 双向）
- 结构：
  1) embedding + dropout
  2) attention（把 decoder 表示与 encoder outputs 对齐）
  3) GRU 更新 hidden
  4) （可选）投影到 embed dim
  5) output projection 到 vocab
- 支持 `incremental_state`：推理时只喂最后一个 token，并缓存 `prev_hiddens`，加速逐步生成。

### 4.4 Seq2Seq（Encoder-Decoder 封装）
类：`Seq2Seq(FairseqEncoderDecoderModel)`
- `forward()`：encoder → decoder，返回 logits。

### 4.5 切换 Transformer 的关键改动点（Notebook 的 TODO）
函数：`build_model(args, task)`
- 已导入：`TransformerEncoder/TransformerDecoder`
- TODO 实际上就是把：
  - `encoder = RNNEncoder(...)` / `decoder = RNNDecoder(...)`
  - 换成 `TransformerEncoder(...)` / `TransformerDecoder(...)`

并且需要补齐 transformer 相关 args（Notebook 给了 `add_transformer_args()` 补丁函数，并提示参考 transformer-base 超参）。

---

## 5. 训练细节：Loss、优化器、AMP、梯度裁剪、验证 BLEU

### 5.1 Label Smoothing Cross Entropy
类：`LabelSmoothedCrossEntropyCriterion`
- 思想：把一部分概率质量从正确标签“分配”给其他标签，缓解 over-confidence。
- 公式直观上相当于：

$$\mathcal{L}= (1-\epsilon)\,\mathcal{L}_{\text{NLL}} + \epsilon\,\mathcal{L}_{\text{smooth}}$$

- PAD 位置通过 `ignore_index` 排除。

### 5.2 Noam 学习率调度（Medium baseline 的关键点）
Notebook 实现了 Noam 风格学习率：

$$\text{lrate}=d_{model}^{-0.5}\cdot\min(\text{step}^{-0.5},\ \text{step}\cdot \text{warmup}^{-1.5})$$

- `NoamOpt` 作为 optimizer wrapper，内部维护 step 并更新 `param_groups[].lr`。
- PDF 也明确指出：Medium baseline 要把常数 lr 改成上式并训练更久。

### 5.3 train_one_epoch：梯度累积 + 自动混合精度（AMP）
函数：`train_one_epoch(...)`
- 用 `GroupedIterator(..., accum_steps)` 做梯度累积
- `autocast()` + `GradScaler()` 做 mixed precision
- 每个 update 前：
  - `optimizer.multiply_grads(1/sample_size)` 归一化
  - `clip_grad_norm_(..., config.clip_norm)` 裁剪梯度，防止爆炸

### 5.4 validate：计算 valid loss + 生成翻译 + BLEU
- 使用 fairseq 的 `sequence_generator = task.build_generator([model], config)`
- `inference_step()` 收集：source / hypothesis / reference
- `sacrebleu.corpus_bleu(hyps, [refs])` 得到 BLEU

### 5.5 checkpoint：保存、载入、平均权重
- `validate_and_save()`：每 epoch 验证并保存 `checkpoint{epoch}.pt`，维护 `checkpoint_last.pt` 与 `checkpoint_best.pt`
- `average_checkpoints.py`：把最近 N 个 epoch 平均成一个权重文件（类似 ensemble 的收益）

---

## 6. 推理与生成提交文件

### 6.1 生成测试集预测
函数：`generate_prediction(model, task, split='test', outfile='./prediction.txt')`
- 对 test split 逐 batch 做 beam search 推理
- 按原始样本 id 排序
- 输出每行一条翻译到 `prediction.txt`

### 6.2 JudgeBoi 注意点（PDF）
- 只允许 `.txt`，大小需小于约 700KB
- 每天有提交次数限制
- 不允许手工修改预测文件

---

## 7. Back-translation（Boss baseline）在 Notebook 的落地方式

Notebook 最后给了完整流程说明，并留了几个 TODO：

1) **训练反向模型**（Zh→En）
- 在 `config` 中交换 `source_lang/target_lang`
- 更换 `savedir`（避免覆盖正向模型）

2) **下载中文单语数据**（TED zh corpus）
- 下载 `ted_zh_corpus.deduped.gz` 并解压

3) **TODO：清洗单语数据**
- 用前面相同的 `clean_s()` 思路统一标点、过滤过长过短

4) **TODO：用既有 spm 模型做子词切分**
- 使用正向/反向模型对应的 `spm{vocab}.model` 进行编码

5) **二进制化单语数据**
- `fairseq_cli.preprocess`，并强制使用原任务的词典（`--srcdict/--tgtdict`）以保持 token id 一致

6) **TODO：用反向模型生成合成平行数据**
- 把 monolingual split 的 bin/idx 文件拷贝进 `ted2020` data-bin 目录
- 通过 `generate_prediction(model, task, split='split_name')` 生成“合成英文”

7) **用新数据训练更强的正向模型**
- `TranslationTask.load_dataset(..., combine=True)` 允许把多个 split 合并训练（Notebook 注释：当你有 back-translation 数据时就用 combine）

---

## 8. 读代码时建议抓住的“关键接口”

如果你要基于该 Notebook 自己改模型/调参，优先关注这些点：

- **数据链路**：raw → clean → spm → fairseq preprocess → `TranslationTask`
- **batch 对齐**：`prev_output_tokens`（decoder 输入）与 `target`（监督标签）
- **训练稳定性**：Noam scheduler、label smoothing、gradient clipping、AMP + 梯度累积
- **推理一致性**：beam search generator + post_process（sentencepiece detok）
- **从 RNN 切 Transformer**：`build_model()` 与 transformer args 补丁（heads、normalize_before、max_positions 等）

---

## 9. 这份 Notebook 与 PDF 的对应关系速查

- PDF 的 “Workflow / Baselines / Back-translation” 章节 ↔ Notebook 的数据预处理、`get_rate/NoamOpt`、`TransformerEncoder/Decoder TODO`、Back-translation TODO 区域
- PDF 的 “Report Problem 1/2” ↔ 需要你在训练脚本中额外写可视化与记录（Notebook 目前主要是训练与推理，不含完整作图/日志实现）
- PDF 的 “JudgeBoi Guide/Rules” ↔ `generate_prediction()` 生成的 `prediction.txt`

---

### 附：常见概念小词典
- **Seq2seq**：Encoder 把源序列编码成表示；Decoder 以自回归方式生成目标序列
- **Teacher forcing**：训练时把右移后的真实目标喂给 decoder，让模型学预测下一个 token
- **Beam search**：保留多个候选前缀，通常比 greedy 更好但更慢
- **Back-translation**：用反向模型把目标语单语数据翻成源语，构造合成平行语料增强训练
