# 七阶段中文学习指南

每阶段先读表中的文件，再做一个练习。用 README 的命令生成该阶段的产物作为验收。代码入口使用 `uv run --no-sync python -m industrialllm.<模块>`，执行目录为项目根目录。

| 阶段 | 应先阅读的函数 | 验收产物 | 动手练习 |
|---|---|---|---|
| 1. 数据工程 | `data.convert`、`data.prepare`、`schema.Extraction` | 2,400 / 250 / 400 条 JSONL、哈希清单、排除清单 | 跟踪 `silver:0` 的 token 跨度，手写五字段 JSON，对照审计示例 |
| 2. 模型与基线 | `model.load_base`、`Engine.encode`、`evaluate.run` | 验证集的 Zero-shot 和 Few-shot JSONL | 打印同一条工单的两种 Prompt，数输入 tokens 并解释增量 |
| 3. SFT 与 LoRA | `train.train`、`train.verify_mask` | Loss Mask 检查、训练日志、Adapter 和重载结果 | 将 rank 8 临时改为 4，用新运行目录测可训练参数量，再恢复配置 |
| 4. 效果评测 | `evaluate.score`、`summarize`、`report` | 三组指标、CSV、对比图、典型失败案例 | 手算某类一个 TP、一个 FN、一个 FP 的 F1，再在单测中验证 |
| 5. 推理优化 | `Engine.generate_tokens`、`benchmark.benchmark` | Cache 开关、固定 token 数、延迟分位与资源统计 | 同一 Adapter 分别跑短、长两组，解释速度和显存变化 |
| 6. 部署与审核 | `api.create_app`、`workflow.make_draft`、`workflow.review` | 三个真实 API 调用和工单审核记录 | 用一条空文本验证 422，再请求一条真实工单并查看审核文件 |
| 7. MLOps 与面试 | `model.load_manifest`、`train.current_commit`、CI 文件 | 可追溯版本、回退复验、真实 CI 状态 | 以两个真实 Manifest 启动服务各一次，核对 `/models` 返回版本 |

## 1. 数据工程与字段含义

输入 `<id> replace faulty horn` 的 token 序列是 `["<id>","replace","faulty","horn"]`。维护行为的 `[1:2]` 得到 `replace`，状态 `[2:3]` 得到 `faulty`，对象 `[3:4]` 得到 `horn`。关系里的 head、tail 是实体数组下标。输出保留对象、活动、状态、过程、属性五个根类。Gold 的多级标签沿路径截取根名，Silver 顶层直接映射。关系不进入训练目标。

执行顺序为 `acquire → convert → group_near → select_gold → prepare 输出`。`acquire(raw)` 检查真实下载文件的 SHA-256。`convert(row,item_id)` 以 span 做标签映射并生成含标准答案的对话训练样本。`group_near(keys)` 给同长度、至少五词、一个位置词不同的文本分组。`select_gold(groups,rows,size)` 按组抽取验证和测试，检查五类覆盖。`prepare(raw,output,settings)` 给出统计、排除清单和 JSONL。短于五词的样本做精确归一匹配，语义近义换词或任意多词变形超出这版检测范围。

必须掌握 token跨度、原文对齐、标签投影及测试隔离。可暂时跳过近重复索引和并查集细节，先用两条只换一个词的工单解释为什么要进同一组。

## 2. Tokenizer、Chat Template 与基线

Tokenizer 将字符序列转换为模型词表 ID。Chat Template 先把 system、user、assistant 角色用模型的控制标记连接，再执行 tokenize。维护数据中的 `tokens` 用于标注定位，Qwen 的 `input_ids` 用于运行模型。两者划分不同。

`load_base(settings,device)` 输出 Qwen 模型、tokenizer 和设备。`Engine.encode(text,examples)` 输出两个主要输入张量：`input_ids` 表示 token ID，`attention_mask` 标明哪些位置是实际 token。Few-shot 以真实历史对话消息形式增加三组 Silver 样例，原任务定义保持一致。读取基线结果时，先查 `json_valid` 再看 `schema_valid` 和实体漏抽。可暂时跳过 tokenizer 词表学习细节。

Transformer 每一层用 self-attention 把当前位置的 Query 与允许访问的 Key 做打分，再把 Value 按权重求和。因果 mask 屏蔽未来 token。RoPE 对 Q/K 分量施加依赖位置的旋转，相对位置进入 Q·K 点积。你此前的 Time‑RoPE 与这里都用位置影响 attention，只是时间建模与文本序列的任务不同。Qwen 使用分组查询注意力，同组的多个 Q 共享 K/V 头。

## 3. SFT、LoRA、PEFT、TRL

SFT 是在预训练模型上用带标准答案的样本做有监督微调。LoRA 是参数更新方式：基础矩阵 $W$ 冻结，在所选线性层学习 $\Delta W = (\alpha/r)BA$。较小的 rank $r$ 使更新只包含两个小矩阵。对输入维度 $d_{in}$、输出维度 $d_{out}$ 的矩阵，增量参数为 $r(d_{in}+d_{out})$。Alpha 控制缩放，dropout 提供正则化。

- `q_proj`、`k_proj`、`v_proj` 产生注意力所用的 Q、K、V，`o_proj` 合并多头输出并投影回隐空间。
- Forward 输出每个位置的下一个 token logits，Cross-Entropy 对标准 token 取负对数概率。Backward 求出 LoRA 参数的梯度，Optimizer Step 更新这些参数。
- 梯度累积把 4 个 microbatch 的梯度合在一次更新里，RTX 4070 的默认配置是每个 microbatch 2 条。梯度检查点通过重算早先激活减少内存，占用更多训练时间。
- 训练样本是 assistant 的 JSON 答案。Trainer 把输入 ID 作为对齐后的候选 labels，把 Prompt 和 padding 的 labels 设置为 `-100`。损失忽略这些位置，模型仍能通过注意力阅读 Prompt。有效标签包括应生成的结束标记。
- `verify_mask(trainer,tokenizer,samples,limit)` 检查训练和验证的模板前缀、长度和 EOS，再检查带 padding 的真实 collator batch，解码监督 token 得到 JSON。
- `train(args)` 先用 PEFT 注入四类投影层，再交给 TRL SFTTrainer。训练日志记录 Loss、步数、可训练参数和真实权重变化。检查点可包含训练状态与优化器等用于续训的文件。Adapter 只携带低秩参数与配置，部署仍需要原底座。发生中断时本版 CLI 默认开启一个新运行，如需续训可在代码中把核验过的 checkpoint 路径交给 Trainer 的 `train(resume_from_checkpoint=...)` 并使用原配置。
- 自回归推理的 KV Cache 只用于推理复用。监督训练在一次前向中并行处理整段，保留逐步生成用的缓存会增加内存，且会与梯度检查点路径产生冲突。

必须掌握怎样证明更新真的发生、怎样读 Loss Mask、怎样重载 Adapter。可暂时跳过 PEFT 注入的 hook 细节、Trainer 的回调调度与 Arrow 内部表示。

## 4. 评测与失败分析

`score(row,result)` 输出逐条 JSON 格式、Schema、实体计数与错误标签。`summarize(rows)` 汇总所有样本。`report(output,summaries,examples)` 输出 CSV、柱状图和失败例。Micro 把所有类别的 TP/FP/FN 累加再计算，Macro 先对每类算 F1 再平均。比如真值是 `valve` 和 `hose`，模型抽了 `valve` 和 `filter`，TP、FP、FN 都为 1，Precision、Recall、F1 均为 0.5。

JSON 无法解析或 Schema 失败时，保留整条并当作零预测。格式合规且词边界匹配只说明输出可解析并有原文证据，仍可能漏抽、类别错误、跨度过宽。长文本与缩写标签是复盘提示。用它们定位样本后检查全文，不能仅凭标签判断错误根因。临时可跳过图表绘制实现。

## 5. Prefill、Decode 与性能

首次对完整 Prompt 做前向称为 Prefill。Decode 每步生成一个 token。不开 Cache 时，每次重新处理增长的序列；开 Cache 时，保存各层旧位置的 K/V，新位置只追加并读取之前缓存。显存代价随层数、K/V 头数、头维度、输入和生成长度、batch 与 dtype 增长。Q/K/V 的模型权重与 KV Cache 是两套不同的张量。

`benchmark.benchmark(args)` 先构建相同输入 token，再 warm-up，按轮次交替 cache 开关。`generate_tokens(batch,cache,fixed_tokens)` 在 CUDA 前后同步、贪心生成并验证实际 token 数。`summarize(rows)` 按单请求计时算统计量。完整生成延迟不等同于 TTFT。单请求生成 tokens/s 也不代表并发吞吐。可暂时跳过 fused attention 内核细节。

## 6. FastAPI、审核和生命周期

`create_app(engine_factory)` 让真实服务与 Mock 测试复用同一 API。启动生命周期加载引擎。请求经 Pydantic 验证后调用模型，再经严格 JSON、Schema 和原文校验。只有通过才返回可审阅的 `result`。`/models` 暴露底座和 Adapter 版本。`workflow.make_draft(text,url)` 通过本地 HTTP 获得结果，写待审核或失败草稿。`review(draft,decision,reviewer)` 写明审核人、决定和时间，保留原草稿。

单 worker、单生成锁针对个人机器，避免多份模型占显存。一台企业服务可有多个业务客户端复用统一 JSON Schema，服务端保持版本清晰。Docker 负责携带运行依赖，权重和 Adapter 单独挂载。可暂时跳过网关、权限和集群部署。

## 7. MLOps 和面试准备

Manifest 用相同底座与 revision 验证 Adapter 适配性，存原始 split manifest 哈希、版本、训练数据与配置关联。正式测试后产生附质量指标和报告哈希的发布 Manifest。回退只需选择已有的已验收版本重新启动。W&B offline 保留本地运行信息，CI 普通任务通过小型数据样例与 Mock 测逻辑，真实模型运行作为独立证据。具体完成状态见验收记录。面试中说清已运行的实验条件和未验证范围。
