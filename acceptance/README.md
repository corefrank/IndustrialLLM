# 验收记录

本目录记录 Work 中的实际执行和待在用户 Windows GPU 上完成的验收。Work 使用 Linux、Python 3.12.14、PyTorch 2.6.0+cpu 与真实的 Qwen3‑0.6B 固定 revision。缓存固定来源文件的哈希见 `00-model-sha256.txt`。CUDA、Windows 训练、Docker Desktop 和远端 GitHub Actions 没有在此环境执行。

## 运行状态

| 阶段 | Work 验收 | 证据 |
|---|---|---|
| 1. 真实数据 | 真实下载、哈希校验、五类 Schema 验证、2,400 / 250 / 400 划分；重复运行六个文件逐字节一致 | `01-data.log`、`01-reproducibility.log`、`artifacts/data/` |
| 2. 基线 | 用真实底座在 Gold 验证集前 12 条运行 Zero-shot 和 Few-shot | `02-04-cpu-eval.log`、`artifacts/evaluation/` |
| 3. 训练 | CPU FP32 Smoke 完成四步，完整检查训练与验证模板，记录并重载 LoRA Adapter | `03-template-full.log`、`03-cpu-smoke.log`、`artifacts/cpu-smoke/` |
| 4. 效果 | 在同一验证样本上完成三组真实推理和小规模对照 | `metrics.json`、各模式 JSONL、CSV、PNG、失败案例 |
| 5. 性能 | 真实 Cache 对照和资源统计见下节 | `05-kv-*.log`、`artifacts/performance/` |
| 6. 部署与审核 | 使用 Docker 定义的最小 CPU 依赖子集，在 Work 直接启动 Uvicorn；真实结果见下节 | `06-runtime-deps.log`、`06-api-*.log`、`artifacts/deployment/` |
| 7. MLOps | W&B offline 审验、Manifest 哈希校验；锁文件、Ruff、pytest、actionlint 检查 | `00-dependencies.log`、`07-*.log`、`artifacts/cpu-smoke/` |

## 数据与训练实测

原始 1,076 条 Gold 和 7,000 条 Silver 经过 SHA 校验。排除 7 条组内精确重复、5 条原文与标注跨度不一致、7 条 Gold 与 Silver 精确重叠、131 条符合本版单位置替换规则的 Gold 与 Silver 近重复 Silver 记录。重新生成的两个输出目录逐字节一致。训练及验证的 2,650 条均通过真实 Qwen tokenizer 的模板前缀、长度和 EOS 检查。实际训练样本最大 155 tokens，验证最大 150 tokens。

CPU Smoke 处理 32 条训练和 16 条验证数据，执行 4 次优化更新。LoRA 可训练参数 2,293,760，总参数 598,343,680，可训练比例约 0.383%。224 个 LoRA 参数张量至少有一个元素实际更新。训练约 60.3 秒，随后验证得到 loss 0.5143，峰值进程 RSS 约 4.69 GiB。重载保存的 Adapter 后真实生成 34 tokens。Smoke 结论限于框架、梯度、保存和部署路径。

最初训练在导入阶段暴露了 NumPy 1.26.4 与自动解析的 PyArrow 26.0.1 的版本冲突。修复方式为在训练 extra 固定 PyArrow 22.0.0 并重建锁文件，同一运行环境随后通过导入及实际训练。保留了前后日志。

## 小规模效果诊断

| 模式 | 样本数 | JSON 有效率 | Schema 通过率 | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 12 | 0% | 0% | 0.000 | 0.000 |
| Few-shot | 12 | 100% | 100% | 0.455 | 0.299 |
| LoRA CPU Smoke | 12 | 100% | 100% | 0.429 | 0.348 |

Zero-shot 的十二条响应都被单层 Markdown 代码围栏包裹，按严格原始响应指标判为格式失败。另用已有响应去掉单层围栏做诊断，不改动主评测口径：Zero-shot 的 Schema 率为 100%，Micro-F1 为 0.245。Few-shot 的原始 Schema 与 Micro-F1 均优于或等同于这次 Smoke LoRA。此次 LoRA 使用小数据和四步更新，在 12 条验证短文本上的表现不代表正式训练或冻结测试集的水平。`failure_examples.json` 还保存了错类、漏抽、跨度候选和原文外实体的实际例子。

## KV Cache 实际结果

| 输入组 | 固定生成 tokens | 每侧请求数 | Cache 关平均 / P95 | Cache 开平均 / P95 | 关 / 开 tokens/s |
|---|---:|---:|---:|---:|---:|
| 真实短输入 | 24 | 9 | 14.71 / 15.53 秒 | 5.12 / 5.26 秒 | 1.63 / 4.69 |
| 合成长输入 | 16 | 4 | 22.02 / 25.95 秒 | 4.66 / 5.12 秒 | 0.73 / 3.44 |


短输入 9 对和长输入 4 对的生成 token 序列均一致。每组使用同一 CPU 模型和参数，预热后交替测量。峰值 CUDA 显存字段为空，因为设备是 CPU。JSON 保留每次测量及进程 RSS。短组和长组的生成预算不同，速度差异只在各组内部比较。计时包含完整生成，没有测量 TTFT 或并发吞吐。

## 真实 FastAPI 与审核结果

用仅安装 CPU 推理依赖的隔离环境启动 Uvicorn，加载真实 Qwen 底座与四步 LoRA Adapter。`/health`、`/models`、`/extract` 均收到真实 HTTP 200，空白请求返回 422。抽取返回非空、符合 Schema 且有原文依据的五字段 JSON，版本为 `smoke-771068477b73`。随后通过同一 API 生成待审核草稿，并执行一次人工审批演示。验收响应、原始草稿和审阅记录见 `artifacts/deployment/`。原文匹配可以确认实体有出处，实体分类和覆盖仍需参考 Gold 与人工审核。

## 本地 CI 与版本

- 真实数据已生成的开发目录执行 Ruff 和 pytest，13 例通过。
- 隔离的无训练依赖、无数据副本中执行与 CI 相同的依赖安装、Ruff 和 pytest，12 例通过，真实语料断言按预期跳过。FastAPI Mock 验证的是接口逻辑。
- actionlint 1.7.7 检查 GitHub Actions YAML 通过。远端 runner 需推到用户仓库后运行。
- Dockerfile 已提供。Work 没有 Docker 工具，未构建镜像，也未运行容器。
- Adapter、W&B offline 文件和训练状态已附在 `artifacts/cpu-smoke`。底座来自公开上游，按 README 下载。Smoke Manifest 可以单独选中用于演示。
- CUDA Standard、完整 400 条测试集对照、GPU Cache 与显存、Docker 镜像和远端 CI 仍待本地／远端验收。

## Work 中执行的命令

以下从项目根目录执行。Work 将固定 Qwen 快照置于 `/tmp/industrial-qwen`，证据中的适配器和报告可以独立查看。普通复现使用 README 的项目默认路径。

```bash
uv lock
uv sync --locked --extra cpu --extra model --extra train --extra report --extra tracking
uv run --no-sync python -m industrialllm.data
uv run --no-sync python -m industrialllm.data --out outputs/reproducibility-data
export INDUSTRIALLLM_SNAPSHOT=/tmp/industrial-qwen
export INDUSTRIALLLM_CPU_THREADS=6
uv run --no-sync python -m industrialllm.train --device cpu --profile smoke --run-dir outputs/work-cpu-smoke --wandb
uv run --no-sync python -m industrialllm.evaluate --device cpu --split validation --limit 12 --manifest outputs/work-cpu-smoke/manifest.json --out outputs/work-validation-smoke
uv run --no-sync python -m industrialllm.benchmark --device cpu --manifest outputs/work-cpu-smoke/manifest.json --prompts 3 --repeats 3 --warmup 1 --generated 24 --out outputs/work-kv-short.json
uv run --no-sync python -m industrialllm.benchmark --device cpu --manifest outputs/work-cpu-smoke/manifest.json --long --prompts 2 --repeats 2 --warmup 1 --generated 16 --out outputs/work-kv-long.json
export UV_PROJECT_ENVIRONMENT=/tmp/industrial-infer-venv
uv sync --locked --no-dev --extra cpu --extra model
export INDUSTRIALLLM_DEVICE=cpu INDUSTRIALLLM_MANIFEST=outputs/work-cpu-smoke/manifest.json
uv run --no-sync uvicorn industrialllm.api:app --host 127.0.0.1 --port 8765 --workers 1
```

API 客户端在本地调用 `/health`、`/models`、`/extract`，并用工作流 CLI 录入审核结果。正式 Windows 或 WSL2 GPU 训练命令见根目录 README。
