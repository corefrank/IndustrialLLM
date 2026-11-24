# IndustrialLLM

把英文工业维修短文本提取为五类实体。使用 MaintIE、Qwen3‑0.6B、PEFT LoRA 和 TRL，比较 Zero-shot、Few-shot 和 LoRA，再用 Transformers 测 KV Cache，最后通过 FastAPI 提供抽取服务。数据来自公开语料，业务场景参考 Bosch Shopfloor Agent 中的 Equipment Ticket Agent。接口输出可审核的草稿。

- [七阶段中文学习指南](docs/learning.md)
- [项目架构与业务取舍](docs/architecture.md)
- [面试问题、讲稿与 LaTeX](docs/interview.md)
- [MaintIE 原始数据审计](docs/maintie-audit.md)
- [运行与验收记录](acceptance/README.md)

## Windows 快速开始

从项目根目录运行。Windows 原生是 RTX 4070 Laptop 8 GiB 的主路径。使用 Python 3.12 和 uv，训练使用 BF16，LoRA 底座保持冻结。建议模型、uv 缓存和产物所在磁盘可用空间至少约 12 GiB，首次安装和下载会占用数 GB。配置集中在 [configs/industrialllm.toml](configs/industrialllm.toml)。

### 1. 安装和数据准备

```powershell
$env:PYTHONUTF8 = "1"
uv sync --locked --python 3.12 --extra cuda --extra model --extra train --extra report --extra tracking
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported(including_emulation=False))"
New-Item -ItemType Directory -Force outputs/logs | Out-Null
uv run --no-sync python -m industrialllm.data 2>&1 | Tee-Object outputs/logs/data.log
uv run --no-sync ruff check .
uv run --no-sync pytest -q
```

期望 PyTorch 为 2.6.0+cu124，CUDA 与原生 BF16 均为 `True`。从 MaintIE 固定 commit 下载两份约 5.4 MB 原始 JSON，核验 SHA-256，生成 `data/processed` 下的 train、validation、test、排除清单和审计 JSON。通过条件为 **2,400 / 250 / 400** 条和指定口径下零跨组重复。若 raw 下载遇到网络问题，脚本会尝试 Git。也可将官方两份文件放入 `data/raw`，脚本仍核验哈希。pytest 中的真实数据断言在数据准备后执行。

### 2. 基线与 Smoke

```powershell
uv run --no-sync python -m industrialllm.evaluate --device cuda --split validation --mode all --limit 24 --out outputs/baseline-validation 2>&1 | Tee-Object outputs/logs/baseline.log
uv run --no-sync python -m industrialllm.train --device cuda --profile smoke --run-dir outputs/smoke --wandb 2>&1 | Tee-Object outputs/logs/smoke.log
```

基线作为验证阶段取样运行 Zero-shot、Few-shot，输出逐条 JSONL、`metrics.json`、CSV 和图。给定 3 条 few-shot 示例全部来自训练集。首次模型加载会下载约 1.52 GB 文件。Smoke 用 32 条训练样本、16 条验证样本、4 个真实优化步检查更新、保存和重载。看 `training_report.json`、`trainer_state.json`、`manifest.json`、`reload.json`，以及 `changed_lora_tensors > 0`。若 Smoke 失败，先检查安装和 Loss Mask，服务结果在后续阶段独立验证。W&B 使用 offline 模式。

### 3. 硬件 Pilot 和正式训练

可选 Pilot 使用正式数据和相同 batch，但只运行 30 个优化步，记录实际训练秒数、显存和独立 Adapter。训练计时不含结束后的独立验证和重载。

```powershell
uv run --no-sync python -m industrialllm.train --device cuda --profile standard --steps 30 --run-dir outputs/pilot 2>&1 | Tee-Object outputs/logs/pilot.log
```

正式命令：

```powershell
uv run --no-sync python -m industrialllm.train --device cuda --profile standard --run-dir outputs/standard --wandb 2>&1 | Tee-Object outputs/logs/standard.log
```

训练输入是冻结的 Silver 训练集和 Gold 验证集。默认 `2 × 4` 有效 batch 8、两轮、约 600 optimizer steps。每 100 步保存并评估验证 Loss，保留最佳检查点与最终选中的 Adapter。用 Pilot 的训练时间乘约 20 再加上验证、保存和重载开销安排正式训练。初始排期约 20–75 分钟。显存不足时改用 `--batch-size 1 --accumulation 8` 后重跑 Pilot，仍不足时加 `--gradient-checkpointing`。每次运行用一个新的 `--run-dir`。脚本拒绝覆盖已有产物。

### 4. 正式效果评测

先确认验证集 Prompt、生成设置和评分口径已冻结。再运行：

```powershell
uv run --no-sync python -m industrialllm.evaluate --device cuda --split test --mode all --manifest outputs/standard/manifest.json --out outputs/final-test 2>&1 | Tee-Object outputs/logs/final-test.log
```

正式评测不要传 `--limit`。三组各应有 400 条明细。查看 `metrics.json`、`comparison.csv`、`comparison.png`、`failure_examples.json`。完整三组测试会生成 `outputs/standard/release_manifest.json`，含正式指标和报告哈希。估算 15–45 分钟，可用验证集实测校准。只有你实际运行并读到输出后，才能把数值用于简历。

### 5. KV Cache 和 Windows 本地服务

```powershell
uv run --no-sync python -m industrialllm.benchmark --device cuda --manifest outputs/standard/release_manifest.json --out outputs/kv-short.json 2>&1 | Tee-Object outputs/logs/kv-short.log
uv run --no-sync python -m industrialllm.benchmark --device cuda --manifest outputs/standard/release_manifest.json --long --out outputs/kv-long.json 2>&1 | Tee-Object outputs/logs/kv-long.log
$env:INDUSTRIALLLM_MANIFEST = "outputs/standard/release_manifest.json"
$env:INDUSTRIALLLM_DEVICE = "cuda"
uv run --no-sync uvicorn industrialllm.api:app --host 127.0.0.1 --port 8000 --workers 1
```

默认各条件 5 个输入 × 10 次重复，严格固定生成 64 token。短文本与拼接长输入有各自结果。看 JSON 内每次测量、P50/P95、tokens/s 和峰值显存。服务进程启动一次即加载模型。另开 PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/models
$body = @{text = "replace faulty horn"} | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:8000/extract -Method Post -ContentType "application/json" -Body $body
```

如果只想先演示随包附带的已验收 CPU Smoke Adapter，把 Manifest 改为 `acceptance/artifacts/cpu-smoke/manifest.json`，底座仍由同一固定 revision 加载。

也可访问 [Swagger UI](http://127.0.0.1:8000/docs)。输出不符合 Schema 或包含原文外实体时，接口用 502 表达模型输出失败，不交付草稿。空文本、过长文本返回 422。要在完整评测之前演示真实模型，可显式将环境变量指向 `outputs/smoke/manifest.json`，`/models` 会标明 `smoke`。

### 6. 人工审核、版本回退、Docker

Python 工单流程接受真实 API 的通过结果生成待审核草稿。先查看 `draft.json`，需要时编辑候选再人工审核，最终审批仍验证 Schema 和原文依据。失败记录不进入待审核状态。

```powershell
uv run --no-sync python -m industrialllm.workflow draft --text "replace faulty horn" --out outputs/draft.json
uv run --no-sync python -m industrialllm.workflow review --draft outputs/draft.json --decision approved --reviewer Nan --out outputs/reviewed.json
```

服务验证基础模型 ID、revision 和 Adapter 文件哈希。回退时，把 `INDUSTRIALLLM_MANIFEST` 指向上一个已验收版本并重启。不要只复制 `manifest.json`，同目录下的 `adapter/` 一并保留。版本记录内 Git commit 来自本地仓库。ZIP 解压后的目录若没有 Git 历史，字段如实为 null。

Docker Desktop 以下命令使用 CPU 镜像，挂载模型与正式 Adapter，容器离线加载：

```powershell
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',local_dir='outputs/hf-model',allow_patterns=['*.json','*.safetensors'])"
docker build -t industrialllm:cpu .
$weights = (Resolve-Path outputs/hf-model).Path
$artifacts = (Resolve-Path outputs/standard).Path
docker run --rm -p 127.0.0.1:8001:8000 --mount "type=bind,source=$weights,target=/model,readonly" --mount "type=bind,source=$artifacts,target=/artifacts,readonly" -e INDUSTRIALLLM_SNAPSHOT=/model industrialllm:cpu
```

用端口 8001 重做三接口验证。若用 Smoke 目录演示，把挂载位置换成 Smoke 目录，并传 `-e INDUSTRIALLLM_MANIFEST=/artifacts/manifest.json`。只在容器实际启动并通过后记录 Docker 验收。GitHub Actions 在你将项目推送到已授权的仓库后运行。

### WSL2 与纯 CPU

WSL2 从 Linux 文件系统下运行，同样使用 CUDA extra，前提是 `nvidia-smi` 和 PyTorch CUDA 检查在 WSL2 内通过。命令主体与上文相同，环境变量用 `export KEY=value`，日志用 `set -o pipefail` 配合 `2>&1 | tee file`；原生 Windows 环境的 .venv 不能跨过去使用。

纯 CPU 把安装命令的 `--extra cuda` 换成 `--extra cpu`，运行命令改为 `--device cpu`。CPU 路径使用 FP32，可选 `INDUSTRIALLLM_CPU_THREADS` 控制线程。CPU 适合 Smoke、少量评测和接口验证。严格离线时使用预先下载的固定 revision，设置 `INDUSTRIALLLM_SNAPSHOT` 指向可信目录。

## 数据、模型与参考代码

- [MaintIE 上游与数据许可](https://github.com/nlp-tlp/maintie/tree/825e7710e4129598da997276a38b3cb40b200bc2)，发行语料 MIT，版权与许可保留于本项目文档。
- [MaintIE 论文](https://aclanthology.org/2024.lrec-main.954/)。
- [Qwen3‑0.6B 固定 revision](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)，Apache 2.0。
- [nanochat 参考版本](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd)。
- [Bosch Shopfloor Agent](https://www.bosch.com/stories/agentic-ai-manufacturing-production/)。
