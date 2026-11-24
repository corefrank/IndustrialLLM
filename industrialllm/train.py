"""用 TRL 训练 LoRA，核验真实参数更新，保存并重载 Adapter。"""

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from .data import sha
from .model import TEMPLATE_OPTIONS, Engine, load_base, split_rows
from .schema import config, write_json


def current_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_mask(trainer, tokenizer, samples, limit):
    """用所有样本检查模板边界，再用真实 collator 验证一个有 padding 的 batch。"""
    lengths = []
    for row in samples:
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                               **TEMPLATE_OPTIONS)
        total = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True,
                                              **TEMPLATE_OPTIONS)
        if total[:len(prompt)] != prompt or len(total) > limit or tokenizer.eos_token_id not in total[len(prompt):]:
            raise ValueError(f"Unexpected prompt boundary, EOS or truncated target for {row['id']}")
        lengths.append(len(total))

    dataset = trainer.train_dataset
    ordered = sorted(range(len(dataset)), key=lambda index: len(dataset[index]["input_ids"]))
    examples = [dataset[ordered[0]], dataset[ordered[-1]]]
    batch = trainer.data_collator(examples)
    supervised_counts = []
    for labels, ids, attention, example in zip(batch["labels"], batch["input_ids"],
                                               batch["attention_mask"], examples, strict=True):
        supervised = labels.ne(-100)
        expected = example["completion_mask"] + [0] * (len(labels) - len(example["completion_mask"]))
        if supervised.tolist() != [bool(value) for value in expected] or (supervised & attention.eq(0)).any():
            raise ValueError("Observed labels do not agree with completion and padding masks")
        decoded = tokenizer.decode(ids[supervised], skip_special_tokens=True)
        if not isinstance(json.loads(decoded), dict):
            raise ValueError("No complete JSON target in supervised tokens")
        supervised_counts.append(int(supervised.sum()))
    return {
        "inspected_rows": len(samples), "minimum_tokens": min(lengths), "maximum_tokens": max(lengths),
        "collator_supervised_tokens": supervised_counts,
    }


def train(args):
    import psutil
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    settings = config(args.config)
    hparams = settings["training"]
    smoke = args.profile == "smoke"
    seed = settings["data"]["seed"]
    set_seed(seed)
    train_rows = split_rows(args.data, "train")
    validation_rows = split_rows(args.data, "validation")
    if smoke:
        train_rows = train_rows[:settings["smoke"]["train"]]
        validation_rows = validation_rows[:settings["smoke"]["validation"]]

    # 输出目录必须为空，避免不同运行的检查点或日志被覆盖。
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if any(args.run_dir.iterdir()):
        raise ValueError("Run directory is not empty. Use a new --run-dir.")
    if args.wandb:
        os.environ["WANDB_MODE"] = "offline"
        os.environ.setdefault("WANDB_PROJECT", "IndustrialLLM")
        os.environ["WANDB_DIR"] = str(args.run_dir.resolve())

    model, tokenizer, device = load_base(settings, args.device)
    model.config.use_cache = False
    lora = LoraConfig(
        task_type="CAUSAL_LM", r=hparams["rank"], lora_alpha=hparams["alpha"],
        lora_dropout=hparams["dropout"], target_modules=hparams["target_modules"], bias="none",
    )
    model = get_peft_model(model, lora)
    # Adapter 元数据引用公开底座，便于换机使用本地快照训练得到的权重。
    model.peft_config["default"].base_model_name_or_path = settings["model"]["id"]
    model.peft_config["default"].revision = settings["model"]["revision"]
    trainable, total = model.get_nb_trainable_parameters()
    weights = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}
    if not weights or any("lora_" not in name for name in weights):
        raise RuntimeError("Unexpected set of trainable parameters")

    steps = args.steps or (settings["smoke"]["max_steps"] if smoke else -1)
    # 限步运行只用于验收。标准训练按每 100 步验证和保存，保留最佳 checkpoint。
    periodic = not smoke and args.steps is None
    interval = settings["standard"]["checkpoint_steps"]
    training = SFTConfig(
        output_dir=str(args.run_dir / "checkpoints"), seed=seed, data_seed=seed,
        use_cpu=device == "cpu", bf16=device == "cuda",
        max_length=hparams["max_length"], packing=False, completion_only_loss=True,
        assistant_only_loss=False, eos_token=tokenizer.eos_token,
        per_device_train_batch_size=args.batch_size or hparams["batch_size"],
        per_device_eval_batch_size=args.batch_size or hparams["batch_size"],
        gradient_accumulation_steps=args.accumulation or hparams["gradient_accumulation"],
        gradient_checkpointing=args.gradient_checkpointing, learning_rate=hparams["learning_rate"],
        num_train_epochs=settings["standard"]["epochs"], max_steps=steps,
        eval_strategy="steps" if periodic else "no", save_strategy="steps" if periodic else "no",
        eval_steps=interval, save_steps=interval, save_total_limit=2,
        load_best_model_at_end=periodic, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1 if not periodic else 10, prediction_loss_only=True, optim="adamw_torch",
        dataloader_num_workers=0, report_to=["wandb"] if args.wandb else [],
        run_name=args.run_dir.name, disable_tqdm=True,
    )
    trainer = SFTTrainer(model=model, args=training, processing_class=tokenizer,
                         train_dataset=Dataset.from_list(train_rows),
                         eval_dataset=Dataset.from_list(validation_rows))
    mask_check = verify_mask(trainer, tokenizer, train_rows + validation_rows, hparams["max_length"])
    print(json.dumps({"loss_mask": mask_check, "trainable": trainable, "total": total}), flush=True)

    process = psutil.Process()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = trainer.train()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    updated = sum(not torch.equal(weights[k], p.detach().cpu()) for k, p in model.named_parameters() if k in weights)
    if not updated or trainer.state.global_step == 0:
        raise RuntimeError("No Adapter parameter changed in training")
    validation = trainer.evaluate()
    adapter = args.run_dir / "adapter"
    trainer.save_model(str(adapter))
    trainer.state.save_to_json(str(args.run_dir / "trainer_state.json"))
    if args.wandb:
        import wandb

        wandb.finish()
    run_profile = "smoke" if smoke else "pilot" if args.steps else "standard"
    report = {
        "profile": run_profile, "device": device, "torch": torch.__version__, "python": platform.python_version(),
        "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.processor() or "CPU",
        "train_rows": len(train_rows), "validation_rows": len(validation_rows), "global_steps": trainer.state.global_step,
        "epoch": trainer.state.epoch, "trainable_parameters": trainable, "total_parameters": total,
        "trainable_ratio": trainable / total, "changed_lora_tensors": updated, "mask_check": mask_check,
        "train_seconds": elapsed, "peak_cuda_allocated_bytes": peak,
        "rss_after_training_bytes": process.memory_info().rss,
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "metrics": result.metrics | validation, "training_args": training.to_dict(),
    }
    if device == "cpu" and platform.system() == "Linux":
        import resource

        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    manifest = {
        "version": f"{run_profile}-{sha(adapter / 'adapter_model.safetensors')[:12]}",
        "profile": run_profile, "base_model": settings["model"]["id"],
        "model_revision": settings["model"]["revision"], "git_commit": current_commit(),
        "data_manifest_sha256": sha(args.data / "split_manifest.json"), "config_sha256": sha(args.config),
        "lora": {key: hparams[key] for key in ("rank", "alpha", "dropout", "target_modules")},
        "seed": seed, "training_seconds": elapsed, "training_metrics": report["metrics"],
        "adapter": "adapter", "adapter_sha256": {name: sha(adapter / name) for name in (
            "adapter_config.json", "adapter_model.safetensors"
        )},
        "inference": {k: v for k, v in settings["model"].items() if k not in ("id", "revision")},
    }
    write_json(args.run_dir / "training_report.json", report)
    path = args.run_dir / "manifest.json"
    write_json(path, manifest)

    # 释放训练模型后按真实部署路径重载，验证保存产物。
    del trainer, model, weights
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    engine = Engine(settings, device=device, manifest=path)
    output = engine.predict(validation_rows[0]["text"], greedy=True)
    if output["generated_tokens"] == 0:
        raise RuntimeError("Reloaded Adapter generated no tokens")
    write_json(args.run_dir / "reload.json", {"version": engine.version, "generation": output})
    print(json.dumps({"manifest": str(path), "steps": report["global_steps"],
                      "updated_tensors": updated, "validation": validation,
                      "generated_tokens_after_reload": output["generated_tokens"]}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=Path("configs/industrialllm.toml"))
    cli.add_argument("--data", type=Path, default=Path("data/processed"))
    cli.add_argument("--profile", default="smoke", choices=("smoke", "standard"))
    cli.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    cli.add_argument("--run-dir", required=True, type=Path)
    cli.add_argument("--steps", type=int, help="Optional shorter run for measurement (pilot)")
    cli.add_argument("--batch-size", type=int)
    cli.add_argument("--accumulation", type=int)
    cli.add_argument("--gradient-checkpointing", action="store_true")
    cli.add_argument("--wandb", action="store_true", help="Record to a local offline W&B run")
    train(cli.parse_args())
