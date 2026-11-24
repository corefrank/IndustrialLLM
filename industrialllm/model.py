"""固定 revision 加载、共享推理及部署版本检查。"""

import json
import os
import time
from pathlib import Path

from .data import sha
from .schema import messages, read_jsonl

TEMPLATE_OPTIONS = {"enable_thinking": False}


class InputLimitError(ValueError):
    """Tokenizer input exceeded the configured limit."""


def split_rows(directory, split):
    path = directory / f"{split}.jsonl"
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["jsonl_sha256"][path.name] != sha(path):
        raise ValueError(f"Dataset file does not match split_manifest.json: {path}")
    return read_jsonl(path)


def load_manifest(path, settings):
    path = Path(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    current = settings["model"]
    if (saved["base_model"], saved["model_revision"]) != (current["id"], current["revision"]):
        raise ValueError("Manifest base model or revision does not match the config")
    adapter = path.parent / saved["adapter"]
    if not saved["adapter_sha256"]:
        raise ValueError("Adapter checksums are missing")
    for name, expected in saved["adapter_sha256"].items():
        if sha(adapter / name) != expected:
            raise ValueError(f"Adapter checksum mismatch: {name}")
    return adapter, saved


def load_base(settings, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use the CUDA extra or pass --device cpu for a CPU smoke test.")
    if device == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This CUDA profile requires native BF16 support.")
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("INDUSTRIALLLM_CPU_THREADS", "4")))
    model_settings = settings["model"]
    local = os.getenv("INDUSTRIALLLM_SNAPSHOT")
    source = local or model_settings["id"]
    options = {"local_files_only": True} if local else {"revision": model_settings["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    model = AutoModelForCausalLM.from_pretrained(
        source, **options, use_safetensors=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    return model, tokenizer, device


class Engine:
    def __init__(self, settings, device="auto", manifest=None):
        import torch

        self.torch = torch
        self.options = settings["model"].copy()
        adapter, saved = load_manifest(manifest, settings) if manifest else (None, None)
        started = time.perf_counter()
        self.model, self.tokenizer, self.device = load_base(settings, device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.options.update(saved["inference"])
        self.model.eval()
        self.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.version = saved["version"] if saved else f"base-{self.options['revision'][:12]}"
        self.profile = saved["profile"] if saved else "base"

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def encode(self, text, examples=()):
        batch = self.tokenizer.apply_chat_template(
            messages(text, examples), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **TEMPLATE_OPTIONS,
        )
        if batch["input_ids"].shape[-1] > self.options["max_input_tokens"]:
            raise InputLimitError("Rendered input exceeds the configured tokenizer limit")
        return batch.to(self.device)

    def generate_tokens(self, batch, *, cache=True, fixed_tokens=None, seed=None, greedy=False):
        # 质量评测使用固定种子采样。缓存对照使用贪心及固定输出长度。
        args = {"use_cache": cache, "max_new_tokens": self.options["max_new_tokens"]}
        if fixed_tokens is not None:
            args.update(min_new_tokens=fixed_tokens, max_new_tokens=fixed_tokens)
        if fixed_tokens is not None or greedy:
            args.update(do_sample=False, temperature=1.0, top_p=1.0, top_k=50)
        else:
            self.torch.manual_seed(self.options["seed"] if seed is None else seed)
            args.update(do_sample=True, temperature=self.options["temperature"],
                        top_p=self.options["top_p"], top_k=self.options["top_k"])
        self.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**batch, **args)
        self.synchronize()
        return output[0, batch["input_ids"].shape[-1]:], time.perf_counter() - start

    def predict(self, text, examples=(), seed=None, greedy=False):
        batch = self.encode(text, examples)
        ids, seconds = self.generate_tokens(batch, seed=seed, greedy=greedy)
        return {
            "raw": self.tokenizer.decode(ids, skip_special_tokens=True),
            "input_tokens": batch["input_ids"].shape[-1], "generated_tokens": ids.numel(),
            "generation_seconds": seconds,
        }
