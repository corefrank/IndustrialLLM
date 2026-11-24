"""固定 token 预算的 KV Cache 开关性能对照。"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from .model import Engine, split_rows
from .schema import config, write_json


def summarize(rows):
    latencies = sorted(row["generation_seconds"] for row in rows)
    peaks = [row["peak_cuda_allocated_bytes"] for row in rows if row["peak_cuda_allocated_bytes"] is not None]
    return {
        "requests": len(rows), "mean_seconds": mean(latencies), "p50_seconds": median(latencies),
        "p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "generated_tokens_per_second": sum(row["generated_tokens"] for row in rows) / sum(latencies),
        "peak_cuda_allocated_bytes": max(peaks, default=None),
        "max_sampled_process_rss_bytes": max(row["process_rss_bytes"] for row in rows),
    }


def benchmark(args):
    import psutil

    settings = config(args.config)
    engine = Engine(settings, device=args.device, manifest=args.manifest)
    tickets = split_rows(args.data, "validation")
    texts = [row["text"] for row in tickets[:args.prompts]]
    if args.long:
        # 长输入组是受控合成文本，仅用于性能对照。
        texts = [" ".join([text] * 24) for text in texts]
    batches = [engine.encode(text) for text in texts]
    for cache in (False, True):
        for _ in range(args.warmup):
            engine.generate_tokens(batches[0], cache=cache, fixed_tokens=args.generated)

    rows, pair_matches = [], []
    process = psutil.Process()
    for round_no in range(args.repeats):
        for index, batch in enumerate(batches):
            seen = {}
            for cache in ((False, True) if (round_no + index) % 2 == 0 else (True, False)):
                if engine.device == "cuda":
                    engine.synchronize()
                    engine.torch.cuda.reset_peak_memory_stats()
                ids, seconds = engine.generate_tokens(batch, cache=cache, fixed_tokens=args.generated)
                if ids.numel() != args.generated:
                    raise RuntimeError(f"Generated {ids.numel()} tokens, expected {args.generated}")
                seen[cache] = ids.tolist()
                rows.append({
                    "round": round_no, "prompt_index": index, "cache": cache,
                    "input_tokens": batch["input_ids"].shape[-1], "generated_tokens": ids.numel(),
                    "generation_seconds": seconds, "process_rss_bytes": process.memory_info().rss,
                    "peak_cuda_allocated_bytes": (
                        engine.torch.cuda.max_memory_allocated() if engine.device == "cuda" else None
                    ),
                })
            pair_matches.append(seen[False] == seen[True])

    result = {
        "device": engine.device, "torch": engine.torch.__version__,
        "device_name": engine.torch.cuda.get_device_name() if engine.device == "cuda" else "CPU",
        "model_version": engine.version, "model_revision": settings["model"]["revision"],
        "profile": engine.profile, "load_seconds": engine.load_seconds, "warmup_per_condition": args.warmup,
        "prompts": len(batches), "repeats": args.repeats, "long_synthetic": args.long,
        "same_token_ids_pairs": sum(pair_matches), "total_pairs": len(pair_matches), "measurements": rows,
        "conditions": {
            name: summarize([row for row in rows if row["cache"] == enabled])
            for name, enabled in (("cache_off", False), ("cache_on", True))
        },
    }
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "measurements"}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=Path("configs/industrialllm.toml"))
    cli.add_argument("--data", type=Path, default=Path("data/processed"))
    cli.add_argument("--manifest", required=True, type=Path)
    cli.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cli.add_argument("--prompts", type=int, default=5)
    cli.add_argument("--warmup", type=int, default=2)
    cli.add_argument("--repeats", type=int, default=10)
    cli.add_argument("--generated", type=int, default=64)
    cli.add_argument("--long", action="store_true")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if min(args.prompts, args.warmup, args.repeats, args.generated) < 1:
        cli.error("All benchmark counts must be positive")
    benchmark(args)
