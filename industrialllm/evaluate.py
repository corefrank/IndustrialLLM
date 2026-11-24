"""相同样本上运行 Zero-shot、Few-shot 和 LoRA，保留逐条结果。"""

import argparse
import gc
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from pydantic import ValidationError

from .data import sha
from .model import Engine, load_manifest, split_rows
from .schema import FIELDS, Extraction, config, grounded, normalize, strict_json, write_json


def entities(value):
    return {(field, normalize(span)) for field in FIELDS for span in value.get(field, [])}


def rates(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def score(row, result):
    raw = result["raw"]
    output, parsed, errors = {}, None, set()
    try:
        parsed = strict_json(raw)
    except ValueError:
        errors.add("invalid_json")
    else:
        try:
            output = Extraction.model_validate(parsed).model_dump()
        except ValidationError:
            errors.add("schema_mismatch")

    expected, predicted = entities(row["target"]), entities(output)
    missed, wrong = expected - predicted, predicted - expected
    for field, span in wrong:
        errors.add("wrong_entity")
        if not grounded(span, row["text"]):
            errors.add("ungrounded_entity")
        if any(span == gold and field != gold_field for gold_field, gold in missed):
            errors.add("wrong_category")
        if any(span != gold and (grounded(span, gold) or grounded(gold, span))
               for gold_field, gold in missed if gold_field == field):
            errors.add("boundary_candidate")
    if missed:
        errors.add("missing_entity")
    if errors and re.search(r"<(?:id|num|date|sensitive)>|\b[A-Z]{2,}\b", row["text"]):
        errors.add("contains_placeholder_or_abbreviation")
    if errors and (len(expected) >= 4 or len(row["text"]) > 50):
        errors.add("longer_or_multi_entity_input")
    matches = expected & predicted
    return {
        "id": row["id"], "text": row["text"], "target": row["target"], **result,
        "parsed": parsed, "prediction": output or None, "json_valid": "invalid_json" not in errors,
        "schema_valid": bool(output), "exact_match": bool(output) and expected == predicted,
        "score": rates(len(matches), len(wrong), len(missed)), "errors": sorted(errors),
        "by_field": {
            field: [sum(category == field for category, _ in values) for values in (matches, wrong, missed)]
            for field in FIELDS
        },
    }


def summarize(rows):
    if not rows:
        raise ValueError("Cannot score an empty set")
    by_field = {field: rates(*(sum(row["by_field"][field][i] for row in rows) for i in range(3)))
                for field in FIELDS}
    return {
        "count": len(rows), "json_valid_rate": mean(row["json_valid"] for row in rows),
        "schema_compliance_rate": mean(row["schema_valid"] for row in rows),
        "exact_match": mean(row["exact_match"] for row in rows),
        "micro": rates(*(sum(f[k] for f in by_field.values()) for k in ("tp", "fp", "fn"))),
        "macro_f1": mean(field["f1"] for field in by_field.values()), "fields": by_field,
        "mean_input_tokens": mean(r["input_tokens"] for r in rows),
        "mean_generated_tokens": mean(r["generated_tokens"] for r in rows),
        "mean_generation_seconds": mean(r["generation_seconds"] for r in rows),
        "errors": dict(Counter(error for row in rows for error in row["errors"])),
    }


def report(output, summaries, examples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    summary = pd.DataFrame({
        name: {
            "Micro-F1": mode["micro"]["f1"], "Macro-F1": mode["macro_f1"],
            "JSON valid": mode["json_valid_rate"], "Schema": mode["schema_compliance_rate"],
            **{key: value["f1"] for key, value in mode["fields"].items()},
        } for name, mode in summaries.items()
    })
    summary.to_csv(output / "comparison.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    labels = {"physical_objects": "Objects", "activities": "Acts", "states": "States",
              "processes": "Processes", "properties": "Properties"}
    colors = ["#61549a", "#217f79", "#a56a28"]
    summary.iloc[:4].plot.bar(ax=axes[0], rot=12, ylim=(0, 1), color=colors[:len(summaries)])
    summary.iloc[4:].rename(index=labels).plot.bar(ax=axes[1], rot=12, ylim=(0, 1),
                                                  color=colors[:len(summaries)], legend=False)
    axes[0].set_title("Overall extraction results")
    axes[1].set_title("F1 by entity category")
    axes[0].set_ylabel("Rate / F1")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=140)
    plt.close(figure)
    write_json(output / "failure_examples.json", examples)


def run(args):
    settings = config(args.config)
    selected = split_rows(args.data, args.split)
    rows = selected[:args.limit]
    train = split_rows(args.data, "train")
    # 固定训练示例，禁止从验证或测试样本挑选 few-shot 示例。
    shots = train[:3]
    modes = [args.mode] if args.mode != "all" else ["zero", "few", "lora"] if args.manifest else ["zero", "few"]
    if "lora" in modes:
        if not args.manifest:
            raise ValueError("--manifest is required for LoRA")
        _, saved = load_manifest(args.manifest, settings)
        if saved["data_manifest_sha256"] != sha(args.data / "split_manifest.json"):
            raise ValueError("Adapter was trained against a different split manifest")
        if any(settings["model"].get(k) != v for k, v in saved["inference"].items()):
            raise ValueError("Restore the base decoding settings recorded in the Adapter manifest")

    args.out.mkdir(parents=True, exist_ok=False)
    summary, failures = {}, {}
    for mode in modes:
        engine = Engine(settings, device=args.device, manifest=args.manifest if mode == "lora" else None)
        examples = shots if mode == "few" else []
        results = []
        # 每条记录即时保存，长时间评测中断后仍能读取已完成结果。
        with (args.out / f"{mode}.jsonl").open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                result = score(row, engine.predict(row["text"], examples, seed=args.seed + index))
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                results.append(result)
                if (index + 1) % 25 == 0:
                    print(f"{mode}: {index + 1}/{len(rows)}", flush=True)
        summary[mode] = summarize(results) | {
            "model_version": engine.version, "profile": engine.profile,
            "device": engine.device, "load_seconds": engine.load_seconds,
        }
        tags = summary[mode]["errors"]
        failures[mode] = {tag: next(r for r in results if tag in r["errors"]) for tag in tags}
        print(json.dumps({mode: summary[mode]}, indent=2), flush=True)
        del engine
        gc.collect()

    combined = {
        "split": args.split, "limited": len(rows) < len(selected), "seed": args.seed,
        "split_sha256": sha(args.data / f"{args.split}.jsonl"), "few_shot_ids": [x["id"] for x in shots],
        "model_settings": settings["model"], "modes": summary,
    }
    write_json(args.out / "metrics.json", combined)
    report(args.out, summary, failures)
    if (args.manifest and args.split == "test" and not combined["limited"]
            and modes == ["zero", "few", "lora"] and summary["lora"]["profile"] == "standard"):
        version = json.loads(args.manifest.read_text(encoding="utf-8"))
        version["test_metrics"] = combined
        version["test_report_sha256"] = sha(args.out / "metrics.json")
        write_json(args.manifest.with_name("release_manifest.json"), version)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=Path("configs/industrialllm.toml"))
    cli.add_argument("--data", type=Path, default=Path("data/processed"))
    cli.add_argument("--split", choices=("validation", "test"), default="validation")
    cli.add_argument("--mode", choices=("all", "zero", "few", "lora"), default="all")
    cli.add_argument("--manifest", type=Path)
    cli.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--limit", type=int)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.limit is not None and args.limit < 1:
        cli.error("--limit must be positive")
    run(args)
