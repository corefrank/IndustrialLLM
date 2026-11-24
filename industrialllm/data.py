"""读取固定版本 MaintIE，产出可审计的 SFT 数据和切分清单。"""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from .schema import (
    FIELDS,
    LABEL_FIELDS,
    Extraction,
    config,
    grounded,
    messages,
    normalize,
    write_json,
    write_jsonl,
)

SOURCE = "https://github.com/nlp-tlp/maintie"
REVISION = "825e7710e4129598da997276a38b3cb40b200bc2"
FILES = {
    "gold": "ef811cf4ffd0c90e6a13b1369e24e9d1d08f9a294d873ab4fb1a580096d7d6e0",
    "silver": "af8f785e5c03fdaefeb9b94dad7d24cbd030c2fa4b53ee452069d035f4bf4877",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def acquire(raw):
    raw.mkdir(parents=True, exist_ok=True)
    for source, expected in FILES.items():
        name = f"{source}_release.json"
        path = raw / name
        if not path.is_file():
            try:
                url = f"https://raw.githubusercontent.com/nlp-tlp/maintie/{REVISION}/data/{name}"
                with urlopen(url, timeout=25) as response:
                    path.write_bytes(response.read())
            except OSError:
                with tempfile.TemporaryDirectory() as directory:
                    subprocess.run(["git", "init", "-q", directory], check=True)
                    subprocess.run(
                        ["git", "-C", directory, "fetch", "-q", "--depth", "1", SOURCE, REVISION],
                        check=True,
                    )
                    path.write_bytes(subprocess.check_output(
                        ["git", "-C", directory, "show", f"FETCH_HEAD:data/{name}"]
                    ))
        if sha(path) != expected:
            raise ValueError(f"SHA-256 mismatch for {path}; remove the incorrect local copy and rerun")


def convert(row, item_id):
    text, tokens = row["text"], row["tokens"]
    if not isinstance(text, str) or not text.strip() or not tokens:
        raise ValueError("empty text or tokens")
    answer = {field: [] for field in FIELDS}
    seen = set()
    for entity in row["entities"]:
        start, end = entity["start"], entity["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(tokens):
            raise ValueError("invalid token span")
        field = LABEL_FIELDS[entity["type"].split("/", 1)[0]]
        span = " ".join(tokens[start:end])
        if span not in text or not grounded(span, text):
            raise ValueError("span is not grounded in text")
        key = field, normalize(span)
        if key not in seen:
            answer[field].append(span)
            seen.add(key)
    target = Extraction.model_validate(answer).model_dump()
    return {
        "id": item_id, "text": text, "target": target, "prompt": messages(text),
        "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def signatures(text):
    """同长度且不少于五个词，单个同位置词不同的样本共享一个签名。"""
    words = tuple(text.split())
    if len(words) >= 5:
        for index in range(len(words)):
            yield index, words[:index], words[index + 1:]


def group_near(keys):
    parent = dict.fromkeys(keys)
    index = {}

    def root(key):
        while parent[key] is not None:
            key = parent[key]
        return key

    for key in keys:
        for signature in signatures(key):
            neighbor = index.setdefault(signature, key)
            left, right = root(key), root(neighbor)
            if left != right:
                parent[left] = right
    groups = defaultdict(list)
    for key in keys:
        groups[root(key)].append(key)
    return list(groups.values())


def labels(group, rows):
    return {field for key in group for field in FIELDS if rows[key]["target"][field]}


def select_gold(groups, rows, size):
    selected, remaining = [], list(groups)
    counts = Counter(field for group in groups for field in labels(group, rows))
    covered = set()
    for field in sorted(FIELDS, key=lambda item: (counts[item], item)):
        if field in covered:
            continue
        candidates = (group for group in remaining
                      if len(group) <= size - len(selected) and field in labels(group, rows))
        group = next(candidates, None)
        if group:
            remaining.remove(group)
            selected.extend(group)
            covered.update(labels(group, rows))
    for group in remaining[:]:
        if len(group) <= size - len(selected):
            selected.extend(group)
            remaining.remove(group)
        if len(selected) == size:
            break
    if len(selected) != size or labels(selected, rows) != set(FIELDS):
        raise ValueError("Gold split cannot reach the requested size with all five labels")
    return selected, remaining


def prepare(raw, output, settings):
    acquire(raw)
    output.mkdir(parents=True, exist_ok=True)
    exclusions, corpus, stats = [], {}, {}
    for source in FILES:
        rows = json.loads((raw / f"{source}_release.json").read_bytes())
        unique = {}
        for index, row in enumerate(rows):
            item_id = f"{source}:{index}"
            key = normalize(row["text"])
            try:
                sample = convert(row, item_id)
            except (KeyError, TypeError, ValueError) as error:
                exclusions.append({"id": item_id, "reason": str(error)})
                continue
            if key in unique:
                exclusions.append({"id": item_id, "reason": "exact_normalized_duplicate", "of": unique[key]["id"]})
                continue
            unique[key] = sample
        corpus[source] = unique
        stats[source] = {
            "raw": len(rows), "usable_unique": len(unique),
            "raw_entity_counts": dict(Counter(e["type"] for r in rows for e in r["entities"])),
            "raw_relation_counts": dict(Counter(e["type"] for r in rows for e in r["relations"])),
        }

    gold, silver = corpus["gold"], corpus["silver"]
    gold_signatures = {sig for key in gold for sig in signatures(key)}
    rng = random.Random(settings["seed"])
    groups = group_near(gold)
    rng.shuffle(groups)
    validation, groups = select_gold(groups, gold, settings["validation"])
    test, reserve = select_gold(groups, gold, settings["test"])
    train_pool = []
    for key, row in silver.items():
        if key in gold:
            exclusions.append({"id": row["id"], "reason": "gold_exact_overlap"})
        elif any(sig in gold_signatures for sig in signatures(key)):
            exclusions.append({"id": row["id"], "reason": "gold_near_overlap"})
        else:
            train_pool.append(key)
    if len(train_pool) < settings["train"]:
        raise ValueError("Too few eligible Silver rows")
    keys_by_split = {"train": rng.sample(train_pool, settings["train"]), "validation": validation, "test": test}

    hashes, manifest, split_stats = {}, {}, {}
    for split, keys in keys_by_split.items():
        rows = [corpus["silver" if split == "train" else "gold"][key] for key in keys]
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        hashes[path.name] = sha(path)
        manifest[split] = [row["id"] for row in rows]
        split_stats[split] = {
            "rows": len(rows),
            "entity_counts": {field: sum(len(r["target"][field]) for r in rows) for field in FIELDS},
        }

    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        left, right = keys_by_split[a], keys_by_split[b]
        if set(left) & set(right) or {s for k in left for s in signatures(k)} & {
            s for k in right for s in signatures(k)
        }:
            raise RuntimeError(f"Data leakage between {a} and {b}")
    write_jsonl(output / "excluded.jsonl", exclusions)
    write_json(output / "split_manifest.json", {
        "source": SOURCE, "revision": REVISION, "raw_sha256": FILES, "seed": settings["seed"],
        "splits": manifest, "jsonl_sha256": hashes, "gold_near_groups": [
            [gold[key]["id"] for key in group] for group in group_near(gold) if len(group) > 1
        ],
        "gold_reserve": [gold[key]["id"] for group in reserve for key in group],
    })
    report = {
        "source_revision": REVISION, "corpora": stats, "splits": split_stats,
        "exclusions_by_reason": dict(Counter(row["reason"] for row in exclusions)),
        "leakage_check": "zero normalized exact matches or equal-length one-token candidates across splits",
        "normalization": "NFKC + casefold + whitespace collapse",
    }
    write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=Path("configs/industrialllm.toml"))
    cli.add_argument("--raw", type=Path, default=Path("data/raw"))
    cli.add_argument("--out", type=Path, default=Path("data/processed"))
    cli.add_argument("--copy-raw-from", type=Path, help="Use existing official data files before verifying SHA-256")
    args = cli.parse_args()
    if args.copy_raw_from:
        args.raw.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copyfile(args.copy_raw_from / f"{name}_release.json", args.raw / f"{name}_release.json")
    result = prepare(args.raw, args.out, config(args.config)["data"])
    print(json.dumps({key: result[key] for key in ("splits", "exclusions_by_reason", "leakage_check")}, indent=2))
