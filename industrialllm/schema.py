"""维修工单 Schema、严格 JSON 解析和文本比较规则。"""

import json
import re
import tomllib
import unicodedata
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

DEFAULT_CONFIG = Path("configs/industrialllm.toml")
LABEL_FIELDS = {
    "PhysicalObject": "physical_objects",
    "Activity": "activities",
    "State": "states",
    "Process": "processes",
    "Property": "properties",
}
FIELDS = tuple(LABEL_FIELDS.values())
SYSTEM_PROMPT = (
    "Extract entities explicitly mentioned in the English maintenance work order. Return one JSON "
    "object with exactly these keys: physical_objects (equipment or parts), activities (actions), "
    "states (conditions or symptoms), processes (processes such as wear), properties (attributes). "
    "Every value is an array of verbatim spans. Use [] when absent. Do not infer causes or "
    "recommend repairs. Return only the JSON."
)
Span = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    physical_objects: list[Span]
    activities: list[Span]
    states: list[Span]
    processes: list[Span]
    properties: list[Span]


def config(path=DEFAULT_CONFIG):
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def grounded(span, text):
    """词边界防止把 oil 在 coil 中当作有依据的实体。"""
    return bool(re.search(r"(?<!\w)" + re.escape(normalize(span)) + r"(?!\w)", normalize(text)))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def strict_json(raw):
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON key")
        return result

    def reject_constant(value):
        raise ValueError(f"non-JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)


def messages(text, examples=()):
    result = [{"role": "system", "content": SYSTEM_PROMPT}]
    for row in examples:
        result.extend(row["prompt"][-1:] + row["completion"])
    return [*result, {"role": "user", "content": text}]
