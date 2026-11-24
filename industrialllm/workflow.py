"""工单草稿 → 人工审核。只管理草稿和审核记录，不执行维修操作。"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .schema import Extraction, grounded


def make_draft(text, base_url):
    request = Request(f"{base_url.rstrip('/')}/extract", data=json.dumps({"text": text}).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
    except HTTPError as error:
        result = json.load(error)
    ready = result.get("schema_valid") and result.get("grounded")
    return {"status": "pending_review" if ready else "extraction_failed", "extraction": result}


def review(draft, decision, reviewer):
    if draft["status"] != "pending_review" or decision not in ("approved", "rejected"):
        raise ValueError("Only a pending draft can receive an approve/reject decision")
    if not reviewer.strip():
        raise ValueError("A reviewer is required")
    record = draft["extraction"]
    validated = Extraction.model_validate(record["result"])
    if any(not grounded(span, record["text"]) for spans in validated.model_dump().values() for span in spans):
        raise ValueError("Draft contains an entity unsupported by the source text")
    return {**draft, "status": decision, "reviewer": reviewer, "reviewed_at": datetime.now(UTC).isoformat()}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    draft_cmd = sub.add_parser("draft")
    draft_cmd.add_argument("--text", required=True)
    draft_cmd.add_argument("--url", default="http://127.0.0.1:8000")
    draft_cmd.add_argument("--out", type=Path, required=True)
    review_cmd = sub.add_parser("review")
    review_cmd.add_argument("--draft", type=Path, required=True)
    review_cmd.add_argument("--decision", choices=("approved", "rejected"), required=True)
    review_cmd.add_argument("--reviewer", required=True)
    review_cmd.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "draft":
        result = make_draft(args.text, args.url)
    else:
        result = review(json.loads(args.draft.read_text(encoding="utf-8")), args.decision, args.reviewer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(result["status"])
