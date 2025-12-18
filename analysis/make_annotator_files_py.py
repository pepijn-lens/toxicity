#!/usr/bin/env python3
import argparse
import json
import os
import random
from itertools import combinations
from typing import Any, Dict, List, Tuple

TOX_PRESENCE = ["no", "borderline", "yes"]
TOX_TYPES = [
    "insult",
    "profanity_obscene",
    "threat_violence",
    "hate_speech",
    "harassment_dehumanisation",
    "other",
]
TARGETS = ["none", "individual", "group_protected", "group_non_protected", "institution", "self", "unknown"]
IMPLICITNESS = ["explicit", "implicit", "context_dependent"]

def read_json_or_jsonl(path: str) -> Tuple[List[Dict[str, Any]], str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return [], "jsonl"
    if text[0] == "[":
        return json.loads(text), "json"
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows, "jsonl"

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def blank_annotation() -> Dict[str, Any]:
    return {
        "toxicity_presence": None,   # one of: no/borderline/yes
        "toxicity_types": [],        # multi-label list from TOX_TYPES
        "target": None,              # one of TARGETS
        "severity": None,            # int 1..5
        "implicitness": None,        # explicit/implicit/context_dependent
        "notes": ""
    }

def canonicalize_item(r: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure required fields exist; keep extras if present
    item_id = str(r.get("item_id", r.get("idx", "")))
    if item_id == "":
        raise ValueError("Each row must have item_id or idx.")
    out = {
        "item_id": item_id,
        "idx": r.get("idx", None),
        "prompt": r.get("prompt", ""),
        "continuation": r.get("continuation", ""),
        "completion": r.get("completion", ""),
        "score": r.get("score", None),
        "annotations": blank_annotation(),
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON/JSONL with prompt+continuation+completion+score")
    ap.add_argument("--outdir", default="RQ1/annotations", help="Output directory for annotator files")
    ap.add_argument("--mode", choices=["all", "two"], default="all",
                    help="Assignment mode: all=all annotators label all items; two=2 annotators per item")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (used for mode=two)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle items before assignment (recommended)")
    args = ap.parse_args()

    rows, _ = read_json_or_jsonl(args.input)
    items = [canonicalize_item(r) for r in rows]

    if args.shuffle:
        random.Random(args.seed).shuffle(items)

    annotators = ["A", "B", "C", "D"]
    assignments = {a: [] for a in annotators}

    if args.mode == "all":
        # Everyone gets every item
        for it in items:
            for a in annotators:
                # deep copy so each annotator edits independently
                assignments[a].append(json.loads(json.dumps(it)))
    else:
        # Each item assigned to exactly 2 annotators
        rng = random.Random(args.seed)
        pairs = list(combinations(annotators, 2))
        for it in items:
            a1, a2 = rng.choice(pairs)
            assignments[a1].append(json.loads(json.dumps(it)))
            assignments[a2].append(json.loads(json.dumps(it)))

    # Write files + a small README
    os.makedirs(args.outdir, exist_ok=True)
    for a in annotators:
        out_path = os.path.join(args.outdir, f"annotator_{a}.jsonl")
        write_jsonl(out_path, assignments[a])
        print(f"✅ Wrote {len(assignments[a])} items -> {out_path}")

    # Optional: label schema helper file
    schema = {
        "toxicity_presence": TOX_PRESENCE,
        "toxicity_types": TOX_TYPES,
        "target": TARGETS,
        "severity": [1, 2, 3, 4, 5],
        "implicitness": IMPLICITNESS,
        "notes": "free text"
    }
    schema_path = os.path.join(args.outdir, "label_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"🧾 Wrote label schema -> {schema_path}")

if __name__ == "__main__":
    main()
