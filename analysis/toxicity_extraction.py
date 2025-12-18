#!/usr/bin/env python3
import argparse
import json
from typing import List, Dict, Tuple, Any

def read_json_or_jsonl(path: str) -> Tuple[List[Dict[str, Any]], str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return [], "jsonl"
    if text[0] == "[":
        return json.loads(text), "json"
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows, "jsonl"

def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_prompt_and_continuation(decodingtrust_toxic_jsonl: str) -> List[Dict[str, str]]:
    rows, _ = read_json_or_jsonl(decodingtrust_toxic_jsonl)
    out: List[Dict[str, str]] = []
    for r in rows:
        prompt = r.get("prompt", {}).get("text", None)
        continuation = r.get("continuation", {}).get("text", None)

        if prompt is None:
            raise ValueError(f"Missing prompt.text in a row: keys={list(r.keys())}")
        if continuation is None:
            # Some datasets might miss it; keep as empty string rather than crash
            continuation = ""

        out.append({"prompt": prompt, "continuation": continuation})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input scored completions JSON/JSONL (needs completion + score)")
    ap.add_argument("--output", required=True, help="Output file (json or jsonl)")
    ap.add_argument("--context", default="RQ1/toxic.jsonl",
                    help="DecodingTrust toxic jsonl containing prompt+continuation (default: RQ1/toxic.jsonl)")
    ap.add_argument("--k", type=int, default=100, help="How many items to extract (default 100)")
    ap.add_argument("--score_key", default="score", help="Key name for toxicity score in input (default: score)")
    ap.add_argument("--keep_format", action="store_true",
                    help="If set, output format matches input format (ignores output extension).")
    ap.add_argument("--strict_length_check", action="store_true",
                    help="If set, error if #context != #completions. Recommended.")
    args = ap.parse_args()

    context = load_prompt_and_continuation(args.context)
    rows, in_format = read_json_or_jsonl(args.input)

    if args.strict_length_check and len(context) != len(rows):
        raise ValueError(
            f"Length mismatch: context={len(context)} vs completions={len(rows)}. "
            "This script assumes row-wise alignment."
        )

    n = min(len(context), len(rows))
    enriched: List[Dict[str, Any]] = []
    for i in range(n):
        r = rows[i]
        enriched.append({
            "idx": i,
            "prompt": context[i]["prompt"],
            "continuation": context[i]["continuation"],
            "completion": r.get("completion", ""),
            "score": r.get(args.score_key, None),
        })

    def get_score(r):
        try:
            return float(r.get("score"))
        except (TypeError, ValueError):
            return float("-inf")

    topk = sorted(enriched, key=get_score, reverse=True)[: args.k]

    out_format = in_format if args.keep_format else ("jsonl" if args.output.lower().endswith(".jsonl") else "json")
    if out_format == "jsonl":
        write_jsonl(args.output, topk)
    else:
        write_json(args.output, topk)

    if topk:
        scores = [get_score(r) for r in topk]
        print(f"Wrote top {len(topk)} to {args.output}")
        print(f"Top score: {scores[0]:.6f} | Kth score: {scores[-1]:.6f}")
        if len(context) != len(rows):
            print(f"WARNING: context={len(context)} completions={len(rows)}; used first n={n} by index alignment.")
    else:
        print("No rows found to write.")

if __name__ == "__main__":
    main()
