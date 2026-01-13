#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# -------------------------
# IO
# -------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_item_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in rows:
        out[str(r["item_id"])] = r
    return out


# -------------------------
# Agreement metrics
# -------------------------
def cohen_kappa(labels1: List[Any], labels2: List[Any]) -> float:
    """Cohen's kappa for categorical labels."""
    assert len(labels1) == len(labels2)
    n = len(labels1)
    if n == 0:
        return float("nan")

    # observed agreement
    po = sum(1 for a, b in zip(labels1, labels2) if a == b) / n

    # expected agreement
    c1 = Counter(labels1)
    c2 = Counter(labels2)
    pe = 0.0
    for k in set(c1) | set(c2):
        pe += (c1[k] / n) * (c2[k] / n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha_nominal(matrix: List[List[Optional[Any]]]) -> float:
    """
    Krippendorff's alpha (nominal).
    matrix: list of items; each item is list of ratings across annotators (None allowed).
    """
    # Collect categories
    cats = sorted({v for row in matrix for v in row if v is not None})
    if len(cats) <= 1:
        return float("nan")

    # Observed disagreement Do
    Do_num = 0.0
    Do_den = 0.0
    overall = Counter()

    for row in matrix:
        vals = [v for v in row if v is not None]
        m = len(vals)
        if m < 2:
            continue
        cnt = Counter(vals)
        overall.update(cnt)

        # nominal distance: 1 if different else 0
        for c in cats:
            for k in cats:
                if c == k:
                    continue
                Do_num += cnt[c] * cnt[k]
        Do_den += m * (m - 1)

    if Do_den == 0:
        return float("nan")
    Do = Do_num / Do_den

    # Expected disagreement De
    N = sum(overall.values())
    if N < 2:
        return float("nan")

    De_num = 0.0
    for c in cats:
        for k in cats:
            if c == k:
                continue
            De_num += overall[c] * overall[k]
    De_den = N * (N - 1)
    De = De_num / De_den

    if De == 0:
        return 1.0
    return 1.0 - (Do / De)


# -------------------------
# Analysis helpers
# -------------------------
def get_field(row: Dict[str, Any], field: str) -> Any:
    return row.get("annotations", {}).get(field)


def get_multilabel(row: Dict[str, Any], field: str) -> List[str]:
    v = row.get("annotations", {}).get(field, [])
    if v is None:
        return []
    return list(v) if isinstance(v, list) else []


def compute_item_agreement(values: List[Any]) -> float:
    """
    Agreement proportion for one item across annotators for a single field.
    e.g., 1.0 if all same, 0.5 if 2 of 4 match most common, etc.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan")
    c = Counter(vals)
    return max(c.values()) / len(vals)


def plot_percent_bar(counter: Counter, title: str, outpath: Optional[str] = None) -> None:
    labels = list(counter.keys())
    counts = [counter[k] for k in labels]
    total = sum(counts) if counts else 1
    perc = [100.0 * c / total for c in counts]

    plt.figure()
    plt.bar(labels, perc)
    plt.title(title)
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    else:
        plt.show()
    plt.close()


def plot_topk_percent(counter: Counter, title: str, k: int = 10, outpath: Optional[str] = None) -> None:
    items = counter.most_common(k)
    labels = [x for x, _ in items]
    counts = [c for _, c in items]
    total = sum(counter.values()) if counter else 1
    perc = [100.0 * c / total for c in counts]

    plt.figure()
    plt.bar(labels, perc)
    plt.title(title)
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    else:
        plt.show()
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Annotator A JSONL")
    ap.add_argument("--b", required=True, help="Annotator B JSONL")
    # ap.add_argument("--c", required=True, help="Annotator C JSONL")
    # ap.add_argument("--d", required=True, help="Annotator D JSONL")
    ap.add_argument("--n", type=int, default=15, help="Only analyze first N items by item_id sort (default 15)")
    ap.add_argument("--outdir", default="RQ1/annotation_reports", help="Where to save charts + JSON summary")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    # files = {"A": args.a, "B": args.b, "C": args.c, "D": args.d}
    files = {"A": args.a, "B": args.b}

    data = {k: index_by_item_id(read_jsonl(path)) for k, path in files.items()}

    # Common items across annotators
    common_ids = set.intersection(*(set(m.keys()) for m in data.values()))
    item_ids = sorted(common_ids)[: args.n]

    if not item_ids:
        raise SystemExit("No common item_ids across annotators. Check that they annotated the same items.")

    fields = ["toxicity_presence", "target", "implicitness", "severity"]

    # Agreement computations
    agreement = {}

    for field in fields:
        # Build label vectors
        labels_by_annot = {ann: [get_field(data[ann][iid], field) for iid in item_ids] for ann in data.keys()}

        # Pairwise Cohen's kappa
        kappas = {}
        for a1, a2 in combinations(sorted(labels_by_annot.keys()), 2):
            kappas[f"{a1}-{a2}"] = cohen_kappa(labels_by_annot[a1], labels_by_annot[a2])

        # Krippendorff alpha (nominal)
        matrix = []
        for i in range(len(item_ids)):
            matrix.append([labels_by_annot[ann][i] for ann in sorted(labels_by_annot.keys())])
        alpha = krippendorff_alpha_nominal(matrix)

        # Simple observed agreement (mean of per-item majority share)
        per_item = [compute_item_agreement([labels_by_annot[ann][i] for ann in labels_by_annot]) for i in range(len(item_ids))]
        mean_majority_share = sum(x for x in per_item if x == x) / len(per_item)  # NaN-safe enough here

        agreement[field] = {
            "krippendorff_alpha_nominal": alpha,
            "cohen_kappa_pairwise": kappas,
            "mean_majority_share": mean_majority_share,
        }

    # Multi-label toxicity_types: compute "alpha" not standard for multilabel without binarization,
    # so we provide distribution + per-item overlap stats.
    tox_type_counter = Counter()
    tox_type_overlap = []  # Jaccard similarity across annotators per item
    for iid in item_ids:
        sets = []
        # for ann in ["A", "B", "C", "D"]:
        for ann in ["A", "B"]:

            s = set(get_multilabel(data[ann][iid], "toxicity_types"))
            sets.append(s)
            tox_type_counter.update(s)
        # Jaccard of union/intersection across all annotators (coarse)
        union = set.union(*sets) if sets else set()
        inter = set.intersection(*sets) if sets else set()
        j = (len(inter) / len(union)) if union else 1.0
        tox_type_overlap.append(j)

    # Distributions (percent charts)
    # We'll pool across annotators to show what labels are being used overall.
    pooled = {f: Counter() for f in fields}
    severity_only = Counter()

    for iid in item_ids:
        for ann in ["A", "B"]:
        # for ann in ["A", "B", "C", "D"]:

            row = data[ann][iid]
            for f in fields:
                v = get_field(row, f)
                if v is not None:
                    pooled[f][str(v)] += 1
            sev = get_field(row, "severity")
            if sev is not None:
                severity_only[str(sev)] += 1

    # Charts
    plot_percent_bar(pooled["toxicity_presence"], "Toxicity presence (pooled across annotators)",
                     outpath=os.path.join(args.outdir, "toxicity_presence_percent.png"))
    plot_percent_bar(severity_only, "Severity (1–5) distribution (pooled across annotators)",
                     outpath=os.path.join(args.outdir, "severity_percent.png"))
    plot_percent_bar(pooled["target"], "Target distribution (pooled across annotators)",
                     outpath=os.path.join(args.outdir, "target_percent.png"))
    plot_percent_bar(pooled["implicitness"], "Implicitness distribution (pooled across annotators)",
                     outpath=os.path.join(args.outdir, "implicitness_percent.png"))
    plot_topk_percent(tox_type_counter, "Toxicity types (multi-label counts, pooled)", k=20,
                      outpath=os.path.join(args.outdir, "toxicity_types_percent_top20.png"))

    # Agreement-by-item chart for toxicity_presence
    presence_by_item = []
    for iid in item_ids:
        # vals = [get_field(data[ann][iid], "toxicity_presence") for ann in ["A", "B", "C", "D"]]
        vals = [get_field(data[ann][iid], "toxicity_presence") for ann in ["A", "B"]]
        presence_by_item.append(compute_item_agreement(vals))

    plt.figure()
    plt.bar([str(i) for i in range(len(item_ids))], [100.0 * x for x in presence_by_item])
    plt.title("Per-item agreement: toxicity_presence (majority share)")
    plt.ylabel("Agreement (%)")
    plt.xlabel("Item index (within subset)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "agreement_by_item_presence.png"), dpi=200)
    plt.close()

    # Summary JSON
    summary = {
        "n_items_analyzed": len(item_ids),
        "item_ids": item_ids,
        "agreement": agreement,
        "toxicity_types": {
            "pooled_counts": dict(tox_type_counter),
            "mean_jaccard_overlap_across_annotators": sum(tox_type_overlap) / len(tox_type_overlap) if tox_type_overlap else None,
        },
        "label_distributions_pooled": {
            "toxicity_presence": dict(pooled["toxicity_presence"]),
            "severity": dict(severity_only),
            "target": dict(pooled["target"]),
            "implicitness": dict(pooled["implicitness"]),
        }
    }

    out_json = os.path.join(args.outdir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print quick results
    print(f"Analyzed {len(item_ids)} common items (first n={args.n}).")
    print(f"Wrote charts + summary to: {args.outdir}")
    for field, stats in agreement.items():
        print(f"\nField: {field}")
        print(f"  Krippendorff's α (nominal): {stats['krippendorff_alpha_nominal']:.4f}")
        print(f"  Mean majority share:        {100*stats['mean_majority_share']:.1f}%")
        for k, v in stats["cohen_kappa_pairwise"].items():
            print(f"  Cohen's κ {k}: {v:.4f}")


if __name__ == "__main__":
    main()
