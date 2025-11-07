"""
IR Retrieval Evaluation on Spider 2.0 (Table-level)

Measures how well the CHESS-style IR (run via EPFL's wrapper) retrieves
the ground-truth tables on Spider 2.0. Ground-truth tables are taken from the
dataset if present; otherwise, they are extracted from the gold SQL using sqlglot.

Usage (example):
  python scripts/eval_ir_spider.py \
    --spider_json "<path>/dev.json" \
    --db_root_path "./data" \
    --mode dev \
    --output ir_eval_spider_dev.json \
    --limit 200

This script does not require CHESS Agent loop; it calls IR deterministically
through `ir.ir_integration.run_ir_and_prune`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot

from ir.ir_integration import run_ir_and_prune


def load_spider_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_gold_tables(sample: Dict[str, Any]) -> Set[str]:
    # Prefer explicit gold tables if present (Spider 2.0 often provides these)
    for key in ("gold_tables", "tables", "gold_table_names"):
        if key in sample and sample[key]:
            vals = sample[key]
            if isinstance(vals, list):
                return {str(x) for x in vals}
    # Fallback: parse from gold SQL
    sql = sample.get("SQL") or sample.get("query") or sample.get("gold_sql")
    if not sql:
        return set()
    try:
        node = sqlglot.parse_one(sql)
        tables = {t.alias_or_name for t in node.find_all(sqlglot.exp.Table)}
        return {str(x) for x in tables}
    except Exception:
        return set()


def evaluate_table_retrieval(
    gold: Set[str], retrieved: Set[str]
) -> Tuple[float, float, float, int, int, int]:
    if not gold and not retrieved:
        return 1.0, 1.0, 1.0, 0, 0, 0
    inter = gold & retrieved
    tp = len(inter)
    fp = len(retrieved - gold)
    fn = len(gold - retrieved)
    recall = tp / len(gold) if gold else 0.0
    precision = tp / len(retrieved) if retrieved else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return recall, precision, f1, tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description="Evaluate IR table retrieval on Spider 2.0")
    ap.add_argument("--spider_json", type=str, required=True, help="Path to Spider 2.0 split JSON (e.g., dev.json)")
    ap.add_argument("--db_root_path", type=str, default="./data", help="DB root in CHESS layout")
    ap.add_argument("--mode", type=str, default="dev", help="Data mode (dev/test/train)")
    ap.add_argument("--output", type=str, default="ir_eval_spider.json", help="Output JSON path for results")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of samples")

    # IR settings (match defaults in IRConfig)
    ap.add_argument("--ir_template", type=str, default="extract_keywords")
    ap.add_argument("--ir_engine", type=str, default="gpt-4o-mini")
    ap.add_argument("--ir_temperature", type=float, default=0.2)
    ap.add_argument("--ir_parser", type=str, default="python_list_output_parser")
    ap.add_argument("--ir_topk", type=int, default=5)

    args = ap.parse_args()

    data_path = Path(args.spider_json)
    if not data_path.exists():
        raise FileNotFoundError(f"Spider JSON not found: {data_path}")

    samples = load_spider_json(data_path)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    results: List[Dict[str, Any]] = []
    agg = {"n": 0, "sum_recall": 0.0, "sum_precision": 0.0, "sum_f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    for i, sample in enumerate(samples):
        qid = sample.get("question_id", i)
        question = sample.get("question") or sample.get("utterance") or ""
        db_id = sample.get("db_id") or sample.get("database_id")
        if not db_id:
            # Skip if DB id missing
            continue

        gold_tables = extract_gold_tables(sample)

        # Run IR deterministically
        pruned_schema, artifacts = run_ir_and_prune(
            question=question,
            db_id=str(db_id),
            data_mode=args.mode,
            db_root_path=args.db_root_path,
            extract_keywords_template=args.ir_template,
            extract_keywords_engine=args.ir_engine,
            extract_keywords_temperature=args.ir_temperature,
            extract_keywords_parser=args.ir_parser,
            retrieve_context_top_k=args.ir_topk,
        )
        retrieved_tables = set(pruned_schema.keys())

        recall, precision, f1, tp, fp, fn = evaluate_table_retrieval(gold_tables, retrieved_tables)

        results.append(
            {
                "question_id": qid,
                "db_id": db_id,
                "question": question,
                "gold_tables": sorted(gold_tables),
                "retrieved_tables": sorted(retrieved_tables),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

        agg["n"] += 1
        agg["sum_recall"] += recall
        agg["sum_precision"] += precision
        agg["sum_f1"] += f1
        agg["tp"] += tp
        agg["fp"] += fp
        agg["fn"] += fn

    summary = {
        "count": agg["n"],
        "macro_recall": (agg["sum_recall"] / agg["n"]) if agg["n"] else 0.0,
        "macro_precision": (agg["sum_precision"] / agg["n"]) if agg["n"] else 0.0,
        "macro_f1": (agg["sum_f1"] / agg["n"]) if agg["n"] else 0.0,
        "micro_tp": agg["tp"],
        "micro_fp": agg["fp"],
        "micro_fn": agg["fn"],
        "micro_recall": (agg["tp"] / (agg["tp"] + agg["fn"])) if (agg["tp"] + agg["fn"]) > 0 else 0.0,
        "micro_precision": (agg["tp"] / (agg["tp"] + agg["fp"])) if (agg["tp"] + agg["fp"]) > 0 else 0.0,
    }

    out = {"summary": summary, "details": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("IR table retrieval evaluation (Spider 2.0)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

