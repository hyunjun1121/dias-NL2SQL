"""
IR Value/Context Support Evaluation (Spider 2.0)

This script computes simple coverage-based metrics for:
  - Value Coverage: does IR-retrieved example values cover literals in the gold SQL?
  - Context Coverage: do IR-retrieved descriptions cover gold (table,column) pairs?

Optional: embedding-based question–description similarity (requires OpenAI Embeddings).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ir.ir_integration import run_ir_and_prune


def load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dbm(db_root_path: str, mode: str, db_id: str):
    from runner.database_manager import DatabaseManager  # type: ignore
    DatabaseManager(db_mode=mode, db_id=db_id)
    return DatabaseManager()


def gold_literals(dbm, sql: str) -> Set[str]:
    try:
        vals = dbm.get_sql_condition_literals(sql=sql) or []
        norm = {str(v).strip("'\" ").lower() for v in vals}
        return {v for v in norm if v}
    except Exception:
        return set()


def flatten_examples(examples: Any) -> Set[str]:
    """Normalize schema_with_examples to a flat set of strings."""
    out: Set[str] = set()
    if isinstance(examples, dict):
        for _t, cols in examples.items():
            if isinstance(cols, dict):
                for _c, vals in cols.items():
                    if isinstance(vals, list):
                        for v in vals:
                            out.add(str(v).strip("'\" ").lower())
                    else:
                        out.add(str(vals).strip("'\" ").lower())
            elif isinstance(cols, list):
                for v in cols:
                    out.add(str(v).strip("'\" ").lower())
    elif isinstance(examples, list):
        for v in examples:
            out.add(str(v).strip("'\" ").lower())
    return {v for v in out if v}


def value_coverage(gold: Set[str], ret: Set[str]) -> Tuple[float, float, float, int, int, int]:
    if not gold and not ret:
        return 1.0, 1.0, 1.0, 0, 0, 0
    inter = gold & ret
    tp = len(inter)
    fp = len(ret - gold)
    fn = len(gold - ret)
    recall = tp / len(gold) if gold else 0.0
    precision = tp / len(ret) if ret else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return recall, precision, f1, tp, fp, fn


def context_coverage(descriptions: Any, gold_pairs: Set[Tuple[str, str]]) -> float:
    """Compute fraction of gold (table,column) pairs that have any retrieved description."""
    if not descriptions or not gold_pairs:
        return 0.0
    covered = 0
    if isinstance(descriptions, dict):
        for (t, c) in gold_pairs:
            t_dict = descriptions.get(t) or descriptions.get(t.lower()) or {}
            if isinstance(t_dict, dict):
                has = False
                for k in (c, c.lower()):
                    if k in t_dict:
                        has = True
                        break
                covered += 1 if has else 0
    denom = len(gold_pairs)
    return covered / denom if denom else 0.0


def main():
    ap = argparse.ArgumentParser(description="IR value/context support evaluation (Spider 2.0)")
    ap.add_argument("--spider_json", type=str, required=True)
    ap.add_argument("--db_root_path", type=str, default="./data")
    ap.add_argument("--mode", type=str, default="dev")
    ap.add_argument("--output", type=str, default="results/ir_eval_value_context.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = load_samples(Path(args.spider_json))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    details: List[Dict[str, Any]] = []
    agg = {
        "n": 0,
        "sum_val_recall": 0.0,
        "sum_val_precision": 0.0,
        "sum_val_f1": 0.0,
        "sum_ctx_cov": 0.0,
    }

    for i, s in enumerate(samples):
        qid = s.get("question_id", i)
        question = s.get("question") or s.get("utterance") or ""
        db_id = s.get("db_id") or s.get("database_id")
        if not db_id:
            continue

        dbm = ensure_dbm(args.db_root_path, args.mode, str(db_id))
        sql = s.get("SQL") or s.get("query") or s.get("gold_sql") or ""
        gold_vals = gold_literals(dbm, sql)

        pruned_schema, artifacts = run_ir_and_prune(
            question=question,
            db_id=str(db_id),
            data_mode=args.mode,
            db_root_path=args.db_root_path,
            extract_keywords_template="extract_keywords",
            extract_keywords_engine="gpt-4o-mini",
            extract_keywords_temperature=0.2,
            extract_keywords_parser="python_list_output_parser",
            retrieve_context_top_k=5,
        )

        retrieved_vals = flatten_examples(artifacts.get("schema_with_examples"))

        # Build gold (table,column) pairs via DatabaseManager helper
        gt_pairs = set()
        try:
            t2c = dbm.get_sql_columns_dict(sql=sql) or {}
            for t, cols in t2c.items():
                for c in cols:
                    gt_pairs.add((str(t), str(c)))
        except Exception:
            pass

        val_recall, val_precision, val_f1, _tp, _fp, _fn = value_coverage(gold_vals, retrieved_vals)
        ctx_cov = context_coverage(artifacts.get("schema_with_descriptions"), gt_pairs)

        details.append(
            {
                "question_id": qid,
                "db_id": db_id,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1": val_f1,
                "ctx_coverage": ctx_cov,
                "gold_values": sorted(list(gold_vals)),
                "retrieved_values": sorted(list(retrieved_vals)),
            }
        )

        agg["n"] += 1
        agg["sum_val_recall"] += val_recall
        agg["sum_val_precision"] += val_precision
        agg["sum_val_f1"] += val_f1
        agg["sum_ctx_cov"] += ctx_cov

    summary = {
        "count": agg["n"],
        "macro_val_recall": (agg["sum_val_recall"] / agg["n"]) if agg["n"] else 0.0,
        "macro_val_precision": (agg["sum_val_precision"] / agg["n"]) if agg["n"] else 0.0,
        "macro_val_f1": (agg["sum_val_f1"] / agg["n"]) if agg["n"] else 0.0,
        "macro_ctx_coverage": (agg["sum_ctx_cov"] / agg["n"]) if agg["n"] else 0.0,
    }

    out = {"summary": summary, "details": details}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("IR value/context support evaluation (Spider 2.0)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

