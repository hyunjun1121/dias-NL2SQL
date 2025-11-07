"""
IR Column-Level Retrieval Evaluation on Spider 2.0

This script evaluates the column-level retrieval quality of the deterministic
CHESS-style IR used in the Triple Kim pipeline. Ground-truth columns are taken
from the dataset if available (e.g., a field like 'gold_columns'); otherwise,
they are derived from gold SQL by leveraging CHESS DatabaseManager's SQL
parsing helpers (get_sql_columns_dict, get_sql_tables) and the database schema.

Usage example:
  python scripts/eval_ir_spider_columns.py \
    --spider_json "<path>/dev.json" \
    --db_root_path "./data" \
    --mode dev \
    --output results/ir_eval_spider_columns_dev.json \
    --limit 200

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ir.ir_integration import run_ir_and_prune
import sqlglot


def load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dbm(db_root_path: str, mode: str, db_id: str):
    # Late import to ensure CHESS src on path (ir_integration does this)
    from runner.database_manager import DatabaseManager  # type: ignore

    DatabaseManager(db_mode=mode, db_id=db_id)
    return DatabaseManager()


def gold_columns_from_dataset(sample: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
    # Prefer explicit fields if present; normalize to {table: {columns}}
    for key in ("gold_columns", "columns", "gold_table_columns"):
        if key in sample and sample[key]:
            raw = sample[key]
            # Expect list of "table.column" strings or mapping
            if isinstance(raw, list):
                mapping: Dict[str, Set[str]] = {}
                for item in raw:
                    try:
                        table, col = str(item).split(".", 1)
                    except ValueError:
                        # Unqualified; skip until we can map
                        continue
                    mapping.setdefault(table, set()).add(col)
                return mapping
            elif isinstance(raw, dict):
                return {t: set(map(str, cols)) for t, cols in raw.items()}
    return None


def _alias_map_from_ast(node) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    try:
        for t in node.find_all(sqlglot.exp.Table):
            real = t.name if hasattr(t, "name") else None
            al = t.alias_or_name if hasattr(t, "alias_or_name") else None
            if real and al:
                alias_map[str(al).lower()] = str(real)
    except Exception:
        pass
    return alias_map


def gold_columns_from_sql(dbm, sql: str, variant: str = "lenient") -> Dict[str, Set[str]]:
    """Extract gold columns from SQL with strict/lenient handling.

    - Base: use DatabaseManager.get_sql_columns_dict
    - Star expansion: expand to referenced tables' full columns
    - Strict: unqualified columns assigned only if single referenced table; use AST to unwrap functions/aliases
    - Lenient: unqualified columns assigned to all referenced tables
    """
    mapping: Dict[str, Set[str]] = {}
    try:
        table_to_cols = dbm.get_sql_columns_dict(sql=sql) or {}
        for table, cols in table_to_cols.items():
            mapping.setdefault(str(table), set()).update({str(c) for c in cols})
        # Collect referenced tables
        try:
            tables_list = dbm.get_sql_tables(sql=sql) or []
        except Exception:
            tables_list = []
        # Expand wildcard
        wildcard = "*" in [c for cols in table_to_cols.values() for c in cols]
        if wildcard and tables_list:
            full = dbm.get_db_schema() or {}
            for t in tables_list:
                cols = full.get(t, [])
                mapping.setdefault(str(t), set()).update({str(c) for c in cols})

        # AST refinement (unwrap functions / aliases)
        try:
            node = sqlglot.parse_one(sql)
            alias_map = _alias_map_from_ast(node)
            referenced = set(str(x) for x in tables_list)
            # Gather columns from AST
            col_nodes = list(node.find_all(sqlglot.exp.Column))
            for cn in col_nodes:
                col_name = str(cn.name) if hasattr(cn, "name") else None
                tbl = None
                try:
                    t = cn.table
                    tbl = str(t) if t else None
                except Exception:
                    tbl = None
                if not col_name:
                    continue
                if tbl:
                    # Resolve alias if any
                    real_tbl = alias_map.get(tbl.lower(), tbl)
                    mapping.setdefault(real_tbl, set()).add(col_name)
                else:
                    # Unqualified
                    if variant == "strict":
                        if len(referenced) == 1:
                            only_tbl = list(referenced)[0]
                            mapping.setdefault(only_tbl, set()).add(col_name)
                        # else: ignore
                    else:
                        # lenient: assign to all referenced tables
                        for rt in referenced:
                            mapping.setdefault(rt, set()).add(col_name)
        except Exception:
            pass
    except Exception:
        pass
    return mapping


def normalize_pairs(mapping: Dict[str, Set[str]]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for t, cols in mapping.items():
        for c in cols:
            pairs.add((str(t).lower(), str(c).lower()))
    return pairs


def retrieved_columns_from_pruned_schema(pruned_schema: Dict[str, Any]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for t, spec in pruned_schema.items():
        cols = spec.get("columns", []) if isinstance(spec, dict) else []
        for entry in cols:
            if isinstance(entry, dict) and "name" in entry:
                pairs.add((str(t).lower(), str(entry["name"]).lower()))
            elif isinstance(entry, str):
                pairs.add((str(t).lower(), entry.lower()))
    return pairs


def evaluate_pairs(gold: Set[Tuple[str, str]], ret: Set[Tuple[str, str]]) -> Tuple[float, float, float, int, int, int]:
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


def main():
    ap = argparse.ArgumentParser(description="Evaluate IR column retrieval on Spider 2.0")
    ap.add_argument("--spider_json", type=str, required=True)
    ap.add_argument("--db_root_path", type=str, default="./data")
    ap.add_argument("--mode", type=str, default="dev")
    ap.add_argument("--output", type=str, default="results/ir_eval_spider_columns.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode_variant", type=str, choices=["strict", "lenient"], default="lenient",
                    help="Column extraction/matching strictness for gold SQL parsing")

    # IR knobs
    ap.add_argument("--ir_template", type=str, default="extract_keywords")
    ap.add_argument("--ir_engine", type=str, default="gpt-4o-mini")
    ap.add_argument("--ir_temperature", type=float, default=0.2)
    ap.add_argument("--ir_parser", type=str, default="python_list_output_parser")
    ap.add_argument("--ir_topk", type=int, default=5)

    args = ap.parse_args()
    samples = load_samples(Path(args.spider_json))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    details: List[Dict[str, Any]] = []
    agg = {"n": 0, "sum_recall": 0.0, "sum_precision": 0.0, "sum_f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    for i, s in enumerate(samples):
        qid = s.get("question_id", i)
        question = s.get("question") or s.get("utterance") or ""
        db_id = s.get("db_id") or s.get("database_id")
        if not db_id:
            continue

        # Prepare DB manager
        dbm = ensure_dbm(args.db_root_path, args.mode, str(db_id))

        # Ground-truth columns
        gt_map = gold_columns_from_dataset(s)
        if gt_map is None:
            sql = s.get("SQL") or s.get("query") or s.get("gold_sql") or ""
            gt_map = gold_columns_from_sql(dbm, sql, variant=args.mode_variant)
        gold_pairs = normalize_pairs(gt_map)

        # IR retrieved columns (from pruned schema)
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
        ret_pairs = retrieved_columns_from_pruned_schema(pruned_schema)

        recall, precision, f1, tp, fp, fn = evaluate_pairs(gold_pairs, ret_pairs)
        details.append(
            {
                "question_id": qid,
                "db_id": db_id,
                "question": question,
                "gold_columns": sorted([f"{t}.{c}" for t, c in gold_pairs]),
                "retrieved_columns": sorted([f"{t}.{c}" for t, c in ret_pairs]),
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

    out = {"summary": summary, "details": details}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("IR column retrieval evaluation (Spider 2.0)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
