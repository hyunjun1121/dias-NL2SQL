"""
Main runner script for EPFL Hyunjun's NL2SQL Pipeline.
"""

import json
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import get_default_config, get_bird_config
from pipeline.main_pipeline import EPFLHyunjunPipeline


def load_dataset(data_path: str, split: str = "dev"):
    """Load dataset (BIRD or Spider format)."""
    data_file = Path(data_path) / f"{split}.json"

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def load_schema(db_id: str, schema_dir: str):
    """Load database schema."""
    schema_file = Path(schema_dir) / db_id / "database_schema.json"

    if schema_file.exists():
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema

    # Fallback: return empty schema
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run EPFL Hyunjun's NL2SQL Pipeline")
    parser.add_argument("--dataset", type=str, default="bird", help="Dataset name (bird/spider)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--db_path", type=str, required=True, help="Path to databases")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split")
    parser.add_argument("--output", type=str, default="results.json", help="Output file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")

    args = parser.parse_args()

    # Load config
    if args.dataset == "bird":
        config = get_bird_config()
    else:
        config = get_default_config()

    # Initialize pipeline
    pipeline = EPFLHyunjunPipeline(config)

    # Load dataset
    print(f"Loading {args.dataset} {args.split} set...")
    data = load_dataset(args.data_path, args.split)

    if args.limit:
        data = data[:args.limit]

    print(f"Processing {len(data)} examples...")

    # Process each example
    results = []
    for idx, example in enumerate(data):
        print(f"\n[{idx+1}/{len(data)}] Processing: {example.get('question', 'N/A')}")

        try:
            # Get database info
            db_id = example.get('db_id', example.get('database_id'))
            db_path = Path(args.db_path) / db_id / f"{db_id}.sqlite"

            # Load schema
            schema = load_schema(db_id, Path(args.db_path))

            # Run pipeline
            output = pipeline.run(
                nl_query=example['question'],
                schema=schema,
                db_path=str(db_path)
            )

            # Save result
            result = {
                'question': example['question'],
                'db_id': db_id,
                'predicted_sql': output.final_sql,
                'gold_sql': example.get('SQL', example.get('query')),
                'execution_success': output.execution_success,
                'semantic_correctness': output.semantic_correctness.overall_score,
                'total_reward': output.total_reward,
                'execution_time': output.total_time,
                'num_iterations': output.num_iterations
            }

            results.append(result)

            print(f"  Predicted SQL: {output.final_sql}")
            print(f"  Execution: {'SUCCESS' if output.execution_success else 'FAILED'}")
            print(f"  Semantic Correctness: {output.semantic_correctness.overall_score:.2%}")
            print(f"  Total Reward: {output.total_reward:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'question': example['question'],
                'db_id': example.get('db_id', 'unknown'),
                'predicted_sql': None,
                'gold_sql': example.get('SQL', example.get('query')),
                'execution_success': False,
                'error': str(e)
            })

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    num_success = sum(1 for r in results if r.get('execution_success', False))
    avg_semantic = sum(r.get('semantic_correctness', 0) for r in results) / len(results)
    avg_reward = sum(r.get('total_reward', 0) for r in results) / len(results)

    print(f"\nSummary:")
    print(f"  Total: {len(results)}")
    print(f"  Execution Success: {num_success}/{len(results)} ({num_success/len(results):.2%})")
    print(f"  Avg Semantic Correctness: {avg_semantic:.2%}")
    print(f"  Avg Total Reward: {avg_reward:.3f}")


if __name__ == "__main__":
    main()
