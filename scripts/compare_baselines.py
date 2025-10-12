"""
Compare our progressive approach vs CHASE-SQL baseline.

Usage:
    python scripts/compare_baselines.py --data_path /path/to/bird --output comparison_results.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from config.config import get_bird_config
from pipeline.main_pipeline import EPFLHyunjunPipeline
from baseline.chase_sql import CHASESQLBaseline
from utils.llm_client import LLMClient
from utils.database_executor import DatabaseExecutor


def load_validation_set(data_path: str, limit: int = 50) -> List[Dict]:
    """
    Load validation set from BIRD dataset.

    Args:
        data_path: Path to BIRD dataset
        limit: Number of examples to load (default: 50)

    Returns:
        List of examples with query, schema, db_path, gold_sql
    """
    # TODO: Implement actual BIRD data loading
    # For now, return empty list - will implement when we have the dataset
    print(f"Loading {limit} examples from {data_path}...")
    return []


def evaluate_example(
    example: Dict,
    our_pipeline: EPFLHyunjunPipeline,
    chase_baseline: CHASESQLBaseline
) -> Dict:
    """
    Evaluate single example with both methods.

    Returns:
        {
            'nl_query': ...,
            'our_method': {...},
            'chase_sql': {...},
            'comparison': {...}
        }
    """
    nl_query = example['query']
    schema = example['schema']
    db_path = example['db_path']
    gold_sql = example.get('gold_sql', None)

    results = {
        'nl_query': nl_query,
        'gold_sql': gold_sql
    }

    # Method 1: Our progressive approach
    print(f"  [Our Method] Processing...")
    start = time.time()
    try:
        our_output = our_pipeline.run(nl_query, schema, db_path)
        our_time = time.time() - start
        results['our_method'] = {
            'sql': our_output.final_sql,
            'execution_success': our_output.execution_success,
            'total_reward': our_output.total_reward,
            'semantic_correct': our_output.semantic_correctness.is_correct if our_output.semantic_correctness else None,
            'num_iterations': our_output.num_iterations,
            'time': our_time
        }
    except Exception as e:
        results['our_method'] = {'error': str(e)}

    # Method 2: CHASE-SQL baseline
    print(f"  [CHASE-SQL] Processing...")
    start = time.time()
    try:
        chase_output = chase_baseline.generate_sql(nl_query, schema)
        chase_time = time.time() - start
        results['chase_sql'] = {
            'sql': chase_output['sql'],
            'execution_success': chase_output['success'],
            'time': chase_time
        }
    except Exception as e:
        results['chase_sql'] = {'error': str(e)}

    # Comparison
    results['comparison'] = {
        'both_success': (
            results.get('our_method', {}).get('execution_success', False) and
            results.get('chase_sql', {}).get('execution_success', False)
        ),
        'our_faster': (
            results.get('our_method', {}).get('time', float('inf')) <
            results.get('chase_sql', {}).get('time', float('inf'))
        )
    }

    return results


def run_comparison(args):
    """Run full comparison."""
    print("="*80)
    print("Comparison: Our Progressive Approach vs CHASE-SQL Baseline")
    print("="*80)

    # Load config
    config = get_bird_config()

    # Initialize LLM client
    # TODO: Use open-source model from cluster
    llm_client = LLMClient(
        model_name=config.llm.model_name,
        api_key=config.llm.api_key
    )

    # Initialize pipelines
    our_pipeline = EPFLHyunjunPipeline(config)
    chase_baseline = CHASESQLBaseline(
        llm_client,
        DatabaseExecutor(args.db_path) if args.db_path else None,
        config.progressive_execution.__dict__
    )

    # Load validation set
    validation_set = load_validation_set(args.data_path, args.limit)
    print(f"\nLoaded {len(validation_set)} examples")

    if len(validation_set) == 0:
        print("\n⚠️  No examples loaded. Please implement load_validation_set()")
        print("    with actual BIRD dataset loading.")
        return

    # Evaluate all examples
    all_results = []
    for i, example in enumerate(validation_set):
        print(f"\n[{i+1}/{len(validation_set)}] {example['query'][:60]}...")
        result = evaluate_example(example, our_pipeline, chase_baseline)
        all_results.append(result)

    # Aggregate statistics
    stats = {
        'total_examples': len(all_results),
        'our_method': {
            'execution_success': sum(1 for r in all_results if r.get('our_method', {}).get('execution_success', False)),
            'semantic_correct': sum(1 for r in all_results if r.get('our_method', {}).get('semantic_correct', False)),
            'avg_time': sum(r.get('our_method', {}).get('time', 0) for r in all_results) / len(all_results),
            'avg_iterations': sum(r.get('our_method', {}).get('num_iterations', 0) for r in all_results) / len(all_results)
        },
        'chase_sql': {
            'execution_success': sum(1 for r in all_results if r.get('chase_sql', {}).get('execution_success', False)),
            'avg_time': sum(r.get('chase_sql', {}).get('time', 0) for r in all_results) / len(all_results)
        },
        'comparison': {
            'both_success': sum(1 for r in all_results if r['comparison']['both_success']),
            'our_faster': sum(1 for r in all_results if r['comparison']['our_faster'])
        }
    }

    # Save results
    output = {
        'config': config.to_dict(),
        'stats': stats,
        'results': all_results
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nOur Method:")
    print(f"  Execution Success: {stats['our_method']['execution_success']}/{stats['total_examples']}")
    print(f"  Semantic Correct: {stats['our_method']['semantic_correct']}/{stats['total_examples']}")
    print(f"  Avg Time: {stats['our_method']['avg_time']:.2f}s")
    print(f"  Avg Iterations: {stats['our_method']['avg_iterations']:.1f}")

    print(f"\nCHASE-SQL Baseline:")
    print(f"  Execution Success: {stats['chase_sql']['execution_success']}/{stats['total_examples']}")
    print(f"  Avg Time: {stats['chase_sql']['avg_time']:.2f}s")

    print(f"\nComparison:")
    print(f"  Both Success: {stats['comparison']['both_success']}/{stats['total_examples']}")
    print(f"  Our Method Faster: {stats['comparison']['our_faster']}/{stats['total_examples']}")

    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare progressive approach vs CHASE-SQL')
    parser.add_argument('--data_path', type=str, required=True, help='Path to BIRD dataset')
    parser.add_argument('--db_path', type=str, help='Path to database directory')
    parser.add_argument('--output', type=str, default='comparison_results.json', help='Output file')
    parser.add_argument('--limit', type=int, default=50, help='Number of examples to evaluate')

    args = parser.parse_args()

    run_comparison(args)


if __name__ == '__main__':
    main()
