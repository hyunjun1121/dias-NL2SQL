"""
Test NL2SQL pipeline with HuggingFace Qwen models.
Integrates the large models into the existing pipeline.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.hf_llm_client import HFLLMClientAdapter
from utils.database_connector import DatabaseConnector
from pipeline.main_pipeline import NL2SQLPipeline
from config.config import Config

def test_single_query(
    nl_query: str,
    db_path: str,
    model_type: str = "thinking",
    verbose: bool = True
) -> dict:
    """
    Test a single NL query with HF model.

    Args:
        nl_query: Natural language query
        db_path: Path to database
        model_type: 'thinking' or 'coder'
        verbose: Print detailed output

    Returns:
        Test results dictionary
    """
    # Select model
    if model_type == "thinking":
        model_name = "hf:qwen3-235b-thinking"
    else:
        model_name = "hf:qwen3-480b-coder"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing with {model_name}")
        print(f"Query: {nl_query}")
        print(f"Database: {db_path}")
        print(f"{'='*60}\n")

    # Initialize components
    try:
        # Create HF LLM client
        llm_client = HFLLMClientAdapter(model_name)

        # Database connector
        db_connector = DatabaseConnector(db_path)
        schema = db_connector.get_schema()

        # Create config with HF model
        config = Config()
        config.llm_config.model_name = model_name

        # Initialize pipeline with HF client
        pipeline = NL2SQLPipeline(config)
        # Replace LLM clients with HF adapter
        pipeline.llm_client = llm_client
        pipeline.subtask_extractor.llm_client = llm_client
        pipeline.query_plan_generator.llm_client = llm_client
        pipeline.progressive_executor.llm_client = llm_client
        pipeline.semantic_reward_model.llm_client = llm_client

        # Run pipeline
        start_time = time.time()
        output = pipeline.run(nl_query, schema, db_connector)
        execution_time = time.time() - start_time

        # Prepare results
        result = {
            "success": True,
            "nl_query": nl_query,
            "predicted_sql": output.final_sql,
            "execution_time": execution_time,
            "execution_success": output.execution_success,
            "total_reward": output.total_reward,
            "num_iterations": output.num_iterations,
            "num_branches": output.num_branches_explored,
            "model": model_name
        }

        if verbose:
            print(f"SQL Generated: {output.final_sql}")
            print(f"Execution Success: {output.execution_success}")
            print(f"Total Reward: {output.total_reward}")
            print(f"Time: {execution_time:.2f}s")

            if output.execution_success and output.execution_result:
                print(f"\nResults Preview:")
                print(output.execution_result[:200])

    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "nl_query": nl_query,
            "model": model_name
        }

        if verbose:
            print(f"Error: {e}")

    finally:
        # Cleanup
        if 'llm_client' in locals():
            llm_client.cleanup()

    return result

def test_batch_queries(
    queries: list,
    db_path: str,
    model_type: str = "thinking",
    output_file: str = None
) -> list:
    """
    Test multiple queries.

    Args:
        queries: List of NL queries
        db_path: Database path
        model_type: Model type to use
        output_file: Optional file to save results

    Returns:
        List of results
    """
    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing query...")
        result = test_single_query(query, db_path, model_type)
        results.append(result)

    # Calculate statistics
    successful = sum(1 for r in results if r.get("success", False))
    exec_successful = sum(1 for r in results if r.get("execution_success", False))
    avg_time = sum(r.get("execution_time", 0) for r in results) / len(results) if results else 0
    avg_reward = sum(r.get("total_reward", 0) for r in results) / len(results) if results else 0

    stats = {
        "total_queries": len(queries),
        "successful": successful,
        "execution_successful": exec_successful,
        "average_time": avg_time,
        "average_reward": avg_reward,
        "model": model_type
    }

    print(f"\n{'='*60}")
    print("Batch Test Summary")
    print(f"{'='*60}")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Successful: {stats['successful']}/{stats['total_queries']}")
    print(f"Execution Success: {stats['execution_successful']}/{stats['total_queries']}")
    print(f"Average Time: {stats['average_time']:.2f}s")
    print(f"Average Reward: {stats['average_reward']:.2f}")

    # Save results
    if output_file:
        output_data = {
            "statistics": stats,
            "results": results
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Test NL2SQL with HuggingFace models")
    parser.add_argument("--model", choices=["thinking", "coder"], default="thinking",
                      help="Model type to use")
    parser.add_argument("--query", type=str,
                      help="Single query to test")
    parser.add_argument("--queries_file", type=str,
                      help="JSON file with list of queries")
    parser.add_argument("--db_path", type=str, required=True,
                      help="Path to SQLite database")
    parser.add_argument("--output", type=str,
                      help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                      help="Verbose output")

    args = parser.parse_args()

    # Determine queries to test
    if args.query:
        # Single query
        result = test_single_query(
            args.query,
            args.db_path,
            args.model,
            args.verbose
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)

    elif args.queries_file:
        # Multiple queries from file
        with open(args.queries_file, 'r') as f:
            queries = json.load(f)

        if isinstance(queries, dict):
            # Extract queries from dict (e.g., BIRD format)
            queries = [q.get("question", q) for q in queries.get("queries", queries)]

        test_batch_queries(
            queries,
            args.db_path,
            args.model,
            args.output
        )

    else:
        # Default test queries
        test_queries = [
            "What is the total number of employees?",
            "Find the average salary by department",
            "List the top 5 highest paid employees with their departments"
        ]

        print("No query specified, using default test queries:")
        for q in test_queries:
            print(f"  - {q}")

        test_batch_queries(
            test_queries,
            args.db_path,
            args.model,
            args.output
        )

if __name__ == "__main__":
    main()