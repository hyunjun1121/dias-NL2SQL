"""
Debug version of Qwen model testing with detailed LLM I/O logging.
Tests single sample with full pipeline step tracking.
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DebugLLMClient:
    """
    Wrapper for LLM client that logs all inputs and outputs.
    """

    def __init__(self, base_client, log_dir: str = "debug_logs"):
        self.base_client = base_client
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.call_counter = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create session log file
        self.session_log_file = self.log_dir / f"session_{self.session_id}.json"
        self.session_logs = []

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with logging."""
        self.call_counter += 1

        # Log input
        input_log = {
            "call_id": self.call_counter,
            "timestamp": datetime.now().isoformat(),
            "type": "input",
            "prompt": prompt,
            "kwargs": kwargs
        }

        print(f"\n{'='*80}")
        print(f"[LLM Call #{self.call_counter}]")
        print(f"{'='*80}")
        print(f"PROMPT ({len(prompt)} chars):")
        print("-" * 40)
        print(prompt[:1000])  # First 1000 chars
        if len(prompt) > 1000:
            print(f"... (truncated, total {len(prompt)} chars)")
        print("-" * 40)

        # Call actual LLM
        start_time = time.time()
        try:
            response = self.base_client.generate(prompt, **kwargs)
            elapsed_time = time.time() - start_time
            success = True
            error = None
        except Exception as e:
            response = ""
            elapsed_time = time.time() - start_time
            success = False
            error = str(e)
            print(f"ERROR: {e}")

        # Log output
        output_log = {
            "call_id": self.call_counter,
            "timestamp": datetime.now().isoformat(),
            "type": "output",
            "response": response,
            "elapsed_time": elapsed_time,
            "success": success,
            "error": error
        }

        print(f"\nRESPONSE ({len(response)} chars):")
        print("-" * 40)
        print(response[:1000])  # First 1000 chars
        if len(response) > 1000:
            print(f"... (truncated, total {len(response)} chars)")
        print("-" * 40)
        print(f"Time: {elapsed_time:.2f}s")
        print(f"{'='*80}\n")

        # Save logs
        call_log = {
            "call_id": self.call_counter,
            "input": input_log,
            "output": output_log
        }
        self.session_logs.append(call_log)

        # Write individual call log
        call_log_file = self.log_dir / f"call_{self.session_id}_{self.call_counter:03d}.json"
        with open(call_log_file, 'w', encoding='utf-8') as f:
            json.dump(call_log, f, indent=2, ensure_ascii=False)

        # Update session log
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "total_calls": self.call_counter,
                "logs": self.session_logs
            }, f, indent=2, ensure_ascii=False)

        return response

    def get_summary(self) -> Dict:
        """Get summary of all calls."""
        return {
            "session_id": self.session_id,
            "total_calls": self.call_counter,
            "log_dir": str(self.log_dir),
            "session_log": str(self.session_log_file)
        }


def test_single_sample_with_debug(
    model_type: str = "thinking",
    use_hf: bool = True,
    use_ollama: bool = False
) -> Dict:
    """
    Test single sample with detailed debugging.

    Args:
        model_type: 'thinking' or 'coder'
        use_hf: Use HuggingFace models
        use_ollama: Use Ollama (fallback)

    Returns:
        Test results with debug information
    """

    # Test query and database
    test_query = "Find the total number of employees in the Engineering department"
    test_db = "data/bird/dev/dev_databases/company/company.sqlite"

    print(f"\n{'#'*80}")
    print("NL2SQL Pipeline Debug Test")
    print(f"{'#'*80}")
    print(f"Model Type: {model_type}")
    print(f"Use HuggingFace: {use_hf}")
    print(f"Use Ollama: {use_ollama}")
    print(f"Test Query: {test_query}")
    print(f"Database: {test_db}")
    print(f"{'#'*80}\n")

    # Initialize LLM client
    if use_hf:
        print("Initializing HuggingFace model...")
        from utils.hf_llm_client import HFLLMClientAdapter

        if model_type == "thinking":
            base_client = HFLLMClientAdapter("hf:qwen3-235b-thinking")
        else:
            base_client = HFLLMClientAdapter("hf:qwen3-480b-coder")

    elif use_ollama:
        print("Initializing Ollama model...")
        from utils.llm_client import LLMClient

        if model_type == "thinking":
            base_client = LLMClient("ollama:qwen3:32b")
        else:
            base_client = LLMClient("ollama:qwen3:32b")
    else:
        print("Using mock LLM for testing...")
        # Mock client for testing
        class MockLLMClient:
            def generate(self, prompt: str, **kwargs) -> str:
                return json.dumps({
                    "subtasks": [
                        {
                            "task_id": 1,
                            "operation": "SELECT FROM employees",
                            "operation_type": "table_selection",
                            "confidence": 0.95,
                            "reasoning": "Mock response",
                            "dependencies": []
                        }
                    ]
                })
        base_client = MockLLMClient()

    # Wrap with debug client
    llm_client = DebugLLMClient(base_client, log_dir=f"debug_logs_{model_type}")

    # Initialize pipeline components
    print("\nInitializing pipeline components...")

    from utils.database_connector import DatabaseConnector
    from model.subtask_extractor import ConfidentSubTaskExtractor
    from model.query_plan_generator import QueryPlanGenerator
    from model.progressive_executor import ProgressiveExecutor
    from model.semantic_reward import SemanticRewardModel
    from config.config import Config

    # Database
    if Path(test_db).exists():
        db_connector = DatabaseConnector(test_db)
        schema = db_connector.get_schema()
        print(f"Database loaded: {len(schema)} tables")
    else:
        print(f"Warning: Database not found, using mock schema")
        schema = {
            "employees": {
                "columns": ["id", "name", "department", "salary"]
            }
        }
        db_connector = None

    # Config
    config = Config()

    # Pipeline components with debug client
    print("\n" + "="*60)
    print("STARTING PIPELINE EXECUTION")
    print("="*60)

    results = {
        "query": test_query,
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "steps": {}
    }

    try:
        # Step 1: Sub-task Extraction
        print("\n[Step 1] SUB-TASK EXTRACTION")
        print("-" * 60)

        subtask_extractor = ConfidentSubTaskExtractor(llm_client, config.subtask_config.__dict__)
        subtask_collection = subtask_extractor.extract(test_query, schema)

        results["steps"]["subtask_extraction"] = {
            "success": True,
            "num_tasks": len(subtask_collection.tasks),
            "tasks": [t.to_dict() for t in subtask_collection.tasks]
        }

        print(f"Extracted {len(subtask_collection.tasks)} sub-tasks")
        for task in subtask_collection.tasks:
            print(f"  - Task {task.task_id}: {task.operation} (confidence: {task.confidence})")

    except Exception as e:
        results["steps"]["subtask_extraction"] = {
            "success": False,
            "error": str(e)
        }
        print(f"Error in sub-task extraction: {e}")

    try:
        # Step 2: Query Plan Generation
        print("\n[Step 2] QUERY PLAN GENERATION")
        print("-" * 60)

        query_plan_generator = QueryPlanGenerator(llm_client, config.query_plan_config.__dict__)
        query_plan = query_plan_generator.generate(test_query, schema, subtask_collection)

        results["steps"]["query_plan"] = {
            "success": True,
            "num_steps": len(query_plan.steps) if query_plan else 0,
            "plan": query_plan.to_dict() if query_plan else None
        }

        if query_plan:
            print(f"Generated query plan with {len(query_plan.steps)} steps")
            for step in query_plan.steps:
                print(f"  - Step {step.step_number}: {step.step_type}")

    except Exception as e:
        results["steps"]["query_plan"] = {
            "success": False,
            "error": str(e)
        }
        print(f"Error in query plan generation: {e}")

    try:
        # Step 3: Progressive Execution
        print("\n[Step 3] PROGRESSIVE EXECUTION")
        print("-" * 60)

        if db_connector:
            progressive_executor = ProgressiveExecutor(llm_client, db_connector, config.executor_config.__dict__)
            final_sql, execution_result, context = progressive_executor.execute(
                subtask_collection,
                query_plan
            )

            results["steps"]["progressive_execution"] = {
                "success": True,
                "final_sql": final_sql,
                "execution_success": execution_result.get("success", False),
                "context": context.to_dict() if context else None
            }

            print(f"Final SQL: {final_sql}")
            print(f"Execution success: {execution_result.get('success', False)}")

        else:
            results["steps"]["progressive_execution"] = {
                "success": False,
                "error": "No database connector"
            }
            final_sql = "SELECT COUNT(*) FROM employees WHERE department = 'Engineering'"
            execution_result = {"success": False}

    except Exception as e:
        results["steps"]["progressive_execution"] = {
            "success": False,
            "error": str(e)
        }
        print(f"Error in progressive execution: {e}")
        final_sql = ""
        execution_result = {"success": False}

    try:
        # Step 4: Semantic Reward
        print("\n[Step 4] SEMANTIC REWARD EVALUATION")
        print("-" * 60)

        if final_sql and execution_result.get("success"):
            semantic_reward = SemanticRewardModel(llm_client, config.reward_config.__dict__)
            reward_result = semantic_reward.calculate_reward(
                final_sql,
                test_query,
                schema,
                execution_result
            )

            results["steps"]["semantic_reward"] = {
                "success": True,
                "total_reward": reward_result.get("total_reward", 0),
                "semantic_correctness": reward_result.get("semantic_correctness", False),
                "reasoning": reward_result.get("semantic_reasoning", "")
            }

            print(f"Total reward: {reward_result.get('total_reward', 0)}")
            print(f"Semantic correctness: {reward_result.get('semantic_correctness', False)}")

        else:
            results["steps"]["semantic_reward"] = {
                "success": False,
                "error": "No valid SQL or execution failed"
            }

    except Exception as e:
        results["steps"]["semantic_reward"] = {
            "success": False,
            "error": str(e)
        }
        print(f"Error in semantic reward: {e}")

    # Summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)

    llm_summary = llm_client.get_summary()
    results["llm_summary"] = llm_summary

    print(f"\nLLM Call Summary:")
    print(f"  Total calls: {llm_summary['total_calls']}")
    print(f"  Session ID: {llm_summary['session_id']}")
    print(f"  Logs saved to: {llm_summary['log_dir']}")
    print(f"  Session log: {llm_summary['session_log']}")

    # Save final results
    results_file = Path(f"debug_logs_{model_type}") / f"pipeline_results_{llm_summary['session_id']}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline results saved to: {results_file}")

    # Cleanup
    if hasattr(base_client, 'cleanup'):
        base_client.cleanup()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug test for Qwen models in NL2SQL pipeline")
    parser.add_argument("--model", choices=["thinking", "coder"], default="thinking",
                      help="Model type to test")
    parser.add_argument("--use_hf", action="store_true",
                      help="Use HuggingFace models")
    parser.add_argument("--use_ollama", action="store_true",
                      help="Use Ollama models")
    parser.add_argument("--mock", action="store_true",
                      help="Use mock LLM for testing")

    args = parser.parse_args()

    # Determine which backend to use
    if args.mock:
        use_hf = False
        use_ollama = False
    elif args.use_hf:
        use_hf = True
        use_ollama = False
    elif args.use_ollama:
        use_hf = False
        use_ollama = True
    else:
        # Default to Ollama
        use_hf = False
        use_ollama = True

    # Run test
    results = test_single_sample_with_debug(
        model_type=args.model,
        use_hf=use_hf,
        use_ollama=use_ollama
    )

    # Print final summary
    print("\n" + "#"*80)
    print("TEST COMPLETE")
    print("#"*80)

    for step_name, step_result in results["steps"].items():
        status = "✓" if step_result.get("success") else "✗"
        print(f"{status} {step_name}: {'Success' if step_result.get('success') else 'Failed'}")
        if not step_result.get("success"):
            print(f"    Error: {step_result.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()