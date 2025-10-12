"""
Example: Compare CHASE-SQL vs Our Progressive Approach

Shows the difference between one-shot and progressive execution.
"""

from baseline.chase_sql import CHASESQLBaseline
from pipeline.main_pipeline import EPFLHyunjunPipeline
from config.config import get_bird_config
from utils.llm_client import LLMClient
from utils.database_executor import DatabaseExecutor


def example_comparison():
    """Compare both methods on a single example."""

    # Example query and schema
    nl_query = "Show employees with salary over 50000 in Engineering department"
    schema = {
        "employees": {
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "TEXT"},
                {"name": "salary", "type": "REAL"},
                {"name": "department", "type": "TEXT"}
            ]
        }
    }
    db_path = "example.db"  # Your database path

    print("="*80)
    print("Comparing CHASE-SQL vs Our Progressive Approach")
    print("="*80)

    # Initialize components
    llm_client = LLMClient(model_name="gpt-4o")  # or "deepseek-r1" for cluster
    db_executor = DatabaseExecutor(db_path)
    config = get_bird_config()

    # Method 1: CHASE-SQL (One-shot)
    print("\n[1] CHASE-SQL (One-shot generation)")
    print("-" * 80)

    chase_baseline = CHASESQLBaseline(llm_client, db_executor, config.progressive_execution.__dict__)
    chase_result = chase_baseline.generate_sql(nl_query, schema)

    print(f"Query Plan: {chase_result['query_plan']}")
    print(f"\nGenerated SQL (one-shot):")
    print(f"  {chase_result['sql']}")
    print(f"\nExecution: {'✓ Success' if chase_result['success'] else '✗ Failed'}")
    if chase_result['success']:
        print(f"Rows returned: {chase_result['execution_result'].get('num_rows', 0)}")

    # Method 2: Our Progressive Approach
    print("\n\n[2] Our Progressive Approach")
    print("-" * 80)

    our_pipeline = EPFLHyunjunPipeline(config)
    our_result = our_pipeline.run(nl_query, schema, db_path)

    print(f"Sub-tasks extracted:")
    for task in our_result.subtasks.tasks:
        print(f"  - [{task.confidence:.2f}] {task.operation}")

    print(f"\nQuery Plan: (same 3-step reasoning)")
    print(f"  {our_result.query_plan.to_text()[:100]}...")

    print(f"\nProgressive execution:")
    for i, task in enumerate(our_result.context.completed_tasks, 1):
        print(f"  Iteration {i}: {task.operation}")
        print(f"    SQL: {task.sql_fragment[:50]}...")
        print(f"    Reward: {task.reward}")

    print(f"\nFinal SQL:")
    print(f"  {our_result.final_sql}")
    print(f"\nExecution: {'✓ Success' if our_result.execution_success else '✗ Failed'}")
    print(f"Semantic Correct: {our_result.semantic_correctness.is_correct if our_result.semantic_correctness else 'N/A'}")
    print(f"Total Reward: {our_result.total_reward}")
    print(f"Iterations: {our_result.num_iterations}")

    # Comparison
    print("\n\n[3] Comparison")
    print("-" * 80)
    print(f"{'Aspect':<25} | {'CHASE-SQL':<30} | {'Our Method':<30}")
    print("-" * 80)
    print(f"{'Approach':<25} | {'One-shot generation':<30} | {'Progressive execution':<30}")
    print(f"{'Iterations':<25} | {'1':<30} | {our_result.num_iterations:<30}")
    print(f"{'Context accumulation':<25} | {'No':<30} | {'Yes':<30}")
    print(f"{'Error recovery':<25} | {'No retry':<30} | {'Syntax retry + Semantic':<30}")
    print(f"{'Execution success':<25} | {str(chase_result['success']):<30} | {str(our_result.execution_success):<30}")

    if our_result.semantic_correctness:
        print(f"{'Semantic correctness':<25} | {'N/A (not evaluated)':<30} | {str(our_result.semantic_correctness.is_correct):<30}")

    print("\n" + "="*80)


if __name__ == "__main__":
    example_comparison()
