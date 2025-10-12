"""
Progressive Executor.
Kyungmin's approach: Execute highest confidence task, accumulate context, repeat.
"""

from typing import Dict
from .data_structures import SubTask, SubTaskCollection, ExecutionContext
from .semantic_reward import SemanticRewardModel
from utils.llm_client import LLMClient
from utils.database_executor import DatabaseExecutor


class ProgressiveExecutor:
    """
    Execute sub-tasks progressively with context accumulation.
    Kyungmin's key insight: Execute and build, don't plan then execute.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        db_executor: DatabaseExecutor,
        reward_model: SemanticRewardModel,
        config: Dict
    ):
        self.llm_client = llm_client
        self.db_executor = db_executor
        self.reward_model = reward_model
        self.config = config

    def execute_progressive(
        self,
        subtasks: SubTaskCollection,
        schema: Dict
    ) -> ExecutionContext:
        """
        Execute sub-tasks progressively with error recovery.

        Algorithm (Kyungmin's approach):
        1. Get highest confidence task
        2. Generate SQL fragment
        3. Execute immediately
        4. Error Recovery Hierarchy:
           - 1st: Syntax error check → Retry if syntax error
           - 2nd: Semantic error check → Rollback if semantic error
        5. If good, accumulate context
        6. Repeat
        """
        context = ExecutionContext()
        max_iterations = self.config.get('max_iterations', 10)
        max_syntax_retries = self.config.get('max_syntax_retries', 2)

        for iteration in range(max_iterations):
            # Get next executable task
            executable_tasks = subtasks.get_executable_tasks()
            if not executable_tasks:
                break

            task = executable_tasks[0]  # Highest confidence

            # Generate SQL fragment with syntax error recovery
            task.sql_fragment, exec_result = self._generate_and_execute_with_retry(
                task, context, schema, max_syntax_retries
            )
            task.execution_result = exec_result

            # Step 2: Semantic error check (only if syntax OK)
            if exec_result['success']:
                reward_dict = self.reward_model.calculate_reward(
                    predicted_sql=task.sql_fragment,
                    nl_query=subtasks.nl_query,
                    schema=schema,
                    execution_result=exec_result
                )
                task.reward = reward_dict['total_reward']

                # Update context if semantically correct
                if reward_dict['total_reward'] == 1.0:
                    context.update_from_task(task)
                    context.current_sql = self._assemble_sql(context)
                else:
                    # Semantic error - record but continue
                    # (Could rollback or create alternative branch here)
                    context.metadata['semantic_errors'] = context.metadata.get('semantic_errors', 0) + 1
            else:
                # Syntax error persisted after retries
                task.reward = 0.0
                context.metadata['syntax_errors'] = context.metadata.get('syntax_errors', 0) + 1

        return context

    def _generate_and_execute_with_retry(
        self,
        task: SubTask,
        context: ExecutionContext,
        schema: Dict,
        max_retries: int
    ) -> tuple:
        """
        Generate SQL and retry on syntax errors (Kyungmin's 1st hierarchy).

        Returns:
            (sql, execution_result)
        """
        error_feedback = None

        for attempt in range(max_retries + 1):
            # Generate SQL with error feedback if retry
            sql = self._generate_sql_fragment(
                task, context, schema, error_feedback
            )

            # Step 1: Execute to check syntax
            exec_result = self.db_executor.execute(sql)

            if exec_result['success']:
                # Syntax OK - return
                return sql, exec_result
            else:
                # Syntax error - prepare feedback for retry
                error_msg = exec_result.get('error', 'Unknown error')

                # Check if it's a syntax error (not semantic)
                if self._is_syntax_error(error_msg):
                    error_feedback = f"Previous SQL had syntax error: {error_msg}"
                    # Continue to retry
                else:
                    # Not a syntax error (e.g., table not found = semantic error)
                    # Don't retry, let semantic check handle it
                    return sql, exec_result

        # All retries exhausted
        return sql, exec_result

    def _is_syntax_error(self, error_msg: str) -> bool:
        """Check if error is syntax-related (retriable)."""
        syntax_keywords = [
            'syntax error',
            'near',
            'unrecognized token',
            'incomplete input',
            'mismatched input',
            'expected'
        ]
        error_lower = error_msg.lower()
        return any(kw in error_lower for kw in syntax_keywords)

    def _generate_sql_fragment(
        self,
        task: SubTask,
        context: ExecutionContext,
        schema: Dict,
        error_feedback: str = None
    ) -> str:
        """Generate SQL fragment for task with optional error feedback."""
        prompt = f"""Generate SQL for this sub-task:

Task: {task.operation}
Type: {task.operation_type}

Current Context:
- Completed tasks: {[t.operation for t in context.completed_tasks]}
- Current SQL: {context.current_sql}
- Current tables: {context.current_tables}
- Current filters: {context.current_filters}

Schema:
{self._format_schema(schema)}
"""

        # Add error feedback if retry
        if error_feedback:
            prompt += f"\n{error_feedback}\nPlease fix the syntax error and regenerate the SQL.\n"

        prompt += """
Generate the SQL fragment that accomplishes this task.
If building on previous SQL, extend it appropriately.
Respond with SQL only."""

        sql = self.llm_client.generate(prompt=prompt, temperature=0.0, max_tokens=512)
        return self._clean_sql(sql)

    def _assemble_sql(self, context: ExecutionContext) -> str:
        """Assemble final SQL from context."""
        if context.completed_tasks:
            last_task = context.completed_tasks[-1]
            if last_task.sql_fragment:
                return last_task.sql_fragment
        return context.current_sql

    def _format_schema(self, schema: Dict) -> str:
        """Format schema."""
        formatted = ""
        for table, info in schema.items():
            formatted += f"{table}: {', '.join([c['name'] if isinstance(c, dict) else c for c in info.get('columns', [])])}\n"
        return formatted

    def _clean_sql(self, sql: str) -> str:
        """Clean SQL."""
        sql = sql.strip()
        if sql.startswith('```sql'):
            sql = sql[6:]
        elif sql.startswith('```'):
            sql = sql[3:]
        if sql.endswith('```'):
            sql = sql[:-3]
        return sql.strip()
