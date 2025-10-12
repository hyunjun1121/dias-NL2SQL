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
        Execute sub-tasks progressively.

        Algorithm:
        1. Get highest confidence task
        2. Generate SQL fragment
        3. Execute immediately
        4. Calculate reward
        5. If good, accumulate context
        6. Recalculate remaining confidence
        7. Repeat
        """
        context = ExecutionContext()
        max_iterations = self.config.get('max_iterations', 10)

        for iteration in range(max_iterations):
            # Get next executable task
            executable_tasks = subtasks.get_executable_tasks()
            if not executable_tasks:
                break

            task = executable_tasks[0]  # Highest confidence

            # Generate SQL fragment
            task.sql_fragment = self._generate_sql_fragment(task, context, schema)

            # Execute
            exec_result = self.db_executor.execute(task.sql_fragment)
            task.execution_result = exec_result

            # Calculate reward (returns Dict now)
            reward_dict = self.reward_model.calculate_reward(
                predicted_sql=task.sql_fragment,
                nl_query=subtasks.nl_query,
                schema=schema,
                execution_result=exec_result
            )
            task.reward = reward_dict['total_reward']

            # Update context if successful
            # New threshold: must be 1.0 (both execution and semantic must pass)
            if exec_result['success'] and reward_dict['total_reward'] == 1.0:
                context.update_from_task(task)
                context.current_sql = self._assemble_sql(context)

        return context

    def _generate_sql_fragment(
        self,
        task: SubTask,
        context: ExecutionContext,
        schema: Dict
    ) -> str:
        """Generate SQL fragment for task."""
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
