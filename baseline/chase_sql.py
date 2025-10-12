"""
CHASE-SQL Baseline Implementation.

Reference: CHASE-SQL paper approach
- 3-step reasoning (like our QueryPlanGenerator)
- But: One-shot SQL generation (NOT progressive)
- Used as comparison baseline for our progressive approach
"""

from typing import Dict
from model.query_plan_generator import QueryPlanGenerator
from utils.llm_client import LLMClient
from utils.database_executor import DatabaseExecutor


class CHASESQLBaseline:
    """
    CHASE-SQL baseline for comparison.

    Key difference from our approach:
    - CHASE-SQL: Query plan → One-shot full SQL generation
    - Our approach: Query plan → Progressive execution with context accumulation
    """

    def __init__(self, llm_client: LLMClient, db_executor: DatabaseExecutor, config: Dict):
        self.llm_client = llm_client
        self.db_executor = db_executor
        self.config = config
        self.query_plan_generator = QueryPlanGenerator(llm_client, config)

    def generate_sql(self, nl_query: str, schema: Dict) -> Dict:
        """
        Generate SQL using CHASE-SQL approach.

        Algorithm:
        1. Generate 3-step query plan
        2. Generate full SQL in one shot based on plan
        3. Execute and return result

        Returns:
            {
                'sql': generated SQL,
                'execution_result': execution result,
                'query_plan': query plan,
                'success': bool
            }
        """
        # Step 1: Generate query plan (3-step reasoning)
        query_plan = self.query_plan_generator.generate(nl_query, schema)

        # Step 2: One-shot SQL generation based on plan
        sql = self._generate_full_sql(nl_query, schema, query_plan)

        # Step 3: Execute
        exec_result = self.db_executor.execute(sql)

        return {
            'sql': sql,
            'execution_result': exec_result,
            'query_plan': query_plan.to_dict(),
            'success': exec_result['success']
        }

    def _generate_full_sql(self, nl_query: str, schema: Dict, query_plan) -> str:
        """
        Generate full SQL in one shot (CHASE-SQL style).

        Unlike our progressive approach, this generates the complete SQL at once.
        """
        prompt = f"""You are an expert SQL generator. Generate a complete SQL query based on the query plan.

Natural Language Query:
{nl_query}

Database Schema:
{self._format_schema(schema)}

Query Plan (3 steps):
{query_plan.to_text()}

Task:
Generate a COMPLETE SQL query that accomplishes all steps in the query plan.
Do NOT generate fragments - generate the full, executable SQL query.

Respond with SQL only, no explanations."""

        sql = self.llm_client.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024
        )

        return self._clean_sql(sql)

    def _format_schema(self, schema: Dict) -> str:
        """Format schema for prompt."""
        formatted = ""
        for table, info in schema.items():
            formatted += f"\n{table}:\n"
            if 'columns' in info:
                for col in info['columns']:
                    col_name = col['name'] if isinstance(col, dict) else col
                    col_type = col.get('type', '') if isinstance(col, dict) else ''
                    formatted += f"  - {col_name}"
                    if col_type:
                        formatted += f" ({col_type})"
                    formatted += "\n"
        return formatted

    def _clean_sql(self, sql: str) -> str:
        """Clean SQL output."""
        sql = sql.strip()
        if sql.startswith('```sql'):
            sql = sql[6:]
        elif sql.startswith('```'):
            sql = sql[3:]
        if sql.endswith('```'):
            sql = sql[:-3]
        return sql.strip()
