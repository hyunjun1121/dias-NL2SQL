"""
Query Plan Generator based on CHASE-SQL.
Human-readable 3-step reasoning process.
"""

import json
from typing import Dict, List
from .data_structures import QueryPlan, QueryPlanStep
from utils.llm_client import LLMClient


class QueryPlanGenerator:
    """
    Generate query plan using CHASE-SQL style 3-step reasoning.

    Steps:
    1. Find relevant tables
    2. Perform operations (filter, join, aggregate)
    3. Select final columns and return results
    """

    def __init__(self, llm_client: LLMClient, config: Dict):
        self.llm_client = llm_client
        self.config = config

    def generate(self, nl_query: str, schema: Dict) -> QueryPlan:
        """
        Generate query plan with 3-step reasoning.

        Args:
            nl_query: Natural language query
            schema: Database schema

        Returns:
            QueryPlan with 3 steps
        """
        prompt = self._build_query_plan_prompt(nl_query, schema)
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=2048
        )

        steps = self._parse_query_plan(response)
        return QueryPlan(steps=steps, nl_query=nl_query)

    def _build_query_plan_prompt(self, nl_query: str, schema: Dict) -> str:
        """Build prompt for query plan generation."""
        prompt = """You are creating a human-readable query execution plan.

Natural Language Query:
{query}

Database Schema:
{schema}

Instructions:
Create a 3-step query plan following this structure:

Step 1: Find Relevant Tables
- Identify which tables are needed
- Explain why each table is relevant

Step 2: Perform Operations
- Describe filtering conditions (WHERE)
- Describe join operations if needed
- Describe aggregations if needed
- Describe grouping if needed

Step 3: Select Final Columns
- Identify which columns to return
- Describe any ordering or limiting

Output Format (JSON):
{{
  "steps": [
    {{
      "step_number": 1,
      "step_type": "find_tables",
      "description": "Find the employees table",
      "reasoning": "The query asks about employees, so we need the employees table",
      "entities": ["employees"]
    }},
    {{
      "step_number": 2,
      "step_type": "perform_operations",
      "description": "Filter by department='Engineering' AND salary>50000",
      "reasoning": "Query specifies Engineering department with high salary",
      "entities": ["department", "salary", "filter", "AND"]
    }},
    {{
      "step_number": 3,
      "step_type": "select_columns",
      "description": "Return all columns for matching employees",
      "reasoning": "Query says 'show employees' implying all information",
      "entities": ["*"]
    }}
  ]
}}

Respond with JSON only.
""".format(
            query=nl_query,
            schema=self._format_schema(schema)
        )
        return prompt

    def _format_schema(self, schema: Dict) -> str:
        """Format schema for prompt."""
        formatted = ""
        for table_name, table_info in schema.items():
            formatted += f"\n{table_name}:\n"
            if 'columns' in table_info:
                for col in table_info['columns']:
                    col_name = col['name'] if isinstance(col, dict) else col
                    formatted += f"  - {col_name}\n"
        return formatted

    def _parse_query_plan(self, response: str) -> List[QueryPlanStep]:
        """Parse LLM response to QueryPlanStep objects."""
        try:
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            data = json.loads(response)
            steps_data = data.get('steps', [])

            steps = []
            for step_data in steps_data:
                step = QueryPlanStep(
                    step_number=step_data['step_number'],
                    step_type=step_data['step_type'],
                    description=step_data['description'],
                    reasoning=step_data['reasoning'],
                    entities=step_data['entities']
                )
                steps.append(step)

            return steps

        except json.JSONDecodeError as e:
            print(f"Error parsing query plan: {e}")
            print(f"Response: {response}")
            return []
