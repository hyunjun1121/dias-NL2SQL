"""
Confident Sub-task Extractor.
Kyungmin's approach: LLM generates sub-tasks with confidence scores.
"""

import json
from typing import Dict, List
from .data_structures import SubTask, SubTaskCollection
from utils.llm_client import LLMClient


class ConfidentSubTaskExtractor:
    """
    Extract sub-tasks with confidence scores using LLM.
    Kyungmin's key insight: LLM directly generates confidence.
    """

    def __init__(self, llm_client: LLMClient, config: Dict):
        self.llm_client = llm_client
        self.config = config

    def extract(self, nl_query: str, schema: Dict) -> SubTaskCollection:
        """
        Extract confident sub-tasks from natural language query.

        Args:
            nl_query: Natural language query
            schema: Database schema

        Returns:
            SubTaskCollection with tasks sorted by confidence
        """
        prompt = self._build_extraction_prompt(nl_query, schema)
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=2048
        )

        tasks = self._parse_subtasks(response)
        return SubTaskCollection(tasks=tasks, nl_query=nl_query, schema=schema)

    def recalculate_confidence(
        self,
        remaining_tasks: List[SubTask],
        completed_context: Dict
    ) -> List[SubTask]:
        """
        Recalculate confidence for remaining tasks with updated context.
        Kyungmin's approach: After each task, recalculate remaining confidence.

        Args:
            remaining_tasks: Tasks not yet executed
            completed_context: Context from completed tasks

        Returns:
            Updated tasks with new confidence scores
        """
        prompt = self._build_recalculation_prompt(remaining_tasks, completed_context)
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024
        )

        updated_confidences = self._parse_confidence_updates(response)

        # Update tasks
        for task in remaining_tasks:
            if task.task_id in updated_confidences:
                task.confidence = updated_confidences[task.task_id]

        return remaining_tasks

    def _build_extraction_prompt(self, nl_query: str, schema: Dict) -> str:
        """Build prompt for sub-task extraction."""
        prompt = """You are an expert at breaking down SQL queries into confident sub-tasks.

Your task: Analyze the natural language query and database schema, then extract atomic sub-tasks with confidence scores.

Natural Language Query:
{query}

Database Schema:
{schema}

Instructions:
1. Break the query into atomic operations (table selection, filtering, joining, aggregation, etc.)
2. For each sub-task, assign a confidence score (0.0 to 1.0) based on:
   - How clearly the operation is specified in the query
   - Whether required schema elements exist
   - Whether the operation is unambiguous
3. Identify dependencies between tasks (which tasks must complete before others)

Output Format (JSON):
{{
  "subtasks": [
    {{
      "task_id": 1,
      "operation": "SELECT FROM employees",
      "operation_type": "table_selection",
      "confidence": 0.95,
      "reasoning": "Table 'employees' explicitly mentioned and exists in schema",
      "dependencies": []
    }},
    {{
      "task_id": 2,
      "operation": "WHERE department = 'Engineering'",
      "operation_type": "filter",
      "confidence": 0.92,
      "reasoning": "Column 'department' exists and value 'Engineering' is clear",
      "dependencies": [1]
    }}
  ]
}}

Respond with JSON only.
""".format(
            query=nl_query,
            schema=self._format_schema(schema)
        )
        return prompt

    def _build_recalculation_prompt(
        self,
        remaining_tasks: List[SubTask],
        completed_context: Dict
    ) -> str:
        """Build prompt for confidence recalculation."""
        prompt = """You are recalculating confidence scores for remaining SQL sub-tasks.

Context from completed tasks:
{context}

Remaining tasks:
{tasks}

Instructions:
Recalculate confidence (0.0 to 1.0) for each remaining task based on:
1. Information gained from completed tasks
2. Whether dependencies are satisfied
3. Whether the task is now clearer or more ambiguous

Output Format (JSON):
{{
  "updated_confidences": {{
    "task_id": new_confidence,
    ...
  }}
}}

Respond with JSON only.
""".format(
            context=json.dumps(completed_context, indent=2),
            tasks=json.dumps([t.to_dict() for t in remaining_tasks], indent=2)
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

    def _parse_subtasks(self, response: str) -> List[SubTask]:
        """Parse LLM response to SubTask objects."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            # Try to extract JSON from thinking mode text
            import re
            json_match = re.search(r'\{.*"subtasks".*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            data = json.loads(response)
            subtasks = data.get('subtasks', [])

            tasks = []
            for st in subtasks:
                task = SubTask(
                    task_id=st['task_id'],
                    operation=st['operation'],
                    operation_type=st['operation_type'],
                    confidence=st['confidence'],
                    reasoning=st['reasoning'],
                    dependencies=st.get('dependencies', [])
                )
                tasks.append(task)

            return tasks

        except json.JSONDecodeError as e:
            print(f"Error parsing subtasks: {e}")
            print(f"Response: {response}")
            return []

    def _parse_confidence_updates(self, response: str) -> Dict[int, float]:
        """Parse confidence update response."""
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
            updates = data.get('updated_confidences', {})

            # Convert string keys to int
            return {int(k): float(v) for k, v in updates.items()}

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing confidence updates: {e}")
            return {}
