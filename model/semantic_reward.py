"""
Semantic Reward Model.
Updated approach:
1. Execution Success (필수) - SQL 실행 성공 여부
2. Semantic Correctness (실행 성공 시에만) - LLM이 binary 판단
"""

from typing import Dict, Optional
from utils.llm_client import LLMClient


class SemanticRewardModel:
    """
    Calculate reward based on:
    1. Execution Success (required)
    2. Semantic Correctness (LLM binary judgment)

    No efficiency metric - focus on correctness only.
    """

    def __init__(self, llm_client: LLMClient, config: Dict):
        self.llm_client = llm_client
        self.config = config

    def calculate_reward(
        self,
        predicted_sql: str,
        nl_query: str,
        schema: Dict,
        execution_result: Dict
    ) -> Dict:
        """
        Calculate reward score.

        Args:
            predicted_sql: Generated SQL
            nl_query: Natural language query
            schema: Database schema
            execution_result: Result from SQL execution

        Returns:
            Dictionary with:
            - execution_success: bool
            - semantic_correctness: bool (only if execution succeeded)
            - semantic_reasoning: str (only if execution succeeded)
            - total_reward: float (1.0 if both pass, 0.0 otherwise)
        """
        # Step 1: Check execution success
        execution_success = execution_result.get('success', False)

        if not execution_success:
            # Execution failed - stop here
            return {
                'execution_success': False,
                'execution_error': execution_result.get('error', 'Unknown error'),
                'semantic_correctness': None,
                'semantic_reasoning': None,
                'total_reward': 0.0
            }

        # Step 2: LLM judges semantic correctness (only if execution succeeded)
        semantic_result = self._llm_judge_semantic_correctness(
            predicted_sql=predicted_sql,
            nl_query=nl_query,
            execution_result=execution_result,
            schema=schema
        )

        return {
            'execution_success': True,
            'execution_error': None,
            'semantic_correctness': semantic_result['correct'],
            'semantic_reasoning': semantic_result['reasoning'],
            'total_reward': 1.0 if semantic_result['correct'] else 0.0
        }

    def _llm_judge_semantic_correctness(
        self,
        predicted_sql: str,
        nl_query: str,
        execution_result: Dict,
        schema: Dict
    ) -> Dict:
        """
        Use LLM to judge if SQL is semantically correct.

        Args:
            predicted_sql: Generated SQL
            nl_query: Natural language query
            execution_result: SQL execution result
            schema: Database schema

        Returns:
            {
                'correct': bool,
                'reasoning': str
            }
        """
        prompt = self._build_semantic_judgment_prompt(
            predicted_sql, nl_query, execution_result, schema
        )

        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=512
        )

        # Parse LLM response
        parsed = self._parse_judgment_response(response)
        return parsed

    def _build_semantic_judgment_prompt(
        self,
        predicted_sql: str,
        nl_query: str,
        execution_result: Dict,
        schema: Dict
    ) -> str:
        """Build prompt for LLM semantic judgment."""
        prompt = f"""You are evaluating if a generated SQL query correctly answers a natural language question.

Natural Language Query:
{nl_query}

Database Schema:
{self._format_schema(schema)}

Generated SQL:
{predicted_sql}

Execution Result:
- Success: Yes
- Number of rows: {execution_result.get('num_rows', 0)}
- Sample result (first 3 rows): {str(execution_result.get('result', [])[:3])}

Task:
Determine if the generated SQL SEMANTICALLY CORRECTLY answers the natural language query.

Consider:
1. Are the correct tables used?
2. Are the correct columns selected?
3. Are the filters/conditions correct?
4. Are joins (if any) correct?
5. Are aggregations (if any) correct?
6. Does the result make sense for the question?

Respond in this EXACT format:
CORRECT: [YES/NO]
REASONING: [Your detailed reasoning in 2-3 sentences]

Example:
CORRECT: YES
REASONING: The SQL correctly selects from the employees table and filters by department='Engineering' and salary>50000, which matches the query asking for employees with high salary in Engineering department.

Now evaluate:
"""
        return prompt

    def _format_schema(self, schema: Dict) -> str:
        """Format schema for prompt."""
        formatted = ""
        for table_name, table_info in schema.items():
            formatted += f"\n{table_name}:\n"
            if 'columns' in table_info:
                for col in table_info['columns']:
                    col_name = col['name'] if isinstance(col, dict) else col
                    col_type = col.get('type', 'UNKNOWN') if isinstance(col, dict) else ''
                    formatted += f"  - {col_name}"
                    if col_type:
                        formatted += f" ({col_type})"
                    formatted += "\n"
        return formatted

    def _parse_judgment_response(self, response: str) -> Dict:
        """Parse LLM judgment response."""
        response = response.strip()

        # Extract CORRECT: YES/NO
        correct = False
        if 'CORRECT: YES' in response.upper() or 'CORRECT:YES' in response.upper():
            correct = True
        elif 'CORRECT: NO' in response.upper() or 'CORRECT:NO' in response.upper():
            correct = False
        else:
            # Fallback: check for YES/NO keywords
            if 'YES' in response.upper().split('\n')[0]:
                correct = True

        # Extract REASONING
        reasoning = ""
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if 'REASONING:' in line.upper():
                # Get everything after REASONING:
                reasoning_start = line.upper().index('REASONING:') + len('REASONING:')
                reasoning = line[reasoning_start:].strip()
                # Also include following lines
                if i + 1 < len(lines):
                    reasoning += " " + " ".join(lines[i+1:])
                break

        if not reasoning:
            # Fallback: use entire response
            reasoning = response

        return {
            'correct': correct,
            'reasoning': reasoning.strip()
        }


class RewardScore:
    """
    Simplified reward score.
    Binary: correct (1.0) or incorrect (0.0).
    """

    def __init__(
        self,
        execution_success: bool,
        semantic_correctness: Optional[bool],
        semantic_reasoning: Optional[str],
        execution_error: Optional[str] = None
    ):
        self.execution_success = execution_success
        self.semantic_correctness = semantic_correctness
        self.semantic_reasoning = semantic_reasoning
        self.execution_error = execution_error

        # Calculate total reward
        if not execution_success:
            self.total_reward = 0.0
        elif semantic_correctness is None:
            # Execution success but no semantic judgment (shouldn't happen)
            self.total_reward = 0.5
        elif semantic_correctness:
            self.total_reward = 1.0
        else:
            self.total_reward = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'execution_success': self.execution_success,
            'execution_error': self.execution_error,
            'semantic_correctness': self.semantic_correctness,
            'semantic_reasoning': self.semantic_reasoning,
            'total_reward': self.total_reward
        }

    def __repr__(self) -> str:
        return f"RewardScore(exec={self.execution_success}, semantic={self.semantic_correctness}, reward={self.total_reward})"
