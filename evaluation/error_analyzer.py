"""
Error Analyzer with semantic categorization.
Kyungmin's focus: Semantic errors matter, syntax errors don't.
"""

from typing import Dict
from model.data_structures import ErrorType, ErrorAnalysis, SubTask


class ErrorAnalyzer:
    """Analyze errors with focus on semantic correctness."""

    def __init__(self, config: Dict):
        self.config = config

    def analyze_error(
        self,
        predicted_sql: str,
        execution_result: Dict,
        task: SubTask = None
    ) -> ErrorAnalysis:
        """Analyze error and categorize."""
        if execution_result['success']:
            return None

        error_msg = execution_result.get('error', '')
        error_type = self._categorize_error(error_msg, predicted_sql)

        fixes = self._suggest_fixes(error_type, predicted_sql)

        return ErrorAnalysis(
            error_type=error_type,
            error_message=error_msg,
            failed_task=task,
            suggested_fixes=fixes
        )

    def _categorize_error(self, error_msg: str, sql: str) -> ErrorType:
        """Categorize error type."""
        error_lower = error_msg.lower()

        # Semantic errors (high priority)
        if 'no such table' in error_lower:
            return ErrorType.WRONG_TABLE
        elif 'no such column' in error_lower:
            return ErrorType.WRONG_COLUMN
        elif 'ambiguous' in error_lower or 'join' in error_lower:
            return ErrorType.WRONG_JOIN

        # Syntax errors (low priority)
        elif 'syntax' in error_lower:
            return ErrorType.SYNTAX_ERROR

        # Execution errors
        elif 'timeout' in error_lower:
            return ErrorType.EXECUTION_TIMEOUT
        else:
            return ErrorType.INVALID_OPERATION

    def _suggest_fixes(self, error_type: ErrorType, sql: str) -> list:
        """Suggest fixes for error."""
        if error_type == ErrorType.WRONG_TABLE:
            return ["Check table name", "Verify schema", "Try similar table names"]
        elif error_type == ErrorType.WRONG_COLUMN:
            return ["Check column name", "Verify table schema", "Try similar column names"]
        elif error_type == ErrorType.WRONG_JOIN:
            return ["Check join conditions", "Verify foreign keys", "Try different join type"]
        else:
            return ["Review SQL syntax", "Check query logic"]
