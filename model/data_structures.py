"""
Core data structures for EPFL Hyunjun's NL2SQL Pipeline.
Kyungmin's design philosophy: Confident sub-task with progressive execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


# ==================== Sub-task Elements ====================

@dataclass
class SubTask:
    """
    Sub-task with LLM-generated confidence.
    Kyungmin's approach: LLM generates both task and confidence.
    """
    task_id: int
    operation: str  # e.g., "SELECT FROM employees"
    operation_type: str  # e.g., "table_selection", "filter", "join", "aggregate"
    confidence: float  # LLM-generated confidence (0.0 to 1.0)
    reasoning: str  # Why this confidence score
    dependencies: List[int] = field(default_factory=list)  # Depends on which tasks
    sql_fragment: Optional[str] = None  # Generated SQL fragment
    execution_result: Optional[Any] = None  # Result after execution
    reward: Optional[float] = None  # Reward after evaluation

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'operation': self.operation,
            'operation_type': self.operation_type,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'dependencies': self.dependencies,
            'sql_fragment': self.sql_fragment,
            'execution_result': str(self.execution_result) if self.execution_result else None,
            'reward': self.reward
        }


@dataclass
class SubTaskCollection:
    """Collection of sub-tasks with confidence ordering."""
    tasks: List[SubTask]
    nl_query: str
    schema: Dict

    def get_highest_confidence_task(self) -> Optional[SubTask]:
        """Get task with highest confidence that hasn't been executed."""
        unexecuted = [t for t in self.tasks if t.execution_result is None]
        if not unexecuted:
            return None
        return max(unexecuted, key=lambda t: t.confidence)

    def get_executable_tasks(self) -> List[SubTask]:
        """Get tasks whose dependencies are satisfied."""
        completed_ids = {t.task_id for t in self.tasks if t.execution_result is not None}
        executable = []
        for task in self.tasks:
            if task.execution_result is not None:
                continue  # Already executed
            if all(dep_id in completed_ids for dep_id in task.dependencies):
                executable.append(task)
        return sorted(executable, key=lambda t: t.confidence, reverse=True)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'nl_query': self.nl_query,
            'tasks': [t.to_dict() for t in self.tasks],
            'num_tasks': len(self.tasks),
            'completed_tasks': len([t for t in self.tasks if t.execution_result is not None])
        }


# ==================== Query Plan Elements (CHASE-SQL style) ====================

@dataclass
class QueryPlanStep:
    """
    Single step in query plan (CHASE-SQL 3-step reasoning).
    Human-readable format.
    """
    step_number: int
    step_type: str  # "find_tables", "perform_operations", "select_columns"
    description: str  # Human-readable description
    reasoning: str  # Why this step
    entities: List[str]  # Tables, columns, or operations involved

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'step_number': self.step_number,
            'step_type': self.step_type,
            'description': self.description,
            'reasoning': self.reasoning,
            'entities': self.entities
        }


@dataclass
class QueryPlan:
    """
    Complete query plan (CHASE-SQL style).
    3-step human-readable reasoning:
    1. Find relevant tables
    2. Perform operations (filter, join, aggregate)
    3. Select final columns and return results
    """
    steps: List[QueryPlanStep]
    nl_query: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'nl_query': self.nl_query,
            'steps': [s.to_dict() for s in self.steps],
            'num_steps': len(self.steps)
        }

    def to_text(self) -> str:
        """Convert to human-readable text."""
        text = f"Query Plan for: {self.nl_query}\n\n"
        for step in self.steps:
            text += f"Step {step.step_number}: {step.step_type}\n"
            text += f"  Description: {step.description}\n"
            text += f"  Reasoning: {step.reasoning}\n"
            text += f"  Entities: {', '.join(step.entities)}\n\n"
        return text


# ==================== Execution Context ====================

@dataclass
class ExecutionContext:
    """
    Context accumulated during progressive execution.
    Kyungmin's approach: Execute and accumulate results step by step.
    """
    completed_tasks: List[SubTask] = field(default_factory=list)
    intermediate_results: List[Any] = field(default_factory=list)
    current_sql: str = ""
    current_tables: List[str] = field(default_factory=list)
    current_columns: List[str] = field(default_factory=list)
    current_filters: List[str] = field(default_factory=list)
    current_joins: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def update_from_task(self, task: SubTask):
        """Update context from completed task."""
        self.completed_tasks.append(task)
        if task.execution_result:
            self.intermediate_results.append(task.execution_result)

        # Update SQL components based on operation type
        if task.operation_type == "table_selection":
            # Extract table names from SQL fragment
            if task.sql_fragment:
                tables = self._extract_tables(task.sql_fragment)
                self.current_tables.extend(tables)

        elif task.operation_type == "filter":
            if task.sql_fragment:
                self.current_filters.append(task.sql_fragment)

        elif task.operation_type == "join":
            if task.sql_fragment:
                self.current_joins.append(task.sql_fragment)

    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL fragment."""
        # Simple extraction (can be improved with proper parsing)
        import re
        match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if match:
            return [match.group(1)]
        return []

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'completed_tasks': len(self.completed_tasks),
            'intermediate_results': len(self.intermediate_results),
            'current_sql': self.current_sql,
            'current_tables': self.current_tables,
            'current_columns': self.current_columns,
            'current_filters': self.current_filters,
            'current_joins': self.current_joins,
            'metadata': self.metadata
        }


# ==================== Reward Elements ====================

@dataclass
class SemanticCorrectness:
    """
    Semantic correctness evaluation - Simplified binary approach.
    LLM judges: Is the SQL semantically correct for the NL query?
    """
    is_correct: bool  # Binary judgment from LLM
    reasoning: str  # LLM's reasoning for the judgment

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_correct': self.is_correct,
            'reasoning': self.reasoning
        }


@dataclass
class RewardScore:
    """
    Complete reward score - Simplified binary approach.
    Execution must succeed, then semantic correctness is judged by LLM.
    Total reward: 1.0 if both pass, 0.0 otherwise.
    """
    execution_success: bool
    semantic_correctness: Optional[SemanticCorrectness]  # Only if execution succeeded
    execution_error: Optional[str] = None  # Error message if execution failed

    total_reward: float = 0.0

    def calculate_total(self):
        """Calculate total reward."""
        if not self.execution_success:
            self.total_reward = 0.0
        elif self.semantic_correctness is None:
            # Execution success but no semantic judgment (shouldn't happen)
            self.total_reward = 0.5
        elif self.semantic_correctness.is_correct:
            self.total_reward = 1.0
        else:
            self.total_reward = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'execution_success': self.execution_success,
            'execution_error': self.execution_error,
            'semantic_correctness': self.semantic_correctness.to_dict() if self.semantic_correctness else None,
            'total_reward': self.total_reward
        }


# ==================== Multi-branch Elements ====================

@dataclass
class Branch:
    """
    Single branch in multi-branch reasoning (MS rStar style).
    Created when error detected.
    """
    branch_id: int
    parent_branch_id: Optional[int]
    tasks: List[SubTask]
    context: ExecutionContext
    cumulative_reward: float = 0.0
    is_active: bool = True
    failure_point: Optional[int] = None  # Which task failed

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'branch_id': self.branch_id,
            'parent_branch_id': self.parent_branch_id,
            'tasks': [t.to_dict() for t in self.tasks],
            'context': self.context.to_dict(),
            'cumulative_reward': self.cumulative_reward,
            'is_active': self.is_active,
            'failure_point': self.failure_point
        }


@dataclass
class BranchCollection:
    """Collection of branches for beam search."""
    branches: List[Branch]
    beam_size: int = 5

    def add_branch(self, branch: Branch):
        """Add branch and maintain beam size."""
        self.branches.append(branch)
        # Sort by reward and keep top-k
        self.branches = sorted(
            self.branches,
            key=lambda b: b.cumulative_reward,
            reverse=True
        )[:self.beam_size]

    def get_best_branch(self) -> Optional[Branch]:
        """Get branch with highest reward."""
        if not self.branches:
            return None
        return self.branches[0]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'branches': [b.to_dict() for b in self.branches],
            'beam_size': self.beam_size,
            'num_branches': len(self.branches)
        }


# ==================== Error Analysis Elements ====================

class ErrorType(Enum):
    """Error types (Kyungmin: semantic errors are priority)."""
    # Semantic errors (high priority)
    WRONG_TABLE = "wrong_table"
    WRONG_COLUMN = "wrong_column"
    WRONG_JOIN = "wrong_join"
    WRONG_FILTER = "wrong_filter"
    WRONG_AGGREGATION = "wrong_aggregation"

    # Execution errors (medium priority)
    EXECUTION_TIMEOUT = "execution_timeout"
    INVALID_OPERATION = "invalid_operation"

    # Syntax errors (low priority - Kyungmin: these don't matter much)
    SYNTAX_ERROR = "syntax_error"


@dataclass
class ErrorAnalysis:
    """
    Error analysis for failed execution.
    Kyungmin's focus: Semantic errors matter, syntax errors don't.
    """
    error_type: ErrorType
    error_message: str
    failed_task: Optional[SubTask]
    suggested_fixes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'error_type': self.error_type.value,
            'error_message': self.error_message,
            'failed_task': self.failed_task.to_dict() if self.failed_task else None,
            'suggested_fixes': self.suggested_fixes
        }


# ==================== Pipeline Output ====================

@dataclass
class PipelineOutput:
    """Final output from the pipeline."""
    final_sql: str
    execution_result: Any
    total_reward: float

    # Intermediate steps
    subtasks: SubTaskCollection
    query_plan: QueryPlan
    context: ExecutionContext
    branches: BranchCollection

    # Evaluation
    semantic_correctness: SemanticCorrectness
    execution_success: bool

    # Metadata
    total_time: float
    num_iterations: int
    num_branches_explored: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'final_sql': self.final_sql,
            'execution_result': str(self.execution_result),
            'total_reward': self.total_reward,
            'subtasks': self.subtasks.to_dict(),
            'query_plan': self.query_plan.to_dict(),
            'context': self.context.to_dict(),
            'branches': self.branches.to_dict(),
            'semantic_correctness': self.semantic_correctness.to_dict(),
            'execution_success': self.execution_success,
            'total_time': self.total_time,
            'num_iterations': self.num_iterations,
            'num_branches_explored': self.num_branches_explored
        }
