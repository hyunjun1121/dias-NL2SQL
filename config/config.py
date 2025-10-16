"""
Configuration for EPFL Hyunjun's NL2SQL Pipeline.
Based on Kyungmin's research direction:
- Agent Pipeline with Reward Design
- Confident Sub-task Progressive Execution
- Semantic Correctness Focus
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None


@dataclass
class SubTaskConfig:
    """Configuration for sub-task extraction."""
    # LLM generates confidence directly
    use_llm_confidence: bool = True

    # Confidence threshold for immediate execution
    high_confidence_threshold: float = 0.85

    # Recalculate confidence after each task completion
    recalculate_after_execution: bool = True

    # Output format
    output_format: str = "json"  # JSON format with task + confidence


@dataclass
class QueryPlanConfig:
    """Configuration for query plan generation (CHASE-SQL style)."""
    # Human-readable 3-step reasoning
    use_three_step_reasoning: bool = True

    # Steps:
    # 1. Find relevant tables
    # 2. Perform operations (filter, join, aggregate)
    # 3. Select final columns
    include_step_reasoning: bool = True


@dataclass
class SemanticRewardConfig:
    """
    Configuration for semantic reward calculation.
    Simplified binary approach:
    1. Execution Success (required)
    2. Semantic Correctness (LLM binary judgment)
    No efficiency metric - focus on correctness only.
    """
    # LLM settings for semantic judgment
    judgment_temperature: float = 0.0
    judgment_max_tokens: int = 512


@dataclass
class ProgressiveExecutionConfig:
    """Configuration for progressive execution."""
    # Execute highest confidence task first
    sort_by_confidence: bool = True

    # Accumulate context after each execution
    accumulate_context: bool = True

    # Maximum iterations
    max_iterations: int = 10

    # Reward threshold for acceptance (now binary: must be 1.0)
    acceptance_threshold: float = 1.0

    # Error recovery (Kyungmin's hierarchy)
    max_syntax_retries: int = 2  # Retry syntax errors up to 2 times


@dataclass
class MultibranchConfig:
    """Configuration for multi-branch reasoning (MS rStar style)."""
    # Enable multi-branch when task fails
    enable_multibranch: bool = True

    # Number of alternative branches to generate
    num_alternatives: int = 3

    # Beam size for branch selection
    beam_size: int = 5

    # Maximum branch depth
    max_branch_depth: int = 3


@dataclass
class ErrorAnalysisConfig:
    """Configuration for error analysis."""
    # Kyungmin's focus: semantic errors, not syntax errors
    prioritize_semantic_errors: bool = True

    # Error categories
    track_table_errors: bool = True
    track_column_errors: bool = True
    track_join_errors: bool = True
    track_filter_errors: bool = True
    track_aggregation_errors: bool = True

    # Syntax errors (low priority)
    track_syntax_errors: bool = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    # Target: BIRD (Spider is too easy - Kyungmin)
    primary_dataset: str = "bird"

    # Also consider Spider v2
    secondary_dataset: str = "spider_v2"

    # Evaluation metrics
    evaluate_execution_accuracy: bool = True
    evaluate_semantic_correctness: bool = True  # Kyungmin's key metric
    evaluate_error_breakdown: bool = True
    evaluate_pipeline_failure_points: bool = True


@dataclass
class IRConfig:
    """Configuration for CHESS-style Information Retriever integration."""

    enabled: bool = True
    db_root_path: Optional[str] = "benchmark"
    data_mode: str = "dev"

    extract_keywords_template: str = "extract_keywords"
    extract_keywords_engine: str = "gpt-4o-mini"
    extract_keywords_temperature: float = 0.2
    extract_keywords_parser: str = "python_list_output_parser"

    retrieve_context_top_k: int = 5


@dataclass
class EPFLHyunjunConfig:
    """Main configuration for EPFL Hyunjun's pipeline."""

    # Component configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    subtask: SubTaskConfig = field(default_factory=SubTaskConfig)
    query_plan: QueryPlanConfig = field(default_factory=QueryPlanConfig)
    semantic_reward: SemanticRewardConfig = field(default_factory=SemanticRewardConfig)
    progressive_execution: ProgressiveExecutionConfig = field(default_factory=ProgressiveExecutionConfig)
    multibranch: MultibranchConfig = field(default_factory=MultibranchConfig)
    error_analysis: ErrorAnalysisConfig = field(default_factory=ErrorAnalysisConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    ir: IRConfig = field(default_factory=IRConfig)

    # Paths
    output_dir: str = "outputs"
    cache_dir: Optional[str] = None
    database_dir: str = "databases"

    # Logging
    verbose: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'llm': self.llm.__dict__,
            'subtask': self.subtask.__dict__,
            'query_plan': self.query_plan.__dict__,
            'semantic_reward': self.semantic_reward.__dict__,
            'progressive_execution': self.progressive_execution.__dict__,
            'multibranch': self.multibranch.__dict__,
            'error_analysis': self.error_analysis.__dict__,
            'benchmark': self.benchmark.__dict__,
            'ir': self.ir.__dict__,
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir,
            'database_dir': self.database_dir,
            'verbose': self.verbose,
            'log_level': self.log_level
        }


def get_default_config() -> EPFLHyunjunConfig:
    """Get default configuration."""
    return EPFLHyunjunConfig()


def get_bird_config() -> EPFLHyunjunConfig:
    """Get configuration optimized for BIRD dataset."""
    config = EPFLHyunjunConfig()
    config.benchmark.primary_dataset = "bird"
    config.progressive_execution.max_iterations = 10
    config.multibranch.num_alternatives = 3
    return config


def get_spider_v2_config() -> EPFLHyunjunConfig:
    """Get configuration optimized for Spider v2 dataset."""
    config = EPFLHyunjunConfig()
    config.benchmark.primary_dataset = "spider_v2"
    config.progressive_execution.max_iterations = 8
    config.multibranch.num_alternatives = 2
    return config
