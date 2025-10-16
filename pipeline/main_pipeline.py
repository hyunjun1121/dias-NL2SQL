"""
Main Pipeline Orchestrator.
Triple Kim's design: Agent Pipeline with Reward-driven Progressive Execution.
"""

import os
import time
from typing import Dict
from model.data_structures import PipelineOutput, BranchCollection, Branch
from model.subtask_extractor import ConfidentSubTaskExtractor
from model.query_plan_generator import QueryPlanGenerator
from model.progressive_executor import ProgressiveExecutor
from model.semantic_reward import SemanticRewardModel
from evaluation.error_analyzer import ErrorAnalyzer
from utils.llm_client import LLMClient
from utils.database_executor import DatabaseExecutor
from ir.ir_integration import run_ir_and_prune


class EPFLHyunjunPipeline:
    """
    Main pipeline implementing Kyungmin's vision:
    - Confident sub-task extraction
    - Progressive execution with context accumulation
    - Semantic reward-driven approach
    """

    def __init__(self, config):
        self.config = config

        # Initialize components
        self.llm_client = LLMClient(
            model_name=config.llm.model_name,
            api_key=config.llm.api_key
        )

        self.subtask_extractor = ConfidentSubTaskExtractor(
            self.llm_client,
            config.subtask.__dict__
        )

        self.query_plan_generator = QueryPlanGenerator(
            self.llm_client,
            config.query_plan.__dict__
        )

        self.reward_model = SemanticRewardModel(
            self.llm_client,
            config.semantic_reward.__dict__
        )

        self.error_analyzer = ErrorAnalyzer(
            config.error_analysis.__dict__
        )

    def run(
        self,
        nl_query: str,
        schema: Dict,
        db_path: str
    ) -> PipelineOutput:
        """
        Run complete pipeline.

        Algorithm:
        1. Extract confident sub-tasks (LLM generates confidence)
        2. Generate query plan (CHASE-SQL 3-step)
        3. Progressive execution:
           - Execute highest confidence task
           - Calculate semantic reward
           - Accumulate context
           - Recalculate remaining confidence
        4. Return best result
        """
        start_time = time.time()

        # Initialize database executor
        db_executor = DatabaseExecutor(db_path)

        # Derive db_id for IR (filename without extension)
        db_id = os.path.splitext(os.path.basename(db_path))[0]

        ir_artifacts = None
        if getattr(self.config, 'ir', None) and self.config.ir.enabled:
            pruned_schema, ir_artifacts = run_ir_and_prune(
                question=nl_query,
                db_id=db_id,
                data_mode=self.config.ir.data_mode,
                db_root_path=self.config.ir.db_root_path,
                extract_keywords_template=self.config.ir.extract_keywords_template,
                extract_keywords_engine=self.config.ir.extract_keywords_engine,
                extract_keywords_temperature=self.config.ir.extract_keywords_temperature,
                extract_keywords_parser=self.config.ir.extract_keywords_parser,
                retrieve_context_top_k=self.config.ir.retrieve_context_top_k,
            )
            schema = pruned_schema

        # Initialize progressive executor
        progressive_executor = ProgressiveExecutor(
            self.llm_client,
            db_executor,
            self.reward_model,
            self.config.progressive_execution.__dict__
        )

        # Step 1: Extract confident sub-tasks
        subtasks = self.subtask_extractor.extract(nl_query, schema)

        # Step 2: Generate query plan
        query_plan = self.query_plan_generator.generate(nl_query, schema)

        # Step 3: Progressive execution
        context = progressive_executor.execute_progressive(subtasks, schema)

        # Get final SQL
        final_sql = context.current_sql

        # Execute final SQL
        final_result = db_executor.execute(final_sql)

        # Calculate final reward (returns Dict now)
        final_reward_dict = self.reward_model.calculate_reward(
            predicted_sql=final_sql,
            nl_query=nl_query,
            schema=schema,
            execution_result=final_result
        )

        # Package output
        total_time = time.time() - start_time

        # Create SemanticCorrectness object from reward dict
        from model.data_structures import SemanticCorrectness
        semantic_correctness = None
        if final_reward_dict['semantic_correctness'] is not None:
            semantic_correctness = SemanticCorrectness(
                is_correct=final_reward_dict['semantic_correctness'],
                reasoning=final_reward_dict['semantic_reasoning']
            )

        output = PipelineOutput(
            final_sql=final_sql,
            execution_result=final_result,
            total_reward=final_reward_dict['total_reward'],
            subtasks=subtasks,
            query_plan=query_plan,
            context=context,
            branches=BranchCollection(branches=[Branch(
                branch_id=0,
                parent_branch_id=None,
                tasks=subtasks.tasks,
                context=context,
                cumulative_reward=final_reward_dict['total_reward']
            )]),
            semantic_correctness=semantic_correctness,
            execution_success=final_result['success'],
            total_time=total_time,
            num_iterations=len(context.completed_tasks),
            num_branches_explored=1,
            ir_artifacts=ir_artifacts
        )

        return output
