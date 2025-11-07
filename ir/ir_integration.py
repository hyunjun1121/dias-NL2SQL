"""Deterministic integration of CHESS Information Retriever (IR) into EPFL pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple


def _ensure_chess_src_on_path() -> Path:
    """Ensure CHESS/src is importable and return the path."""
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # dias-NL2SQL root
    legacy_root = here.parents[2]  # original expectation (/home/USER)

    env_override = os.getenv("CHESS_SRC_PATH")
    candidates = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend([
        repo_root / "CHESS" / "src",
        legacy_root / "CHESS" / "src",
    ])

    for chess_src in candidates:
        if chess_src.exists():
            if str(chess_src) not in sys.path:
                sys.path.insert(0, str(chess_src))
            return chess_src

    raise RuntimeError(
        "Unable to locate CHESS/src. Looked in: "
        + ", ".join(str(p) for p in candidates)
    )


_ensure_chess_src_on_path()

from runner.database_manager import DatabaseManager  # type: ignore
from runner.task import Task  # type: ignore
from workflow.system_state import SystemState  # type: ignore
from workflow.agents.information_retriever.tool_kit.extract_keywords import ExtractKeywords  # type: ignore
from workflow.agents.information_retriever.tool_kit.retrieve_entity import RetrieveEntity  # type: ignore
from workflow.agents.information_retriever.tool_kit.retrieve_context import RetrieveContext  # type: ignore


def _set_db_root_env(db_root_path: str) -> Path:
    path = Path(db_root_path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[1] / path).resolve()
    os.environ.setdefault("DB_ROOT_PATH", str(path))
    return path


def _build_state(question: str, db_id: str, evidence: str, data_mode: str) -> SystemState:
    DatabaseManager(db_mode=data_mode, db_id=db_id)
    tentative_schema = DatabaseManager().get_db_schema()
    task = Task(question_id=0, db_id=db_id, question=question, evidence=evidence)
    state = SystemState(task=task, tentative_schema=tentative_schema, execution_history=[])
    return state


def _run_extract_keywords(state: SystemState, *, template: str, engine_name: str, temperature: float, parser_name: str):
    tool = ExtractKeywords(
        template_name=template,
        engine_config={"engine_name": engine_name, "temperature": temperature},
        parser_name=parser_name,
    )
    tool(state)


def _run_retrieve_entity(state: SystemState):
    tool = RetrieveEntity()
    tool(state)


def _run_retrieve_context(state: SystemState, top_k: int):
    tool = RetrieveContext(top_k=top_k)
    tool(state)


def _epfl_pruned_schema(state: SystemState) -> Dict[str, Any]:
    source = state.similar_columns or state.tentative_schema
    pruned: Dict[str, Any] = {}
    for table, columns in source.items():
        pruned[table] = {"columns": [{"name": col} for col in columns]}
    return pruned


def run_ir_and_prune(
    *,
    question: str,
    db_id: str,
    data_mode: str,
    db_root_path: str,
    extract_keywords_template: str,
    extract_keywords_engine: str,
    extract_keywords_temperature: float,
    extract_keywords_parser: str,
    retrieve_context_top_k: int,
    evidence: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute CHESS IR deterministically and return pruned schema + artifacts."""

    _set_db_root_env(db_root_path)
    state = _build_state(question=question, db_id=db_id, evidence=evidence, data_mode=data_mode)

    _run_extract_keywords(
        state,
        template=extract_keywords_template,
        engine_name=extract_keywords_engine,
        temperature=extract_keywords_temperature,
        parser_name=extract_keywords_parser,
    )
    _run_retrieve_entity(state)
    _run_retrieve_context(state, top_k=retrieve_context_top_k)

    pruned_schema = _epfl_pruned_schema(state)
    artifacts = {
        "keywords": state.keywords,
        "similar_columns": state.similar_columns,
        "schema_with_examples": state.schema_with_examples,
        "schema_with_descriptions": state.schema_with_descriptions,
    }
    return pruned_schema, artifacts
