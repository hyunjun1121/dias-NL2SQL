# Information Retriever (IR) Integration Dossier

이 문서는 CHESS 레포지터리(`../CHESS`)로부터 가져온 구성요소와 본 프로젝트(Triple Kim NL2SQL 파이프라인)에서 새로 구현한 부분을 명확히 구분하고, 재현 가능한 통합 절차를 제공한다. 전문 용어는 영어로 유지한다.

## Imported Directly from CHESS
- `workflow/agents/information_retriever/tool_kit/extract_keywords.py`
- `workflow/agents/information_retriever/tool_kit/retrieve_entity.py`
- `workflow/agents/information_retriever/tool_kit/retrieve_context.py`
- `workflow/system_state.py`
- `runner/database_manager.py`
- `runner/task.py`
- `llm/models.py`, `llm/prompts.py`, `llm/parsers.py`
- Templates (located under `../CHESS/templates`):
  - `template_extract_keywords.txt`
  - `template_agent_prompt.txt` (보존용, deterministic 실행에서는 사용하지 않음)
- Parser: `python_list_output_parser`

이들 파일은 복사하지 않고, `sys.path`를 이용해 런타임 import 한다.

## Implemented in Triple Kim Repository
- `ir/ir_integration.py`: CHESS IR Tool들을 deterministic 순서(ExtractKeywords → RetrieveEntity → RetrieveContext)로 실행하고, schema를 EPFL 프롬프트 포맷({table: {"columns": [{"name": ...}]}})으로 변환.
- `config/config.py`: `IRConfig` dataclass 추가 및 `EPFLHyunjunConfig`에 편입.
- `pipeline/main_pipeline.py`: Sub-task extraction 이전에 IR를 호출하여 pruned schema 사용. IR 산출물을 `PipelineOutput.ir_artifacts`에 저장.
- `model/data_structures.py`: `PipelineOutput`에 `ir_artifacts` 필드 추가.
- `requirements.txt`: langchain, langchain-openai, langchain-chroma, chromadb 의존성 추가.
- `docs/CHESS_TEMPLATES.md`: 사용 템플릿/파서에 대한 요약.

## Execution Flow
1. `run_ir_and_prune`가 CHESS IR Tool을 호출하여 keywords, similar columns, value examples, column descriptions를 수집한다.
2. 수집된 결과를 기반으로 schema를 pruning하여 Sub-task extractor와 Query plan generator에 전달한다.
3. IR 산출물(`ir_artifacts`)은 최종 결과물에 저장되어 향후 분석 또는 프롬프트 강화 단계에서 활용 가능하다.

## Required Preparation
1. Benchmark 압축 해제: `benchmark/bird.zip`, `benchmark/Spider2.zip`을 해제하여 `<benchmark>/<dataset>` 구조로 구성한다.
2. CHESS 전처리(`../CHESS/src/preprocess.py`)를 실행하여 각 데이터베이스에 대해 MinHash LSH(`preprocessed/*.pkl`)와 Chroma Vector DB(`context_vector_db/`)를 생성한다.
3. OpenAI Embeddings `text-embedding-3-small` 사용을 위한 `OPENAI_API_KEY` 환경 변수 설정.

## Configuration Snippet
```python
ir.enabled = True
ir.db_root_path = "./data"
ir.data_mode = "dev"
ir.extract_keywords_template = "extract_keywords"
ir.extract_keywords_engine = "gpt-4o-mini"
ir.extract_keywords_temperature = 0.2
ir.extract_keywords_parser = "python_list_output_parser"
ir.retrieve_context_top_k = 5
```

## Failure Handling
- 필요한 템플릿 또는 인덱스가 누락된 경우 파이프라인은 예외를 발생시키고 즉시 중단한다.
- pipeline output에는 마지막으로 사용된 schema와 IR 아티팩트가 남지 않으므로, 문제 해결 시 해당 파일과 환경 변수를 확인한다.

