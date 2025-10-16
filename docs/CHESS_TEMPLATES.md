# CHESS Templates & Parsers Overview

본 문서는 IR 통합에서 재사용하는 템플릿과 파서의 역할을 요약한다.

## Templates (from ../CHESS/templates)
- `template_extract_keywords.txt`: 질문과 증거에서 핵심 keyword를 추출하는 Prompt
- `template_agent_prompt.txt`: CHESS Agent loop용 system prompt (현 통합에서는 사용하지 않지만, 추후 agent 기반 실행을 위해 유지)

## Parsers (from ../CHESS/src/llm/parsers.py)
- `python_list_output_parser`: LLM 응답을 Python list로 변환하는 parser (keywords 추출에 사용)

## Usage in Triple Kim Pipeline
IR deterministic 경로는 `template_extract_keywords.txt`와 `python_list_output_parser` 조합만을 사용한다. Agent loop를 활성화하려면 `template_agent_prompt.txt`를 비롯한 추가 템플릿을 그대로 재사용하면 된다.

