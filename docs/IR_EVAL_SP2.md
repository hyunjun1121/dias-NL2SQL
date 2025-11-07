#+ IR Evaluation on Spider 2.0 (Table-Level)

본 문서는 Spider 2.0 데이터셋에서 제공되는 ground-truth table 정보를 활용하여, CHESS 스타일 IR(Information Retriever)의 table retrieval 성능을 측정하는 방법을 기록한다. 전문 용어는 영어로 유지한다.

## Imported vs Implemented
- Imported (via runtime import from `../CHESS/src`): IR tools, Database utilities (see `docs/IR_INTEGRATION.md`).
- Implemented (here): `scripts/eval_ir_spider.py` — deterministic IR 호출 후 table-level recall/precision/F1 산출.

## Ground-Truth Tables
- Spider 2.0 샘플에서 `gold_tables`(또는 동등 필드)가 제공되면 이를 직접 사용한다.
- 해당 필드가 없으면, gold SQL을 `sqlglot`으로 parse하여 table set을 추출한다(보수적 fallback).

## Metrics
- Table Recall@all = |Retrieved ∩ Gold| / |Gold|
- Table Precision@all = |Retrieved ∩ Gold| / |Retrieved|
- F1 = 2PR / (P + R)
- Macro averages: per-sample 평균
- Micro aggregates: 전체 TP/FP/FN 기반 집계

## Running the Evaluator

```bash
python scripts/eval_ir_spider.py \
  --spider_json <path>/dev.json \
  --db_root_path ./data \
  --mode dev \
  --output results/ir_eval_spider_dev.json \
  --limit 200
```

- `--db_root_path`는 CHESS 레이아웃의 root (`<root>/<mode>_databases/<db_id>/...`)를 가리킨다.
- IR 설정(템플릿/엔진/파서/temperature/top-k)은 CLI 옵션으로 조정 가능하며, 기본값은 `IRConfig`와 일치.

## Output Structure

- Summary: macro/micro recall/precision/F1, 표본 수
- Details: 각 질문별 retrieved/gold table set과 per-sample score

```json
{
  "summary": {
    "count": 200,
    "macro_recall": 0.84,
    "macro_precision": 0.62,
    "macro_f1": 0.71,
    "micro_tp": 330,
    "micro_fp": 202,
    "micro_fn": 62,
    "micro_recall": 0.84,
    "micro_precision": 0.62
  },
  "details": [
    {"question_id": 12, "db_id": "concert_singer", ...}
  ]
}
```

## Notes
- 평가 전제: CHESS 전처리(LSH/Chroma) 완료, `OPENAI_API_KEY` 설정, `./data` 경로가 올바르게 구성되어야 한다.
- Spider 2.0의 table ground-truth가 없는 샘플은 fallback(SQL parse)로 대체된다.
