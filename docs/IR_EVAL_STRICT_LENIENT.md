# Column-Level Evaluation Modes (Strict vs. Lenient)

본 문서는 column-level 평가에서 SQL 파싱/매칭의 엄밀도를 선택하는 옵션을 설명한다.

## Strict Mode
- Alias-aware: sqlglot AST로 `table alias → real table` 매핑.
- Function unwrap: `LOWER(col)`, `ABS(col)` 등 함수 감싸기 내부의 원 컬럼을 인식.
- Unqualified column: 참조 테이블이 1개일 때만 해당 테이블로 귀속, 그 외엔 무시.
- Star(`*`) 확장: 참조 테이블에 한정하여 실제 열로 확장.

## Lenient Mode
- Alias-aware: 동일.
- Function unwrap: 동일.
- Unqualified column: 참조 테이블 셋 전체에 매핑(관대한 매칭).
- Star 확장: 참조 테이블 전체로 확장(보수적 recall 강조).

## 사용 방법

```bash
python scripts/eval_ir_spider_columns.py \
  --spider_json <path>/dev.json \
  --db_root_path benchmark \
  --mode dev \
  --output results/ir_eval_spider_columns_dev.json \
  --mode_variant strict   # 또는 lenient (기본)
```
