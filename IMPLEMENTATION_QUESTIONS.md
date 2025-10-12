# Implementation Questions and Confirmation Needed

## Date: 2025-10-11

구현 과정에서 결정이 필요하거나 확인이 필요한 부분들을 정리합니다.

---

## Category 1: Confidence Calculation

### Question 1.1: LLM Confidence Generation Format
**Current Implementation**:
```python
# LLM에게 JSON format으로 confidence를 생성하도록 요청
{
  "subtasks": [
    {"task_id": 1, "confidence": 0.95, "reasoning": "..."}
  ]
}
```

**Questions**:
1. **Temperature 설정**: 현재 temperature=0.0 (deterministic)으로 설정했습니다.
   - 장점: Consistent confidence scores
   - 단점: Exploration 부족
   - **Confirm**: Temperature 0.0이 맞나요? 아니면 0.2-0.3 정도로 약간의 variation을 주는게 나을까요?

2. **Confidence 범위**: 0.0-1.0 범위로 설정했습니다.
   - Alternative: 0-100 (percentage)
   - Alternative: Low/Medium/High (categorical)
   - **Confirm**: 0.0-1.0 범위가 적절한가요?

3. **Reasoning 필수 여부**: 현재 reasoning을 필수로 요구합니다.
   - 장점: Interpretability, debugging 용이
   - 단점: Token 사용량 증가
   - **Confirm**: Reasoning을 계속 요구할까요, 아니면 optional로 만들까요?

### Question 1.2: Confidence Recalculation Timing
**Current Implementation**:
```python
# 각 task 완료 후 매번 recalculate
for task in tasks:
    execute(task)
    recalculate_confidence(remaining_tasks, context)
```

**Questions**:
1. **Recalculation 빈도**:
   - Current: 매 task마다
   - Alternative 1: N개 task마다 (e.g., 2개마다)
   - Alternative 2: Reward가 낮을 때만
   - **Concern**: 매번 recalculate하면 API call이 많아짐 (cost, latency)
   - **Confirm**: 매번 recalculate가 맞나요?

2. **Context 전달 방식**:
   - Current: 모든 completed tasks의 operation + result 전달
   - Alternative: Summary만 전달 (e.g., "Completed: table selection, 1000 rows")
   - **Confirm**: Full context vs Summary?

---

## Category 2: SQL Generation

### Question 2.1: SQL Fragment vs Full SQL
**Current Implementation**:
```python
# Progressive: 각 task마다 incremental SQL 생성
Iteration 1: "SELECT * FROM employees"
Iteration 2: "SELECT * FROM employees WHERE department='Engineering'"
Iteration 3: "... AND salary>50000"
```

**Questions**:
1. **Generation 방식**:
   - Current: 이전 SQL을 extend
   - Alternative: 각 task의 fragment를 생성하고 나중에 assemble
   - **Trade-off**:
     - Extend: Context 유지, but 에러 누적 가능
     - Fragment: 독립적, but assembly logic 필요
   - **Confirm**: Extend 방식이 맞나요?

2. **Validation**: 각 iteration의 SQL을 바로 실행하는데, syntax error가 나면?
   - Current: 그냥 실패로 기록
   - Alternative: Syntax check 후 재생성 시도
   - **Confirm**: Syntax error handling strategy?

### Question 2.2: SQL Optimization
**Current Implementation**:
- Optimization 없음 (LLM이 생성한 그대로 사용)

**Questions**:
1. **Query Optimization**:
   - 예: `SELECT *` → `SELECT id, name, salary` (필요한 column만)
   - 예: Filter order optimization
   - **Concern**: Optimization이 semantic correctness를 해칠 수 있음
   - **Confirm**: Optimization을 추가할까요, 아니면 LLM에게 맡길까요?

---

## Category 3: Reward Calculation

### Question 3.1: LLM Semantic Judgment
**Current Implementation**:
```python
# Binary approach
# Step 1: Check execution success
# Step 2: LLM judges semantic correctness (if execution succeeded)
# Reward: 1.0 if both pass, 0.0 otherwise
```

**Questions**:
1. **LLM Judgment Temperature**: 현재 temperature=0.0 (deterministic)으로 설정했습니다.
   - 장점: Consistent judgments
   - 단점: No variation in edge cases
   - **Confirm**: Temperature 0.0이 맞나요?

2. **Prompt Design**: 현재 6가지 고려사항을 포함한 prompt 사용
   - 테이블, 컬럼, 조인, 필터, 집계, 결과 타당성
   - **Confirm**: Prompt에 더 추가하거나 제거할 내용이 있나요?

3. **Reasoning 활용**: LLM이 제공한 reasoning을 어떻게 활용할까요?
   - 단순 기록만?
   - Error recovery에 활용?
   - **Confirm**: Reasoning을 다음 iteration에 반영할까요?

### Question 3.2: Ground Truth Comparison
**Current Implementation**:
- Ground truth 없이 heuristic으로 평가
- 예: Table name이 query에 나오면 correct

**Questions**:
1. **Ground Truth Schema Links**:
   - `remaining_tasks.md`에 기록했듯이, ground truth 확보 방안 미정
   - **Options**:
     - Gold SQL을 parse해서 추출
     - Manual annotation
     - 없이 진행 (heuristic만 사용)
   - **Confirm**: 당장 어떻게 진행할까요?

2. **Evaluation 기준**:
   - Current: NL query와 schema만 보고 판단
   - Alternative: Gold SQL과 비교
   - **Confirm**: Gold SQL comparison을 구현해야 하나요?

---

## Category 4: Error Handling

### Question 4.1: Error Recovery Strategy
**Current Implementation**:
```python
if execution_result['success'] == False:
    # Just record the error, move to next iteration
    error_analysis = error_analyzer.analyze(...)
```

**Questions**:
1. **Retry Strategy**:
   - Current: No retry (한 번 실패하면 그냥 진행)
   - Alternative 1: 같은 task를 다른 방식으로 재시도
   - Alternative 2: Task를 분해해서 더 작은 task로
   - **Confirm**: Retry를 구현해야 하나요?

2. **Error Feedback to LLM**:
   - Current: Error message를 단순히 기록
   - Alternative: Error message를 다음 generation에 활용
   - **Example**:
     ```python
     prompt += f"Previous attempt failed: {error_msg}. Try a different approach."
     ```
   - **Confirm**: Error feedback loop를 구현해야 하나요?

### Question 4.2: Multi-branch Reasoning
**Current Implementation**:
- 단순 구현 (single branch만)
- `BranchCollection` data structure는 있지만 사용 안 함

**Questions**:
1. **Multi-branch 우선순위**:
   - **Kyungmin mentioned**: MS rStar 스타일 multi-branch
   - Current: Not implemented
   - **Confirm**: 지금 당장 구현해야 하나요, 아니면 나중에?

2. **Branch Creation 시점**:
   - Option 1: Error 발생 시
   - Option 2: Reward가 낮을 때 (e.g., < 0.5)
   - Option 3: 처음부터 multiple branches 유지
   - **Confirm**: 어느 시점에 branch를 만들어야 하나요?

---

## Category 5: Performance and Efficiency

### Question 5.1: API Call Optimization
**Current Implementation**:
```python
# API calls per query:
# 1. Sub-task extraction (1 call)
# 2. Query plan generation (1 call)
# 3. SQL generation per task (N calls, N=avg 3-5)
# 4. Confidence recalculation per task (N calls)
# Total: ~10-15 calls per query
```

**Questions**:
1. **Call Reduction**:
   - **Concern**: 너무 많은 API call → high cost, high latency
   - **Options**:
     - Batch multiple tasks in one call
     - Skip confidence recalculation
     - Cache LLM responses
   - **Confirm**: Cost/latency가 문제가 될까요? Optimization이 필요한가요?

2. **Parallel Execution**:
   - Current: Sequential (한 task씩)
   - Alternative: Independent tasks를 parallel로 실행
   - **Example**: Task 2와 Task 3가 모두 Task 1에만 depend하면, 2와 3을 parallel로
   - **Confirm**: Parallel execution을 구현할까요?

### Question 5.2: Caching Strategy
**Current Implementation**:
- No caching

**Questions**:
1. **Cache Target**:
   - Schema embedding
   - Sub-task extraction results (for similar queries)
   - Query plan results
   - **Confirm**: 무엇을 cache해야 할까요?

2. **Cache Invalidation**:
   - Schema 변경 시
   - Query 변경 시
   - **Confirm**: Cache strategy가 필요한가요?

---

## Category 6: Evaluation and Metrics

### Question 6.1: Evaluation Benchmark
**Current Implementation**:
- BIRD, Spider v2 준비됨
- Evaluation script: `scripts/run_pipeline.py`

**Questions**:
1. **Evaluation Split**:
   - BIRD dev set: ~500 examples
   - **Concern**: 전체를 다 돌리면 시간/비용이 많이 듦
   - **Options**:
     - Subset (e.g., 50-100 examples)
     - Full evaluation
     - Stratified sampling
   - **Confirm**: 처음에 몇 개로 테스트할까요?

2. **Evaluation Metrics**:
   - Current implementation:
     - Execution Accuracy (binary)
     - Semantic Correctness (0-1)
     - Total Reward (0-1)
   - **Missing**:
     - Exact match accuracy?
     - Component-wise accuracy (table/column/join)?
     - Error breakdown by type?
   - **Confirm**: 어떤 metric을 추가로 계산해야 하나요?

### Question 6.2: Baseline Comparison
**Current Implementation**:
- Baseline과의 비교 없음

**Questions**:
1. **Comparison Target**:
   - 기존 6개 구현 (RASL, PNEUMA, etc.)과 비교해야 하나요?
   - GPT-4o baseline (zero-shot, few-shot)?
   - **Confirm**: 어떤 baseline과 비교할까요?

2. **Ablation Study**:
   - Progressive execution vs one-shot
   - With confidence recalculation vs without
   - 60/20/20 reward vs execution-only
   - **Confirm**: Ablation study를 해야 하나요?

---

## Category 7: Code Quality and Maintenance

### Question 7.1: Type Hints and Validation
**Current Implementation**:
- Partial type hints
- Minimal input validation

**Questions**:
1. **Type Checking**:
   - Add full type hints everywhere?
   - Use mypy for type checking?
   - **Confirm**: Type safety가 중요한가요?

2. **Input Validation**:
   - Schema validation
   - Query validation
   - Config validation
   - **Confirm**: 어느 정도 validation이 필요한가요?

### Question 7.2: Testing Strategy
**Current Implementation**:
- No unit tests yet

**Questions**:
1. **Test Coverage**:
   - Unit tests for each component?
   - Integration tests for pipeline?
   - End-to-end tests on BIRD samples?
   - **Confirm**: 어떤 test가 우선순위인가요?

2. **Mock vs Real LLM**:
   - Mock LLM responses for fast testing?
   - Real LLM calls for accurate testing?
   - **Confirm**: Testing strategy는?

---

## Category 8: Configuration and Hyperparameters

### Question 8.1: Hyperparameter Tuning
**Current Implementation**:
```python
high_confidence_threshold = 0.85
acceptance_threshold = 0.7
max_iterations = 10
semantic_weight = 0.6
# ... etc
```

**Questions**:
1. **Tuning 방법**:
   - Manual tuning on dev set?
   - Grid search?
   - Bayesian optimization?
   - **Confirm**: Hyperparameter tuning을 어떻게 할까요?

2. **Dataset-specific Config**:
   - BIRD vs Spider v2 다른 config?
   - Simple vs complex query 다른 config?
   - **Confirm**: Dataset/query type별 config를 준비해야 하나요?

### Question 8.2: LLM Model Selection
**Current Implementation**:
- Default: GPT-4o
- Also supports: Claude

**Questions**:
1. **Model Comparison**:
   - GPT-4o vs Claude-3.5-Sonnet 성능 비교?
   - DeepSeek-R1 지원 추가?
   - **Confirm**: 어떤 모델로 main evaluation을 할까요?

2. **Cost vs Performance**:
   - GPT-4o: Expensive but accurate
   - GPT-4o-mini: Cheaper but less accurate
   - **Confirm**: Cost budget이 있나요?

---

## Category 9: Integration with Existing Work

### Question 9.1: 기존 6개 구현 활용
**Current Implementation**:
- 개념적으로만 참고
- 실제 코드 재사용 없음

**Questions**:
1. **Component 재사용**:
   - RSL-SQL의 BSL을 schema linking에 사용?
   - Reward-SQL의 PRM을 reward calculation에 사용?
   - LinkAlign의 multi-agent를 alternative generation에 사용?
   - **Confirm**: 기존 구현을 더 직접적으로 활용할까요?

2. **Ensemble Approach**:
   - 여러 방법을 combine해서 사용?
   - Voting mechanism?
   - **Confirm**: Ensemble이 필요한가요?

---

## Category 10: Deployment and Usage

### Question 10.1: Production Readiness
**Current Implementation**:
- Research prototype
- No production features

**Questions**:
1. **Production Features**:
   - Logging and monitoring?
   - Error reporting?
   - Rate limiting?
   - **Confirm**: Production을 고려해야 하나요, 아니면 research only?

2. **API Service**:
   - REST API로 wrapping?
   - Batch processing support?
   - **Confirm**: Service 형태로 deploy할 계획이 있나요?

---

## Priority Ranking

제가 생각하는 우선순위별 정리:

### 🔴 High Priority (즉시 결정 필요)
1. **Confidence Recalculation 빈도** (Category 1.2)
   - API cost/latency에 직접 영향
2. **Ground Truth Comparison** (Category 3.2)
   - Evaluation 정확도에 영향
3. **Evaluation Subset Size** (Category 6.1)
   - 다음 단계 진행에 필요
4. **Error Recovery Strategy** (Category 4.1)
   - 현재 failure handling이 너무 단순

### 🟡 Medium Priority (개선 시 고려)
5. **LLM Semantic Judgment Prompt** (Category 3.1)
   - Prompt 개선으로 judgment 정확도 향상
6. **Multi-branch Implementation** (Category 4.2)
   - Kyungmin이 언급했지만 당장은 아닐 수도
7. **API Call Optimization** (Category 5.1)
   - Cost 절감
8. **Hyperparameter Tuning** (Category 8.1)
   - 성능 최적화

### 🟢 Low Priority (나중에 고려)
9. **Type Hints** (Category 7.1)
   - Code quality
10. **Production Features** (Category 10.1)
    - Research → Production 전환 시

---

## Decisions Needed From You

다음 사항들에 대해 결정/확인이 필요합니다:

1. **Confidence recalculation**: 매번? N번마다? Reward 낮을 때만?
2. **Ground truth**: 지금 당장 어떻게 확보? 아니면 heuristic만?
3. **Error recovery**: Retry 구현? 아니면 단순히 기록만?
4. **Multi-branch**: 지금 구현? 나중에?
5. **Evaluation size**: BIRD dev set 몇 개로 시작?
6. **Baseline comparison**: 필요? 어떤 baseline?
7. **Hyperparameter tuning**: Manual? Automated?
8. **Model selection**: GPT-4o? Claude? 둘 다?

---

**Date**: 2025-10-11
**Status**: Awaiting decisions on priority questions
