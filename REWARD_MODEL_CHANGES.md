# Reward Model Changes - Simplified Approach

## Date: 2025-10-11

## Changes Made

### 1. Removed Efficiency Metric

**Reason**: 경민님 언급 없음, 불필요한 복잡도

**Before**:
```python
Semantic Correctness: 60%
Execution Success: 20%
Efficiency: 20%
```

**After**:
```python
Execution Success: 필수 (먼저 체크)
Semantic Correctness: LLM binary judgment (실행 성공 시에만)
```

### 2. Simplified Semantic Correctness

**Before**: 5가지 component로 세분화
- Table correctness (25%)
- Column correctness (25%)
- Join correctness (20%)
- Filter correctness (20%)
- Aggregation correctness (10%)

**After**: LLM이 한 번에 binary 판단
- Input: NL query + Generated SQL + Execution result + Schema
- Output: CORRECT: YES/NO + REASONING

### 3. New Reward Calculation Flow

```python
Step 1: Execute SQL
├─ Success?
│  ├─ Yes → Go to Step 2
│  └─ No → total_reward = 0.0, STOP

Step 2: LLM judges semantic correctness
├─ Prompt includes:
│  ├─ NL query
│  ├─ Generated SQL
│  ├─ Execution result (sample rows)
│  └─ Schema
│
└─ LLM Response:
   ├─ CORRECT: YES → total_reward = 1.0
   └─ CORRECT: NO → total_reward = 0.0
```

### 4. Updated Files

#### model/semantic_reward.py
- New `SemanticRewardModel` class
- `calculate_reward()` returns Dict not RewardScore object
- `_llm_judge_semantic_correctness()` method
- Prompt template with EXACT format requirement

#### model/data_structures.py
**Need to update**:
- `SemanticCorrectness` class → simplified (is_correct: bool, reasoning: str)
- `RewardScore` class → simplified (execution_success, semantic_correctness, total_reward)
- `PipelineOutput` class → use new SemanticCorrectness

#### model/progressive_executor.py
**Need to update**:
- `calculate_reward()` call returns Dict
- Extract values from Dict
- Pass llm_client to reward_model

#### pipeline/main_pipeline.py
**Need to update**:
- Initialize SemanticRewardModel with llm_client
- Handle new reward format

### 5. Example LLM Prompt for Semantic Judgment

```
You are evaluating if a generated SQL query correctly answers a natural language question.

Natural Language Query:
Show employees with salary over 50000 in Engineering department

Database Schema:
employees:
  - id (INTEGER)
  - name (TEXT)
  - salary (REAL)
  - department (TEXT)

Generated SQL:
SELECT * FROM employees WHERE department = 'Engineering' AND salary > 50000

Execution Result:
- Success: Yes
- Number of rows: 85
- Sample result: [(1, 'Alice', 75000, 'Engineering'), ...]

Task:
Determine if the generated SQL SEMANTICALLY CORRECTLY answers the natural language query.

Consider:
1. Are the correct tables used?
2. Are the correct columns selected?
3. Are the filters/conditions correct?
4. Are joins (if any) correct?
5. Are aggregations (if any) correct?
6. Does the result make sense for the question?

Respond in this EXACT format:
CORRECT: [YES/NO]
REASONING: [Your detailed reasoning in 2-3 sentences]
```

### 6. Example LLM Response

**Case 1: Correct**
```
CORRECT: YES
REASONING: The SQL correctly selects from the employees table and applies both required filters: department='Engineering' and salary>50000. The execution result shows 85 matching employees, which is reasonable and the sample data confirms employees from Engineering department with salaries above 50000.
```

**Case 2: Incorrect**
```
CORRECT: NO
REASONING: The SQL uses the wrong table name 'employes' instead of 'employees', which caused an execution error. Even though the filter logic for department and salary would be correct, the fundamental table reference is wrong.
```

### 7. Benefits of New Approach

1. **Simpler**: Binary yes/no instead of 5-component weighted average
2. **More Accurate**: LLM sees execution result, can judge holistically
3. **Easier to Debug**: Clear reasoning from LLM
4. **Fewer API Calls**: One judgment instead of multiple heuristics
5. **Aligned with Kyungmin's Vision**: Focus on semantic correctness, execution is just prerequisite

### 8. Remaining Tasks

- [ ] Update `model/data_structures.py` - Simplify SemanticCorrectness and RewardScore
- [ ] Update `model/progressive_executor.py` - Use new reward format
- [ ] Update `pipeline/main_pipeline.py` - Pass llm_client to reward model
- [ ] Update `config/config.py` - Remove efficiency weight config
- [ ] Update `PIPELINE_HIGH_LEVEL.md` - Reflect new reward calculation
- [ ] Update `PIPELINE_DETAILED_STEPS.md` - Update Stage 4 details
- [ ] Update `IMPLEMENTATION_QUESTIONS.md` - Remove efficiency-related questions

### 9. API Call Count Impact

**Before**:
- Per query: ~10-15 calls (subtask extraction, plan, generation, confidence recalc)

**After**:
- Same, but clearer: 1 call for semantic judgment (only if execution succeeds)
- Actually saves calls when execution fails (no need for 5-component evaluation)

### 10. Threshold Decision

**Before**:
```python
if total_reward >= 0.7:  # Accept
```

**After**:
```python
if total_reward == 1.0:  # Both execution and semantic must pass
```

More strict, but clearer criterion.

---

**Status**: semantic_reward.py updated, other files pending
