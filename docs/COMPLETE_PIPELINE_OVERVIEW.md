# Complete NL2SQL Pipeline Overview

## 📋 Complete Pipeline Stages (0-5)

### **Stage 0: CHESS Information Retrieval (IR)** 🔍
**Component**: `ir/ir_integration.py`

#### Purpose
Schema pruning - 대규모 데이터베이스에서 관련 테이블/컬럼만 추출

#### Process
1. **Extract Keywords**
   - LLM이 자연어 질문에서 키워드 추출
   - Template-based prompt

2. **Retrieve Entity**
   - 키워드와 매칭되는 테이블/컬럼 찾기
   - Vector similarity search (ChromaDB)

3. **Retrieve Context**
   - Top-K 관련 컬럼 선택
   - 예시 값과 설명 포함

#### Input
```python
{
    "question": "Show employees with salary over 50000 in Engineering",
    "db_id": "company",
    "full_schema": {
        "employees": [...],
        "departments": [...],
        "projects": [...],
        # ... 100+ tables
    }
}
```

#### Output (Pruned Schema)
```python
{
    "employees": {
        "columns": [
            {"name": "id"},
            {"name": "name"},
            {"name": "salary"},
            {"name": "department"}
        ]
    }
}
```

#### Why Important?
- 대규모 DB (100+ tables)에서 관련 테이블만 선택
- LLM context 절약
- SQL 생성 정확도 향상

---

### **Stage 1: Confident Sub-task Extraction** 🎯
**Component**: `model/subtask_extractor.py`

#### Purpose
자연어 쿼리를 atomic sub-tasks로 분해 + confidence score 생성

#### Input
- Pruned schema (from Stage 0)
- Natural language question

#### Output
```python
[
    SubTask(id=1, conf=0.95, op="SELECT FROM employees"),
    SubTask(id=2, conf=0.92, op="WHERE department='Engineering'"),
    SubTask(id=3, conf=0.90, op="WHERE salary>50000")
]
```

---

### **Stage 2: Query Plan Generation** 📝
**Component**: `model/query_plan_generator.py`

#### Purpose
Human-readable 3-step query plan (CHASE-SQL methodology)

#### Output
```python
QueryPlan(
    steps=[
        Step(1, "find_tables", "Find employees table"),
        Step(2, "perform_operations", "Filter by dept and salary"),
        Step(3, "select_columns", "Return all columns")
    ]
)
```

---

### **Stage 3: Progressive Execution Loop** 🔄
**Component**: `model/progressive_executor.py`

#### Kyungmin's Core Innovation
- Execute highest confidence task first
- **Immediate execution** (not deferred)
- Accumulate context from results
- Recalculate remaining task confidence

#### Process
```python
for iteration in range(max_iterations):
    task = get_highest_confidence_task()
    sql = generate_sql_fragment(task, context)
    result = db_executor.execute(sql)  # Execute immediately!

    if result.success:
        reward = calculate_semantic_reward(sql, result)
        if reward > threshold:
            context.update(task, result)
```

---

### **Stage 4: Semantic Reward Evaluation** ⭐
**Component**: `model/semantic_reward.py`

#### Binary Approach (Simplified)
```python
if execution_fails:
    reward = 0.0
elif LLM_judges_semantic_as_incorrect:
    reward = 0.0
else:
    reward = 1.0  # Perfect!
```

#### LLM Judgment
```
CORRECT: YES/NO
REASONING: [Detailed explanation]
```

---

### **Stage 5: Error Analysis** 🐛
**Component**: `evaluation/error_analyzer.py`

#### Error Priority
1. **High**: Semantic errors (wrong table, wrong column, wrong join)
2. **Low**: Syntax errors (easily fixable)
3. **Medium**: Execution errors (timeout, invalid ops)

---

## 🔧 LLM Integration Points

### Current LLMClient Architecture
**File**: `utils/llm_client.py`

#### Supported Backends
- OpenAI (GPT-4o)
- Anthropic (Claude)
- vLLM cluster
- HuggingFace Inference API
- Ollama
- **Transformers (local)** ← 여기에 새 모델 추가!

#### Stage-specific LLM Usage
```python
class EPFLHyunjunPipeline:
    def __init__(self, config):
        # Each stage can use different models!
        self.subtask_llm = LLMClient(config.subtask.model_name)
        self.plan_llm = LLMClient(config.query_plan.model_name)
        self.sql_llm = LLMClient(config.progressive_execution.sql_model_name)
        self.reward_llm = LLMClient(config.semantic_reward.model_name)
```

---

## 🎯 모델 통합 전략

### Tested Models Performance (NL2SQL)

| Model | Memory | SQL Quality | Speed | Best For |
|-------|--------|-------------|-------|----------|
| **Qwen3-480B-Coder** | 340GB | ✅ Perfect SQL | 259s | **SQL Generation** |
| **Qwen3-235B-Thinking** | 220GB | ❌ No SQL (thinking only) | 77s | Sub-task reasoning |
| **MiniMax-M2** | 214GB | ❌ No SQL (thinking only) | 53s | Sub-task reasoning |

### Recommended Pipeline Configuration

```python
# Stage 0: IR (keyword extraction)
ir_model = "gpt-4o-mini"  # Fast & cheap for keyword extraction

# Stage 1: Sub-task Extraction
subtask_model = "Qwen3-235B-Thinking"  # or MiniMax-M2
# Benefit: Detailed reasoning about sub-tasks

# Stage 2: Query Plan
plan_model = "Qwen3-235B-Thinking"  # or MiniMax-M2
# Benefit: Step-by-step planning

# Stage 3: SQL Generation (CRITICAL!)
sql_model = "Qwen3-480B-Coder"
# Benefit: Specialized for code/SQL, produces complete queries

# Stage 4: Semantic Reward
reward_model = "gpt-4o-mini"  # or any thinking model
# Benefit: Just needs to judge correctness
```

### Why Qwen3-480B-Coder is Essential

1. **Only model that generates complete SQL** ✅
2. **No thinking tags** - direct SQL output
3. **Code-specialized** - understands SQL syntax perfectly
4. **Works on 4 GPUs** - practical deployment

### Integration Steps

1. **Update LLMClient** (`utils/llm_client.py`)
   ```python
   def _detect_backend(self):
       if model_lower in ["qwen3-480b-coder", "qwen3-235b-thinking", "minimax-m2"]:
           return "transformers_local"
   ```

2. **Add Local Transformers Handler**
   ```python
   elif self.backend == "transformers_local":
       return self._load_qwen_model()
   ```

3. **Update Config** (`config/config.py`)
   ```python
   class ProgressiveExecutionConfig:
       sql_model_name: str = "qwen3-480b-coder"
   ```

---

## 📊 Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ INPUT: Natural Language Query + Database Path          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 0: CHESS IR (Schema Pruning)                     │
│ ├─ Extract Keywords (LLM)                              │
│ ├─ Retrieve Entity (Vector DB)                         │
│ └─ Retrieve Context (Top-K)                            │
│ Output: Pruned Schema                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Sub-task Extraction                           │
│ LLM: Qwen3-235B-Thinking / MiniMax-M2                 │
│ Output: [Task1(0.95), Task2(0.92), Task3(0.90)]       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Query Plan Generation                         │
│ LLM: Qwen3-235B-Thinking / MiniMax-M2                 │
│ Output: 3-step plan                                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Progressive Execution (LOOP)                  │
│ LLM: **Qwen3-480B-Coder** ← CRITICAL!                 │
│                                                          │
│ For each task (highest confidence first):              │
│   1. Generate SQL fragment                             │
│   2. Execute immediately                               │
│   3. Calculate reward                                  │
│   4. Update context if good                            │
│                                                          │
│ Output: Final SQL                                       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Semantic Reward Evaluation                    │
│ LLM: Any thinking model (GPT-4o-mini)                  │
│                                                          │
│ Binary Decision:                                        │
│   Execution OK? + Semantically Correct? → 1.0 : 0.0   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 5: Error Analysis (if reward = 0)                │
│ Categorize: Semantic > Execution > Syntax              │
│ Suggest fixes                                           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT: PipelineOutput                                  │
│ ├─ final_sql: str                                      │
│ ├─ execution_result: Dict                              │
│ ├─ semantic_correctness: bool                          │
│ ├─ total_reward: 1.0 or 0.0                           │
│ └─ execution_time: float                               │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Next Steps

1. **Extend LLMClient** to support local Qwen models
2. **Test Stage 3** with Qwen3-480B-Coder
3. **Benchmark** against baseline (GPT-4o)
4. **Optimize** memory management for 4-GPU setup
5. **Deploy** on HPC cluster

---

**Created**: 2025-11-15
**Status**: Ready for Qwen model integration