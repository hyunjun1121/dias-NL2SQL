# CHASE-SQL vs Our Progressive Approach

## Visual Comparison

### CHASE-SQL Flow (One-shot)
```
Natural Language Query
         ↓
   Query Plan (3-step)
    Step 1: Find tables
    Step 2: Perform operations
    Step 3: Select columns
         ↓
   Generate FULL SQL
   (One LLM call, generates complete SQL)
         ↓
      Execute
         ↓
   Success or Fail
   (No retry, no semantic check)
```

### Our Progressive Approach
```
Natural Language Query
         ↓
   Extract Sub-tasks + Confidence
   [0.95] SELECT FROM employees
   [0.92] WHERE department='Engineering'
   [0.90] WHERE salary>50000
         ↓
   Query Plan (3-step)
   (Same as CHASE-SQL)
         ↓
   Progressive Execution Loop:

   Iteration 1:
   └─ Task: [0.95] SELECT FROM employees
   └─ Generate: "SELECT * FROM employees"
   └─ Execute: ✓ 1000 rows
   └─ Syntax Error? No → Continue
   └─ Semantic Check: LLM judges → Correct
   └─ Reward: 1.0
   └─ Accumulate context: {tables: [employees]}

   Iteration 2:
   └─ Task: [0.92] WHERE department='Engineering'
   └─ Generate with context: "SELECT * FROM employees WHERE department='Engineering'"
   └─ Execute: ✓ 300 rows
   └─ Syntax Error? No → Continue
   └─ Semantic Check: LLM judges → Correct
   └─ Reward: 1.0
   └─ Accumulate context: {filters: ["department='Engineering'"]}

   Iteration 3:
   └─ Task: [0.90] WHERE salary>50000
   └─ Generate with context: "... AND salary>50000"
   └─ Execute: ✓ 85 rows
   └─ Syntax Error? No → Continue
   └─ Semantic Check: LLM judges → Correct
   └─ Reward: 1.0
   └─ Final SQL ready
         ↓
   Total Reward: 1.0 (all passed)
```

---

## Detailed Example

### Input
```sql
Query: "Show employees with salary over 50000 in Engineering"
Schema: employees(id, name, salary, department)
```

### CHASE-SQL Output (1 iteration)
```python
# Step 1: Generate full SQL in one shot
sql = """
SELECT * FROM employees
WHERE department = 'Engineering'
AND salary > 50000
"""

# Step 2: Execute
result = execute(sql)
# → 85 rows returned

# Done! No semantic verification
```

### Our Method Output (3 iterations)
```python
# Iteration 1
task_1 = "SELECT FROM employees"
sql_1 = "SELECT * FROM employees"
execute(sql_1)  # → 1000 rows
llm_judge(sql_1, query, result)  # → "CORRECT: YES"
reward_1 = 1.0
context = {tables: [employees]}

# Iteration 2 (uses context from iteration 1)
task_2 = "WHERE department='Engineering'"
sql_2 = "SELECT * FROM employees WHERE department='Engineering'"
execute(sql_2)  # → 300 rows
llm_judge(sql_2, query, result)  # → "CORRECT: YES"
reward_2 = 1.0
context = {tables: [employees], filters: ["department='Engineering'"]}

# Iteration 3 (uses accumulated context)
task_3 = "WHERE salary>50000"
sql_3 = "SELECT * FROM employees WHERE department='Engineering' AND salary>50000"
execute(sql_3)  # → 85 rows
llm_judge(sql_3, query, result)  # → "CORRECT: YES"
reward_3 = 1.0

# Final SQL = sql_3
# Total reward = 1.0 (all 3 iterations passed)
```

---

## When Each Method Works Better

### CHASE-SQL is better when:
1. **Simple queries** (1-2 operations)
   - Example: "SELECT * FROM users WHERE age > 18"
   - One-shot is faster, no need for progressive

2. **Clear, unambiguous queries**
   - Example: "Count total sales"
   - Straightforward, low error risk

3. **Speed is critical**
   - One LLM call vs multiple
   - Lower latency

### Our Method is better when:
1. **Complex queries** (3+ operations)
   - Example: "Join 3 tables, filter by 2 conditions, aggregate by month"
   - Progressive allows error catching at each step

2. **Error-prone domains**
   - Complex schemas
   - Ambiguous column names
   - Syntax error recovery needed

3. **Semantic correctness is critical**
   - Each step is verified
   - Context accumulation reduces errors

4. **Using smaller models**
   - Task 잘게 쪼개짐 → 작은 모델도 효과적
   - DeepSeek-R1, Qwen2.5 등

---

## Experimental Comparison

### Metrics to Compare
```python
{
    'execution_accuracy': {
        'chase_sql': '?%',
        'our_method': '?%'
    },
    'semantic_correctness': {
        'chase_sql': 'N/A (not evaluated)',
        'our_method': '?%'
    },
    'avg_time': {
        'chase_sql': '? seconds',
        'our_method': '? seconds'
    },
    'api_calls': {
        'chase_sql': '~3-4 per query',
        'our_method': '~10-15 per query'
    }
}
```

### Expected Results
- **Simple queries**: CHASE-SQL wins (faster, same accuracy)
- **Complex queries**: Our method wins (higher accuracy, error recovery)
- **Cost**: CHASE-SQL cheaper (fewer API calls)
- **Semantic correctness**: Our method better (explicit verification)

---

## Usage Guide

### Quick Test
```python
from baseline.chase_sql import CHASESQLBaseline
from utils.llm_client import LLMClient
from utils.database_executor import DatabaseExecutor

# CHASE-SQL
chase = CHASESQLBaseline(llm_client, db_executor, config)
result = chase.generate_sql(query, schema)
print(result['sql'])  # One-shot SQL
```

### Full Comparison
```bash
# Compare both methods on validation set
python scripts/compare_baselines.py \
    --data_path /path/to/bird \
    --limit 50 \
    --output comparison_results.json
```

### Analyze Results
```python
import json

with open('comparison_results.json') as f:
    results = json.load(f)

stats = results['stats']
print(f"Our method execution success: {stats['our_method']['execution_success']}/50")
print(f"CHASE-SQL execution success: {stats['chase_sql']['execution_success']}/50")
print(f"Our method semantic correct: {stats['our_method']['semantic_correct']}/50")
```

---

## Key Takeaways

1. **CHASE-SQL = Baseline**
   - Simple, fast, one-shot
   - Good for comparison

2. **Our Method = Innovation**
   - Progressive execution
   - Error recovery
   - Semantic verification
   - Context accumulation

3. **Goal**: Show our method > CHASE-SQL
   - Especially on complex queries
   - Especially with smaller models
   - Lower cost, higher accuracy

4. **Paper contribution**:
   - Ablation: CHASE-SQL → +Progressive → +Confidence → +Error Recovery
   - Each step shows improvement
