# Examples

## Quick Start Examples

### 1. Test CHASE-SQL Baseline
```bash
cd examples
python test_chase_sql.py
```

**What it does:**
- Creates a toy database with 8 employees
- Runs CHASE-SQL one-shot generation
- Shows query plan, generated SQL, and results

**Output:**
```
📝 Query: Show employees with salary over 50000 in Engineering
🚀 Running CHASE-SQL (one-shot generation)...
💻 Generated SQL: SELECT * FROM employees WHERE department='Engineering' AND salary>50000
✓ Success! 3 rows returned
```

---

### 2. Compare Both Methods
```bash
cd examples
python compare_methods.py
```

**What it does:**
- Runs same query with CHASE-SQL and Our Method
- Shows side-by-side comparison
- Highlights differences (iterations, context, error handling)

**Output:**
```
[1] CHASE-SQL: 1 iteration, no context
[2] Our Method: 3 iterations, context accumulation
Comparison: Both succeeded, our method has semantic verification
```

---

### 3. Full Baseline Comparison (Requires BIRD dataset)
```bash
python scripts/compare_baselines.py \
    --data_path /path/to/bird \
    --db_path /path/to/bird/databases \
    --limit 50 \
    --output comparison_results.json
```

**What it does:**
- Evaluates 50 examples from BIRD dev set
- Compares execution accuracy, semantic correctness, time
- Saves detailed results to JSON

---

## File Overview

| File | Purpose | Usage |
|------|---------|-------|
| `test_chase_sql.py` | Test CHASE-SQL alone | Quick demo of one-shot generation |
| `compare_methods.py` | Side-by-side comparison | Show differences between methods |
| `../scripts/compare_baselines.py` | Full evaluation | Benchmark on BIRD dataset |

---

## Expected Results

### test_chase_sql.py
- Should successfully generate SQL for simple query
- Returns 3-4 employees from Engineering with salary > 50000
- Shows CHASE-SQL's one-shot approach

### compare_methods.py
- Both methods should succeed on simple example
- Our method shows 3 iterations vs CHASE-SQL's 1
- Our method includes semantic verification

### compare_baselines.py (on BIRD)
- Our method should have higher semantic correctness
- CHASE-SQL may be faster (fewer API calls)
- Our method better on complex queries

---

## Modifying Examples

### Change LLM Model
```python
# In any example file, change:
llm_client = LLMClient(model_name="gpt-4o")

# To open-source:
llm_client = LLMClient(
    model_name="deepseek-r1",
    base_url="http://cluster:8000/v1"
)
```

### Add Your Own Query
```python
# In test_chase_sql.py or compare_methods.py:
nl_query = "Your custom query here"
schema = {
    "your_table": {
        "columns": [...]
    }
}
```

---

## Troubleshooting

### "No module named 'baseline'"
```bash
# Run from project root:
cd E:\Project\nl2sql-baseline\EPFL_hyunjun
python examples/test_chase_sql.py
```

### API Key Issues
```bash
# Set environment variable:
export OPENAI_API_KEY=your-key

# Or in Python:
llm_client = LLMClient(
    model_name="gpt-4o",
    api_key="your-key"
)
```

### Database Not Found
```bash
# test_chase_sql.py creates its own database
# For compare_baselines.py, provide actual BIRD dataset path
```
