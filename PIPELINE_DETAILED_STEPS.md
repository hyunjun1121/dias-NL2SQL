# EPFL Hyunjun NL2SQL Pipeline - Detailed Step-by-Step Explanation

## Table of Contents
1. [Stage 1: Confident Sub-task Extraction](#stage-1-confident-sub-task-extraction)
2. [Stage 2: Query Plan Generation](#stage-2-query-plan-generation)
3. [Stage 3: Progressive Execution Loop](#stage-3-progressive-execution-loop)
4. [Stage 4: Semantic Reward Evaluation](#stage-4-semantic-reward-evaluation)
5. [Stage 5: Error Analysis](#stage-5-error-analysis)

---

## Stage 1: Confident Sub-task Extraction

### Component
**File**: `model/subtask_extractor.py`
**Class**: `ConfidentSubTaskExtractor`

### Purpose
LLM이 자연어 쿼리를 atomic sub-tasks로 분해하고, 각 task에 대한 confidence score를 직접 생성.

### Input
```python
{
    "nl_query": "Show employees with salary over 50000 in Engineering department",
    "schema": {
        "employees": {
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "TEXT"},
                {"name": "salary", "type": "REAL"},
                {"name": "department", "type": "TEXT"},
                {"name": "hire_date", "type": "DATE"}
            ]
        }
    }
}
```

### Process

#### Step 1.1: Build Extraction Prompt
```python
prompt = f"""You are an expert at breaking down SQL queries into confident sub-tasks.

Natural Language Query:
{nl_query}

Database Schema:
{formatted_schema}

Instructions:
1. Break the query into atomic operations
2. Assign confidence score (0.0-1.0) for each
3. Identify dependencies between tasks

Output Format (JSON):
{{
  "subtasks": [...]
}}
"""
```

#### Step 1.2: LLM Generates Sub-tasks with Confidence
```python
response = llm_client.generate(prompt, temperature=0.0)
```

**Example LLM Response**:
```json
{
  "subtasks": [
    {
      "task_id": 1,
      "operation": "SELECT FROM employees",
      "operation_type": "table_selection",
      "confidence": 0.95,
      "reasoning": "Table 'employees' explicitly mentioned and exists in schema",
      "dependencies": []
    },
    {
      "task_id": 2,
      "operation": "WHERE department = 'Engineering'",
      "operation_type": "filter",
      "confidence": 0.92,
      "reasoning": "Column 'department' exists and value 'Engineering' is clear",
      "dependencies": [1]
    },
    {
      "task_id": 3,
      "operation": "WHERE salary > 50000",
      "operation_type": "filter",
      "confidence": 0.90,
      "reasoning": "Column 'salary' exists and numeric condition is clear",
      "dependencies": [1]
    },
    {
      "task_id": 4,
      "operation": "Combine filters with AND",
      "operation_type": "logical_operator",
      "confidence": 0.85,
      "reasoning": "Both filters should apply simultaneously",
      "dependencies": [2, 3]
    }
  ]
}
```

#### Step 1.3: Parse and Validate
```python
tasks = []
for st in parsed_response['subtasks']:
    task = SubTask(
        task_id=st['task_id'],
        operation=st['operation'],
        operation_type=st['operation_type'],
        confidence=st['confidence'],
        reasoning=st['reasoning'],
        dependencies=st['dependencies']
    )
    tasks.append(task)
```

#### Step 1.4: Create SubTaskCollection
```python
collection = SubTaskCollection(
    tasks=tasks,  # Sorted by confidence
    nl_query=nl_query,
    schema=schema
)
```

### Output
```python
SubTaskCollection(
    tasks=[
        SubTask(id=1, conf=0.95, op="SELECT FROM employees"),
        SubTask(id=2, conf=0.92, op="WHERE department='Engineering'"),
        SubTask(id=3, conf=0.90, op="WHERE salary>50000"),
        SubTask(id=4, conf=0.85, op="Combine with AND")
    ]
)
```

### Key Design Decisions

1. **Why LLM-generated confidence?**
   - LLM understands semantic clarity better than heuristics
   - Can consider schema availability, value examples, linguistic ambiguity
   - Kyungmin's insight: Let LLM judge its own confidence

2. **Why JSON format?**
   - Structured, parseable output
   - Easy to validate and debug
   - Clear separation of concerns

3. **Why track dependencies?**
   - Some tasks must complete before others (e.g., table selection before filtering)
   - Enables parallel execution of independent tasks
   - Maintains logical ordering

---

## Stage 2: Query Plan Generation

### Component
**File**: `model/query_plan_generator.py`
**Class**: `QueryPlanGenerator`

### Purpose
Generate human-readable 3-step query plan following CHASE-SQL methodology.

### Input
```python
{
    "nl_query": "Show employees with salary over 50000 in Engineering",
    "schema": {...}
}
```

### Process

#### Step 2.1: Build Query Plan Prompt
```python
prompt = f"""Create a 3-step query plan:

Step 1: Find Relevant Tables
Step 2: Perform Operations (filter, join, aggregate)
Step 3: Select Final Columns

Query: {nl_query}
Schema: {schema}

Output JSON with reasoning for each step.
"""
```

#### Step 2.2: LLM Generates Plan
**Example Response**:
```json
{
  "steps": [
    {
      "step_number": 1,
      "step_type": "find_tables",
      "description": "Find the employees table",
      "reasoning": "Query asks about employees, so need employees table",
      "entities": ["employees"]
    },
    {
      "step_number": 2,
      "step_type": "perform_operations",
      "description": "Filter by department='Engineering' AND salary>50000",
      "reasoning": "Query specifies both conditions must be satisfied",
      "entities": ["department", "salary", "WHERE", "AND", ">", "="]
    },
    {
      "step_number": 3,
      "step_type": "select_columns",
      "description": "Return all columns (*) for matching employees",
      "reasoning": "'Show employees' implies all information needed",
      "entities": ["*"]
    }
  ]
}
```

#### Step 2.3: Parse to QueryPlan Object
```python
steps = []
for step_data in response['steps']:
    step = QueryPlanStep(
        step_number=step_data['step_number'],
        step_type=step_data['step_type'],
        description=step_data['description'],
        reasoning=step_data['reasoning'],
        entities=step_data['entities']
    )
    steps.append(step)

query_plan = QueryPlan(steps=steps, nl_query=nl_query)
```

### Output
```python
QueryPlan(
    steps=[
        QueryPlanStep(1, "find_tables", "Find employees", ...),
        QueryPlanStep(2, "perform_operations", "Filter by dept and salary", ...),
        QueryPlanStep(3, "select_columns", "Return all columns", ...)
    ]
)
```

### Purpose of Query Plan

1. **Human Interpretability**: Clear reasoning for each step
2. **Context for Generation**: Helps SQL generation in next stages
3. **Error Diagnosis**: If generation fails, can trace back to plan
4. **Evaluation Reference**: Compare generated SQL against intended plan

---

## Stage 3: Progressive Execution Loop

### Component
**File**: `model/progressive_executor.py`
**Class**: `ProgressiveExecutor`

### Purpose
Kyungmin's핵심: Execute highest confidence task → Accumulate context → Repeat

### Input
```python
{
    "subtasks": SubTaskCollection,
    "schema": {...},
    "db_path": "path/to/database.sqlite"
}
```

### Process

#### Iteration Loop
```python
context = ExecutionContext()
for iteration in range(max_iterations):
    # Get next task
    task = subtasks.get_highest_confidence_task()
    if not task:
        break

    # Generate SQL
    sql = generate_sql_fragment(task, context, schema)

    # Execute immediately
    result = db_executor.execute(sql)

    # Calculate reward
    reward = reward_model.calculate(sql, nl_query, schema, result)

    # Update context if good
    if result['success'] and reward > threshold:
        context.update_from_task(task)
        context.current_sql = assemble_sql(context)
```

### Detailed Iteration Example

#### **Iteration 1**: Table Selection

**Task**:
```python
SubTask(
    id=1,
    operation="SELECT FROM employees",
    confidence=0.95,
    dependencies=[]
)
```

**Step 3.1.1: Generate SQL Fragment**
```python
prompt = f"""Generate SQL for this task:

Task: SELECT FROM employees
Type: table_selection

Current Context:
- Completed: []
- Current SQL: ""
- Current tables: []

Schema:
employees: id, name, salary, department, hire_date

Generate SQL fragment.
"""

sql_fragment = llm.generate(prompt)
# Returns: "SELECT * FROM employees"
```

**Step 3.1.2: Execute SQL**
```python
result = db_executor.execute("SELECT * FROM employees")
# Returns: {
#   'success': True,
#   'result': [(1, 'Alice', 75000, 'Engineering', '2020-01-15'), ...],
#   'num_rows': 1000,
#   'execution_time': 0.05
# }
```

**Step 3.1.3: Calculate Reward**
```python
reward = reward_model.calculate_reward(
    predicted_sql="SELECT * FROM employees",
    nl_query=original_query,
    schema=schema,
    execution_result=result
)
# Returns: RewardScore(
#   semantic_correctness=SemanticCorrectness(
#     table_correctness=1.0,  # Correct table
#     column_correctness=1.0,  # SELECT * is acceptable
#     ...
#     overall=0.95
#   ),
#   execution_success=True,
#   efficiency=0.9,
#   total_reward=0.6*0.95 + 0.2*1.0 + 0.2*0.9 = 0.75
# )
```

**Step 3.1.4: Update Context**
```python
context.update_from_task(task)
# Context now contains:
# {
#   'completed_tasks': [task1],
#   'current_tables': ['employees'],
#   'current_sql': "SELECT * FROM employees",
#   'intermediate_results': [result1]
# }
```

#### **Iteration 2**: First Filter

**Task**:
```python
SubTask(
    id=2,
    operation="WHERE department = 'Engineering'",
    confidence=0.92,
    dependencies=[1]  # Depends on task 1
)
```

**Step 3.2.1: Generate SQL Fragment with Context**
```python
prompt = f"""Generate SQL for this task:

Task: WHERE department = 'Engineering'
Type: filter

Current Context:
- Completed: ["SELECT FROM employees"]
- Current SQL: "SELECT * FROM employees"
- Current tables: ['employees']
- Current filters: []

Schema:
employees: id, name, salary, department, hire_date

Extend the current SQL with this filter.
"""

sql_fragment = llm.generate(prompt)
# Returns: "SELECT * FROM employees WHERE department = 'Engineering'"
```

**Step 3.2.2: Execute**
```python
result = db_executor.execute("SELECT * FROM employees WHERE department = 'Engineering'")
# Returns: {
#   'success': True,
#   'num_rows': 300,
#   'execution_time': 0.04
# }
```

**Step 3.2.3: Calculate Reward**
```python
reward = 0.6*0.98 + 0.2*1.0 + 0.2*0.92 = 0.772
```

**Step 3.2.4: Update Context**
```python
context.current_filters.append("department = 'Engineering'")
context.current_sql = "SELECT * FROM employees WHERE department = 'Engineering'"
```

#### **Iteration 3**: Second Filter

**Task**:
```python
SubTask(
    id=3,
    operation="WHERE salary > 50000",
    confidence=0.90,
    dependencies=[1]
)
```

**Step 3.3.1: Generate with Accumulated Context**
```python
prompt = f"""Generate SQL for this task:

Task: WHERE salary > 50000
Type: filter

Current Context:
- Completed: ["SELECT FROM employees", "WHERE department='Engineering'"]
- Current SQL: "SELECT * FROM employees WHERE department = 'Engineering'"
- Current filters: ["department = 'Engineering'"]

Add this filter to existing SQL.
"""

sql_fragment = llm.generate(prompt)
# Returns: "SELECT * FROM employees WHERE department = 'Engineering' AND salary > 50000"
```

**Step 3.3.2: Execute**
```python
result = db_executor.execute(sql_fragment)
# Returns: {
#   'success': True,
#   'num_rows': 85,
#   'execution_time': 0.03
# }
```

**Step 3.3.3: Final Context**
```python
context = {
    'completed_tasks': [task1, task2, task3],
    'current_sql': "SELECT * FROM employees WHERE department = 'Engineering' AND salary > 50000",
    'current_filters': ["department='Engineering'", "salary>50000"],
    'final_reward': 0.87
}
```

### Key Features of Progressive Execution

#### 1. **Immediate Execution**
```python
# NOT this:
task1_sql = generate(task1)
task2_sql = generate(task2)
task3_sql = generate(task3)
final_sql = combine(task1_sql, task2_sql, task3_sql)
result = execute(final_sql)  # Execute once at the end

# BUT this (Kyungmin's way):
for task in tasks:
    sql = generate(task, context)
    result = execute(sql)  # Execute immediately!
    if good:
        context.update(result)
```

#### 2. **Context Accumulation**
Each iteration builds on previous results:
- Iteration 1: Learn base table
- Iteration 2: Add first filter, see how many rows remain
- Iteration 3: Add second filter with knowledge of current state

#### 3. **Confidence Recalculation** (After each iteration)
```python
# After task 1 completes successfully:
remaining_tasks = [task2, task3, task4]
updated_confidences = subtask_extractor.recalculate_confidence(
    remaining_tasks=remaining_tasks,
    completed_context={
        'completed': ['SELECT FROM employees'],
        'result': '1000 rows from employees table'
    }
)

# Task 2 confidence might increase: 0.92 → 0.96
# (because we now know employees table works)
# Task 4 confidence might decrease: 0.85 → 0.75
# (because filters already handle the logic)
```

---

## Stage 4: Semantic Reward Evaluation

### Component
**File**: `model/semantic_reward.py`
**Class**: `SemanticRewardModel`

### Purpose
Calculate reward with **binary approach** (Kyungmin's simplified requirement):
1. Execution must succeed (required)
2. LLM judges semantic correctness (only if execution succeeded)

### Input
```python
{
    "predicted_sql": "SELECT * FROM employees WHERE department='Engineering' AND salary>50000",
    "nl_query": "Show employees with salary over 50000 in Engineering",
    "schema": {...},
    "execution_result": {'success': True, 'num_rows': 85, 'result': [...]}
}
```

### Process

#### Step 4.1: Check Execution Success
```python
execution_success = execution_result.get('success', False)

if not execution_success:
    # Stop here - no need to check semantic correctness
    return {
        'execution_success': False,
        'execution_error': execution_result.get('error'),
        'semantic_correctness': None,
        'semantic_reasoning': None,
        'total_reward': 0.0
    }
```

#### Step 4.2: LLM Judges Semantic Correctness (Only if execution succeeded)

##### 4.2.1: Build LLM Judgment Prompt
```python
prompt = f"""You are evaluating if a generated SQL query correctly answers a natural language question.

Natural Language Query:
{nl_query}

Database Schema:
{formatted_schema}

Generated SQL:
{predicted_sql}

Execution Result:
- Success: Yes
- Number of rows: {execution_result.get('num_rows', 0)}
- Sample result (first 3 rows): {str(execution_result.get('result', [])[:3])}

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

Example:
CORRECT: YES
REASONING: The SQL correctly selects from the employees table and filters by department='Engineering' and salary>50000, which matches the query asking for employees with high salary in Engineering department.

Now evaluate:
"""
```

##### 4.2.2: LLM Generates Judgment
```python
response = llm_client.generate(
    prompt=prompt,
    temperature=0.0,
    max_tokens=512
)
```

**Example LLM Response**:
```
CORRECT: YES
REASONING: The SQL correctly selects from the employees table and applies both required filters: department='Engineering' and salary>50000. The execution result shows 85 matching employees, which is reasonable and the sample data confirms employees from Engineering department with salaries above 50000.
```

##### 4.2.3: Parse LLM Response
```python
def parse_judgment_response(response):
    response = response.strip()

    # Extract CORRECT: YES/NO
    correct = False
    if 'CORRECT: YES' in response.upper() or 'CORRECT:YES' in response.upper():
        correct = True
    elif 'CORRECT: NO' in response.upper() or 'CORRECT:NO' in response.upper():
        correct = False

    # Extract REASONING
    reasoning = ""
    lines = response.split('\n')
    for i, line in enumerate(lines):
        if 'REASONING:' in line.upper():
            reasoning_start = line.upper().index('REASONING:') + len('REASONING:')
            reasoning = line[reasoning_start:].strip()
            # Include following lines
            if i + 1 < len(lines):
                reasoning += " " + " ".join(lines[i+1:])
            break

    return {
        'correct': correct,
        'reasoning': reasoning.strip()
    }

# For our example:
# {
#   'correct': True,
#   'reasoning': "The SQL correctly selects from the employees table..."
# }
```

#### Step 4.3: Calculate Total Reward
```python
# Binary reward calculation
if execution_success and semantic_result['correct']:
    total_reward = 1.0
else:
    total_reward = 0.0

return {
    'execution_success': True,
    'execution_error': None,
    'semantic_correctness': semantic_result['correct'],  # True
    'semantic_reasoning': semantic_result['reasoning'],
    'total_reward': 1.0
}
```

### Output
```python
{
    'execution_success': True,
    'execution_error': None,
    'semantic_correctness': True,
    'semantic_reasoning': "The SQL correctly selects from the employees table and applies both required filters...",
    'total_reward': 1.0  # ← Binary: 1.0 or 0.0!
}
```

### Decision Logic
```python
if total_reward == 1.0:  # Perfect - both execution and semantic pass
    # Accept this SQL as final result
    return SQL
else:
    # Reject and try alternatives
    create_alternative_branches()
```

### Comparison: Before vs After

**Before (60/20/20 weighted)**:
```python
total_reward = 0.6*semantic + 0.2*execution + 0.2*efficiency
# Could be: 0.6*0.8 + 0.2*1.0 + 0.2*0.9 = 0.74 (pass)
# Complex calculation, hard to interpret
```

**After (Binary approach)**:
```python
if execution fails:
    total_reward = 0.0
elif LLM judges semantic as incorrect:
    total_reward = 0.0
else:
    total_reward = 1.0
# Simple, clear, interpretable
```

---

## Stage 5: Error Analysis

### Component
**File**: `evaluation/error_analyzer.py`
**Class**: `ErrorAnalyzer`

### Purpose
Categorize errors with **semantic errors as priority** (Kyungmin's requirement: syntax errors don't matter).

### Input
```python
{
    "predicted_sql": "SELECT * FROM employes WHERE salary > 50000",
    "execution_result": {
        'success': False,
        'error': "no such table: employes"
    },
    "task": SubTask(...)
}
```

### Process

#### Step 5.1: Categorize Error Type
```python
def categorize_error(error_msg, sql):
    error_lower = error_msg.lower()

    # HIGH PRIORITY: Semantic Errors
    if 'no such table' in error_lower:
        return ErrorType.WRONG_TABLE  # ← High priority!

    elif 'no such column' in error_lower:
        return ErrorType.WRONG_COLUMN  # ← High priority!

    elif 'ambiguous' in error_lower or 'join' in error_lower:
        return ErrorType.WRONG_JOIN  # ← High priority!

    # LOW PRIORITY: Syntax Errors
    elif 'syntax' in error_lower:
        return ErrorType.SYNTAX_ERROR  # ← Low priority (Kyungmin: don't matter)

    # MEDIUM: Execution Errors
    elif 'timeout' in error_lower:
        return ErrorType.EXECUTION_TIMEOUT

    else:
        return ErrorType.INVALID_OPERATION

# For our example:
# "no such table: employes" → ErrorType.WRONG_TABLE
```

#### Step 5.2: Suggest Fixes
```python
def suggest_fixes(error_type, sql):
    if error_type == ErrorType.WRONG_TABLE:
        return [
            "Check table name spelling",
            "Verify table exists in schema",
            "Try similar table names (e.g., 'employes' → 'employees')"
        ]

    elif error_type == ErrorType.WRONG_COLUMN:
        return [
            "Check column name spelling",
            "Verify column exists in table schema",
            "Try similar column names"
        ]

    elif error_type == ErrorType.WRONG_JOIN:
        return [
            "Check join conditions",
            "Verify foreign key relationships",
            "Try different join type (INNER/LEFT/RIGHT)"
        ]

    elif error_type == ErrorType.SYNTAX_ERROR:
        return ["Review SQL syntax"]  # Low priority

    # ... other cases

# For our example:
# fixes = [
#   "Check table name spelling",
#   "Verify table exists in schema",
#   "Try 'employes' → 'employees'"
# ]
```

#### Step 5.3: Create Error Analysis
```python
error_analysis = ErrorAnalysis(
    error_type=ErrorType.WRONG_TABLE,
    error_message="no such table: employes",
    failed_task=SubTask(id=1, operation="SELECT FROM employes"),
    suggested_fixes=[
        "Check table name spelling",
        "Verify table exists in schema",
        "Try 'employes' → 'employees'"
    ]
)
```

### Output
```python
ErrorAnalysis(
    error_type=ErrorType.WRONG_TABLE,  # HIGH PRIORITY
    error_message="no such table: employes",
    failed_task=SubTask(...),
    suggested_fixes=[...]
)
```

### Error Priority Levels

#### High Priority (Semantic Errors)
1. **WRONG_TABLE**: Incorrect table name
2. **WRONG_COLUMN**: Incorrect column name
3. **WRONG_JOIN**: Incorrect join logic
4. **WRONG_FILTER**: Incorrect filter conditions
5. **WRONG_AGGREGATION**: Incorrect aggregation

**Why high priority?** These indicate **semantic misunderstanding** of the query.

#### Low Priority (Syntax Errors)
1. **SYNTAX_ERROR**: SQL syntax mistakes

**Why low priority?** Kyungmin's insight: "Syntax errors don't matter much, they're easily fixable."

#### Medium Priority (Execution Errors)
1. **EXECUTION_TIMEOUT**: Query too slow
2. **INVALID_OPERATION**: Runtime errors

---

## Integration: How Stages Work Together

### Full Pipeline Flow with Example

```
INPUT:
Query: "Show employees with salary over 50000 in Engineering"
Schema: {employees: [id, name, salary, department, hire_date]}
DB: company.sqlite

↓

STAGE 1: Sub-task Extraction
→ Task 1 [0.95]: SELECT FROM employees
→ Task 2 [0.92]: WHERE department='Engineering'
→ Task 3 [0.90]: WHERE salary>50000
→ Task 4 [0.85]: Combine with AND

↓

STAGE 2: Query Plan
→ Step 1: Find employees table
→ Step 2: Filter by dept and salary
→ Step 3: Return all columns

↓

STAGE 3: Progressive Execution
→ Iteration 1: Execute task 1
  SQL: "SELECT * FROM employees"
  Result: 1000 rows ✓
  Reward: 0.75
  Context: {tables: [employees]}

→ Iteration 2: Execute task 2
  SQL: "SELECT * FROM employees WHERE department='Engineering'"
  Result: 300 rows ✓
  Reward: 0.77
  Context: {filters: ["dept='Engineering'"]}

→ Iteration 3: Execute task 3
  SQL: "SELECT * FROM employees WHERE department='Engineering' AND salary>50000"
  Result: 85 rows ✓
  Reward: 0.87
  Context: {filters: ["dept='Engineering'", "salary>50000"]}

↓

STAGE 4: Semantic Reward
→ Step 1: Execution Success: ✓
→ Step 2: LLM Semantic Judgment:
  CORRECT: YES
  REASONING: "The SQL correctly selects from employees and applies
  both required filters (department='Engineering' and salary>50000).
  Result shows 85 matching employees which is reasonable."
→ Total Reward: 1.0 ✓ (perfect)

↓

STAGE 5: Error Analysis (skipped - no errors)

↓

OUTPUT:
Final SQL: "SELECT * FROM employees WHERE department='Engineering' AND salary>50000"
Execution: SUCCESS (85 rows)
Semantic Correctness: True (LLM judged as correct)
Total Reward: 1.0
```

---

## Key Implementation Details

### 1. Context Accumulation Structure
```python
class ExecutionContext:
    completed_tasks: List[SubTask] = []
    intermediate_results: List[Any] = []
    current_sql: str = ""
    current_tables: List[str] = []
    current_columns: List[str] = []
    current_filters: List[str] = []
    current_joins: List[str] = []
    metadata: Dict = {}
```

### 2. Confidence Recalculation Logic
```python
# After task completion, recalculate confidence of remaining tasks
remaining_tasks = [t for t in tasks if not t.execution_result]
updated_confidences = llm.recalculate_confidence(
    remaining_tasks=remaining_tasks,
    completed_context={
        'completed': [t.operation for t in context.completed_tasks],
        'results': [t.execution_result for t in context.completed_tasks]
    }
)
```

### 3. Reward Threshold Decision
```python
if reward.total_reward == 1.0:  # Binary: must be perfect
    # Accept result
    return PipelineOutput(final_sql=context.current_sql, ...)
else:
    # Reject and retry or create alternatives
    if config.enable_multibranch:
        alternatives = create_alternative_branches(failed_task)
        best_alternative = select_best_by_reward(alternatives)
        return execute_alternative(best_alternative)
    else:
        return PipelineOutput(final_sql=context.current_sql, ...)
```

---

**Implementation Date**: 2025-10-11
**Status**: ✅ Complete with detailed step-by-step breakdown
