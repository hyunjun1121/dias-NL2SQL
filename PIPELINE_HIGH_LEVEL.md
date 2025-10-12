# EPFL Hyunjun NL2SQL Pipeline - High Level Overview

## Core Philosophy

> **"Fine-tuning 없이 추론만으로, Agent Pipeline과 Reward 설계가 핵심"**
>
> **"Confident한 부분부터 하나씩 실행하면서 쌓아가기"**
>
> **"의미적 정확성이 진짜 중요한 지표"**

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT                                      │
│  - Natural Language Query                                     │
│  - Database Schema                                            │
│  - Database Path                                              │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: Confident Sub-task Extraction                      │
│  ────────────────────────────────────────                    │
│  Component: ConfidentSubTaskExtractor                         │
│  Method: LLM generates tasks with confidence scores           │
│                                                               │
│  Input:  NL query + Schema                                   │
│  Output: List of sub-tasks with confidence (0.0-1.0)         │
│                                                               │
│  Example:                                                     │
│  Query: "Show employees with salary > 50000 in Engineering"  │
│                                                               │
│  Sub-tasks:                                                   │
│  1. [0.95] SELECT FROM employees                             │
│  2. [0.92] WHERE department = 'Engineering'                  │
│  3. [0.90] WHERE salary > 50000                              │
│  4. [0.85] Combine with AND                                  │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: Query Plan Generation                              │
│  ────────────────────────────────                            │
│  Component: QueryPlanGenerator                                │
│  Method: CHASE-SQL style 3-step reasoning                    │
│                                                               │
│  Step 1: Find Relevant Tables                                │
│  → "Find employees table"                                    │
│                                                               │
│  Step 2: Perform Operations                                  │
│  → "Filter by department='Engineering' AND salary>50000"     │
│                                                               │
│  Step 3: Select Columns                                      │
│  → "Return all columns for matching employees"               │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 3: Progressive Execution Loop                          │
│  ────────────────────────────────────                        │
│  Component: ProgressiveExecutor                               │
│  Method: Execute highest confidence → Accumulate → Repeat    │
│                                                               │
│  Iteration 1:                                                 │
│  ├─ Task: [0.95] SELECT FROM employees                       │
│  ├─ Generate: "SELECT * FROM employees"                      │
│  ├─ Execute: ✓ 1000 rows                                     │
│  ├─ Reward: 0.30 (semantic=1.0, exec=1.0, eff=0.9)          │
│  └─ Context: {tables: [employees], rows: 1000}               │
│                                                               │
│  Iteration 2:                                                 │
│  ├─ Task: [0.92] WHERE department = 'Engineering'            │
│  ├─ Generate: "... WHERE department = 'Engineering'"         │
│  ├─ Execute: ✓ 300 rows                                      │
│  ├─ Reward: 0.28 (semantic=0.95, exec=1.0, eff=0.9)         │
│  └─ Context: {filters: ["department='Engineering'"], ...}    │
│                                                               │
│  Iteration 3:                                                 │
│  ├─ Task: [0.90] WHERE salary > 50000                        │
│  ├─ Generate: "... AND salary > 50000"                       │
│  ├─ Execute: ✓ 85 rows                                       │
│  ├─ Reward: 0.29 (semantic=1.0, exec=1.0, eff=0.9)          │
│  └─ Context: {filters: [..., "salary>50000"], rows: 85}      │
│                                                               │
│  Recalculate Confidence: (After each iteration)              │
│  └─ Remaining tasks' confidence updated with new context     │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 4: Semantic Reward Evaluation                          │
│  ────────────────────────────────────                        │
│  Component: SemanticRewardModel                               │
│  Method: Binary approach (Execution + Semantic)              │
│                                                               │
│  Step 1: Check Execution Success                             │
│  └─ Execute SQL → Success: ✓                                 │
│                                                               │
│  Step 2: LLM Judges Semantic Correctness (only if exec ✓)   │
│  Prompt: "Does this SQL correctly answer the NL query?"      │
│  Input:                                                       │
│  ├─ NL Query: "Show employees with salary > 50000..."       │
│  ├─ Generated SQL: "SELECT * FROM employees WHERE..."        │
│  ├─ Execution Result: 85 rows                                │
│  └─ Schema: employees(id, name, salary, department)         │
│                                                               │
│  LLM Response:                                                │
│  CORRECT: YES                                                 │
│  REASONING: The SQL correctly selects from employees table   │
│  and applies both required filters (department='Engineering' │
│  and salary>50000). Result shows 85 matching employees.      │
│                                                               │
│  Total Reward = 1.0 (both execution and semantic pass)       │
│                                                               │
│  Note: If execution fails → reward = 0.0, no semantic check  │
│        If semantic incorrect → reward = 0.0                  │
└──────────────────────────────────────────────────────────────┘
                          ↓
         ┌────────────────────────────┐
         │  Reward == 1.0 (Perfect)?   │
         └────────────────────────────┘
                ↓               ↓
              YES              NO
                ↓               ↓
         Accept Result    Generate Alternatives
                ↓               (Multi-branch)
                ↓               ↓
                └───────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 5: Error Analysis (If Failed)                         │
│  ────────────────────────────────────                        │
│  Component: ErrorAnalyzer                                     │
│  Method: Categorize semantic errors (NOT syntax!)            │
│                                                               │
│  Priority Errors:                                             │
│  1. Wrong Table      (semantic)                              │
│  2. Wrong Column     (semantic)                              │
│  3. Wrong Join       (semantic)                              │
│  4. Wrong Filter     (semantic)                              │
│  5. Wrong Aggregation (semantic)                             │
│                                                               │
│  Low Priority:                                                │
│  - Syntax errors (경민님: don't matter)                        │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│                    OUTPUT                                     │
│  ────────────────────────────────                        │
│  - Final SQL                                                  │
│  - Execution Result                                           │
│  - Total Reward (1.0)                                        │
│  - Semantic Correctness (is_correct=True, reasoning=...)     │
│  - All Sub-tasks with confidence                             │
│  - Query Plan (3 steps)                                      │
│  - Execution Context                                          │
│  - Metadata (time, iterations, etc.)                         │
└──────────────────────────────────────────────────────────────┘
```

## Key Differentiators

### 1. LLM-Generated Confidence
```
Traditional: Fixed confidence or no confidence
Kyungmin:    LLM generates confidence scores in JSON
             Recalculated after each execution
```

### 2. Progressive Execution
```
Traditional: Plan → Generate → Execute (one-shot)
Kyungmin:    Execute → Accumulate → Execute (iterative)
```

**Example**:
```python
# Traditional
plan = create_full_plan(query)           # All at once
sql = generate_sql(plan)                 # All at once
result = execute(sql)                    # All at once

# Kyungmin
while has_tasks():
    task = get_highest_confidence()      # One at a time
    sql = generate(task, context)        # Use accumulated context
    result = execute(sql)                # Immediate execution
    if good: context.update(result)      # Accumulate
    recalculate_confidence(context)      # Update remaining
```

### 3. Semantic-First Reward
```
Traditional: Execution accuracy (binary)
Kyungmin:    Binary approach
             - Step 1: Execution must succeed
             - Step 2: LLM judges semantic correctness
             - Reward: 1.0 if both pass, 0.0 otherwise
```

### 4. Error Prioritization
```
Traditional: All errors treated equally
Kyungmin:    Semantic errors >> Syntax errors
             Focus on meaning, not syntax
```

## Data Flow

```
NL Query ──→ Sub-tasks ──→ Progressive Execution ──→ Rewards
   ↓            ↓                    ↓                   ↓
Schema    Confidence         Context Accumulation   Semantic
   ↓            ↓                    ↓               Evaluation
Database  Dependencies        SQL Fragments            ↓
   ↓            ↓                    ↓              Accept/Retry
   └────────────┴────────────────────┴──────────────────┘
```

## Component Interaction

```
┌─────────────────┐
│  LLM Client     │←──────────────────────────────┐
└─────────────────┘                                │
         ↓                                         │
┌─────────────────┐        ┌─────────────────┐    │
│ SubTask         │───────→│  Query Plan     │    │
│ Extractor       │        │  Generator      │    │
└─────────────────┘        └─────────────────┘    │
         ↓                          ↓              │
         └──────────┬───────────────┘              │
                    ↓                              │
         ┌─────────────────┐                       │
         │  Progressive    │←──────────────────────┤
         │  Executor       │                       │
         └─────────────────┘                       │
                ↓                                   │
         ┌─────────────────┐                       │
         │  Database       │                       │
         │  Executor       │                       │
         └─────────────────┘                       │
                ↓                                   │
         ┌─────────────────┐                       │
         │  Semantic       │                       │
         │  Reward Model   │                       │
         └─────────────────┘                       │
                ↓                                   │
         ┌─────────────────┐                       │
         │  Error          │───────────────────────┘
         │  Analyzer       │  (if failed)
         └─────────────────┘
```

## Execution Modes

### Normal Mode (Default)
- Execute highest confidence tasks sequentially
- Accumulate context progressively
- Accept if reward > threshold

### Multi-branch Mode (Future)
- When error detected, create alternative branches
- Beam search with multiple hypotheses
- Select best branch by cumulative reward

## Performance Characteristics

### Strengths
1. **High Semantic Correctness**: LLM judgment ensures meaningful SQL
2. **Adaptive**: Confidence recalculation after each step
3. **Interpretable**: Clear sub-tasks and reasoning
4. **Robust**: Progressive execution catches errors early

### Trade-offs
1. **API Calls**: Multiple LLM calls (sub-tasks, confidence, generation)
2. **Latency**: Progressive execution takes longer than one-shot
3. **Cost**: More token usage due to multiple calls

## Configuration Points

### Key Hyperparameters
- `high_confidence_threshold`: 0.85 (execute immediately if above)
- `acceptance_threshold`: 1.0 (accept only if both execution and semantic pass)
- `max_iterations`: 10 (maximum progressive execution steps)
- `judgment_temperature`: 0.0 (LLM temperature for semantic judgment)
- `judgment_max_tokens`: 512 (max tokens for LLM judgment response)

### Tunable Components
- LLM temperature for confidence generation
- LLM temperature for semantic judgment
- Context accumulation strategy
- Error recovery strategy

## Success Criteria

### Primary Metric (Kyungmin's Focus)
**Semantic Correctness**: Are the tables, columns, joins, filters correct?
- LLM judges: Does the SQL correctly answer the NL query?
- Binary: Correct or Incorrect

### Secondary Metrics
- Execution Accuracy: Does it run and return correct results?
- Task Success Rate: How many sub-tasks succeeded?

### Error Analysis
- Breakdown by error type (semantic vs syntax)
- Pipeline failure point identification
- Sub-task success rate

## Extension Points

### 1. Multi-branch Reasoning (Planned)
- Implement beam search when errors detected
- Generate alternative sub-tasks
- Select best branch by reward

### 2. Schema Linking Ground Truth (Future)
- Annotate or extract ground truth schema links
- Evaluate schema linking recall/precision
- Improve confidence calculation

### 3. Advanced Error Recovery (Future)
- More sophisticated error categorization
- Automatic fix suggestions
- Learning from error patterns

---

**Implementation Date**: 2025-10-11
**Status**: ✅ Complete and ready for evaluation
**Target Benchmarks**: BIRD dev, Spider v2
