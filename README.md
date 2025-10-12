# EPFL Hyunjun's NL2SQL Pipeline

Research implementation based on Kyungmin's direction:
**"Agent Pipeline with Reward Design - Confident Sub-task Progressive Execution"**

## Core Philosophy

> "Fine-tuning 없이 추론만으로, Agent Pipeline과 Reward 설계가 핵심"
> "Confident한 부분부터 하나씩 실행하면서 쌓아가기"

## Key Innovations

### 1. Confident Sub-task Extraction
- LLM generates sub-tasks with confidence scores
- Execute highest-confidence tasks first
- Recalculate confidence after each execution

### 2. Progressive Execution
- NOT "plan first, execute later"
- BUT "execute and accumulate step by step"
- Each sub-task builds on previous results

### 3. Semantic Reward Model
- **60% Semantic Correctness** (most important!)
- 20% Execution Success
- 20% Efficiency
- Focus: Meaning matters, syntax doesn't

### 4. Multi-branch Reasoning
- When error detected, create alternative branches
- MS rStar style beam search
- Select best branch by reward

## Architecture

```
NL Query + Schema
       ↓
┌─────────────────────────────────────┐
│ 1. Confident Sub-task Extraction    │
│    (LLM generates tasks + confidence)│
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 2. Query Plan Generation            │
│    (CHASE-SQL 3-step reasoning)     │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 3. Progressive Execution             │
│    Loop:                             │
│    - Execute highest confidence task │
│    - Calculate reward                │
│    - Accumulate context              │
│    - Recalculate remaining confidence│
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 4. Semantic Reward Evaluation        │
│    - Table/Column/Join/Filter check  │
│    - Execution success check         │
│    - Efficiency calculation          │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 5. Multi-branch (if needed)          │
│    - Create alternatives on failure  │
│    - Beam search best branch         │
└─────────────────────────────────────┘
       ↓
   Final SQL
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Set API key
export OPENAI_API_KEY=your_key_here

# Run on BIRD dev set
python scripts/run_pipeline.py \
    --dataset bird \
    --split dev \
    --output results.json
```

## Configuration

Edit `config/config.py`:

```python
# Semantic reward weights (Kyungmin's specification)
semantic_correctness_weight: 0.6  # Most important
execution_success_weight: 0.2
efficiency_weight: 0.2

# Sub-task confidence threshold
high_confidence_threshold: 0.85

# Multi-branch settings
num_alternatives: 3
beam_size: 5
```

## Project Structure

```
EPFL_hyunjun/
├── config/
│   └── config.py              # Configuration
├── model/
│   ├── data_structures.py     # Core data structures
│   ├── subtask_extractor.py   # Confident sub-task extraction
│   ├── query_plan_generator.py # CHASE-SQL style query plan
│   ├── semantic_reward.py     # Reward model (60/20/20)
│   ├── progressive_executor.py # Progressive execution
│   └── multibranch_reasoner.py # Multi-branch reasoning
├── pipeline/
│   └── main_pipeline.py       # Main orchestrator
├── evaluation/
│   ├── error_analyzer.py      # Semantic error analysis
│   └── bird_evaluator.py      # BIRD benchmark
├── utils/
│   ├── llm_client.py          # LLM API client
│   └── database_executor.py   # SQL execution
├── scripts/
│   └── run_pipeline.py        # Main runner
└── README.md                  # This file
```

## Key Differences from Existing Methods

### vs Traditional Approaches
```python
# Traditional: Plan → Execute
plan = create_full_plan(query)
sql = generate_sql(plan)
result = execute(sql)

# Kyungmin's: Execute → Accumulate → Execute
for task in sorted_by_confidence(subtasks):
    result = execute(task, context)
    context.update(result)
    recalculate_confidence(remaining_tasks, context)
```

### vs CHASE-SQL
- Uses 3-step query plan for reasoning
- BUT adds confidence-based progressive execution
- AND semantic reward model for evaluation

### vs RSL-SQL
- Uses bidirectional schema linking concept
- BUT focuses on sub-task confidence
- AND progressive accumulation

## Target Benchmarks

- **Primary**: BIRD (Spider too easy)
- **Secondary**: Spider v2
- **Focus**: Semantic correctness over execution accuracy

## Future Work

See `remaining_tasks.md` for:
- Schema linking ground truth acquisition
- Advanced error recovery strategies
- Optimization techniques

## Citation

Research implementation for EPFL Hyunjun's NL2SQL project.

## License

Academic research purposes.
