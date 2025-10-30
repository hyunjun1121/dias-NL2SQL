# EPFL Triple Kim NL2SQL Pipeline

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
- Execution success gate + constraint verification + LLM semantic judgment

### 4. Multi-branch Reasoning (Future)\n- Beam-style exploration is a planned extension; current code runs single-branch

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
│   ├── semantic_reward.py     # Reward model (execution + constraints + LLM)
│   ├── progressive_executor.py # Progressive execution
│   └── ├── pipeline/
│   └── main_pipeline.py       # Main orchestrator
├── evaluation/
│   ├── error_analyzer.py      # Semantic error analysis
│   └── ├── utils/
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

Research implementation for EPFL triple kim's NL2SQL project.

## License

Academic research purposes.

## IR Evaluation Scripts

- Table-level (Spider 2.0): `scripts/eval_ir_spider.py`
- Column-level (Spider 2.0): `scripts/eval_ir_spider_columns.py`
  - Modes: `--mode_variant strict|lenient` (strict: 엄밀 / lenient: 관대)
- Value/Context support: `scripts/eval_ir_value_context.py`

See `docs/IR_EVAL_SP2.md`, `docs/IR_EVAL_STRICT_LENIENT.md` for details.


## Configuration (Detailed)

Edit `config/config.py` (key fields):

```python
# Information Retriever (CHESS)
ir.enabled = True
ir.db_root_path = "benchmark"      # EPFL_hyunjun/benchmark (CHESS layout)
ir.data_mode = "dev"               # dev/test/train
ir.extract_keywords_template = "extract_keywords"
ir.extract_keywords_engine = "gpt-4o-mini"
ir.extract_keywords_temperature = 0.2
ir.extract_keywords_parser = "python_list_output_parser"
ir.retrieve_context_top_k = 5

# Sub-task extraction
subtask.high_confidence_threshold = 0.85

# Progressive execution
progressive_execution.max_iterations = 10
progressive_execution.acceptance_threshold = 1.0

# Multi-branch (planned; current code runs single-branch)
multibranch.enable_multibranch = True
multibranch.num_alternatives = 3
multibranch.beam_size = 5
```

Note: Runner requires both `--data_path` (dataset json) and `--db_path` (root of databases). The earlier minimal example omitted these flags.

## Run Examples

### BIRD (dev)

```bash
export OPENAI_API_KEY=your_key

python scripts/run_pipeline.py \
  --dataset bird \
  --data_path <path_to_bird>/dev.json \
  --db_path   <path_to_bird_databases_root> \
  --split dev \
  --output results_bird_dev.json \
  --limit 100
```

### Spider 2.0 (dev)

```bash
export OPENAI_API_KEY=your_key

python scripts/run_pipeline.py \
  --dataset spider \
  --data_path <path_to_spider2>/dev.json \
  --db_path   <path_to_spider2_databases_root> \
  --split dev \
  --output results_spider2_dev.json \
  --limit 100
```

## Architecture (Updated)

```
NL Query + Schema
       ↓
────────────────────────────────────────────────────────────────────────────
 0.5 CHESS IR (deterministic)
   - ExtractKeywords / RetrieveEntity / RetrieveContext
   - Pruned schema + examples + descriptions
────────────────────────────────────────────────────────────────────────────
       ↓
────────────────────────────────────────────────────────────────────────────
 1. Confident Sub-task Extraction
   - LLM-generated tasks with confidence/dependencies
────────────────────────────────────────────────────────────────────────────
       ↓
────────────────────────────────────────────────────────────────────────────
 2. Query Plan Generation (CHASE-SQL style)
   - Tables → Operations → Final projection
────────────────────────────────────────────────────────────────────────────
       ↓
────────────────────────────────────────────────────────────────────────────
 3. Progressive Execution
   - Execute → Evaluate → Accumulate (iteration)
────────────────────────────────────────────────────────────────────────────
       ↓
────────────────────────────────────────────────────────────────────────────
 4. Semantic Reward Evaluation
   - Execution success → Constraint check → LLM semantic judgment
────────────────────────────────────────────────────────────────────────────
       ↓
   Final SQL
```
