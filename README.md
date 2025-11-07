# EPFL Triple Kim NL2SQL Pipeline

This repository implements a research-grade text-to-SQL pipeline guided by the "Triple Kim" philosophy: confidence-driven decomposition, progressive execution, and semantic reward prioritisation. Each stage leverages a HuggingFace Qwen 3 model specialised for the task (IR remains the original CHESS implementation).

## Pipeline Stages

1. **CHESS IR (Stage 0.5)** – Deterministic execution of ExtractKeywords, RetrieveEntity, RetrieveContext to produce a pruned schema plus value/description artefacts.
2. **Confident Sub-task Extraction** – LLM generates sub-tasks (JSON) with confidence scores and dependencies (Qwen3-235B-A22B-Thinking).
3. **CHASE-SQL Query Plan Generation** – Three-step reasoning (tables → operations → final projection) with the same Thinking model.
4. **Progressive Execution** – Iteratively generate SQL fragments, execute, and accumulate context (Qwen3-Coder-480B-A35B-Instruct).
5. **Semantic Reward Evaluation** – Execution gate, constraint verification, and binary LLM judgement (Qwen3-235B-A22B-Instruct).

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

## Installation

```bash
pip install -r requirements.txt
```

Make sure `HUGGINGFACEHUB_API_TOKEN` is available for the Qwen 3 models.

## Configuration (config/config.py)

```python
# Global default (used if a stage override is None)
llm.model_name = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"

# Stage-specific overrides
subtask.model_name = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
query_plan.model_name = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
progressive_execution.sql_model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
semantic_reward.model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Other key knobs
subtask.high_confidence_threshold = 0.85
progressive_execution.max_iterations = 10
progressive_execution.acceptance_threshold = 1.0
multibranch.enable_multibranch = True   # Planned extension (current code runs single branch)
```

**Stage-specific defaults**
- Sub-task & Query Plan: `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8`
- SQL Fragment Generation: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`
- Semantic Reward Judgement: `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`

## Run Examples

### BIRD (dev)
```bash
export OPENAI_API_KEY=your_key
export HUGGINGFACEHUB_API_TOKEN=your_hf_token
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
export HUGGINGFACEHUB_API_TOKEN=your_hf_token
python scripts/run_pipeline.py \
  --dataset spider \
  --data_path <path_to_spider2>/dev.json \
  --db_path   <path_to_spider2_databases_root> \
  --split dev \
  --output results_spider2_dev.json \
  --limit 100
```

## Project Structure

```
config/                     # Dataclass configuration (IRConfig, etc.)
docs/                       # IR integration & evaluation notes
ir/                         # Deterministic CHESS IR wrapper
model/                      # Data structures, progressive executor, reward model
pipeline/                   # Main orchestrator (stage-specific LLM clients)
scripts/                    # Runner + IR evaluation utilities
utils/                      # LLM client (OpenAI, HF inference, etc.)
data/                       # (local) BIRD/Spider2 datasets + indices (ignored by Git)
```

## IR Evaluation Scripts

- Table-level (Spider 2.0): `scripts/eval_ir_spider.py`
- Column-level (Spider 2.0): `scripts/eval_ir_spider_columns.py`
  - Modes: `--mode_variant strict|lenient`
- Value/Context support: `scripts/eval_ir_value_context.py`

See `docs/IR_EVAL_SP2.md` and `docs/IR_EVAL_STRICT_LENIENT.md` for methodology.

## Notes

- Benchmark archives (`data/bird.zip`, `data/Spider2.zip`) are ignored by Git. See `docs/BENCHMARK_INVENTORY.md` for expected layout and CHESS preprocessing instructions.
- IR integration details (imported vs implemented) are summarised in `docs/IR_INTEGRATION.md`.
- Templates/Parsers sourced from CHESS are catalogued in `docs/CHESS_TEMPLATES.md`.

## License

Academic research only.
