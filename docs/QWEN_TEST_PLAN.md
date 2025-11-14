# Qwen Large Model Testing Plan

## Overview
This document outlines the testing plan for running Qwen3-235B and Qwen3-480B models on H100 GPUs for NL2SQL tasks.

## Models Under Test

### 1. Qwen3-235B-A22B-Thinking-2507-FP8
- **Purpose**: Complex reasoning and thinking tasks
- **Parameters**: 235B total, 22B active (MoE)
- **Quantization**: FP8
- **Context Length**: 262,144 tokens
- **Special Feature**: Thinking mode with `<think>` tags

### 2. Qwen3-Coder-480B-A35B-Instruct-FP8
- **Purpose**: Code generation and SQL queries
- **Parameters**: 480B total, 35B active (MoE)
- **Quantization**: FP8
- **Context Length**: 262,144 tokens
- **Special Feature**: Optimized for coding tasks

## GPU Requirements

### Estimated Requirements (H100 80GB)

| Model | Min GPUs | Recommended | Optimal | Memory/GPU |
|-------|----------|-------------|---------|------------|
| Qwen3-235B-Thinking | 4 | 8 | 8 | ~60GB |
| Qwen3-Coder-480B | 8 | 16 | 16 | ~60GB |

### Calculation Basis
- FP8 quantization: 1 byte per parameter
- KV cache: ~0.1GB per 1K tokens
- Activation memory: ~0.5GB per 1B active params
- Overhead: 20% for inference

## Testing Scripts

### 1. Basic Model Test (`test_qwen_hf.py`)
Tests model loading, inference speed, and memory usage.

**Usage**:
```bash
# Test Thinking model with 4 GPUs
python scripts/test_qwen_hf.py --model thinking --num_gpus 4 --max_tokens 1024

# Test Coder model with 8 GPUs
python scripts/test_qwen_hf.py --model coder --num_gpus 8 --max_tokens 2048
```

### 2. GPU Requirement Estimation (`estimate_gpu_requirements.py`)
Calculates optimal GPU allocation for different configurations.

**Usage**:
```bash
# Estimate requirements for both models
python scripts/estimate_gpu_requirements.py --model both --gpu H100

# Compare different GPU types
python scripts/estimate_gpu_requirements.py --compare_gpus

# Generate SLURM scripts
python scripts/estimate_gpu_requirements.py --generate_slurm
```

### 3. NL2SQL Pipeline Integration (`test_nl2sql_with_hf.py`)
Integrates HuggingFace models with the NL2SQL pipeline.

**Usage**:
```bash
# Single query test
python scripts/test_nl2sql_with_hf.py \
    --model thinking \
    --query "Find top 5 customers by total purchase" \
    --db_path data/bird/dev/dev_databases/retail/retail.sqlite

# Batch test
python scripts/test_nl2sql_with_hf.py \
    --model coder \
    --queries_file test_queries.json \
    --db_path data/bird/dev/dev_databases/retail/retail.sqlite \
    --output results.json
```

## SLURM Job Submission

### Thinking Model Test (4 GPUs)
```bash
sbatch scripts/slurm_test_thinking.sh
```

### Coder Model Test (8 GPUs)
```bash
sbatch scripts/slurm_test_coder.sh
```

### Scalability Test (Variable GPUs)
```bash
sbatch scripts/slurm_test_scalability.sh
```

## Test Workflow

### Phase 1: Basic Functionality (Day 1)
1. Test model loading with minimal GPUs
2. Verify inference works
3. Measure memory usage
4. Document any errors

### Phase 2: Scalability Testing (Day 2)
1. Test with different GPU counts (2, 4, 8, 16)
2. Measure performance scaling
3. Find optimal configuration
4. Compare inference speed

### Phase 3: NL2SQL Integration (Day 3)
1. Integrate with existing pipeline
2. Test on BIRD dev set queries
3. Compare with Ollama baseline
4. Measure end-to-end performance

### Phase 4: Production Testing (Day 4)
1. Long-context queries (>100K tokens)
2. Batch processing
3. Error recovery
4. Resource optimization

## Expected Results

### Performance Metrics
- **Loading Time**: 30-60 seconds
- **Tokens/Second**:
  - Thinking: ~50-100 TPS with 8 GPUs
  - Coder: ~30-70 TPS with 8 GPUs
- **Memory Usage**:
  - 60-70GB per GPU at full utilization
  - Lower with dynamic batching

### Quality Metrics
- **SQL Accuracy**: Compare with GPT-4 baseline
- **Execution Success Rate**: >80% expected
- **Semantic Correctness**: >70% expected

## Monitoring and Logging

### Key Metrics to Track
1. GPU memory allocation/usage
2. Inference latency
3. Token generation speed
4. Error rates
5. SQL execution success

### Log Files
- Model loading: `logs/qwen_{model}_{job_id}.log`
- Test results: `results/{model}_test_{job_id}.json`
- Performance metrics: `results/scale_{model}_{gpus}gpu_{job_id}.json`

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Solution: Increase GPU count or reduce batch size
   - Use `--max_tokens` to limit generation length

2. **Slow Loading**
   - Solution: Use faster storage (NVMe SSD)
   - Pre-download models to local cache

3. **CUDA Errors with FP8**
   - Solution: Set `CUDA_LAUNCH_BLOCKING=1`
   - Update CUDA drivers to 12.1+

4. **Distributed Inference Issues**
   - Solution: Ensure all GPUs visible
   - Check tensor parallel configuration

## Cost-Benefit Analysis

### GPU Hour Estimates
- Thinking Model: 4-8 H100 hours per test session
- Coder Model: 8-16 H100 hours per test session
- Total for comprehensive testing: ~100 H100 hours

### Expected Benefits
- No API costs for inference
- Full control over model behavior
- Ability to fine-tune if needed
- Better data privacy

## Recommendations

### For NL2SQL Tasks
1. **Start with Qwen3-235B-Thinking** for complex reasoning queries
2. **Use Qwen3-480B-Coder** for straightforward SQL generation
3. **Allocate 8 GPUs minimum** for production use
4. **Implement caching** for common queries

### For Production Deployment
1. Use **vLLM or SGLang** for better serving performance
2. Implement **request batching** for efficiency
3. Set up **model sharding** across nodes if needed
4. Monitor GPU utilization continuously

## Next Steps

1. **Immediate**: Run basic functionality tests
2. **Week 1**: Complete scalability testing
3. **Week 2**: Optimize for NL2SQL pipeline
4. **Month 1**: Production readiness assessment

## Contact and Support

For issues or questions:
- Check model documentation: https://huggingface.co/Qwen/
- Review transformers issues: https://github.com/huggingface/transformers
- Contact HPC support for GPU allocation

---

*Document Version: 1.0*
*Last Updated: 2025*