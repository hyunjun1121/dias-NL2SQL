# Model Usage Guide

## Supported Models

### 1. Proprietary Models (API)

#### OpenAI GPT-4o
```python
from utils.llm_client import LLMClient

client = LLMClient(
    model_name="gpt-4o",
    api_key="your-openai-key"
)
```

#### Anthropic Claude
```python
client = LLMClient(
    model_name="claude-3.5-sonnet",
    api_key="your-anthropic-key"
)
```

---

### 2. Open-Source Models (Recommended for cost)

#### vLLM Cluster (Kyungmin's recommendation)
**Best for: School cluster deployment**

```python
# DeepSeek-R1 (recommended by Kyungmin)
client = LLMClient(
    model_name="deepseek-r1",
    base_url="http://your-cluster:8000/v1"
)

# Qwen2.5
client = LLMClient(
    model_name="qwen2.5-72b-instruct",
    base_url="http://your-cluster:8000/v1"
)

# Llama-3.3
client = LLMClient(
    model_name="llama-3.3-70b-instruct",
    base_url="http://your-cluster:8000/v1"
)
```

**Environment setup:**
```bash
export VLLM_BASE_URL=http://your-cluster:8000/v1
```

#### HuggingFace Transformers (Local)
**Best for: Single GPU, small models**

```python
client = LLMClient(
    model_name="hf:Qwen/Qwen2.5-7B-Instruct"
)
```

Requirements:
```bash
pip install transformers torch accelerate
```

#### Ollama (Local)
**Best for: Quick local testing**

```python
client = LLMClient(
    model_name="ollama:qwen2.5"
)
```

Requirements:
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5
pip install ollama-python
```

---

### 3. Custom Cluster Endpoint

```python
client = LLMClient(
    model_name="cluster:your-model",
    base_url="http://your-endpoint:8000/v1"
)
```

---

## Usage in Pipeline

```python
from config.config import get_bird_config
from pipeline.main_pipeline import EPFLHyunjunPipeline

# Option 1: Use config (edit config/config.py)
config = get_bird_config()
config.llm.model_name = "deepseek-r1"  # or "gpt-4o", "qwen2.5", etc.
pipeline = EPFLHyunjunPipeline(config)

# Option 2: Direct initialization
from utils.llm_client import LLMClient
llm_client = LLMClient(
    model_name="deepseek-r1",
    base_url="http://cluster:8000/v1"
)
```

---

## Recommended Models for Research

### Based on Kyungmin's feedback:

1. **Primary: Open-source models (학교 클러스터)**
   - DeepSeek-R1 (best performance/cost)
   - Qwen2.5-72B-Instruct (good alternative)
   - Llama-3.3-70B (stable option)

2. **Development: Proprietary (개인 환급 가능)**
   - GPT-4o-mini (fast prototyping)
   - GPT-4o (when performance needed)

3. **Production: Cost-effective open-source**
   - 중국 open-source models (잘게 쪼개진 task → 저비용 고성능)

---

## Budget Considerations

**Current budget: $500**

**Strategy:**
1. Use open-source for bulk evaluation (BIRD dev set)
2. Use proprietary only for:
   - Initial prototyping
   - Performance comparison
   - Final benchmark run (if needed)

**Cost savings with our approach:**
- Task가 잘게 쪼개짐 → 작은 모델도 잘 작동
- Open-source models: ~free (cluster) vs $100+ (GPT-4o for full benchmark)

---

## Performance Comparison (To be filled)

| Model | Execution Acc | Semantic Acc | Avg Time | Cost/Query |
|-------|--------------|--------------|----------|------------|
| GPT-4o | TBD | TBD | TBD | $0.05-0.10 |
| DeepSeek-R1 | TBD | TBD | TBD | ~$0 (cluster) |
| Qwen2.5-72B | TBD | TBD | TBD | ~$0 (cluster) |

---

## Troubleshooting

### vLLM Connection Issues
```python
# Test connection
import requests
response = requests.get("http://your-cluster:8000/v1/models")
print(response.json())
```

### Cluster Setup (For IT)
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4
```

### Memory Issues (Transformers)
```python
# Use 8-bit quantization
client = LLMClient(
    model_name="hf:Qwen/Qwen2.5-7B-Instruct"
)
# Modify _load_transformers_model to add load_in_8bit=True
```
