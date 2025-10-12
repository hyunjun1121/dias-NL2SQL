# Kyungmin's Feedback Implementation Summary

## Date: 2025-10-12

경민님의 피드백을 바탕으로 구현한 내용 정리

---

## ✅ Implemented Features

### 1. Error Recovery Hierarchy ⚠️

**경민님 요청사항:**
> "Error들에 대해 잘 hierarchy를 구성해야 하는데, SQL이 잘못된 syntax error 검사를 1차적으로 하고, 살아남은 경우에 semantic error를 2차로"

**구현 내용:**
```python
# model/progressive_executor.py
def execute_progressive(...):
    # 1st level: Syntax error check + retry
    sql, exec_result = self._generate_and_execute_with_retry(
        task, context, schema, max_syntax_retries=2
    )

    # 2nd level: Semantic error check (only if syntax OK)
    if exec_result['success']:
        reward_dict = self.reward_model.calculate_reward(...)
```

**특징:**
- **1차 (Syntax)**: 실행해서 syntax error → 최대 2회 재시도 with error feedback
- **2차 (Semantic)**: Execution 성공 시 → LLM이 의미적 correctness 판단
- Error message를 prompt에 포함하여 retry 시 개선

**Reference**: https://www.vldb.org/pvldb/vol13/p1737-kim.pdf Figure 9

---

### 2. CHASE-SQL Baseline Implementation 📊

**경민님 요청사항:**
> "CHASE-SQL 포함 baseline들 확실하게 이길 것 같다 하면 없이 우리 것만 나중에 benchmark 전체 돌려서 더 좋은 숫자 보여주면 충분하고요"

**구현 내용:**
- `baseline/chase_sql.py`: CHASE-SQL one-shot approach
- `scripts/compare_baselines.py`: Head-to-head comparison

**차이점:**
| Aspect | CHASE-SQL | Our Method |
|--------|-----------|------------|
| Approach | Query plan → One-shot SQL generation | Query plan → Progressive execution |
| Context | No context accumulation | Context accumulation per task |
| Error handling | Single attempt | Syntax retry + Semantic check |
| Iterations | 1 (one-shot) | Multiple (progressive) |

**사용법:**
```bash
python scripts/compare_baselines.py \
    --data_path /path/to/bird \
    --limit 50 \
    --output comparison_results.json
```

---

### 3. Open-Source Model Support 🤖

**경민님 요청사항:**
> "우선 학교 클러스터에서 open-source 모델을 써야할 것 같아요. 외부 API 비용에 대해서는 보수적이라.."

**구현 내용:**
```python
# utils/llm_client.py - Unified LLMClient

# vLLM cluster (recommended)
client = LLMClient(
    model_name="deepseek-r1",  # or qwen2.5, llama-3.3
    base_url="http://cluster:8000/v1"
)

# HuggingFace Transformers
client = LLMClient(model_name="hf:Qwen/Qwen2.5-7B-Instruct")

# Ollama local
client = LLMClient(model_name="ollama:qwen2.5")
```

**지원 모델:**
- ✅ DeepSeek-R1 (recommended by Kyungmin)
- ✅ Qwen2.5-72B-Instruct
- ✅ Llama-3.3-70B
- ✅ GPT-4o, Claude (for comparison)

**Cost strategy:**
- Primary: Open-source on school cluster (~$0)
- Development: Proprietary for prototyping (개인 환급)
- Budget: $500 (모니터링하다 필요하면 늘릴 수 있음)

---

### 4. Confidence Recalculation Strategy 🎯

**경민님 답변:**
> "Input string이 업데이트 될 때마다 계산하는게 맞는거같긴 한데, 별도 prompt로 가져올 경우 이건 줄이긴 해야겠네요"

**구현 방향:**
```python
# Option A: Token logprobs 사용 (추후 구현)
# GPT-4o API logprobs로 confidence 계산

# Option B: 별도 prompt (현재)
# 2-3개 task마다 recalculate로 줄이기
```

**현재 설정:**
- Config에서 설정 가능하도록 준비
- Default: 매번 recalculate (정확도 우선)
- 비용 문제 시 조정 가능

---

### 5. Schema Linking Strategy 🔗

**경민님 답변:**
> "단순하게는 지금 공개된 schema linking 구현을 쓰되, 이쪽을 파려면 실제 예제들 분석하는 건 필요해보이네요. + 네 현재 못하는 것을 실제로 보이고 풀면 이것만으로 논문이 됩니다"

**구현 방향:**
1. 일단 기존 SOTA schema linking 사용 (RESDSQL, DAIL-SQL)
2. Validation set에서 bottleneck 파악
3. 문제 있으면 개선 (논문 가능성)

**현재 상태:**
- QueryPlanGenerator에서 schema 전달
- 개선 필요 시 별도 component 추가 예정

---

## 📋 Current Status

### Completed ✅
1. ✅ Error recovery hierarchy (syntax + semantic)
2. ✅ Syntax error retry mechanism (max 2 times)
3. ✅ CHASE-SQL baseline implementation
4. ✅ Open-source model support (vLLM, Transformers, Ollama)
5. ✅ Comparison framework (scripts/compare_baselines.py)

### In Progress 🔄
1. Validation set creation (BIRD dev set 50-100 examples)
2. Confidence recalculation optimization
3. Schema linking analysis

### Pending ⏳
1. Multi-branch reasoning (MS rStar style)
   - **Decision**: 나중에 (일단 single branch로 baseline 확보)
2. Token logprobs for confidence
3. Ground truth schema links
4. Full BIRD benchmark evaluation

---

## 🎯 Next Steps

### Immediate (이번 주)
1. **Validation set 구성**
   - BIRD dev에서 50-100 examples 선정
   - Stratified sampling (simple vs complex)

2. **Baseline 비교 실행**
   - Our method vs CHASE-SQL
   - Metrics: execution accuracy, semantic correctness, time

3. **Bottleneck 파악**
   - Schema linking 성능
   - Syntax error 빈도
   - Semantic error 빈도

### Short-term (다음 주)
1. **Ablation study 설계**
   - Confidence score의 영향
   - Progressive execution vs one-shot
   - Error recovery의 영향

2. **학교 클러스터 셋업**
   - vLLM 서버 접근
   - DeepSeek-R1 or Qwen2.5 테스트

### Long-term (이후)
1. Multi-branch reasoning (if needed)
2. Schema linking 개선 (if bottleneck)
3. Full BIRD benchmark
4. Paper writing

---

## 📊 Implementation Strategy (Kyungmin's guidance)

> "가장 단순한 형태로 구현 후에(새로운 아이디어가 아닌 부분은 이전 코드에서 최대한 가져오고), 단계별로, 즉 confidence score가 정확도에 미치는 영향, schema linking이 미치는 영향, syntax error recovery가 미치는 영향 등을 보고 개선하면 될 것 같아요"

### Ablation Study Plan
1. **Baseline**: CHASE-SQL (one-shot)
2. **+Progressive**: Our progressive execution
3. **+Confidence**: With LLM confidence scores
4. **+Error Recovery**: With syntax/semantic recovery
5. **+Schema Linking**: With improved schema linking (if needed)

각각이 optimization subsection이 됨!

---

## 💡 Key Insights from Kyungmin

### 1. Cost & Model Selection
- "Proprietary, large model 안써도 성능 따라잡을 수 있고, 얼마만큼 시간 및 비용 줄일수있다고 보여주면 베스트"
- "Task가 매우 잘게 쪼개진 덕에 비용은 낮은데, 성능은 잘 나올거 같긴 합니다"

### 2. Error Hierarchy
- "Syntax error 개선이 훨씬 쉬운데(돌아가는지/아닌지만 판단하면 되니까)"
- "Semantic error는 실제 NL question 의미에 합당한 결과인지를 판단해야 합니다"

### 3. Evaluation Strategy
- "매번 전체 benchmark를 돌리기보다, 특정 validation set을 뽑고 잘 된다는 확신이 들 때에만 test set을 키워나가면 될 것 같아요"
- "$500인데, 계속 모니터링하다 필요할 경우 늘릴 수 있습니다"

### 4. Paper Strategy
- "실제 bottleneck이 무엇인지 파악 후에 우리 아이디어가 이걸 잘 해소해주냐를 보고"
- "Schema linking을 잘한다거나 하면 이 부분에 대해서는 더 개선하기보다 다른 부분을 파고"

---

## 📂 File Structure

```
EPFL_hyunjun/
├── baseline/
│   ├── chase_sql.py              # CHASE-SQL one-shot baseline
│   └── __init__.py
├── docs/
│   └── MODEL_USAGE.md            # Open-source model usage guide
├── model/
│   ├── progressive_executor.py   # ✨ Error recovery added
│   └── semantic_reward.py        # Binary reward approach
├── scripts/
│   └── compare_baselines.py      # Comparison framework
├── utils/
│   └── llm_client.py             # ✨ Multi-backend support
└── KYUNGMIN_FEEDBACK_IMPLEMENTATION.md  # This file
```

---

## 🔗 References

1. Error hierarchy: https://www.vldb.org/pvldb/vol13/p1737-kim.pdf Figure 9
2. CHASE-SQL: 3-step reasoning approach
3. vLLM: https://github.com/vllm-project/vllm

---

**Last Updated**: 2025-10-12
**Status**: ✅ Core features implemented, ready for validation
