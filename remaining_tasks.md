# Remaining Tasks and Future Considerations

## Schema Linking Ground Truth

**Status**: To be decided later

**Challenge**:
- How to efficiently and accurately obtain ground truth for schema linking?
- Need reliable annotation method for evaluation

**Potential Approaches**:
1. Parse gold SQL to extract referenced tables/columns
2. Manual annotation by domain experts
3. Automatic extraction with validation
4. Hybrid approach: auto-extract + human verification

**Action**: Revisit this after initial pipeline implementation

---

## Notes

### Cell Value Numeric Handling
- Decision: Don't pass numeric cell values to LLM as input
- Reasoning: Numbers as cell values provide limited semantic information
- Exception: May still use for statistics (min/max/avg) if needed

### Query Plan Definition (Based on CHASE-SQL)
- Not traditional database execution plan
- Human-readable 3-step reasoning process:
  1. Find relevant tables
  2. Perform operations (filter, join, aggregate)
  3. Select final columns and return results
- Reference: "CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL"

### Sub-task Confidence Calculation
- Method: LLM generates confidence scores directly
- Output format: JSON with task + confidence
- Re-calculation: After completing highest-confidence task, recalculate remaining tasks with updated context
