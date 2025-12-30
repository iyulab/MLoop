# ADR-001: Incremental Sampling Preprocessing Architecture

## Status

**Accepted** - 2024-12-30

## Context

MLoop's current preprocessing agents (DataAnalystAgent, PreprocessingExpertAgent) analyze datasets by loading all data into memory. This approach becomes impractical for large-scale datasets (100K+ records) due to:

1. **Memory constraints**: Loading 100K+ records for analysis
2. **Time constraints**: Full dataset analysis is slow
3. **Human review impossibility**: Cannot manually verify all records
4. **Rule discovery uncertainty**: Need statistical confidence in discovered patterns

The development team identified a need for **incremental sampling-based preprocessing** that:
- Discovers patterns through progressive sampling
- Builds confidence in rules before full application
- Involves human judgment for business logic decisions
- Produces reusable preprocessing artifacts

## Decision

We will implement an **Incremental Preprocessing Agent** with a 5-stage workflow:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                IncrementalPreprocessingAgent                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Sampling    │  │    Rule      │  │  Confidence  │          │
│  │  Strategy    │  │  Discovery   │  │  Calculator  │          │
│  │              │  │   Engine     │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│                    ┌──────▼───────┐                             │
│                    │    State     │                             │
│                    │   Manager    │                             │
│                    │  (5 stages)  │                             │
│                    └──────┬───────┘                             │
│                           │                                     │
│  ┌────────────────────────┼────────────────────────┐           │
│  │                        │                        │           │
│  ▼                        ▼                        ▼           │
│ HITL               FilePrepper              Deliverables       │
│ Interface          Integration              Generator          │
└─────────────────────────────────────────────────────────────────┘
```

### 5-Stage Workflow

| Stage | Sample % | Purpose | Output |
|-------|----------|---------|--------|
| 1. Initial Exploration | 0.1% (100) | Schema discovery, obvious patterns | Initial rule candidates |
| 2. Pattern Expansion | 0.5% (500) | Pattern validation, auto-fixes | Confirmed rules |
| 3. HITL Decision | 1.5% (1500) | Business logic decisions | User-approved rules |
| 4. Confidence Checkpoint | 2.5% (2500) | Statistical validation | Final rule set |
| 5. Bulk Processing | 100% | Full application | Cleaned dataset |

### Key Components

1. **SamplingStrategy**: Determines sample sizes and selection methods
   - Random sampling for initial exploration
   - Stratified sampling for pattern validation
   - Confidence-based sampling for validation

2. **RuleDiscoveryEngine**: Identifies preprocessing patterns
   - Missing value patterns (nulls, "N/A", empty strings)
   - Type inconsistencies (mixed numeric/string)
   - Format variations (date formats, encodings)
   - Outlier patterns (statistical anomalies)

3. **ConfidenceCalculator**: Measures rule stability
   - Rule consistency across samples
   - Exception rate tracking
   - Statistical significance testing
   - Convergence detection (no new patterns in N samples)

4. **InteractivePromptBuilder**: HITL interface
   - Multiple choice questions for ambiguous decisions
   - Context-rich explanations
   - Default recommendations with rationale

### Rule Types

```csharp
public enum PreprocessingRuleType
{
    // Auto-resolvable (no HITL needed)
    DateFormatStandardization,    // Multiple date formats → ISO-8601
    EncodingNormalization,        // Mixed encodings → UTF-8
    WhitespaceNormalization,      // Trim, collapse spaces

    // Requires HITL
    MissingValueStrategy,         // Delete vs Impute vs Default
    OutlierHandling,              // Keep vs Remove vs Cap
    CategoryMapping,              // Unknown categories handling
    BusinessLogicDecision         // Domain-specific rules
}
```

### Integration Points

1. **FilePrepper**: Leverages DataPipeline for efficient row-by-row processing
2. **Ironbees ConversationService**: Manages HITL conversation state
3. **PreprocessingExpertAgent**: Generates C# scripts from discovered rules

## Consequences

### Positive

- **Scalability**: Can process datasets of any size with constant memory
- **Confidence**: Statistical validation before full application
- **Transparency**: Human oversight for critical decisions
- **Reusability**: Generated scripts work for future datasets
- **Efficiency**: 2.5% sample typically sufficient for rule discovery

### Negative

- **Complexity**: More complex than batch processing
- **Latency**: Multiple HITL interactions add time
- **State Management**: Need to track workflow progress
- **Edge Cases**: Very heterogeneous data may need more sampling

### Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Rules don't generalize | Medium | High | Confidence threshold (98%) before full apply |
| HITL fatigue | Low | Medium | Batch similar questions, smart defaults |
| Memory issues in bulk phase | Low | High | Streaming processing via FilePrepper |
| Sampling bias | Medium | Medium | Stratified sampling, bias detection |

## Alternatives Considered

1. **Full dataset analysis with pagination**: Rejected - still requires full scan
2. **Pure ML-based rule discovery**: Rejected - lacks explainability for HITL
3. **Fixed sampling percentage**: Rejected - doesn't adapt to data complexity
4. **No HITL, fully automatic**: Rejected - business logic needs human judgment

## Implementation Plan

### Phase 11.1: Core Infrastructure (Week 1)
- [ ] SamplingStrategy implementation
- [ ] RuleDiscoveryEngine basic patterns
- [ ] ConfidenceCalculator metrics

### Phase 11.2: State Machine (Week 2)
- [ ] PreprocessingWorkflowState enum
- [ ] Stage transition logic
- [ ] Progress persistence

### Phase 11.3: HITL Interface (Week 3)
- [ ] InteractivePromptBuilder
- [ ] CLI integration
- [ ] Decision recording

### Phase 11.4: Integration & Deliverables (Week 4)
- [ ] FilePrepper streaming integration
- [ ] Script generation from rules
- [ ] Exception log and reports

## References

- [MLoop Philosophy](../README.md#philosophy-excellent-mlops-with-minimum-cost)
- [Ironbees Workflow Documentation](https://github.com/iyulab/ironbees)
- [FilePrepper Pipeline API](https://github.com/iyulab/FilePrepper)
- [Internal Design Meeting Notes](./internal/preprocessing-agent-design.md)
