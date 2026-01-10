# MLoop Roadmap

**Mission**: Excellent MLOps with Minimum Cost

This roadmap aligns all development with MLoop's core philosophy: enabling production-quality ML models with minimal coding, minimal ML expertise, and minimal operational complexity.

---

## Current Status (v0.2.0 - January 2026)

### Core Platform
- ML.NET 5.0 with AutoML 0.23.0
- Filesystem-based MLOps with git-friendly experiment tracking
- Multi-process concurrent training support
- Production model promotion and discovery
- Batch prediction with auto-discovery
- CLI with comprehensive command set
- .NET 10.0 + C# 13 modern codebase

### AI Agent System (Ironbees v0.4.1)
- Multi-provider LLM support (OpenAI, Anthropic, Google, AWS, Azure, Ollama)
- YAML-based agent templates (agent.yaml + system-prompt.md)
- 5 specialized agents: data-analyst, model-architect, preprocessing-expert, experiment-explainer, ml-tutor
- Semantic conversation memory (MemoryIndexer v0.6.0)

### Quality
- 464+ tests passing (Core + API + AIAgent + Pipeline)
- 15 LLM integration tests for AI agents

---

## Completed Phases

### Phase 0: Data Preparation Excellence
**Goal**: Enable 100% dataset coverage with minimal user effort

**Delivered**:
- `IPreprocessingScript` interface with sequential execution model
- `ScriptLoader` with Roslyn compilation and DLL caching (<50ms)
- `PreprocessingEngine` orchestration layer
- `mloop preprocess` CLI command (manual + validation modes)
- Auto-preprocessing in `mloop train` (transparent execution)
- 7 example preprocessing scripts (datetime, unpivot, feature engineering, cleaning, encoding, imputation, outliers)
- 36 unit tests, all passing

### Phase 1: Extensibility System
**Goal**: Enable expert users to customize while maintaining simplicity

**Delivered**:
- `IMLoopHook` interface with lifecycle hooks (pre-train, post-train, pre-predict, post-evaluate)
- `HookEngine` integrated into training pipeline (<1ms overhead)
- `mloop new hook` CLI command with templates
- `IMLoopMetric` interface for business metrics
- `MetricEngine` with script discovery
- 4 example hooks (validation, MLflow, gates, deployment)
- 3 example metrics (profit, churn, ROI)

### Phase 2: AI Agent Enhancements
**Goal**: Reduce knowledge cost through intelligent assistance

**Delivered**:
- **data-analyst**: Dataset analysis, preprocessing strategy recommendations, class imbalance detection
- **model-architect**: Complexity-based time budgets, intelligent AutoML configuration
- **preprocessing-expert**: Feature engineering with domain-specific patterns (e-commerce, healthcare, finance)
- **experiment-explainer**: Algorithm explanation, metric interpretation
- **ml-tutor**: Interactive ML tutorials, Q&A mode
- Conversation context awareness and proactive assistance
- 15 LLM integration tests

### Phase 3: FilePrepper Integration
**Goal**: Simplify data preparation for common cases

**Delivered**:
- `DataQualityAnalyzer` with 7 issue types (encoding, missing values, duplicates, outliers, etc.)
- `PreprocessingScriptGenerator` for automatic script generation
- `--analyze-data` and `--generate-script` CLI options
- 4 preprocessing recipe scripts

### Additional Deliverables
- `mloop docker` command (Dockerfile, docker-compose.yml, .dockerignore generation)
- 3 end-to-end tutorials (Iris, Sentiment Analysis, Housing Prices)
- RECIPE-INDEX.md with 22 organized recipes

---

## Future Considerations (P3 LOW)

### Advanced Features
- Automated feature selection
- Multi-model ensemble support
- Transfer learning integration
- Real-time prediction streaming

### Enterprise Features
- Team collaboration features
- Experiment comparison dashboard
- Model registry with governance
- Audit logging and compliance

### Platform Expansion
- Python SDK for MLoop
- VS Code extension
- Azure ML / AWS SageMaker integration
- Cloud-native deployment

---

## Release Schedule

| Version | Date | Focus | Status |
|---------|------|-------|--------|
| **v0.1.0** | Nov 2025 | ML.NET 5.0 + Core | ✅ Complete |
| **v0.2.0** | Jan 2026 | Preprocessing + Extensibility + AI Agents | ✅ Complete |
| **v1.0.0** | TBD | Production-Ready Release | 🎯 Target |

---

## Success Metrics

**Core Mission: Excellent MLOps with Minimum Cost**

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Development Cost | 2-4 weeks | 1-2 hours | ✅ Achieved |
| Knowledge Required | ML degree | Basic CSV | ✅ Achieved |
| Operational Cost | Docker + K8s + MLOps | CLI + filesystem | ✅ Achieved |
| Time to Production | Weeks | Hours | ✅ Achieved |

---

## Contributing

To propose changes:

1. **Philosophy Alignment**: Does the feature reduce cost (development/knowledge/operational)?
2. **User Impact**: How many users benefit? How significantly?
3. **Complexity**: Does it maintain simplicity or require trade-offs?

Submit proposals via GitHub Issues with `roadmap` label.

---

**Last Updated**: January 10, 2026
**Version**: 0.2.0
