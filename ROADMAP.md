# MLoop Roadmap

**Mission**: Excellent MLOps with Minimum Cost

This roadmap aligns all development with MLoop's core philosophy: enabling production-quality ML models with minimal coding, minimal ML expertise, and minimal operational complexity.

---

## Current Status (v0.1.0 - November 2025)

### ✅ Completed Features

**Core Functionality**
- ✅ ML.NET 5.0 migration complete with AutoML 0.23.0
- ✅ Filesystem-based MLOps with git-friendly experiment tracking
- ✅ Multi-process concurrent training support
- ✅ Production model promotion and discovery
- ✅ Batch prediction with auto-discovery
- ✅ CLI with comprehensive command set

**AI Agent System** (Ironbees v0.4.1)
- ✅ Multi-provider LLM support (OpenAI, Anthropic, Google, AWS, Azure, Ollama)
- ✅ Interactive ML assistance and guidance
- ✅ Context-aware experiment analysis
- ✅ Conversation history management via FileSystemConversationStore
- ✅ YAML-based agent templates (agent.yaml + system-prompt.md)
- ✅ Enhanced agents (MLOOP-006):
  - data-analyst: Dataset analysis with preprocessing strategy recommendations
  - model-architect: Intelligent AutoML configuration with complexity-based time budgets
  - preprocessing-expert: Feature engineering assistant with domain-specific patterns
  - All agents: Conversation memory, context awareness, proactive assistance

**Quality & Testing**
- ✅ 436 tests passing (Core + API + AIAgent + Pipeline)
- ✅ 15 LLM integration tests for AI agents (data-analyst, model-architect, preprocessing-expert)
- ✅ Real-world integration testing validated
- ✅ .NET 10.0 + C# 13 modern codebase

---

## Priority Framework

All features are prioritized based on their impact on the core mission:

- **P0 CRITICAL**: Essential for "minimum cost" goal - blocks basic workflow
- **P1 HIGH**: Significantly reduces development/knowledge/operational cost
- **P2 MEDIUM**: Enhances value but not essential for core mission
- **P3 LOW**: Nice-to-have, deferred until P0-P1 complete

---

## Phase 0: Data Preparation Excellence (P0 CRITICAL)
**Timeline**: Weeks 1-2 (Nov 11-22, 2025)
**Goal**: Enable 100% dataset coverage with minimal user effort

### Problem Statement
Current MLoop/FilePrepper handles only **50% of real-world datasets** (3/6 from analysis). Critical gaps:
- Multi-file operations (joins, merges, concatenation)
- Wide-to-Long transformations (unpivot operations)
- Complex feature engineering beyond basic preprocessing

### Tasks

#### Week 1: Core Infrastructure
- [x] **T0.1**: Design `IPreprocessingScript` interface and execution model ✅
  - Input: CSV path, Output: Transformed CSV path
  - Sequential execution: `01_*.cs` → `02_*.cs` → `03_*.cs`
  - Context API: File I/O, FilePrepper integration, logging
  - Implementation: `MLoop.Extensibility.Preprocessing.IPreprocessingScript`

- [x] **T0.2**: Implement `ScriptCompiler` with Roslyn integration ✅
  - Compile `.cs` scripts with full ML.NET and MLoop.Core references
  - Cache compiled DLLs for performance (<50ms cached load)
  - Graceful degradation: Script failure doesn't break AutoML
  - Implementation: `MLoop.Core.Scripting.ScriptLoader`

- [x] **T0.3**: Build `PreprocessingEngine` orchestration layer ✅
  - Script discovery and execution ordering
  - Temporary file management (`.mloop/temp/`)
  - Progress reporting and error handling
  - Implementation: `MLoop.Core.Preprocessing.PreprocessingEngine`

- [x] **T0.4**: Unit tests for preprocessing system (>90% coverage) ✅
  - PreprocessingEngine tests: 13 tests
  - ScriptLoader tests: 9 tests
  - CsvHelper tests: 14 tests
  - Total: 36 tests, all passing

#### Week 2: CLI Integration & Examples
- [x] **T0.5**: CLI command: `mloop preprocess` ✅
  - Manual preprocessing for debugging: `mloop preprocess --input raw.csv --output train.csv`
  - Validation mode: `mloop preprocess --validate`
  - Implementation: `MLoop.CLI.Commands.PreprocessCommand` (traditional + incremental workflows)

- [x] **T0.6**: Auto-preprocessing in `mloop train` ✅
  - Detect `.mloop/scripts/preprocess/*.cs`
  - Execute preprocessing pipeline before AutoML training
  - Transparent to user: Just works™
  - Implementation: `MLoop.CLI.Commands.TrainCommand` (lines 200-220)

- [x] **T0.7**: Example preprocessing scripts ✅
  - DateTime extraction example (01_datetime_features.cs) - Dataset 005 pattern
  - Wide-to-Long unpivot example (02_unpivot_shipments.cs) - Dataset 006 pattern
  - Feature engineering example (03_feature_engineering.cs) - Production efficiency metrics
  - Comprehensive README.md with patterns, testing, debugging tips
  - Implementation: `examples/preprocessing-scripts/` directory (all scripts verified)

- [x] **T0.8**: Documentation: Preprocessing guide ✅
  - Fixed GUIDE.md incorrect API example (replaced ExtractDateTimeFeaturesAsync with correct ReadAsync/WriteAsync)
  - examples/preprocessing-scripts/README.md provides comprehensive guide
  - Covers: Script structure, sequential execution, context APIs, testing patterns, best practices
  - Implementation: `docs/GUIDE.md` (lines 395-442), `examples/preprocessing-scripts/README.md` (348 lines)

**Success Criteria**:
- 100% dataset coverage (6/6 datasets trainable)
- Zero breaking changes to existing workflows
- <1ms overhead when no preprocessing scripts present
- Comprehensive examples for common scenarios

---

## Phase 1: Extensibility System (P1 HIGH)
**Timeline**: Weeks 3-4 (Nov 25 - Dec 6, 2025)
**Goal**: Enable expert users to customize while maintaining simplicity

### Week 3: Lifecycle Hooks
- [ ] **T1.1**: Design `IMLoopHook` interface
  - Hook points: `pre-train`, `post-train`, `pre-predict`, `post-evaluate`
  - `HookContext` with data access, metadata, logger
  - `HookResult`: Continue, Abort, ModifyConfig

- [ ] **T1.2**: Integrate hooks into training pipeline
  - Hook discovery and compilation (reuse ScriptCompiler)
  - Execution at appropriate lifecycle points
  - Performance impact: <50ms per hook

- [ ] **T1.3**: Example hooks
  - Data validation hook (minimum rows, class balance)
  - MLflow logging integration
  - Model performance gate (abort if accuracy < threshold)
  - Automated deployment trigger

- [ ] **T1.4**: CLI: `mloop new hook --name <name> --type <pre-train|post-train>`
  - Generate template hook script with boilerplate
  - Interactive wizard for common use cases

### Week 4: Custom Metrics
- [x] **T1.5**: Design `IMLoopMetric` interface ✅
  - Business metric calculation from ML predictions
  - Post-evaluation approach (not AutoML integration)
  - Implementation: `MLoop.Extensibility.Metrics.IMLoopMetric`, `MetricContext`, `MetricResult`

- [x] **T1.6**: MetricEngine implementation ✅
  - Script discovery from `.mloop/scripts/metrics/*.cs`
  - Execution with ScriptLoader integration
  - Zero-overhead when no metrics present (<1ms)
  - Implementation: `MLoop.Core.Metrics.MetricEngine`

- [x] **T1.7**: Example custom metrics ✅
  - Profit maximization (TP profit - FP cost)
  - Churn prevention value (LTV - intervention cost)
  - ROI optimization
  - Implementation: `docs/examples/metrics/` with 3 examples

- [x] **T1.8**: Documentation: Extensibility guide ✅
  - Comprehensive README.md with business metrics patterns
  - Usage examples, testing patterns, debugging tips
  - Integration with mloop evaluate workflow
  - Implementation: `docs/examples/metrics/README.md` (350+ lines)

**Success Criteria**:
- Hooks execute at correct pipeline stages
- Custom metrics guide AutoML optimization
- Zero overhead when extensions not used
- Backward compatibility: 100% (all v0.1.0 workflows unchanged)

---

## Phase 2: AI Agent Enhancements (P1 HIGH)
**Timeline**: Weeks 5-6 (Dec 9-20, 2025)
**Goal**: Reduce knowledge cost further through intelligent assistance

### Week 5: Intelligent Optimization
- [x] **T2.1**: Dataset analysis and recommendations (✅ Completed MLOOP-006)
  - Auto-detect data quality issues (missing values, outliers, imbalance)
  - Suggest preprocessing strategies
  - Recommend appropriate ML task type (classification, regression)
  - Added capabilities: preprocessing-strategy-recommendation, class-imbalance-detection, ml-task-type-recommendation
  - Created DataAnalystAgentTests with 5 LLM integration tests

- [x] **T2.2**: Hyperparameter suggestion system (✅ Completed MLOOP-006)
  - Analyze dataset characteristics (size, features, task)
  - Suggest AutoML time budget based on complexity
  - Recommend metric optimization based on problem type
  - Implemented complexity-based time budget calculation with feature count multipliers, data quality adjustments
  - Added capabilities: complexity-based-time-budget, dataset-characteristics-analysis, intelligent-configuration-recommendation
  - Created ModelArchitectAgentTests with 5 LLM integration tests

- [x] **T2.3**: Feature engineering assistance (✅ Completed MLOOP-006)
  - Identify datetime columns → suggest time-based features (temporal components, cyclical encoding)
  - Detect categorical features → suggest encoding strategies (target/frequency/hash encoding, interactions)
  - Recommend interaction features for high cardinality
  - Added domain-specific patterns (e-commerce, healthcare, finance, real estate)
  - Added capabilities: datetime-feature-extraction, interaction-feature-suggestion, polynomial-feature-generation, domain-specific-feature-engineering
  - Created PreprocessingExpertAgentTests with 5 LLM integration tests

### Week 6: Learning and Explanation
- [x] **T2.4**: Experiment analysis and explanation ✅
  - Explain why AutoML selected specific algorithm
  - Interpret model metrics for non-experts
  - Suggest improvements based on metrics
  - Created `experiment-explainer` agent with algorithm explanation, metric interpretation, performance analysis capabilities
  - Implementation: `examples/mloop-agents/.mloop/agents/experiment-explainer/` (agent.yaml + system-prompt.md)

- [x] **T2.5**: Interactive tutorials via agent ✅
  - "Teach me ML basics" conversation mode
  - "What does F1 score mean?" explanations
  - "How do I improve my model?" guidance
  - Created `ml-tutor` agent with interactive learning modes (tutorial, Q&A, guided practice)
  - Implementation: `examples/mloop-agents/.mloop/agents/ml-tutor/` (agent.yaml + system-prompt.md)

- [x] **T2.6**: Agent memory and context (✅ Completed MLOOP-006)
  - Remember user's ML experience level and preferences
  - Track common issues and provide proactive help
  - Learn from user's dataset patterns
  - Enhanced all agents with conversation context awareness, proactive assistance, learning from interactions
  - Added capabilities: conversation-context-awareness, user-experience-level-adaptation, proactive-assistance, pattern-learning-from-history

**Success Criteria**:
- AI agents reduce "time to first model" by 50%
- Users without ML background successfully train models
- Agents provide actionable, specific recommendations
- Educational value: Users learn ML concepts through interaction

---

## Phase 3: FilePrepper Integration (P2 MEDIUM)
**Timeline**: Weeks 7-8 (Jan 2025)
**Goal**: Simplify data preparation for common cases

### Tasks
- [ ] **T3.1**: Automatic FilePrepper detection
  - Detect when CSV needs preprocessing (encoding issues, missing values)
  - Auto-suggest FilePrepper transformations

- [ ] **T3.2**: FilePrepper → Preprocessing script bridge
  - Generate preprocessing script from FilePrepper operations
  - Allow users to customize auto-generated scripts

- [ ] **T3.3**: Common preprocessing recipes
  - CSV encoding normalization (UTF-8 conversion)
  - Missing value imputation strategies
  - Outlier detection and handling
  - Duplicate row removal

**Success Criteria**:
- 80% of datasets trainable without manual preprocessing
- FilePrepper operations exposed as customizable scripts
- Performance: 20x faster than pandas (maintained)

---

## Phase 4: Production Deployment (P2 MEDIUM)
**Timeline**: Weeks 9-10 (Feb 2025)
**Goal**: Minimize operational cost for deployment

### Tasks
- [ ] **T4.1**: `mloop serve` enhancements
  - Auto-generate API documentation from schema
  - Health check endpoint for monitoring
  - Request/response logging

- [ ] **T4.2**: Containerization support
  - Generate Dockerfile for model serving
  - Docker Compose for multi-model serving
  - Kubernetes deployment manifests (optional)

- [ ] **T4.3**: Model monitoring
  - Prediction drift detection
  - Data drift monitoring
  - Performance degradation alerts

- [ ] **T4.4**: A/B testing support
  - Multi-model serving with traffic splitting
  - Automatic metric comparison
  - Promotion based on live performance

**Success Criteria**:
- One command deploys model to production
- Monitoring and alerting work out-of-box
- A/B testing enables data-driven decisions
- Zero DevOps expertise required

---

## Phase 5: Enhanced Documentation & Examples (P1 HIGH)
**Timeline**: Ongoing (parallel with all phases)
**Goal**: Minimize learning cost through excellent documentation

### Tasks
- [ ] **T5.1**: End-to-end tutorials
  - [ ] Binary classification: Sentiment analysis
  - [ ] Regression: Housing price prediction
  - [ ] Multiclass: Iris classification
  - [ ] Time series: Sales forecasting
  - [ ] With preprocessing: Multi-file join workflow
  - [ ] With AI agent: Complete beginner walkthrough

- [ ] **T5.2**: Recipe library
  - [ ] Common preprocessing patterns
  - [ ] Useful hook examples
  - [ ] Business metric templates
  - [ ] Deployment configurations

- [ ] **T5.3**: Video content
  - [ ] "MLoop in 5 minutes" quickstart
  - [ ] "Zero to Production" complete workflow
  - [ ] "AI Agent Assistant" demonstration

- [ ] **T5.4**: API reference
  - [ ] Complete CLI command reference
  - [ ] Extensibility API documentation
  - [ ] Configuration reference
  - [ ] Troubleshooting guide

**Success Criteria**:
- New users productive in <15 minutes
- All common use cases covered with examples
- Video tutorials reach 10K+ views
- <5% of users require support for basic tasks

---

## Future Considerations (P3 LOW)

### Advanced Features (2025 Q2+)
- [ ] Automated feature selection
- [ ] Multi-model ensemble support
- [ ] Transfer learning integration
- [ ] Real-time prediction streaming
- [ ] Advanced visualization dashboard

### Enterprise Features (2025 Q3+)
- [ ] Team collaboration features
- [ ] Experiment comparison dashboard
- [ ] Model registry with governance
- [ ] Audit logging and compliance
- [ ] SSO and access control

### Platform Expansion (2025 Q4+)
- [ ] Python SDK for MLoop
- [ ] VS Code extension
- [ ] Azure ML integration
- [ ] AWS SageMaker integration
- [ ] Cloud-native deployment

---

## Release Schedule

| Version | Target Date | Focus | Status |
|---------|-------------|-------|--------|
| **v0.1.0** | Nov 2025 | ML.NET 5.0 + Core Features | ✅ Complete |
| **v0.2.0** | Dec 2025 | Preprocessing + Extensibility | 🚧 In Progress |
| **v0.3.0** | Jan 2025 | AI Agent Enhancement + FilePrepper | 📋 Planned |
| **v0.4.0** | Feb 2025 | Production Deployment | 📋 Planned |
| **v1.0.0** | Mar 2025 | Production-Ready Release | 🎯 Target |

---

## Success Metrics

**Core Mission: Excellent MLOps with Minimum Cost**

### Development Cost (Target: 80% reduction)
- **Current Baseline**: Traditional ML project = 2-4 weeks
- **MLoop Target**: 1-2 hours from CSV to production
- **Measurement**: Time from project init to deployed model

### Knowledge Cost (Target: Zero ML expertise required)
- **Current Baseline**: Requires ML degree or 6+ months training
- **MLoop Target**: Basic CSV understanding only
- **Measurement**: % of users without ML background who succeed

### Operational Cost (Target: 90% reduction)
- **Current Baseline**: Docker, Kubernetes, MLOps platform, monitoring
- **MLoop Target**: .NET CLI + filesystem only
- **Measurement**: Infrastructure components required

### Value Delivery (Target: Production-quality models)
- **Current Baseline**: Weeks to production-ready model
- **MLoop Target**: Hours to production-ready model
- **Measurement**: Model accuracy vs manual tuning, deployment success rate

---

## Contributing

This roadmap reflects our mission and community priorities. To propose changes:

1. **Philosophy Alignment**: Does the feature reduce cost (development/knowledge/operational)?
2. **User Impact**: How many users benefit? How significantly?
3. **Complexity**: Does it maintain simplicity or require trade-offs?
4. **Priority**: P0 (blocks core workflow) → P1 (major cost reduction) → P2 (enhancement) → P3 (nice-to-have)

Submit proposals via GitHub Issues with `roadmap` label.

---

**Last Updated**: January 7, 2026
**Version**: 0.2.0-draft
**Next Review**: January 31, 2026

**Recent Changes**:
- Phase 0 (Data Preparation Excellence) ✅ COMPLETE: T0.1-T0.8 all tasks finished
  - T0.1-T0.3: Core preprocessing infrastructure (IPreprocessingScript, ScriptLoader, PreprocessingEngine)
  - T0.4: 36 unit tests (PreprocessingEngine, ScriptLoader, CsvHelper) - all passing
  - T0.5-T0.6: CLI integration (mloop preprocess, auto-preprocessing in mloop train)
  - T0.7: Example scripts (datetime, unpivot, feature engineering)
  - T0.8: Documentation (GUIDE.md fix, comprehensive README.md)
- Phase 1 Week 4 (Custom Metrics) ✅ COMPLETE: T1.5-T1.8 all tasks finished
  - T1.5: IMLoopMetric interface design (MetricContext, MetricResult)
  - T1.6: MetricEngine implementation with script discovery
  - T1.7: Example metrics (profit maximization, churn prevention, ROI)
  - T1.8: Comprehensive documentation (350+ lines README.md)
- Phase 2 Week 6 (Learning & Explanation) ✅ COMPLETE: T2.4-T2.5 finished
  - T2.4: experiment-explainer agent (algorithm explanation, metric interpretation)
  - T2.5: ml-tutor agent (interactive ML tutorials, beginner-friendly learning)
- MLOOP-006: Enhanced AI agents with intelligent analysis, configuration, feature engineering, and memory (T2.1, T2.2, T2.3, T2.6)
- Added 15 LLM integration tests for agent validation
- Updated agent architecture to Ironbees v0.4.1 with YAML-based templates
