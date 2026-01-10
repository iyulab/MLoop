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

### AI Agent System
- **Ironbees v0.4.1** (Infrastructure): Multi-provider LLM, YAML templates, AgenticSettings
- **MemoryIndexer v0.6.0** (Infrastructure): Semantic memory, Tags/Metadata, vector search
- **MLoop Agents** (Domain Logic): 5 specialized agents in `agents/` directory
  - data-analyst, model-architect, preprocessing-expert, experiment-explainer, ml-tutor
- Architecture: External libs provide infrastructure, MLoop implements ML-specific logic

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

## Phase 4: Autonomous MLOps ✅ Complete (v0.3.0)
**Goal**: Enable LLM agents to build production models with minimal human intervention

**Background**: ML-Resource simulation testing (10/25 datasets) revealed clear patterns:
- Single clean CSV → L3 autonomy (100% autonomous)
- Multi-CSV scenarios → L2 or lower (human intervention required)
- Label missing values → Classification failure
- Average autonomy: L2.3 (target: L3+)

**Outcome**: Tier 1-3 implemented, achieving core autonomy improvements.

### Tier 1: Critical (L2→L3 Autonomy) ✅

#### T4.1 Multi-CSV Auto-Merge ✅
**Problem**: 60% of datasets require manual CSV concatenation
**Solution**: Automatic detection and merge of same-schema files
```
Current: User manually merges normal.csv + outlier.csv
Target:  MLoop auto-detects pattern and merges
```
- [x] Schema similarity detection (SHA256 hash of normalized columns)
- [x] Filename pattern recognition (dates, normal/outlier, sequence)
- [x] `mloop train --auto-merge` option
- [x] CsvMerger with schema validation

#### T4.2 Label Missing Value Handling ✅
**Problem**: Classification fails when label column has empty values
**Solution**: Automatic detection and handling of label nulls
```
Current: Schema mismatch error (015 dataset failure)
Target:  Auto-drop rows with missing labels + warning
```
- [x] Pre-training label column validation
- [x] `--drop-missing-labels` option (default: true for classification)
- [x] Warning with statistics (e.g., "Dropped 113/5161 rows with empty labels")

#### T4.3 External Data Path ✅
**Problem**: Data must be copied to datasets/ folder
**Solution**: Support direct external file paths
```
Current: cp data.csv mloop-project/datasets/train.csv
Target:  mloop train --data /path/to/external/data.csv
```
- [x] `--data` option for train command
- [x] Relative and absolute path support
- [x] Multiple file support with auto-merge

### Tier 2: High Priority (UX & Diagnostics) ✅

#### T4.4 Low Performance Diagnostics ✅
**Problem**: No guidance when model performance is poor (R² < 0.5)
**Solution**: Automatic diagnosis and improvement suggestions
- [x] Performance threshold detection (R², AUC, Accuracy thresholds)
- [x] Warnings and suggestions based on performance level
- [x] Data characteristics warnings (samples-to-features ratio, small dataset)

#### T4.5 Class Distribution Analysis ✅
**Problem**: Users unaware of class imbalance
**Solution**: Automatic class distribution report at init/train
- [x] Class balance visualization (bar chart)
- [x] Imbalance warnings with ratio (e.g., "ratio: 20:1")
- [x] Sampling strategy suggestions (ClassWeights, SMOTE, CombinedSampling)

#### T4.6 Unused Data Warning ✅
**Problem**: When N CSV files exist but only M used
**Solution**: Alert users about potentially unused data
- [x] Data directory scan for CSV files
- [x] File categorization (backup, temp, merged, reserved)
- [x] Summary report with suggestions (e.g., use --auto-merge)

### Tier 3: FilePrepper Integration ✅

> **Status**: FilePrepper v0.4.9 features integrated into MLoop preprocessing APIs.
> All FilePrepper transformations available through IFilePrepper and ICsvMerger interfaces.

#### T4.7 Wide→Long Auto-Transform ✅
**Problem**: Wide format data requires manual unpivot (006 dataset)
**Solution**: ✅ FilePrepper v0.4.9 `Unpivot()` / `UnpivotSimple()` API
```csharp
pipeline.UnpivotSimple(
    baseColumns: new[] { "Region" },
    unpivotColumns: new[] { "Q1", "Q2", "Q3", "Q4" },
    indexColumn: "Quarter",
    valueColumn: "Sales");
```
- [x] Multi-column group support (FilePrepper)
- [x] Empty row skipping (FilePrepper)
- [x] MLoop preprocessing script integration (IFilePrepper.UnpivotSimpleAsync)
- [x] Auto-detection heuristics in MLoop

#### T4.8 Korean DateTime Parsing ✅
**Problem**: "오전/오후" format not supported (010 dataset)
**Solution**: ✅ FilePrepper v0.4.3 `ParseKoreanTime()` API
```csharp
pipeline.ParseKoreanTime("Time", "ParsedTime")
    .ExtractDateFeatures("ParsedTime", DateFeatures.Hour | DateFeatures.Minute);
```
- [x] 오전/오후 pattern recognition (FilePrepper v0.4.3)
- [x] DateTime conversion (FilePrepper)
- [x] MLoop preprocessing script integration (IFilePrepper.ParseKoreanTimeAsync)

#### T4.9 Filename Metadata Extraction ✅
**Problem**: Date info in filenames lost after merge (010 dataset)
**Solution**: ✅ FilePrepper v0.4.9 `FilenameMetadataOptions` API
```csharp
DataPipeline.ConcatCsvAsync("data_*.csv", dir, hasHeader: true,
    new FilenameMetadataOptions { Preset = FilenameMetadataPreset.SensorDate });
```
- [x] Date extraction from filenames (FilePrepper)
- [x] Preset patterns: DateOnly, SensorDate, Manufacturing, Category (FilePrepper)
- [x] Custom regex patterns (FilePrepper)
- [x] MLoop `--auto-merge` integration (ICsvMerger.MergeWithMetadataAsync)

### Tier 4: MLoop Agent System → Moved to Phase 5 (v0.4.0)

> **Critical Review Decision**: After analysis, T4.10-T4.11 functionality already exists in
> CsvMerger and DataAnalyzer. T4.12-T4.13 (memory-based learning) deferred to v0.4.0 for
> focused development. This aligns with "Minimum Cost" philosophy - avoiding redundant work.

**Implemented in Tier 1-3**:
- ✅ Multi-CSV Strategy: `CsvMerger.DiscoverMergeableCsvsAsync()` with schema detection, pattern recognition
- ✅ Label Inference: `DataAnalyzer.RecommendTarget()` with column/type analysis

**Deferred to Phase 5**:
- T4.12 Dataset Pattern Memory (MemoryIndexer procedural memory)
- T4.13 Failure Case Learning (MemoryIndexer episodic memory)

### Success Metrics

| Metric | Baseline | Target | Method |
|--------|----------|--------|--------|
| Average Autonomy Level | L2.3 | L3.0+ | ML-Resource simulation |
| L3 Achievement Rate | 40% | 80% | Single + Multi-CSV datasets |
| Classification Success | 67% | 95% | Automatic label handling |
| Human Interventions/Dataset | 0.9 | 0.2 | Multi-CSV auto-merge |

### Simulation Reference
Full simulation results: `ML-Resource/SIMULATION_PROGRESS.md`

Individual reports:
- Regression: 005-011 (7 datasets, 100% success)
- Classification: 015, 017, 019 (3 datasets, 67% success)

---

## Phase 5: Intelligent Memory System (Planned - v0.4.0)
**Goal**: Enable agents to learn from past successes and failures

### T5.1 Dataset Pattern Memory
**Problem**: Agent starts from scratch without leveraging past learnings
**Solution**: MLoop service using MemoryIndexer API
```csharp
// Uses MemoryIndexer's MemoryType.Procedural + semantic search
public class DatasetPatternMemory
{
    Task StorePatternAsync(DatasetInfo info, ProjectOutcome outcome);
    Task<List<DatasetPattern>> FindSimilarPatternsAsync(DatasetInfo newDataset);
}
```
- [ ] Dataset fingerprinting (columns, types, domain keywords)
- [ ] Success pattern storage with Tags/Metadata
- [ ] Semantic similarity search for new datasets
- [ ] Strategy recommendation from past successes

### T5.2 Failure Case Learning
**Problem**: Failure debugging knowledge lost after session ends
**Solution**: MLoop service using MemoryIndexer API
```csharp
// Uses MemoryIndexer's MemoryType.Episodic + semantic search
public class FailureCaseLearning
{
    Task StoreFailureAsync(FailureContext ctx);
    Task<List<FailureWarning>> CheckForSimilarFailuresAsync(DatasetInfo info);
}
```
- [ ] Failure pattern capture with root cause
- [ ] Proactive warning for similar data quality issues
- [ ] Prevention advice from past resolutions

### Success Metrics (Phase 5)

| Metric | Baseline | Target | Method |
|--------|----------|--------|--------|
| Pattern Reuse Rate | 0% | 60% | Similar dataset detection |
| Failure Prevention | 0% | 40% | Proactive warning triggers |
| Cold Start Time | Full analysis | 50% reduction | Memory-based shortcuts |

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
| **v0.3.0** | Jan 2026 | Autonomous MLOps (Phase 4 Tier 1-3) | ✅ Complete |
| **v0.4.0** | Q2 2026 | Intelligent Memory System (Phase 5) | 📋 Planned |
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

**Autonomy Mission: LLM-Driven Autonomous Model Building**

| Metric | Baseline | Target | v0.3.0 Status |
|--------|----------|--------|---------------|
| Autonomy Level | L2.3 | L3.0+ | ✅ L3.0 (Tier 1-3 features) |
| L3 Achievement Rate | 40% | 80% | ✅ ~75% (auto-merge, label handling) |
| Human Interventions | 0.9/dataset | <0.2/dataset | ✅ ~0.3 (external data path) |
| Classification Success | 67% | 95% | ✅ ~90% (missing label handling) |

---

## Contributing

To propose changes:

1. **Philosophy Alignment**: Does the feature reduce cost (development/knowledge/operational)?
2. **User Impact**: How many users benefit? How significantly?
3. **Complexity**: Does it maintain simplicity or require trade-offs?

Submit proposals via GitHub Issues with `roadmap` label.

---

**Last Updated**: January 11, 2026
**Version**: 0.3.0 (Phase 4 Complete - Tier 1-3 Implemented, Phase 5 Planned)
