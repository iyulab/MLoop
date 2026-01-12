# MLoop Roadmap

**Mission**: Excellent MLOps with Minimum Cost

This roadmap aligns all development with MLoop's core philosophy: enabling production-quality ML models with minimal coding, minimal ML expertise, and minimal operational complexity.

---

## Current Status (v1.8.0 - January 2026)

### Core Platform
- ML.NET 5.0 with AutoML 0.23.0
- Filesystem-based MLOps with git-friendly experiment tracking
- Multi-process concurrent training support
- Production model promotion and discovery
- Batch prediction with auto-discovery
- Prediction logging and feedback collection
- Data sampling for retraining datasets
- Feedback-based retraining triggers with CLI evaluation
- CLI with comprehensive command set (including `mloop trigger check`)
- SDK packages ready for NuGet distribution
- .NET 10.0 + C# 13 modern codebase
- Zero AI dependencies (pure ML CLI tool)

### AI Integration (External)
> **v1.2.0 Architecture Change**: "AI uses MLoop" instead of "MLoop contains AI"

- **mloop-mcp**: MCP server exposing MLoop CLI to AI clients
  - Repository: https://github.com/iyulab/mloop-mcp
  - Tools: train, predict, list, promote, info, serve
- **mloop-studio**: Web platform for visual ML workflows (planned)
  - Repository: https://github.com/iyulab/mloop-studio

### Project Structure (6 projects)

**SDK (`src/`) - NuGet Package Distribution**
- **MLoop.Core**: Pure ML engine (AutoML, preprocessing, encoding detection)
- **MLoop.DataStore**: Prediction logging, feedback collection, data sampling (JSONL)
- **MLoop.Extensibility**: Hooks, scripts, metrics interfaces
- **MLoop.Ops**: Model comparison, feedback-based retraining triggers

**Tools (`tools/`) - Executable Distribution**
- **MLoop.CLI**: Command-line interface (dotnet tool)
- **MLoop.API**: REST API server (Docker/standalone)

**Usage Pattern:**
- **.NET 앱 (MLoop Studio)**: SDK 직접 참조 (`dotnet add package MLoop.Core`)
- **외부 시스템 (mloop-mcp)**: CLI subprocess 호출
- **비-.NET 시스템**: API HTTP 호출

### Quality
- 429+ tests passing (Core + API + CLI + DataStore + Ops)

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

## Phase 5: Intelligent Memory System ⚠️ DEPRECATED (v1.2.0)
**Goal**: Enable agents to learn from past successes and failures

> ⚠️ **Deprecated in v1.2.0**: This phase was removed as part of the "Zero AI Dependency" refactoring.
> Memory services (DatasetPatternMemoryService, FailureCaseLearningService) were part of MLoop.AIAgent
> which has been completely removed. AI-based pattern learning is now the responsibility of external
> AI clients using mloop-mcp.
>
> **Historical Reference**: The implementation below is preserved for documentation purposes only.

### T5.1 Dataset Pattern Memory ✅
**Problem**: Agent starts from scratch without leveraging past learnings
**Solution**: MLoop service using MemoryIndexer API with Procedural memory
```csharp
// Uses MemoryIndexer's MemoryType.Procedural + semantic search
public class DatasetPatternMemoryService
{
    Task StorePatternAsync(DatasetFingerprint fingerprint, ProcessingOutcome outcome);
    Task<List<ProcessingRecommendation>> FindSimilarPatternsAsync(DatasetFingerprint fingerprint);
    Task<ProcessingRecommendation?> GetRecommendationAsync(DatasetFingerprint fingerprint);
}
```
- [x] Dataset fingerprinting (columns, types, ratios, hash)
- [x] Success pattern storage with MemoryUnit metadata
- [x] Semantic similarity search using embeddings
- [x] Strategy recommendation from past successes
- [x] DI registration via AddIntelligentMemoryServices()
- [x] Comprehensive unit tests (DatasetPatternMemoryServiceTests)

### T5.2 Failure Case Learning ✅
**Problem**: Failure debugging knowledge lost after session ends
**Solution**: MLoop service using MemoryIndexer API with Episodic memory
```csharp
// Uses MemoryIndexer's MemoryType.Episodic + semantic search
public class FailureCaseLearningService
{
    Task StoreFailureAsync(FailureContext context, Resolution resolution);
    Task<List<FailureWarning>> CheckForSimilarFailuresAsync(DatasetInfo info);
    Task<ResolutionSuggestion?> FindResolutionAsync(string errorType, string errorMessage);
}
```
- [x] Failure pattern capture with context and resolution
- [x] Proactive warning for similar data quality issues (WarningLevel: High/Medium/Low)
- [x] Prevention advice from past resolutions
- [x] Importance scoring (verified resolutions, root cause, prevention advice)
- [x] DI registration via AddIntelligentMemoryServices()
- [x] Comprehensive unit tests (FailureCaseLearningServiceTests)

### Technical Implementation

**Architecture Decision**: Direct IMemoryStore usage (simplified from original IDatasetPatternMemory/IFailureCaseLearning abstraction)
- Reduces abstraction layers while maintaining full functionality
- Aligns with "Minimum Cost" philosophy
- Uses MemoryIndexer SDK v0.6.0 with 3-axis memory model (Type × Scope × Tier)

**Key Components**:
- `DatasetFingerprint`: Column names, types, ratios, hash
- `ProcessingOutcome`: Steps, metrics, trainer, success status
- `FailureContext`: Error type, message, phase, dataset context
- `Resolution`: Root cause, fix description, prevention advice, verified flag

### Success Metrics (Phase 5)

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Core Services | 0% | 100% | ✅ T5.1, T5.2 implemented |
| Unit Test Coverage | 0% | 100% | ✅ 42 tests passing |
| DI Integration | No | Yes | ✅ AddIntelligentMemoryServices() |

### Future Enhancements (Post v0.4.0)
- Agent integration (DataAnalyzer using pattern memory) → Phase 6 T6.1
- CLI commands for memory visibility
- Pattern reuse rate and failure prevention metrics tracking

---

## Phase 6: Agent Intelligence & Data Quality (v0.5.0)
**Goal**: Polish and stabilize through intelligent data handling and agent memory integration

> ⚠️ **Partial Deprecation in v1.2.0**: T6.1 (Agent Memory Integration) was removed with MLoop.AIAgent.
> T6.2 (Encoding Auto-Detection) and T6.3 (Dataset Compatibility Check) are preserved in MLoop.Core.

**Background**: ML-Resource simulation analysis (10/25 datasets) revealed:
- Remaining datasets have fundamental incompatibilities (no labels, wrong format)
- Encoding issues block 018-열처리 예지보전 (CP949/EUC-KR)
- Better guidance needed when data doesn't fit supervised learning paradigm

**Philosophy Alignment**: Focus on POLISH, not new features
- Add low-cost, high-value improvements (encoding detection)
- Improve error messages instead of complex workarounds

### T6.1 Agent Memory Integration ⚠️ DEPRECATED
**Status**: Removed in v1.2.0 with MLoop.AIAgent

> This feature was removed as part of the "Zero AI Dependency" refactoring.
> IntelligentDataAnalyzer and memory integration are no longer part of MLoop.

### T6.2 Encoding Auto-Detection ✅
**Problem**: CP949/EUC-KR encoded files cause garbled text (018 dataset)
**Solution**: Automatic charset detection and UTF-8 conversion
```
Current: 㿀₩Ý¢Àà§°ü¬÷Àú¥ó (garbled)
Target:  설비명,설비번호,공정명 (correct Korean)
```
- [x] EncodingDetector with BOM, UTF-8, CP949 detection
- [x] Auto-conversion to UTF-8 with BOM in CsvDataLoader
- [x] ML.NET InferColumns compatibility ensured
- [x] Comprehensive test coverage for Korean text

### T6.3 Dataset Compatibility Check ✅
**Problem**: Unclear error when data lacks required structure
**Solution**: Pre-training compatibility validation with clear guidance
```
Warning: Dataset may not be compatible with supervised learning.
- Label column 'Price' has 2.3% missing values
- Suggestion: Use --drop-missing-labels flag to handle missing label values.
```
- [x] DatasetCompatibilityChecker with severity levels (Critical/Warning/Info)
- [x] Label column validation (exists, missing values, task compatibility)
- [x] Clear error messages with actionable suggestions
- [x] Integrated into IntelligentAnalysisResult.IsMLReady

### Success Metrics (Phase 6)

| Metric | Baseline | Target | Method |
|--------|----------|--------|--------|
| Agent Memory Usage | 0% | 100% | DataAnalyzer integration |
| Encoding Issues | Manual fix | Auto-detect | UTF-8 conversion |
| Error Message Quality | Generic | Actionable | Compatibility checks |

---

## Phase 7: Production Readiness (v1.0.0)
**Goal**: Finalize integration and achieve production-ready release

**Background**: Phase 6 completed intelligent analysis infrastructure. Phase 7 focuses on:
- Connecting IntelligentDataAnalyzer to CLI workflow
- Validating all features work correctly in real scenarios
- Achieving stable, well-tested v1.0.0 release

**Philosophy Alignment**: "Minimum Cost" = Polish existing features, not add new ones

### T7.1 CLI Integration → Deferred to v1.1.0
**Problem**: IntelligentDataAnalyzer exists but is not connected to mloop train
**Critical Review Decision**: DEFER

**Rationale**:
- TrainCommand already has 6 comprehensive analysis components:
  - DataQualityAnalyzer, ClassDistributionAnalyzer, LabelValueHandler
  - PerformanceDiagnostics, UnusedDataScanner, PreprocessingEngine
- Memory services (Pattern Memory, Failure Learning) are empty initially
- Adding would increase UI complexity without immediate user benefit
- "Minimum Cost" philosophy: avoid redundant UI

**Future Implementation (v1.1.0+)**:
- [ ] Optional `--insights` flag to enable memory-based recommendations
- [ ] Background pattern learning during training
- [ ] Failure case capture on training errors

### T7.2 Simulation Validation ✅
**Problem**: ML-Resource simulation shows 40% completion but features exist
**Solution**: Re-validate with correct feature usage knowledge
```
Current: Simulation thinks Multi-CSV is unsupported
Target:  Simulation correctly uses --data file1.csv file2.csv
```
- [x] Update simulation guidance with existing CLI options
- [x] Document --data, --auto-merge, --drop-missing-labels usage
- [x] Re-run problematic datasets with correct approach (018 tested successfully)
- [x] Update SIMULATION_PROGRESS.md with actual results
- [x] **Bugfix**: Add EncodingDetector to InfoCommand for Korean text support

### T7.3 Test Coverage Completion → Minimal Scope for v1.0.0
**Problem**: IntelligentDataAnalyzer and DatasetCompatibilityChecker lack integration tests
**Critical Review Decision**: MINIMAL SCOPE

**Current Coverage (589 tests passing)**:
- MLoop.Core.Tests: 308 tests (EncodingDetector, DataAnalyzer, etc.)
- MLoop.AIAgent.Tests: 222 tests (Memory services, Analyzers)
- MLoop.Tests: 50 tests (CLI infrastructure)
- MLoop.API.Tests: 9 tests (REST endpoints)

**v1.0.0 Scope**: Existing coverage is sufficient
- [x] EncodingDetector unit tests (7 tests covering BOM, UTF-8, CP949)
- [x] DatasetCompatibilityChecker tests via IntelligentDataAnalyzer
- [x] InfoCommand encoding bug fix verified with real dataset (018)

**Future Testing (v1.1.0+)**:
- [ ] E2E tests for full training workflow with various encodings
- [ ] Integration tests for memory services with actual data

### Success Metrics (Phase 7)

| Metric | Baseline | Target | Actual | Status |
|--------|----------|--------|--------|--------|
| Memory Integration | Code only | Infrastructure ready | ✅ Services built, CLI deferred | v1.1.0 |
| Simulation Accuracy | 40% | 80%+ | ✅ CLI docs + 018 bug fix | Complete |
| Test Coverage | 580 tests | 589 tests | ✅ 589 tests passing | Complete |

### v1.0.0 Release Criteria

| Criteria | Status |
|----------|--------|
| Phase 6 Complete (IntelligentDataAnalyzer, EncodingDetector) | ✅ |
| T7.2 Simulation Validation | ✅ |
| All CI tests passing (589 tests) | ✅ |
| InfoCommand encoding bug fixed | ✅ |
| Critical review of T7.1/T7.3 complete | ✅ |
| Feature branch merged to main | ✅ |
| v1.0.0 tag created | ✅ |

---

## Phase 8: Polish & Documentation ✅ Complete (v1.1.0)
**Goal**: Maximize user success with minimal new complexity

**Background**: v1.0.0 provides solid ML training foundation. Phase 8 focuses on:
- Documentation excellence to reduce knowledge cost
- Background infrastructure for future insights features
- Error message improvements for better debugging
- Example projects demonstrating real-world usage

**Philosophy Alignment**: "Minimum Cost" = Polish existing features, not add new ones

**Critical Review Decision**: T7.1 (CLI Insights) deferred to v1.2.0 because memory services are empty initially and provide no immediate value. Extended E2E tests deferred indefinitely due to maintenance cost.

### T8.1 User Documentation ✅
**Problem**: Users lack comprehensive guides for MLoop workflows
**Solution**: Create detailed user-facing documentation
```
Target:
- USER_GUIDE.md with step-by-step workflows
- Troubleshooting guide for common errors
- Quick-start examples in README.md
- Best practices for data preparation
```
- [x] Update docs/GUIDE.md with v1.0.0 features (train options, docker, encoding)
- [x] Add troubleshooting section for common errors
- [x] Update README.md quick-start examples
- [x] Document all CLI commands with examples (train, info, docker)

### T8.2 Background Memory Infrastructure ⚠️ DEPRECATED
**Status**: Removed in v1.2.0 with MLoop.AIAgent

> This feature was removed as part of the "Zero AI Dependency" refactoring.
> TrainingMemoryCollector and memory services are no longer part of MLoop.
> Pattern learning is now the responsibility of external AI clients.

### T8.3 Error Message Improvement ✅
**Problem**: Error messages are generic, lack actionable guidance
**Solution**: Add context-specific suggestions to all error messages
- [x] Review common error scenarios from simulation testing
- [x] Add actionable suggestions (e.g., "Try using --drop-missing-labels")
- [x] Improve encoding detection failure messages
- [x] Better handling of unsupported data formats

**Implementation**:
- `ErrorSuggestions`: Central helper class for actionable error guidance
- Pattern-based suggestion generation for common issues
- Context-aware suggestions for training and prediction errors
- Integrated into TrainCommand and PredictCommand

### T8.4 Example Projects Update ✅
**Problem**: Example projects don't demonstrate v1.0.0 features
**Solution**: Update and expand examples with real-world scenarios
- [x] Update examples/ with v1.0.0 CLI options
- [x] Verify all examples use multi-model mloop.yaml format
- [x] Ensure all examples work out-of-box with `mloop train`

**Status**: Examples already in good shape with multi-model format. All tutorials (iris-classification, housing-prices, sentiment-analysis) and examples (customer-churn, equipment-anomaly-detection) verified working.

### T8.5 Encoding Detection Consistency ✅
**Problem**: CsvHelperImpl lacked encoding detection, causing garbled Korean column names in `mloop train`
**Discovery**: Agent Simulation testing (IMP-001 finding)
**Solution**: Add EncodingDetector to CsvHelperImpl.ReadAsync() and ReadHeadersAsync()
```
Current: 깨진한글 (garbled due to CP949→UTF8 mismatch)
Target:  정상한글 (correct Korean text)
```
- [x] EncodingDetector.ConvertToUtf8WithBom() in ReadAsync()
- [x] EncodingDetector.ConvertToUtf8WithBom() in ReadHeadersAsync()
- [x] Temp file cleanup after encoding conversion
- [x] Consistent behavior with InfoCommand and CsvDataLoader

**Fix Location**: `src/MLoop.Core/Data/CsvHelper.cs`

### Success Metrics (Phase 8)

| Metric | Baseline | Target | Actual | Status |
|--------|----------|--------|--------|--------|
| Documentation Completeness | 30% | 80% | ✅ 80%+ | Complete |
| Memory Data Collection | None | Background active | ✅ Active | Complete |
| Example Project Coverage | 3 | 6+ | ✅ 5 verified | Complete |
| Error Message Quality | Generic | Actionable | ✅ Contextual | Complete |
| Encoding Consistency | Partial | Full | ✅ All commands | Complete |

---

## Phase 9: CLI Intelligence ❌ CANCELLED
**Original Goal**: Enable memory-based intelligent recommendations for users

> ❌ **Cancelled in v1.2.0**: This phase was cancelled as part of the "Zero AI Dependency" refactoring.
> Memory-based features (T9.1 CLI Insights) depended on MLoop.AIAgent which has been removed.
> T9.2 (Label Column Inference) may be reconsidered for future versions as a Core feature.
>
> AI-based recommendations are now the responsibility of external AI clients using mloop-mcp.

### Future Direction

Instead of embedded AI intelligence, MLoop follows the Unix philosophy:
- **MLoop CLI**: Simple, stateless ML operations (like `grep`, `awk`)
- **mloop-mcp**: MCP server exposing CLI to AI clients
- **AI Clients**: Claude, Cursor, or any MCP-compatible AI provides intelligence

This separation of concerns enables:
- Zero AI dependencies in MLoop
- Any AI provider can use MLoop
- Simpler maintenance and testing

---

## Phase 10: DataStore Implementation ✅ Complete (v1.3.0)
**Goal**: Enable prediction logging and monitoring infrastructure

**Background**: Production ML systems need prediction tracking for:
- Monitoring model performance over time
- Debugging prediction issues
- Compliance and audit requirements
- Building feedback loops for retraining

**Philosophy Alignment**: Filesystem-first, JSONL format for simplicity

### T10.1 FilePredictionLogger ✅
**Problem**: No way to track predictions after deployment
**Solution**: JSONL-based prediction logging to `.mloop/logs/{model}/{date}.jsonl`
- [x] IPredictionLogger interface with async logging
- [x] FilePredictionLogger with thread-safe file operations
- [x] Batch logging support for high-throughput scenarios
- [x] Query API with date range and limit filters

### T10.2 CLI Integration ✅
**Problem**: Users need to view prediction history
**Solution**: Add prediction logging capabilities to CLI
- [x] `--log` option in `mloop predict` command
- [x] `mloop logs` command with filtering options
- [x] JSON output format for scripting
- [x] Table output format for human readability

### Success Metrics (Phase 10)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| FilePredictionLogger | Complete | ✅ | Complete |
| CLI Integration | Complete | ✅ | Complete |
| Unit Tests | 8+ tests | 8 tests | ✅ |

---

## Phase 11: Ops Implementation ✅ Complete (v1.4.0)
**Goal**: Enable model comparison and retraining trigger infrastructure

**Background**: MLOps automation requires:
- Comparing model versions to decide on promotion
- Scheduling retraining based on time or performance
- Tracking model lifecycle and performance drift

**Philosophy Alignment**: Minimal scope, filesystem-based, no external dependencies

### T11.1 FileModelComparer ✅
**Problem**: No automated way to compare experiments
**Solution**: Filesystem-based model comparison service
- [x] IModelComparer interface implementation
- [x] Compare experiments by metrics
- [x] Compare candidate with production model
- [x] Find best experiment based on criteria
- [x] Support for higher-is-better and lower-is-better metrics

### T11.2 TimeBasedTrigger ✅
**Problem**: No automated retraining scheduling
**Solution**: Simple time-based retraining trigger
- [x] IRetrainingTrigger interface (partial implementation)
- [x] TimeBased condition evaluation
- [x] Days since last training calculation
- [x] Clear messaging for unsupported conditions

### T11.3 Existing CompareCommand ✅
**Problem**: CLI comparison functionality
**Solution**: Already implemented in Phase 4
- [x] `mloop compare` command with metric highlighting
- [x] Production model indication
- [x] Best value highlighting in table output

### Success Metrics (Phase 11)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| FileModelComparer | Complete | ✅ | Complete |
| TimeBasedTrigger | Complete | ✅ | Complete |
| Unit Tests | 16+ tests | 16 tests | ✅ |

### Deferred to v2.0.0
- **IPromotionManager**: Overlaps with existing `mloop promote` command
- **Full IRetrainingTrigger**: Requires IFeedbackCollector for AccuracyDrop, DataDrift, etc.

---

## Phase 12: Feedback Collection ✅ Complete (v1.5.0)
**Goal**: Enable ground truth feedback collection for model monitoring

**Background**: Production ML systems need feedback loops to:
- Track model accuracy over time
- Enable data-driven retraining decisions
- Build training data from production predictions

**Philosophy Alignment**: Filesystem-first, JSONL format, links to prediction logs

### T12.1 FileFeedbackCollector ✅
**Problem**: No way to record ground truth for predictions
**Solution**: JSONL-based feedback storage linked to predictions
- [x] IFeedbackCollector interface implementation
- [x] RecordFeedbackAsync linking to prediction IDs
- [x] GetFeedbackAsync with filtering
- [x] CalculateMetricsAsync for accuracy calculation

### T12.2 CLI Integration ✅
**Problem**: Users need CLI access to feedback functionality
**Solution**: Add `mloop feedback` command with subcommands
- [x] `mloop feedback add --prediction-id xxx --actual-value yyy`
- [x] `mloop feedback list --model xxx`
- [x] `mloop feedback metrics --model xxx`
- [x] JSON and table output formats

### Success Metrics (Phase 12)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| FileFeedbackCollector | Complete | ✅ | Complete |
| CLI Integration | Complete | ✅ | Complete |
| Unit Tests | 8+ tests | 8 tests | ✅ |

---

## Phase 13: Data Sampling & Triggers ✅ Complete (v1.6.0)
**Goal**: Enable retraining data creation and automated trigger evaluation

**Background**: Feedback collection (v1.5.0) enables:
- Joining predictions with ground truth for retraining datasets
- Calculating accuracy metrics for trigger decisions
- Building data pipelines for continuous improvement

**Philosophy Alignment**: Minimal scope, only implement needed strategies

### T13.1 FileDataSampler ✅
**Problem**: Need to create retraining datasets from predictions + feedback
**Solution**: Sampling service with multiple strategies
- [x] IDataSampler interface implementation
- [x] Random, Recent, FeedbackPriority strategies (Stratified, LowConfidence deferred)
- [x] Join predictions with feedback data
- [x] Export to CSV format for ML.NET training
- [x] GetStatisticsAsync for sampling metadata

### T13.2 FeedbackBasedTrigger ✅
**Problem**: Need accuracy-based retraining decisions
**Solution**: Trigger evaluation using FeedbackMetrics
- [x] IRetrainingTrigger implementation
- [x] AccuracyDrop condition (accuracy < threshold)
- [x] FeedbackVolume condition (count >= threshold)
- [x] Clear messaging for unsupported conditions (DataDrift, PerformanceDegradation)

### T13.3 CLI Integration ✅
**Problem**: Users need CLI access to sampling functionality
**Solution**: Add `mloop sample` command
- [x] `mloop sample create --model xxx --size 1000 --strategy random`
- [x] `mloop sample stats --model xxx`
- [x] Strategies: random, recent, feedback-priority
- [x] Auto-generated output path with timestamp

### Success Metrics (Phase 13)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| FileDataSampler | Complete | ✅ | Complete |
| FeedbackBasedTrigger | Complete | ✅ | Complete |
| CLI Integration | Complete | ✅ | Complete |
| Unit Tests | 16+ tests | 16 tests | ✅ |

---

## Phase 14: Trigger CLI Enhancement ✅ Complete (v1.7.0)
**Goal**: Enable CLI-based retraining trigger evaluation for automation scripts

**Background**: FeedbackBasedTrigger (v1.6.0) provides programmatic trigger evaluation.
Phase 14 exposes this functionality via CLI for:
- Shell script automation (`if mloop trigger check; then mloop train; fi`)
- Cron job integration
- CI/CD pipeline triggers

**Philosophy Alignment**: CLI is the primary interface, SDK is the building block

### T14.1 TriggerCommand ✅
**Problem**: No CLI access to trigger evaluation
**Solution**: Add `mloop trigger check` command
- [x] `mloop trigger check --model xxx` with default conditions
- [x] Custom thresholds: `--accuracy 0.7 --feedback 100`
- [x] JSON output format with `--json` flag
- [x] Exit code: 0 = should retrain, 1 = no retraining, 2 = error
- [x] Integrated with FeedbackBasedTrigger from MLoop.Ops

### T14.2 CLI Integration ✅
**Problem**: TriggerCommand needs project reference to MLoop.Ops
**Solution**: Update MLoop.CLI.csproj
- [x] Add ProjectReference to MLoop.Ops
- [x] Register TriggerCommand in Program.cs
- [x] Add trigger to help banner display

### Success Metrics (Phase 14)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| TriggerCommand | Complete | ✅ | Complete |
| CLI Integration | Complete | ✅ | Complete |
| Unit Tests | Existing | 24 tests | ✅ |

---

## Phase 15: NuGet Package Preparation ✅ Complete (v1.8.0)
**Goal**: Prepare SDK packages for NuGet distribution

**Background**: v2.0.0 Studio Integration requires SDK packages on NuGet.
This phase prepares all SDK projects for public distribution.

**Philosophy Alignment**:
- SDK/Tools separation enables clean package distribution
- NuGet packages enable MLoop Studio direct reference

### T15.1 Common Package Settings ✅
**Status**: Already configured in Directory.Build.props
- [x] Authors, Company, Copyright
- [x] MIT License
- [x] GitHub Repository URL
- [x] Source Link integration
- [x] Symbol packages

### T15.2 SDK Package Metadata ✅
**Problem**: SDK projects need individual package metadata
**Solution**: Add NuGet configuration to each SDK project
- [x] MLoop.Core: PackageId, Description, Tags, README
- [x] MLoop.DataStore: PackageId, Description, Tags, README
- [x] MLoop.Extensibility: PackageId, Description, Tags, README (already complete)
- [x] MLoop.Ops: PackageId, Description, Tags, README

### T15.3 SDK README Files ✅
**Problem**: Each package needs documentation
**Solution**: Create README.md for NuGet package display
- [x] MLoop.Core/README.md
- [x] MLoop.DataStore/README.md
- [x] MLoop.Extensibility/README.md (already exists)
- [x] MLoop.Ops/README.md

### Success Metrics (Phase 15)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Package Metadata | 4 SDKs | ✅ | Complete |
| README Files | 4 SDKs | ✅ | Complete |
| Build Verification | Pass | ✅ | Complete |

### Next Steps (v1.9.0+)

**Project Structure Refactoring**:
- Move `src/MLoop.CLI` → `tools/MLoop.CLI`
- Move `src/MLoop.API` → `tools/MLoop.API`
- Update solution file and project references

**Sampling Enhancements**:
- Stratified Sampling: Sample proportionally by class distribution
- LowConfidence Sampling: Prioritize uncertain predictions

**NuGet Publishing**:
- GitHub Actions workflow for automated publishing
- Version management strategy

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
| **v0.4.0** | Jan 2026 | Intelligent Memory System (Phase 5) | ⚠️ Deprecated |
| **v0.5.0** | Jan 2026 | Agent Intelligence & Data Quality (Phase 6) | ⚠️ Partial |
| **v1.0.0** | Jan 2026 | Production Readiness (Phase 7) | ✅ Released |
| **v1.1.0** | Jan 2026 | Polish & Documentation (Phase 8) | ⚠️ Partial |
| **v1.2.0** | Jan 2026 | Zero AI Dependency Refactoring | ✅ Complete |
| **v1.3.0** | Jan 2026 | DataStore Implementation | ✅ Complete |
| **v1.4.0** | Jan 2026 | Ops Implementation | ✅ Complete |
| **v1.5.0** | Jan 2026 | Feedback Collection | ✅ Complete |
| **v1.6.0** | Jan 2026 | Data Sampling & Triggers | ✅ Complete |
| **v1.7.0** | Jan 2026 | Trigger CLI Enhancement | ✅ Complete |
| **v1.8.0** | Jan 2026 | NuGet Package Preparation | ✅ Complete |
| **v2.0.0** | Q2 2026 | Studio Integration | 📋 Planning |

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

**Last Updated**: January 12, 2026
**Version**: v1.8.0 Complete (NuGet Package Preparation)
**Recent Changes**:
- v1.8.0 NuGet Package Preparation complete
  - SDK package metadata: PackageId, Description, Tags for all SDK projects
  - README files for NuGet package display
  - Ready for NuGet.org or GitHub Packages distribution
- v1.7.0 Trigger CLI Enhancement complete
  - TriggerCommand: `mloop trigger check --model xxx` for CLI-based trigger evaluation
  - Custom thresholds: `--accuracy` and `--feedback` options
  - Exit code semantics for shell script automation
  - MLoop.CLI now references MLoop.Ops for trigger functionality
- v1.6.0 Data Sampling & Triggers implementation complete
  - FileDataSampler: Random, Recent, FeedbackPriority sampling strategies
  - FeedbackBasedTrigger: AccuracyDrop and FeedbackVolume conditions
  - `mloop sample create/stats` commands for CLI access
  - 16 unit tests for sampling and trigger services
- v1.5.0 Feedback Collection implementation complete
  - FileFeedbackCollector: JSONL-based feedback storage linked to predictions
  - `mloop feedback add/list/metrics` commands for CLI access
  - Accuracy calculation from predictions vs ground truth
  - 8 unit tests for feedback services
- v1.4.0 MLoop.Ops implementation complete
  - FileModelComparer: Filesystem-based experiment comparison
  - TimeBasedTrigger: Simple time-based retraining trigger
  - 16 unit tests for Ops services
- v1.3.0 MLoop.DataStore implementation complete
  - FilePredictionLogger: JSONL-based prediction logging
  - `mloop logs` command for viewing prediction history
- v1.2.0 "Zero AI Dependency" refactoring complete
- MLoop.AIAgent removed, AI integration via mloop-mcp
