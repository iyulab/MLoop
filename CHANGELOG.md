# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.11.0] - 2026-03-20

### Added
- **Auto-sampling for large datasets**: `mloop train --max-rows N` with task-aware strategy (stratified for classification, random for others)
- `--sampling-strategy` and `--seed` options for explicit sampling control
- `sample` step type in `mloop prep` pipeline (DataPipelineExecutor)
- Shared PredictionService engine (MLoop.Core.Prediction) used by both CLI and API
- API `/predict` endpoint with real prediction (replacing placeholder)
- SSA forecasting predict via model Transform (T-29)

### Fixed
- BUG-25b: EvaluateCommand Boolean/String label mismatch for multiclass
- BUG-30: PredictCommand fails for unsupervised tasks with empty label
- Spectre.Console markup escaping in ServeCommand output

### Changed
- FilePrepper dependency upgraded 0.5.0 → 0.6.0 (StripCarriageReturn support)
- ConfigureAwait(false) applied to all library async code
- CLI test coverage expanded (17/25 commands covered)

### Validated
- 89 KAMP datasets tested (Sprint-13 + Sprint-14), 0 new bugs
- All 15 ML.NET task types implemented and verified
- 1,595 tests passing, 0 warnings, 0 errors

## [0.10.1] - 2026-03-15

### Fixed
- BUG-25: VectorDataViewType feature detection
- BUG-26: GetRowCount null handling with CountRows helper
- BUG-27: Unsupervised label skip in data validation
- InitCommand updated to allow all 15 task types
- DataQualityValidator updated for unsupervised learning

## [0.10.0] - 2026-03-12

### Added
- ML.NET full 15 task types: Anomaly Detection, Clustering, Ranking, Forecasting (SSA), Time-Series Anomaly, Recommendation, Image Classification, Object Detection, Text Classification, Sentence Similarity, NER, Question Answering
- `mloop runtime` command for on-demand DL runtime management (TorchSharp, TensorFlow)

## [0.6.1-alpha] - 2026-03-10

### Added
- Composite text-likeness heuristic (`LooksLikeText`) for column type detection: average token count, string length, and high-cardinality patterns prevent log messages from being misclassified as categorical
- Text-likeness reclassification in `mloop info` output for consistent info/train behavior
- MCP `mloop_predict`: `unknownStrategy` parameter (auto/error/use-most-frequent/use-missing)
- MCP `mloop_train`: `testSplit`, `noPromote`, `dropMissingLabels` parameters
- 7 unit tests for `LooksLikeText` covering log messages, short codes, numeric IDs, etc.

### Fixed
- Text columns with repeated values (low unique ratio) incorrectly classified as Categorical, causing OneHotEncoding instead of FeaturizeText and prediction failures on unseen text
- Training Configuration display showing "Test Split: 20%" when `--balance` pre-split already isolates test data (now shows "Pre-split (filename)")
- Hidden columns in prediction output selection
- Label inference heuristic and unused data scanner scope
- mloop-mcp submodule: DEP0190 warning, evaluate command, init fixes

### Changed
- Column type detection upgraded from simple 50% unique-ratio threshold to composite heuristic
- MCP-CLI parameter parity improved for predict and train tools

## [0.6.0-alpha] - 2026-02-27

### Added
- `mloop prep run` command for standalone YAML pipeline execution
- Prediction preview (first 5 rows) and distribution display in `mloop predict`
- Data summary and production model comparison in `mloop train`
- Trainer name, metric name, and relative timestamps in `mloop list`
- YAML prep pipeline integration in `mloop predict` (automatic preprocessing)
- Prep step validation and CSV label column check in `mloop validate`
- Centralized error suggestions across all CLI commands (ErrorSuggestions)
- 5 new DataPipelineExecutor operations: extract-date, parse-datetime, normalize, scale, fill-missing
- Pipeline tests for all 14 preprocessing step types
- `mloop status` command with config summary and latest prediction display
- Backup and promotion history in `mloop promote`
- FilePromotionManager for filesystem-based model promotion
- Phase 1 hook/metric extensibility in ScriptDiscovery and AutoMLRunner

### Fixed
- ServeCommand TFM discovery for .NET 10
- DataQualityAnalyzer bug with empty datasets
- PerformanceTests scalability test reliability
- TrainCommand nullable warnings
- Datetime columns auto-excluded from training features
- File I/O optimization in PredictCommand and InfoCommand
- HITL workflow session IDs now shared across decisions in a single workflow execution
- Version consistency: removed hard-coded version from MLoop.Extensibility

### Changed
- FilePrepper integrated via submodule with project reference
- Improved error messages with actionable suggestions and context
- Dead code cleanup and standardized Phase 1 TODO comments
- Assembly version used for accurate SemVer display

## [0.5.1-alpha] - 2025-12-20

### Added
- Self-update system (`mloop update`)
- YAML-based preprocessing pipeline configuration

### Fixed
- ARM64 build targets removed (unsupported by ML.NET)
- NuGet CLI tool publishing disabled in favor of GitHub Releases
- RuntimeIdentifiers conflict with dotnet pack

## [0.5.0-alpha] - 2025-12-15

### Added
- FilePrepper CLI integration for data preprocessing
- Class imbalance detection and `--balance` option
- Custom script extensibility (hooks, metrics, preprocessing)
- Roslyn-based script compilation with DLL caching

### Fixed
- Prediction CSV output formatting
- Cache directory creation for compiled DLLs

## [0.4.0] - 2025-11-01

### Added
- REST API server (`mloop serve`)
- Docker file generation (`mloop docker`)
- Feedback collection system (`mloop feedback`)
- Data sampling utilities (`mloop sample`)
- Retraining trigger evaluation (`mloop trigger check`)

## [0.3.0] - 2025-10-01

### Added
- Multi-model support with namespaced directories
- Experiment promotion workflow (`mloop promote`)
- Dataset profiling (`mloop info`)
- Prediction logging (JSONL)

## [0.1.0] - 2025-09-01

### Added
- Initial release
- ML.NET AutoML training (`mloop train`)
- Prediction with production model (`mloop predict`)
- Experiment management (`mloop list`)
- YAML configuration (`mloop.yaml`)
- Filesystem-based state management

[Unreleased]: https://github.com/iyulab/MLoop/compare/v0.6.1-alpha...HEAD
[0.6.1-alpha]: https://github.com/iyulab/MLoop/compare/v0.6.0-alpha...v0.6.1-alpha
[0.6.0-alpha]: https://github.com/iyulab/MLoop/compare/v0.5.1-alpha...v0.6.0-alpha
[0.5.1-alpha]: https://github.com/iyulab/MLoop/compare/v0.5.0-alpha...v0.5.1-alpha
[0.5.0-alpha]: https://github.com/iyulab/MLoop/compare/v0.4.0...v0.5.0-alpha
[0.4.0]: https://github.com/iyulab/MLoop/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/iyulab/MLoop/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/iyulab/MLoop/releases/tag/v0.1.0
