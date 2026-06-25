# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **`mloop analyze <aspect>` — granular, read-only EDA command group**: decomposes `mloop info --analyze`'s one-shot deep analysis into per-aspect subcommands — `profile` (types/null%/cardinality/constant columns), `correlation` (high-correlation pairs + multicollinearity), `importance` (feature ranking, requires a label), `outliers` (count/rate/isolation-forest threshold), and `distribution` (skewness/kurtosis/normality tests). Each aspect computes only its slice via DataLens `AnalysisOptions`, and `--json` emits a structured `{aspect, available, summary, data, flags}` envelope for agent/LLM consumption. Read-only: never mutates the data file or `mloop.yaml`. (Agent-driven data-enhancement roadmap, Phase 1.)

### Changed
- **Upgraded FilePrepper 0.6.0 → 0.7.0 and DataLens 0.4.0 → 0.13.0** (umbrella version-governance alignment). DataLens's `AnalysisOptions`/`AnalysisResult` surface is backward-compatible (additive); FilePrepper 0.7 adds auto encoding detection, skip-rows, and constant-column removal. Security transitive pins re-verified (0 vulnerable packages).

## [0.16.3] - 2026-06-25

### Fixed
- **`prep:` statistical transforms leaked test statistics into training**: a `mloop.yaml` `prep:` step that fits a statistic over the data (`normalize`, `scale`, `fill-mean`) was applied to the *entire* dataset and baked into the CSV before AutoML's train/validation split — so each cross-validation fold trained on parameters (min/max, mean, std) computed from rows it would later be evaluated against. The raw→AutoML path was always safe (ML.NET's built-in featurizers fit fold-internally), but adding `prep:` reintroduced leakage. Statistical transforms are now routed to ML.NET AutoML's `preFeaturizer` estimator, which the AutoML contract fits *inside* each fold, eliminating the leak. New `PrepStepClassifier` (classifies each step as preFeaturizer / csv-bake / unsupported), `PrepFeaturizerBuilder` (statistical step → ML.NET estimator), and `PrepRouter` (routing + shared leakage warning), wired through `TrainingConfig.PreFeaturizer` → `AutoMLRunner` (three injection sites, task-aware via `SupportsPreFeaturizer`) and `TrainCommand.ApplyPrepAsync`. Transforms ML.NET can't express fold-internally (`median`, `rolling`, `resample`) remain CSV-baked but now emit an explicit leakage warning, and `mloop validate` gained `InspectPrepLeakage` to surface residual-leakage steps ahead of training. Two compounding gaps were caught in review and fixed before release: preFeaturizer-produced columns were dropped by `InferColumns` merging (crashing real training until `preserveColumns` retention was added), and statistical prep was silently dropped on non-classification/regression tasks (now task-aware via `AutoMLRunner.SupportsPreFeaturizer`). Found via deep-research + code verification of the agent-driven feature-engineering design (V2 leakage hypothesis confirmed against MLoop source).
- **Global runtime cache mistaken for a project root, breaking relative paths under `$HOME` (BUG-47)**: after `mloop runtime install` creates the global DL-runtime cache at `~/.mloop/runtimes`, the user-profile directory contains a `.mloop` directory. `ProjectDiscovery.FindRoot` identified a project root solely by the presence of a `.mloop` directory, so walking up from *any* directory under `$HOME` matched that cache and treated `$HOME` itself as the project root. Relative data paths were then resolved against `$HOME` — e.g. `mloop info data.csv` run in a folder that actually contains `data.csv` failed with `File not found: C:\Users\<user>\data.csv` (the file the command was pointed at). `ProjectDiscovery` now excludes the runtime-cache root (the user-profile directory) from project detection — a directory impossible to use as a project anyway, since its `.mloop` would collide with the cache. The established "`.mloop` directory marks a project" contract is otherwise preserved. Found via mloop-agent dogfooding: the agent's `mloop_info data.csv` calls failed-and-retried because the agent's project sat under `$HOME`.

## [0.16.2] - 2026-06-24

### Fixed
- **Misleading "Data file not found" when a directory is passed to `predict`**: pointing image-classification `predict` at a folder of images reported `Data file not found` even though the directory exists, because the path was only checked with `File.Exists`. It now diagnoses the directory honestly — `Expected a CSV file but got a directory` — and, for image classification, explains that predict consumes a CSV with an `ImagePath` column while a labelled image directory belongs to `evaluate`. (Full directory-input prediction remains a separate, demand-driven feature.)
- **Image-classification quality gate skipped, promoting non-converged models (BUG-46)**: a non-converged image model (micro-accuracy below the 1/N random baseline) auto-promoted to production with only a warning, while tabular tasks were correctly blocked. Three compounding gaps: (1) `mloop init --task image-classification` wrote `metric: auto` (the init template's fallthrough), and `ShouldPromoteAsync` computed the threshold from the raw `auto`/alias string — which has no entry in the threshold table — so the gate branch was silently skipped; (2) the directory-based input schema never recorded the label's class count, so even a resolved metric had no 1/N floor; (3) degenerate-model detection keys off `f1_score`, which image classification doesn't emit (only `accuracy`/`micro_accuracy`/`log_loss`). Fixes: `init` now emits `micro_accuracy` for image classification; `ModelRegistry.ResolveCanonicalMetricKey(metric, task, keys)` falls back to the task's canonical metric when the project metric is `auto`, and the gate now derives both the threshold and the error-direction from the *resolved* canonical key (also closing a latent `auto`/alias inconsistency for tabular tasks); `BuildDirectoryInputSchema` populates the label's `UniqueValueCount` from the class-folder count via the new `ImageDirectoryLoader.CountClasses`. A binary image model at micro-accuracy 0.43 (< 0.5) is now blocked; a 6-class model at 0.90 (> 0.167) still promotes. Found via KAMP dogfooding (SEQ036 solder-joint, 48 images, non-converged).
- **Metric-alias mismatch broke auto-promotion and `compare` sorting (BUG-45)**: a non-default optimization metric alias — `--metric f1`, `r2`, `log-loss`, or multiclass `accuracy` — silently failed lookups against the stored metrics dictionary, which uses canonical keys (`f1_score`, `r_squared`, `log_loss`, `micro_accuracy`). Two symptoms shared one root cause:
  - **Auto-promotion**: `ModelRegistry.ShouldPromoteAsync` looked up `experiment.Metrics["f1"]`/`ContainsKey("f1")` and returned `false` at its first branch, so a quality-gate-passing model was left unpromoted with no error (only `mloop list` showing `Production: 0` revealed it).
  - **`mloop compare --sort f1`**: the alias wasn't in the metric-name set, so the sort and the "best experiment" recommendation silently fell back to the first metric (e.g. accuracy) instead of f1.

  A shared `ModelRegistry.ResolveMetricKey(metric, availableKeys)` normalizer now maps aliases to the stored key actually present (exact-match first, then `f1`→`f1_score`, `r2`→`r_squared`, `log-loss`→`log_loss`, `accuracy`→`micro_accuracy`/`macro_accuracy`), returning null when none match. `ShouldPromoteAsync` uses it for the quality gate and production comparison (falling back to promoting a quality-gated model when the key is absent on either side); `CompareCommand` uses it for `--sort`/config-metric resolution. Found via KAMP dogfooding (P051 rubber-molding defect, `--metric f1`).

## [0.16.1] - 2026-06-23

### Fixed
- **`evaluate` encoding gap (BUG-43)**: `mloop evaluate` read the test CSV as forced UTF-8 instead of running it through `EncodingDetector` like `train`/`predict`, so CP949/EUC-KR files (most KAMP data) with Korean column names were garbled — schema validation reported spurious "missing columns / not UTF-8", and `EvaluationEngine` could not find the label column even though training accepted the same file. Both `SchemaValidator` and `EvaluationEngine.LoadTestData` now delegate to `EncodingDetector.ConvertToUtf8WithBom`, matching the loaders.
- **`evaluate` feature-dimension mismatch (BUG-44)**: `EvaluationEngine.LoadTestData` reimplemented test-data preprocessing and diverged from `train`/`predict` — it did not strip unnamed/index columns or the DateTime/constant/sparse columns the trained schema marks as `Exclude`, so the rebuilt feature vector was wider than the model (e.g. `expected Vector<Single,114> got 123`) and `Transform` threw. It now mirrors prediction: `CsvDataLoader.RemoveIndexColumns` + the new shared `CsvDataLoader.RemoveExcludedColumns`.

### Changed
- **Inference preprocessing unified (root cause of BUG-43/44)**: `predict` and `evaluate` each hand-rolled their test-data preprocessing and diverged from each other and from `train`. The sequence is now a single shared `InferenceDataPreprocessor.Prepare` (encoding → flatten multiline → remove index → schema-driven exclude / data-dependent fallback), which `PredictionEngine` and `EvaluationEngine` both call. This collapses the divergence that produced BUG-43/44 and closes two gaps it had hidden: multiline-quoted-field flattening and constant-column removal were present in `train` but missing from both inference paths, and `predict`'s hand-rolled UTF-8 BOM check read CP949 as UTF-8 (the BUG-43 class survived in `predict` even after `evaluate` was fixed). `predict` now handles CP949 test files, multiline CSVs, and constant columns identically to `train`/`evaluate`.
- **`RemoveExcludedColumns` shared (DRY)**: the schema-driven excluded-column removal that `PredictionEngine` kept private is promoted to `CsvDataLoader.RemoveExcludedColumns(path, excludedNames)`, joining its sibling `Remove*` helpers; both the prediction and evaluation paths now call it.
- **Security: pinned transitive `System.Security.Cryptography.Xml`** to `10.0.9` via central-package transitive pinning, resolving the high-severity NU1903 advisories (GHSA-37gx-xxp4-5rgx, GHSA-w3x6-4m5h-cxqf) pulled in through `FilePrepper → EPPlus 8.5.0`.
- Doc-comment cref fix (`ObjectDetectionEvaluator`, CS1574) and de-flaked `PerformanceTests.ReproducibilityOverhead` (ratio-based threshold instead of an absolute ms-difference that was fragile under CPU contention).

## [0.16.0] - 2026-06-22

### Added
- **Object detection — `predict` (detections output)**: `mloop predict` on an object-detection model reads an image directory (COCO/YOLO, auto-detected) and emits per-image detections — class label, confidence score, and bounding box (`x0 y0 x1 y1`) — as JSON (`predictions/<model>-detections-<ts>.json`), with a per-class count and per-image preview. The new `ObjectDetectionPredictor` (Core) extracts detections from the scored data view, the structural twin of `ObjectDetectionEvaluator`.
- **Object detection — `evaluate` (mAP)**: `mloop evaluate` scores an object-detection model with mean Average Precision via ML.NET's built-in `EvaluateObjectDetection` (`ObjectDetectionEvaluator` → `map_50` / `map_50_95`), rather than a hand-rolled IoU/AP implementation. `EvaluationEngine` loads the image directory, transforms, and scores; `EvaluateCommand` accepts an image directory and skips CSV schema validation.
- **Image classification — `evaluate` over a directory**: `mloop evaluate` now evaluates an image-classification model over a labelled image directory (folder = label), reporting multiclass accuracy — previously evaluation only accepted a CSV. The directory-based evaluate path is generalized across image classification and object detection via `DataLoaderFactory`.

### Changed
- **`DataProviderBase` extraction (TD-05)**: the `SplitData` / `ValidateLabelColumn` / `GetSchema` / `GetRowCount` duplication across the five data loaders (`Csv`, `Image`, `Coco`, `Yolo`, `ObjectDetection`, ~220 lines) is unified into a `DataProviderBase` abstract class; each loader implements only `LoadData`. Directory-loader label matching is now `OrdinalIgnoreCase` (matching `CsvDataLoader`), and row counting falls back to a cursor when `GetRowCount` is null.
- Directory-dataset resolution for evaluate/predict is centralized in `DatasetDiscovery.FindDirectoryDataset` (object detection → `datasets/coco` → `datasets/yolo` → `datasets`; image classification → `datasets/images` → `datasets`), mirroring `train`'s convention.

## [0.15.0] - 2026-06-21

### Added
- **Image classification — end-to-end working**: `ImageDirectoryLoader` consumes a `datasets/images/<class>/` layout (folder name = label) with TensorFlow transfer learning (ResNet v2 50). `train` → `promote` → `predict` now complete on real data. `mloop init --task image-classification` scaffolds the directory layout. Verified e2e on a KAMP folder=label dataset (5/5 correct on the held-out test images).
- **Object detection — input path**: `CocoDataLoader` (COCO JSON, `bbox` → `x0 y0 x1 y1`) and `YoloDataLoader` (YOLO `images/` + `labels/*.txt`, normalized → absolute via image dimensions), dispatched by `ObjectDetectionDataLoader` with COCO/YOLO auto-detection. Wired through `RunObjectDetectionAsync` (LoadImages + MapValueToKey + bounding-box column) and the `train`/`init` CLI (`datasets/coco/`, `datasets/yolo/`). Evaluation honestly reports object-detection metrics as not-yet-supported (mAP pending) rather than mis-scoring.
- **On-demand DL runtime management**: `mloop runtime install/list/remove` for the TensorFlow (image) and libtorch (object detection / NLP) native runtimes, downloaded on demand and loaded transparently when a DL task runs.

### Fixed
- **DL runtime install/load chain (BUG-37/38/39)**: `mloop runtime install` crashed on every path — a thread-affine `Mutex` released across `await` threw `ApplicationException` (and masked the real error), the torch version coordinate 404'd / mismatched the TorchSharp ABI, and TorchSharp's native wrapper (`LibTorchSharp.dll`) was not on the search path. Replaced the mutex with a cross-process file lock, pinned libtorch to the binding's `2.2.1.1`, and exposed both the runtime cache and the app's `runtimes/<rid>/native` directory to the native loader.
- **Native DL runtime not loaded on inference paths (BUG-40)**: image/object-detection `predict`, `evaluate`, and `serve /predict` crashed with `DllNotFoundException` because the native runtime was registered only on the training path. ML.NET resolves the native library while deserializing the model, so a new shared `RuntimeManager.EnsureRuntimeForTask(taskType)` (no-op for tabular tasks) is now invoked before every model load — training, CLI predict, evaluate, serve, and model-cache preload.
- **Directory-schema label type (BUG-42)**: the image/OD schema stored its label `dataType` as the raw `"String"`, which fell through `PredictionEngine`'s type-override default and was read as `Single`, breaking the model's `MapValueToKey` at predict time. The directory schema now uses MLoop's canonical vocabulary (`Text` for the image path, `Categorical` for the label), and `PredictionEngine` tolerates `"String"` defensively.
- **Misleading prediction/evaluation diagnostics (BUG-31)**: "feature vector dimension mismatch" messages wrongly attributed the cause to text/value distribution drift. Saved transformers embed their fitted featurizers, so the real cause is a prediction-data column-structure/type mismatch; the four affected messages were corrected.

## [0.14.2] - 2026-06-19

### Fixed
- **IPreprocessingScript contract drift**: public preprocessing examples (`examples/preprocessing-scripts/01–07`), the tutorial/docs code blocks, and MLoop's own `PreprocessingScriptGenerator` referenced types that do not exist in the source — `PreprocessingResult`, `PreprocessingContext`, `context.GetTempPath()` — and the dead bare `MLoop.Extensibility` namespace. The authoritative contract is `Task<string> ExecuteAsync(PreprocessContext)` returning the produced CSV path. Root cause: none of these artifacts live in a csproj, so the build never compiled them and they drifted from the contract that `ScriptLoader` compiles standalone at runtime (no implicit/global usings). Generator-produced scripts therefore failed to compile/load, and copy-pasting the examples broke for consumers.
  - `PreprocessingScriptGenerator` now emits the real contract and accumulates `currentPath` so the returned path always exists (also fixes a latent missing-file return on the encoding-only path).
  - `PreprocessContext` gains null-safe `GetMetadata<T>`/`HasMetadata`, matching `HookContext`/`MetricContext`.
  - Examples 01–07 unified to the real contract; `04_data_cleaning` drops the external CsvHelper dependency in favor of the injected `ctx.Csv` (`ICsvHelper`).
- **`mloop preprocess --help`** now documents the two paths: the default path runs `IPreprocessingScript` files in `.mloop/scripts/preprocess/`, while `--incremental` runs the detector-based rule-discovery workflow.

### Added
- Regression guards that compile artifacts through the real `ScriptLoader`: generated scripts (`PreprocessingScriptGeneratorTests`) and the shipped example files (`ExampleScriptsCompilationTests`), making the examples first-class compile targets.

## [0.14.1] - 2026-06-19

### Fixed
- **ScriptLoader missing BCL references**: user scripts (detectors/hooks/metrics) referencing `Regex` failed to compile (`The type name 'Regex' could not be found ... add a reference`) because `System.Text.RegularExpressions` — a separate assembly, not `System.Private.CoreLib` — was absent from the compilation reference set. Added `System.Text.RegularExpressions` and `System.Text.Json` (with single-file-publish DLL fallbacks). Removed the dead `GetScriptOptions()` code path (the live path is `CompileAndCacheAsync`).
- **RuleApplier silent-success**: all 9 rule-application strategies were no-ops yet reported `Success=true`, so `mloop prep` emitted a "cleaned" dataset byte-for-byte identical to the input while claiming the rules were applied. Introduced `RuleApplicationStatus` (`Applied`/`NotImplemented`/`Failed`); unimplemented strategies now surface `NotImplemented` (`Success=false`) with a clear reason, and the incremental workflow warns when no rows were actually modified. (Built-in application strategies and any custom-apply extension remain backlog — see `claudedocs/plans/2026-06-19-rule-application-engine.md`.)

## [0.14.0] - 2026-06-19

### Added
- **Pluggable pattern detectors**: `IPatternDetector` implementations can now be contributed by consumer apps via `.mloop/scripts/detectors/*.cs` (auto-discovery), mirroring the existing hooks/metrics extension mechanism. Detectors run alongside the built-in 7 in the incremental preprocessing rule-discovery engine — no core modification required.
  - `ScriptDiscovery.DiscoverDetectorsAsync()` + `GetDetectorsDirectory()`; `InitializeDirectories()` now also creates `detectors/`.
  - `RuleDiscoveryEngine` accepts optional injected custom detectors (DI), combined with the built-ins (backward compatible).
  - `ScriptLoader` exposes MLoop.Core + Microsoft.Data.Analysis to compiled scripts so detectors can reference `DataFrameColumn`, `DetectedPattern`, `PatternType`.
  - `mloop preprocess --incremental` wires discovered detectors into the engine (zero-overhead when the directory is absent).

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
