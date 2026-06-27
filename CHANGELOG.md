# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.18.1] - 2026-06-28

### Changed
- **Centralized the task→primary-metric mapping into a single `TaskMetadata` source of truth (TD-06 / D4)**: the canonical metric each ML.NET task optimizes was duplicated across three drifting switch statements — `ModelRegistry.DefaultMetricForTask` (promotion gate), `InitCommand`'s yaml template (`metricExample`), and `TrainingEngine.GetPrimaryMetricValue` (probe-vs-main selection) — plus a hand-maintained allowlist in `ValidateCommand.ValidMetrics`. That duplication caused BUG-46 (init wrote `auto` for a task with a canonical metric, so the gate silently skipped) and F-17 (validate's allowlist fell out of sync with init). They now all read from one `TaskMetadata` table; adding a task in one place keeps every consumer consistent. Converging the table also resolved latent drift in the more-correct direction: `init` now writes `metric: micro_accuracy` for `text-classification` (was `auto` — the same BUG-46 shape), the promotion gate now resolves a canonical metric for `clustering`/`ranking`/`forecasting`/`recommendation`/`time-series-anomaly` so production comparison uses the right metric (blocking is unchanged — those are error/threshold-less metrics), and `validate` accepts every task's canonical metric by deriving the allowlist from `TaskMetadata.AllPrimaryMetrics` so it can never drift again. Pure refactoring (no public API change); 28 new tests pin the converged behavior. (Track 1 tech-debt; roadmap §4 D4.)
- **Upgraded DataLens 0.13.0 → 0.13.1 — restores predictive feature importance for numeric targets (F-06)**: DataLens 0.13.0's `FeatureAnalyzer` left the target column inside the permutation-importance feature matrix, so a model predicted the target from itself and zeroed every real feature's importance — forcing `mloop analyze importance` onto its honest structural fallback (`method=structural`) for numeric targets. The F-01 consumer-side guard kept the output non-misleading, but the true predictive signal was unavailable. DataLens 0.13.1 excludes the target from the matrix (fixed in-session per the local-submodule fast-dev-cycle), so numeric-target importance now reports `method=permutation`. Verified end-to-end on KAMP SEQ026 (양극산화, target `M_X`): `method` flips `structural`→`permutation`, `KWh` correctly ranks ≈0 (matching the C01 finding that dropping it left R² unchanged), 총도금시간/평균온도 rank top. The F-01 degenerate-fallback guard is retained as defense-in-depth. (sprint-35 C17.)
- **`mloop analyze <aspect>` data-file argument is now optional, defaulting to the project's configured train data (F-03/F-13)**: every analyze subcommand required an explicit `<data-file>` positional, while the rest of the CLI (`train`, `info`) defaults to the project's `data.train` (else `datasets/train.csv` discovery) when omitted. The inconsistency was agent-hostile: in a live mloop-agent FE-loop run (sprint-35 C09) the agent — instructed to always pass the project path — put the *project directory* in `dataFile` and burned five tool calls recovering before it found `datasets/train.csv`. `analyze` now reuses `TrainDataValidator.ResolveDataFileAsync` (the same resolver `train` uses), so `mloop analyze profile` with no path analyzes the project's train data; an explicit path still overrides. The MCP `mloop_analyze` bridge mirrors this — `dataFile` is no longer a required parameter (pass `projectPath` so the default resolves). Read-only contract unchanged. Found via KAMP MLOps dogfooding campaign (sprint-35 C09 agent observation).

### Fixed
- **`mloop predict` failed on ranking and recommendation models with "Could not find input column" (F-23)**: a ranking model's pipeline starts with `MapValueToKey("GroupId", group_column)` and a recommendation model maps the user/item columns to keys, so those columns must stay individually addressable at predict time. But `InferColumns` (used by the predict loader) merges adjacent numeric columns into one ranged Features vector, so the standalone `query_id` / `user_id` / `item_id` columns vanished and `model.Transform` threw — every ranking and recommendation model's train→predict slice was broken. The training loader already guards against this by splitting preserved columns (group/user/item, preFeaturizer columns) back out of the merged range; that logic is now a shared `CsvDataLoader.ApplyColumnPreservation` helper, and the predict path calls it too — `PredictCommand` reads the trained group/user/item columns from the experiment config and threads them through `PredictionEngine`. The user/item columns are now also persisted in `ExperimentConfig` (the group column already was). Classification/clustering predict are unaffected (they pass no preserve columns). Verified end-to-end: ranking predict now emits per-row scores and recommendation predict emits per (user, item) rating scores. Found via the ranking + recommendation full-slice dogfooding sweep (Cycle 54-55, synthetic data).
- **Clustering's optimal-K auto-search was silently inert — it always picked the largest K, and reported Davies-Bouldin Index 0 (F-22)**: the K-search trains KMeans for `k=2..maxK` and is meant to select the K with the lowest Davies-Bouldin Index (which, unlike average-distance, balances cluster separation and cohesion to find the *natural* K). But `Clustering.Evaluate` was called with only `scoreColumnName: "Score"` and no `featureColumnName` — and ML.NET only computes the Davies-Bouldin Index when the feature column is supplied. So DBI came back a constant `0`, the `useDbi = dbi > 0` branch was never taken, and the search fell back to average-distance, which decreases monotonically with K and therefore *always* selected the search ceiling (e.g. k=10 for 3 obvious clusters). Users silently got an over-segmented model and a meaningless `DAVIES BOULDIN INDEX 0.0000` in the metrics. Passing `featureColumnName: "Features"` to `Clustering.Evaluate` restores real DBI computation and thus the intended K-search. Verified on synthetic 3-cluster data: `num_clusters` now selects 3 (was 10) and DBI reports a real value (~0.26); a new integration test (`ClusteringKSearchRegressionTests`) pins both. Found via the clustering full-slice dogfooding sweep (Cycle 53, synthetic data).
- **`mloop predict` crashed on clustering models with "Metadata KeyValues does not exist" (F-21)**: the predict path restores original class names by applying `MapKeyToValue` to a key-typed `PredictedLabel` — correct for classification (whose key, produced by `MapValueToKey`, carries a `KeyValues` annotation mapping ids back to label strings). But clustering's KMeans also emits `PredictedLabel` as a key (the cluster id 1..k) with **no** `KeyValues` annotation, so `MapKeyToValue` threw and `mloop predict` failed outright for every clustering model — the train→promote→predict slice was broken end-to-end. The guard now applies `MapKeyToValue` only when the column actually has a `KeyValues` annotation (new shared `PredictionService.HasKeyValues`). The exact same restore logic was duplicated in both predict paths — `PredictionService` (Core) and the CLI's `PredictionEngine` (what `mloop predict` runs) — and both had the bug; both now call the one shared guard so they cannot drift. Verified end-to-end: clustering predict now emits cluster ids, multiclass still restores class names (`High`/`Low`), binary is unaffected (its `PredictedLabel` is a boolean, never a key). Found via the clustering full-slice dogfooding sweep (Cycle 52, synthetic data) — the first time `mloop predict` had been exercised on a clustering model.
- **`mloop validate` did not check task-specific required fields that `train` mandates; a parallel orphan validator that did was dead and drifted (F-20)**: the live `ValidateCommand` (what `mloop validate` runs) never validated that a `forecasting` model has a `horizon`, a `ranking` model has a `group_column`, or a `recommendation` model has `user_column`/`item_column` — all required at train time (`AutoMLRunner` throws without them), so a project missing them passed `validate` and then failed `train`. A *second* config validator, `ConfigValidator.Validate(MLoopConfig)` ("extracted from ValidateCommand for testability"), did contain these checks — but nothing in production ever called it; only its own tests did. Kept alive solely by tests, it had drifted from the live path: its task allowlist had lost the three NLP tasks (`sentence-similarity`/`ner`/`question-answering`), its unsupervised set had lost `time-series-anomaly`, and it still carried the F-19 label bug. The unique required-field checks were ported into the live `ValidateCommand` (with tests), and the orphan `Validate`/`ValidateModel`/`ValidateLabelInCsv`/`ValidTaskTypes` were removed — only the genuinely live `ConfigValidator.ValidatePrepSteps` (used by `PrepRunCommand`) remains. `init` writes a default for each field (`horizon: 10`, `group_column: query_id`, `user_column`/`item_column`), so a freshly-initialized project still validates clean; the new errors only fire when a field is removed. Found via the task-classification duplication audit (Cycle 51) that the F-19 fix motivated.
- **`mloop validate` errored "Label column is required" on valid unsupervised projects, and `mloop train` rejected label-less `time-series-anomaly` — a drifted task-set duplication (F-19)**: the unsupervised, label-optional task set (`anomaly-detection`, `clustering`, `time-series-anomaly` — `CsvDataLoader` loads a dummy label when none is given so every column becomes a feature) was hardcoded as three separate inline `HashSet`s in `ConfigMerger`, `InitCommand`, and `TrainCommand`, and the sets had **already drifted**: `TrainCommand`'s copy omitted `time-series-anomaly`, so a label-less ts-anomaly project that `init`/merge accept failed `train`'s required-field guard. `ValidateCommand` had no notion of the concept at all and unconditionally required a label, so `mloop validate` reported a hard error on a perfectly valid label-less clustering/anomaly project that `mloop train` runs fine — the same train↔validate contradiction shape as F-17. All four now read one source of truth, `AutoMLRunner.RequiresLabel(task)` (sibling to `SupportsPreFeaturizer`/`IsTimeSeriesTask`; unknown tasks conservatively require a label), removing the three inline sets and both drift bugs. Supervised tasks (regression/binary/multiclass/forecasting/ranking/recommendation) still require a label. Found via the unsupervised-task tool-surface dogfooding sweep (Cycle 50, synthetic data) — the same TD-06-shaped triplication that the `TaskMetadata` convergence fixed for task→metric.
- **`mloop validate` did not surface that `test_split` is silently ignored for time-series tasks (F-18)**: forecasting and `time-series-anomaly` feed the full series to training and hold out the last `horizon` rows internally (a random split would break temporal order — `AutoMLRunner` ignores `config.TestSplit` for these tasks), yet `mloop init` writes a uniform `test_split: 0.2` into every project's yaml and `validate` reported "Training settings ✓" without comment. A user configuring forecasting reasonably expects a 20% holdout but gets `horizon` rows (e.g. 10 of 365) — a silent no-op the project's "no silent failures" principle forbids. `validate` now emits an informational warning for time-series tasks that `test_split` has no effect (they use a temporal `horizon` holdout instead), while still erroring on an out-of-range value (a real typo) and leaving tabular-task behavior unchanged. The `forecasting | time-series-anomaly` classification is centralized as `AutoMLRunner.IsTimeSeriesTask` (sibling to `SupportsPreFeaturizer`) and consumed by both the runner's split branch and `validate`, so the two cannot drift. Found via the anomaly/forecasting EDA·FE·validate tool-surface dogfooding sweep (synthetic data) — §1 capability-matrix cells that had never been exercised end-to-end.
- **`mloop validate` flagged its own task-default metrics as "Unknown" (F-17)**: `ValidateCommand.ValidMetrics` listed only user-facing aliases (`accuracy`, `auc`, `f1`, `r2`, `rmse`, …) and omitted the canonical metrics that `mloop init` writes and that AutoML / the promotion gate actually use — so `mloop validate` warned `Unknown metric 'macro_accuracy'` on a freshly `init`-ed multiclass project (and likewise `micro_accuracy` for image/text, `r_squared` for regression). `ModelRegistry.DefaultMetricForTask` returns exactly these (`multiclass→macro_accuracy`, `image/text→micro_accuracy`, `regression→r_squared`), so validate contradicted init/train — a false warning that, in a live mloop-agent FE-loop run (sprint-35 C18/C19/C21), the agent repeatedly dismissed as "a pre-existing config problem." `ValidMetrics` now also includes the canonical forms (`macro_accuracy`, `micro_accuracy`, `r_squared`, `log_loss`, `f1_score`); a Theory test pins every task-default metric as accepted so the two lists cannot silently diverge again. (The deeper fix — centralizing task→metric knowledge into one `TaskMetadata` source so the allowlist *derives* from it — was since done as TD-06, see Changed above.) Found via KAMP MLOps dogfooding campaign (sprint-35 C22): another train/validate consistency gap, same shape as F-16.
- **`mloop analyze profile` did not surface ID/index columns that `train` already warns about (F-16)**: `train` detects strictly-increasing integer columns (`CsvDataLoader.DetectMonotonicColumns`) and warns they are likely ID/index features causing leakage/overfitting — but `analyze profile`, the agent's "eye," was silent: its envelope reported only `uniqueCount`, and `importance` ranks such a column by mutual-info (a *keep* signal), so an LLM FE-loop agent had no basis to apply its "drop ID/index" rule. In a live mloop-agent multiclass run (sprint-35 C18/C19, KAMP P068) the agent received a clean EDA yet judged "no ID/index columns," missing `ncWorkCount` (uniqueCount = row-count, values 1..N). `MapProfile` now accepts the monotonic-column list (reusing the exact detector `train` uses, so the two stay consistent) and emits a `likely-index: <col> (strictly increasing integers; exclude to avoid leakage)` flag plus a `likely-index` count in the summary; the console renders it via the shared `env.Flags` path. Additive — `monotonicColumns` defaults to none, so callers and the JSON shape are unchanged otherwise. Verified E2E on P068: `1 likely-index` / `likely-index: ncWorkCount`. Found via KAMP MLOps dogfooding campaign (sprint-35 C18 finding → C19 confirmation → C20 fix): "agent quality is capped by tool quality" — the eye must report what train sees.
- **`mloop features select --drop` silently recorded non-existent column names (F-14)**: `--drop` passed its column list straight to `FeatureSelector.Drop`, which added an `ignore` override for *any* name — including a column that isn't in the data. So a typo, or an LLM agent's hallucinated column name, produced a *valid-looking but wrong* policy: the phantom override is a no-op at train time while the column the user/agent actually meant stays a feature. (`--keep` was already safe — it computes the complement against the real header.) Observed in a live mloop-agent binary FE-loop run (sprint-35 C14): the agent dropped `검사항기`/`검사항목` when the real columns were `검사호기`/`검사모드`, and the tool accepted both. `features select --drop` now validates requested columns against the train-data header (new pure `FeatureSelector.PartitionByHeader`, case-sensitive to match train-time override-key lookup): unknown columns are skipped with a warning (surfaced in the console and in the `--json` envelope's `warnings`), and only real columns are recorded. Best-effort — when no train data is resolvable, the prior unconditional behaviour is kept. Found via KAMP MLOps dogfooding campaign (sprint-35 C14): a tooling guard that catches agent hallucination ("agent quality is capped by tool quality").
- **Experiment `metadata.json` recorded a hardcoded, stale MLoop version (F-11)**: `ExperimentStore.SaveAsync` wrote `versions.mLoop = "0.2.0"` for every experiment regardless of the actual tool version (currently 0.18.0), so the experiment record — meant to make a run reproducible — misreported the very tool that produced it. It now records the real version via `UpdateChecker.GetCurrentVersion()` (the assembly informational version with the `+commitHash` suffix stripped, the same source `mloop --version` uses). The `mlNet = "5.0.0"` entry is accurate and unchanged. Found via KAMP MLOps dogfooding campaign (sprint-35 C08, while tracing experiment metadata for the FE-policy reproducibility gap F-05).
- **`mloop analyze distribution` flagged low-cardinality/categorical columns as "highly-skewed" (F-02)**: skewness is mathematically defined for any numeric column, but for a binary/categorical column stored as numbers (e.g. a 1/2 rectifier id) it merely reflects the class balance — not an actionable distribution shape. The flag (`highly-skewed: 정류기 (skew=-1.14)`) misled a downstream agent into reasoning about a meaningless transform (observed in a live mloop-agent run: the agent copied the false skew note into its EDA summary). `MapDistribution` now takes per-column cardinality (from MLoop's own `ComputeColumnStats`, the same scan `profile` uses) and suppresses the highly-skewed flag for columns with fewer than 3 distinct values; the per-column `skewness` value is still reported, only the actionable *flag* is gated. Backward compatible when cardinality is unavailable (flags any |skew|>1). Found via KAMP MLOps dogfooding campaign (sprint-35 C04 agent observation → C05 fix).
- **`mloop analyze importance --label <col>` ranked the label against itself and surfaced target-agnostic structural variance instead of predictive importance (F-01)**: the command requires a label — implying a target-aware ranking — but mapped DataLens's target-agnostic `Features.Importance` (a variance/condition-number structural score computed over the *full* numeric matrix). Two consequences: (1) the label column appeared in its own ranking, usually at rank #1 (`top = M_X` where `M_X` is the target), and (2) features were ranked by structural variance, not predictive relevance — on KAMP SEQ026 (양극산화), `KWh` ranked #2 structurally yet contributed nothing to the model (dropping it left R² byte-identical at 0.9402), so an agent/user trusting the ranking would mis-prioritize feature engineering. `AnalyzeJson.RankImportance(FeatureReport, label)` now (a) **always excludes the label** from the ranking, (b) **prefers target-aware sources** — permutation (numeric target) → mutual-info → ANOVA-F (categorical target) — over structural, (c) falls back to structural with an honest `method` tag when the target-aware source is **degenerate** (all-zero among non-label features — DataLens currently leaves the target in the feature matrix, letting a model predict the target from itself and zeroing every real feature's permutation importance; tracked as a DataLens issue), and (d) emits a `method` field (`permutation`/`mutual-info`/`anova-f`/`structural`) so consumers know what the ranking means. Structural diagnostics (condition number, low-variance/high-correlation counts) are still surfaced. Both the `--json` envelope and the console table now share `RankImportance`, so they stay consistent (`InfoCommand`'s deep-analyze rendering is unchanged). Found via KAMP MLOps dogfooding campaign (sprint-35 C01).

## [0.18.0] - 2026-06-26

### Added
- **`mloop prep plan` / `mloop features select` — declarative policy commands (Phase 2)**: record feature-engineering decisions in `mloop.yaml` without touching data. `prep plan --set type[:method] [--columns c1,c2] [--remove type] [--list] [--name model]` upserts prep steps (keyed by type+columns, idempotent) and shows each step's leakage category (✓ fold-safe / ⚠ leak / ℹ csv-stage); `features select --drop <cols> | --keep <cols> | --reset [--name model]` edits column include/exclude via `ColumnOverride{Type:"ignore"}` (`--keep` computes the complement against the train-data header, label always kept). Both commands accept `--json` for a structured envelope (each prep step reports its leakage `category`/`foldSafe`; features reports the resulting `ignored` columns) so AI agents can read the result programmatically. Statistical fit still happens fold-internally at train time; these commands only edit the config. `validate` now surfaces informational model-class context for scaling steps on tasks where AutoML uses the preFeaturizer (informational, not a hard gate). `mloop.yaml` saving now omits null fields for clean round-trips.

## [0.17.0] - 2026-06-25

### Fixed
- **`mloop validate` under-reported prep leakage on tasks that ignore the preFeaturizer (T-B)**: `InspectPrepLeakage` only flagged the always-leaky transforms (`median` fill, `rolling`, `resample`) and was task-agnostic, so it stayed silent on `normalize`/`scale`/`fill-mean` for tasks whose AutoML pipeline ignores the `preFeaturizer` (clustering, anomaly, ranking, recommendation). At train time those steps are CSV-baked globally — leaking test statistics — and `PrepRouter` already warns; `validate` now mirrors that task-aware routing (`AutoMLRunner.SupportsPreFeaturizer`) and emits the same `UnsupportedTaskLeakageWarning`, so `validate` and train agree. PreFeaturizer-category steps on binary/multiclass/regression remain correctly silent (fitted fold-internally). `ValidateModelConfig` passes the model's task through `ValidatePrepSteps`.
- **`mloop analyze` crashed serializing non-finite statistics (e.g. `importance` on multicollinear data)**: `AnalyzeJson.Serialize` used System.Text.Json's default number handling, which throws on `NaN`/`±Infinity`. The `importance` aspect's `conditionNumber` is infinite precisely when features are collinear — the very condition the aspect exists to diagnose — so `mloop analyze importance` (and any aspect emitting a non-finite double) exited with `positive and negative infinity cannot be written as valid JSON` instead of producing output. A new `FiniteDoubleConverter` (registered on the analyze serializer) writes non-finite doubles as JSON `null`, applied centrally to every `double`/`double?` field. Chosen over `JsonNumberHandling.AllowNamedFloatingPointLiterals`, which would emit `Infinity`/`NaN` tokens that are invalid JSON and break strict parsers such as the MCP bridge's `JSON.parse`. The human-readable `flags` entry still reports the infinite condition number, so the multicollinearity signal is preserved while the numeric field stays valid JSON. Found via mloop-mcp `mloop_analyze` bridge dogfooding.

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
