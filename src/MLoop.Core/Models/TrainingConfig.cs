using Microsoft.ML;

namespace MLoop.Core.Models;

/// <summary>
/// Configuration for model training
/// </summary>
/// <remarks>
/// A <c>record</c> so that pipeline stages which need to change one field (e.g. TrainingEngine
/// swapping <see cref="DataFile"/> for an encoding-converted copy) can use <c>with</c> instead of
/// re-listing every property in a hand-written constructor call. The hand-written form silently
/// dropped whatever it forgot — <see cref="TestDataFile"/> and the pre-featurizer fields were lost
/// on every encoding conversion — and each new property re-opened the same hole.
/// </remarks>
public record TrainingConfig
{
    /// <summary>
    /// Model name for multi-model support (defaults to "default")
    /// </summary>
    public required string ModelName { get; init; }

    /// <summary>
    /// Path to training data file
    /// </summary>
    public required string DataFile { get; init; }

    /// <summary>
    /// Label column name in the data
    /// </summary>
    public required string LabelColumn { get; init; }

    /// <summary>
    /// ML task type (binary-classification, multiclass-classification, regression)
    /// </summary>
    public required string Task { get; init; }

    /// <summary>
    /// Training time limit in seconds
    /// </summary>
    public int TimeLimitSeconds { get; init; } = 300;

    /// <summary>
    /// The optimization metric as it was given — the user's own spelling, kept so a message can
    /// quote it back. What the optimizer and the records use is <see cref="CanonicalMetric"/>.
    /// Defaults to <c>auto</c> — the task's primary metric — because a config does not know its task
    /// when this initializer runs, and the old default (<c>accuracy</c>) was a binary-classification
    /// name that every regression config built without a metric silently carried.
    /// </summary>
    public string Metric { get; init; } = Evaluation.MetricNames.Auto;

    /// <summary>
    /// <see cref="Metric"/> in MLoop's canonical vocabulary for this <see cref="Task"/>
    /// (<c>F1Score</c> → <c>f1_score</c>, <c>auto</c> → the task's primary metric), or the given
    /// spelling unchanged when <see cref="Evaluation.MetricNames"/> does not know it. Such a name
    /// never reaches training: <see cref="Evaluation.MetricNames.Rejection"/> refuses it at every
    /// producer, and the optimizer refuses it again rather than substituting the task default.
    /// </summary>
    /// <remarks>
    /// Derived here, at the one gate every producer of a config passes through (both CLI paths and
    /// the API's job runner), so the experiment config, the index and the report all record the
    /// name of the metric that was actually optimized instead of whatever was typed.
    /// </remarks>
    public string CanonicalMetric => Evaluation.MetricNames.Canonical(Task, Metric) ?? Metric;

    /// <summary>
    /// Test data split ratio
    /// </summary>
    public double TestSplit { get; init; } = 0.2;

    /// <summary>
    /// Path to separate test data file (when pre-split, e.g. for balanced training)
    /// </summary>
    public string? TestDataFile { get; init; }

    /// <summary>
    /// Whether to use automatic time estimation (true when --time not specified and YAML has no time_limit_seconds)
    /// </summary>
    public bool UseAutoTime { get; init; } = false;

    /// <summary>
    /// Column type overrides from mloop.yaml.
    /// Key: column name, Value: type string (text, categorical, numeric, ignore)
    /// </summary>
    /// <remarks>
    /// Includes <see cref="WeightColumn"/> as <c>ignore</c>: a weight is not a feature, and every path
    /// that keeps a column out of the features (the saved schema, AutoML's column information) reads
    /// this one map. It is also in <see cref="ClaimedColumns"/>, so no exclusion heuristic removes it from the data. A weight that leaked into the features would let the model
    /// learn the sampling — a 1/π weight correlates with the label wherever the sampling did.
    /// </remarks>
    public Dictionary<string, string>? ColumnOverrides
    {
        get
        {
            if (string.IsNullOrEmpty(WeightColumn))
                return _columnOverrides;
            var merged = _columnOverrides is null
                ? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                : new Dictionary<string, string>(_columnOverrides, StringComparer.OrdinalIgnoreCase);
            merged[WeightColumn] = "ignore";
            return merged;
        }
        init => _columnOverrides = value;
    }

    private readonly Dictionary<string, string>? _columnOverrides;

    /// <summary>
    /// The column whose values weight the rows — the relative frequency each row should have (for a
    /// row sampled with probability π, 1/π). Null for unweighted training. Never a feature.
    /// </summary>
    public string? WeightColumn { get; init; }

    /// <summary>
    /// prep 통계 변환을 ML.NET AutoML에 fold-내 fit으로 위임하기 위한 pre-featurizer.
    /// null이면 미적용. PrepFeaturizerBuilder가 생성한다.
    /// </summary>
    public IEstimator<ITransformer>? PreFeaturizer { get; init; }

    /// <summary>
    /// preFeaturizer가 참조하는 개별 컬럼 이름들. 이 컬럼들은 CsvDataLoader의
    /// InferColumns 병합(Features 벡터로 합쳐짐)에서 제외되어 개별 컬럼으로 유지되어야
    /// preFeaturizer(NormalizeMinMax("age","age") 등)가 입력 컬럼을 찾을 수 있다.
    /// null/빈 목록이면 보존 동작 없음(prep/통계 변환이 없는 경우).
    /// </summary>
    public List<string>? PreFeaturizerColumns { get; init; }

    /// <summary>
    /// Columns featurization drops (DateTime / sparse / constant / identifier), decided once for the whole run by
    /// <see cref="Data.CsvDataLoader.DetermineExcludedColumns"/> and applied to every slice — the
    /// train partition, the test partition, and the saved input schema that predict and evaluate
    /// replay. Null means the decision has not been made upstream, in which case the loader falls
    /// back to deciding from whatever file it is handed.
    /// </summary>
    /// <remarks>
    /// Deciding per slice is the drift this field exists to prevent: a column that is constant only
    /// inside one partition changes that partition's feature width, and the pipeline fitted on the
    /// other width then fails with a "Schema mismatch for feature column 'Features'" error.
    /// </remarks>
    public IReadOnlyCollection<string>? FeatureExclusions { get; init; }

    /// <summary>
    /// String feature columns that hold categories and are one-hot encoded rather than featurized as
    /// text — decided once for the whole run by <see cref="Data.CategoryColumns.Decide"/>, like
    /// <see cref="FeatureExclusions"/>, and recorded as <c>Categorical</c> in the saved schema. A
    /// column the user typed keeps its declared type. Null means no decision was made upstream (an
    /// in-process caller), and every untyped string column is featurized as text, as before.
    /// </summary>
    public IReadOnlyCollection<string>? CategoryColumns { get; init; }

    /// <summary>
    /// The type every load of this run reads the label as — decided once from the saved schema
    /// (<see cref="LabelKinds"/>), like <see cref="FeatureExclusions"/>. Null leaves each load to infer
    /// it, which two partitions of a file sorted by its label do differently.
    /// </summary>
    public Microsoft.ML.Data.DataKind? LabelKind { get; init; }

    /// <summary>
    /// Train only with steps an ONNX runtime computes exactly, so the model can be exported
    /// (<c>mloop export onnx</c>): text columns are featurized by
    /// <see cref="AutoML.PortableTextFeaturizer"/> instead of <c>FeaturizeText</c>.
    /// Binary classification, multiclass classification and regression.
    /// </summary>
    public bool Portable { get; init; }

    /// <summary>
    /// Number of clusters for clustering task (0 = auto-select via silhouette search)
    /// </summary>
    public int NumClusters { get; init; } = 0;

    /// <summary>
    /// Group column name for ranking task (groups rows into query contexts)
    /// </summary>
    public string? GroupColumn { get; init; }

    /// <summary>
    /// Forecast horizon — number of future time steps to predict (forecasting task)
    /// </summary>
    public int Horizon { get; init; } = 0;

    /// <summary>
    /// SSA window size for time series decomposition (0 = auto: series_length / 4)
    /// </summary>
    public int WindowSize { get; init; } = 0;

    /// <summary>
    /// Number of past data points to consider in SSA model (0 = auto: total rows)
    /// </summary>
    public int SeriesLength { get; init; } = 0;

    /// <summary>
    /// User column name for recommendation task
    /// </summary>
    public string? UserColumn { get; init; }

    /// <summary>
    /// Item column name for recommendation task
    /// </summary>
    public string? ItemColumn { get; init; }

    /// <summary>
    /// The seed and threads MLoop's search runs with; <see langword="null"/> for its defaults. Internal:
    /// the search's measurements vary the seed.
    /// </summary>
    internal MLoop.Search.Candidates.SearchSettings? SearchSettings { get; init; }

    /// <summary>
    /// At most this many trials; <see langword="null"/> for the time budget alone. Internal: the search's
    /// measurements hold the trial count fixed.
    /// </summary>
    internal int? SearchMaxTrials { get; init; }

    /// <summary>
    /// Columns this run has claimed for a purpose of its own — the ranking group, the recommendation
    /// user and item, the row weight, pre-featurizer inputs, and every column the user gave an
    /// explicit type override other than <c>ignore</c>. No feature-exclusion heuristic drops one of
    /// these: a stated intent outranks a guess about the data. Handed to
    /// <see cref="Data.CsvDataLoader.DetermineExcludedColumns"/> as its protected set, from one
    /// place so the runner and the CLI cannot derive two different sets.
    /// </summary>
    public IReadOnlyList<string> ClaimedColumns
    {
        get
        {
            var claimed = new List<string>();
            if (!string.IsNullOrEmpty(GroupColumn)) claimed.Add(GroupColumn);
            if (!string.IsNullOrEmpty(UserColumn)) claimed.Add(UserColumn);
            if (!string.IsNullOrEmpty(ItemColumn)) claimed.Add(ItemColumn);
            if (!string.IsNullOrEmpty(WeightColumn)) claimed.Add(WeightColumn);
            if (PreFeaturizerColumns is { Count: > 0 }) claimed.AddRange(PreFeaturizerColumns);
            if (ColumnOverrides is not null)
            {
                claimed.AddRange(ColumnOverrides
                    .Where(kv => !string.Equals(kv.Value?.Trim(), "ignore", StringComparison.OrdinalIgnoreCase))
                    .Select(kv => kv.Key));
            }
            return claimed;
        }
    }
}
