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
    /// Optimization metric
    /// </summary>
    public string Metric { get; init; } = "accuracy";

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
    public Dictionary<string, string>? ColumnOverrides { get; init; }

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
    /// Columns featurization drops (DateTime / sparse / constant), decided once for the whole run by
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
}
