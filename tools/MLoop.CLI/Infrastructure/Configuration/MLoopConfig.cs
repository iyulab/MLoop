using MLoop.Core.Preprocessing;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// MLoop project configuration supporting multiple models
/// </summary>
public class MLoopConfig
{
    /// <summary>
    /// Project name
    /// </summary>
    public string? Project { get; set; }

    /// <summary>
    /// Model definitions keyed by model name.
    /// "default" is used when --name is omitted from CLI commands.
    /// </summary>
    public Dictionary<string, ModelDefinition> Models { get; set; } = new();

    /// <summary>
    /// Shared data path settings
    /// </summary>
    public DataSettings? Data { get; set; }

    /// <summary>
    /// How long prediction logs and feedback are kept. Absent: kept until removed by hand.
    /// </summary>
    public RetentionSettings? Retention { get; set; }
}

/// <summary>
/// <c>retention:</c> in mloop.yaml — the days of prediction logs and feedback to keep. <c>mloop logs prune</c>
/// removes older day-files, and <c>mloop serve</c> does so once a day while it runs.
/// </summary>
public class RetentionSettings
{
    /// <summary>Days kept besides today (UTC); at least 1.</summary>
    public int? Days { get; set; }
}

/// <summary>
/// Definition of a single ML model within the project
/// </summary>
public class ModelDefinition
{
    /// <summary>
    /// ML task type: regression, binary-classification, multiclass-classification
    /// </summary>
    public required string Task { get; set; }

    /// <summary>
    /// Label column name in the dataset
    /// </summary>
    public required string Label { get; set; }

    /// <summary>
    /// Model-specific training settings (overrides project defaults)
    /// </summary>
    public TrainingSettings? Training { get; set; }

    /// <summary>
    /// Model description for documentation purposes
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Preprocessing pipeline steps executed before training via FilePrepper DataPipeline API.
    /// Steps are executed sequentially in order.
    /// </summary>
    public List<PrepStep>? Prep { get; set; }

    /// <summary>
    /// Column type overrides. Allows explicit specification of how columns should be treated.
    /// Overrides auto-detected column types during training and prediction.
    /// Key: column name, Value: column override settings
    /// </summary>
    public Dictionary<string, ColumnOverride>? Columns { get; set; }

    /// <summary>
    /// Number of clusters for clustering task.
    /// 0 or null = auto-select via silhouette search (k=2..10).
    /// </summary>
    public int? NumClusters { get; set; }

    /// <summary>
    /// Column whose values weight the rows in training: the relative frequency each row should have
    /// (for a row sampled with probability π, 1/π). Never used as a feature. Supported for
    /// binary-classification, multiclass-classification and regression.
    /// </summary>
    public string? Weight { get; set; }

    /// <summary>
    /// The image encoder an <c>image-anomaly</c> model embeds its images with (an installed encoder's id).
    /// Unset: the only image encoder installed.
    /// </summary>
    public string? Encoder { get; set; }

    /// <summary>
    /// The most patches an <c>image-anomaly</c> model's bank holds. Unset: every patch of the images.
    /// </summary>
    public int? BankSize { get; set; }

    /// <summary>
    /// Train only with steps an ONNX runtime computes exactly, so the model can be exported with
    /// <c>mloop export onnx</c>: text columns are featurized by character n-grams instead of
    /// ML.NET's text featurizer. Binary classification, multiclass classification and regression.
    /// </summary>
    public bool? Portable { get; set; }

    /// <summary>
    /// Group column name for ranking task.
    /// Groups rows into query contexts for learning-to-rank.
    /// </summary>
    public string? GroupColumn { get; set; }

    /// <summary>
    /// Forecast horizon — number of future time steps to predict.
    /// Required for forecasting task.
    /// </summary>
    public int? Horizon { get; set; }

    /// <summary>
    /// SSA window size (0 or null = auto: series_length / 4).
    /// </summary>
    public int? WindowSize { get; set; }

    /// <summary>
    /// Number of past data points for SSA model (0 or null = auto: total rows).
    /// </summary>
    public int? SeriesLength { get; set; }

    /// <summary>
    /// User column for recommendation task.
    /// </summary>
    public string? UserColumn { get; set; }

    /// <summary>
    /// Item column for recommendation task.
    /// </summary>
    public string? ItemColumn { get; set; }

    /// <summary>
    /// Rules a model must also pass to be promoted, judged on an evaluation set at the moment of
    /// promotion — on top of the fixed quality gate. <c>null</c> applies the fixed gate alone.
    /// </summary>
    public GatePolicy? Gate { get; set; }
}

/// <summary>
/// A project's own promotion rules (<c>gate:</c> under a model): judged on <see cref="Data"/> each time
/// the model would be promoted — by <c>promote</c>, by a training run, or through the REST API. A rule
/// the evaluation cannot judge refuses: a policy that silently passed when its data went missing would
/// be no policy.
/// </summary>
public class GatePolicy
{
    /// <summary>The labeled evaluation set the rules are judged on, relative to the project root.</summary>
    public string? Data { get; set; }

    /// <summary>Column of <see cref="Data"/> that weights its rows; defaults as <c>mloop evaluate --weight</c> does.</summary>
    public string? Weight { get; set; }

    /// <summary>
    /// Refuse a model that is worse than the one in production on <see cref="Data"/> — judged row by row
    /// (<c>mloop evaluate --against production</c>): refused when the paired interval says production is better.
    /// </summary>
    public bool NoWorseThanProduction { get; set; }

    /// <summary>Class-set limits a classifier must stay within, overall and in every slice.</summary>
    public List<GateLimit>? Limits { get; set; }

    /// <summary>Also apply <see cref="Limits"/> to each value of this column of <see cref="Data"/> (values with 30+ rows).</summary>
    public string? SliceBy { get; set; }

    /// <summary>Whether the policy has any rule to judge.</summary>
    [YamlDotNet.Serialization.YamlIgnore]
    [System.Text.Json.Serialization.JsonIgnore]
    public bool HasRules => NoWorseThanProduction || Limits is { Count: > 0 };
}

/// <summary>
/// Of the rows labeled with one of <see cref="Actual"/>, at most <see cref="Max"/> may be answered with
/// one of <see cref="Predicted"/> — a class that must be caught, let through as a harmless one.
/// </summary>
public class GateLimit
{
    /// <summary>The label classes the limit is about.</summary>
    public List<string> Actual { get; set; } = [];

    /// <summary>The answers that count against it.</summary>
    public List<string> Predicted { get; set; } = [];

    /// <summary>The largest share allowed, from 0 to 1. Required: a limit without it cannot be judged, so it refuses.</summary>
    public double? Max { get; set; }
}

/// <summary>
/// Column type override for a single column.
/// Allows users to explicitly specify column treatment when auto-detection is incorrect.
/// </summary>
public class ColumnOverride
{
    /// <summary>
    /// Column type: "text", "categorical", "numeric", or "ignore".
    /// - text: Treated as free-form text (TF-IDF featurization)
    /// - categorical: Treated as categorical (OneHotEncoding)
    /// - numeric: Treated as numeric feature
    /// - ignore: Excluded from features entirely
    /// </summary>
    public required string Type { get; set; }

    /// <summary>
    /// A text column's text encoder (<c>mloop encoder install multilingual-minilm</c>): the column is also read as its
    /// sentence vector, beside its word and character n-grams — for what the n-grams cannot carry, such as meaning across
    /// paraphrases. Needs <c>type: text</c>; binary, multiclass and regression only.
    /// </summary>
    public string? Encoder { get; set; }

    /// <summary>
    /// Optional description for documentation
    /// </summary>
    public string? Description { get; set; }
}

/// <summary>
/// Training configuration settings
/// </summary>
public class TrainingSettings
{
    /// <summary>
    /// Maximum training time in seconds (default: 300)
    /// </summary>
    public int? TimeLimitSeconds { get; set; }

    /// <summary>
    /// Optimization metric (e.g., accuracy, auc, r2, rmse)
    /// </summary>
    public string? Metric { get; set; }

    /// <summary>
    /// Fraction of data to use for testing (default: 0.2)
    /// </summary>
    public double? TestSplit { get; set; }
}

/// <summary>
/// Data path settings
/// </summary>
public class DataSettings
{
    /// <summary>
    /// Path to training data file
    /// </summary>
    public string? Train { get; set; }

    /// <summary>
    /// Path to test data file
    /// </summary>
    public string? Test { get; set; }

    /// <summary>
    /// Path to prediction input file
    /// </summary>
    public string? Predict { get; set; }
}

/// <summary>
/// Default values and constants for configuration
/// </summary>
public static class ConfigDefaults
{
    // The name itself lives with the model-name rule, which Ops and the API also read.
    public const string DefaultModelName = MLoop.Core.Storage.ModelName.Default;
    public const int DefaultTimeLimitSeconds = 300;
    public const double DefaultTestSplit = 0.2;
    public const string DefaultMetric = "auto";

    /// <summary>
    /// Creates default training settings
    /// </summary>
    public static TrainingSettings CreateDefaultTrainingSettings() => new()
    {
        // TimeLimitSeconds intentionally left null: "unspecified" means auto-time (estimated
        // from data size), not a fixed 300s. The fixed 300s is only the fallback applied
        // downstream (`?? DefaultTimeLimitSeconds`) when auto-time is explicitly disabled.
        TimeLimitSeconds = null,
        Metric = DefaultMetric,
        TestSplit = DefaultTestSplit
    };
}
