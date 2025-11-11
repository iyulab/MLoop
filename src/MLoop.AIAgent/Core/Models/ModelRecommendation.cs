namespace MLoop.AIAgent.Core.Models;

/// <summary>
/// ML model configuration recommendation
/// </summary>
public class ModelRecommendation
{
    /// <summary>
    /// Recommended ML problem type
    /// </summary>
    public required MLProblemType ProblemType { get; init; }

    /// <summary>
    /// Target column name
    /// </summary>
    public required string TargetColumn { get; init; }

    /// <summary>
    /// Feature column names
    /// </summary>
    public required List<string> FeatureColumns { get; init; }

    /// <summary>
    /// Recommended training time in seconds
    /// </summary>
    public required int RecommendedTrainingTimeSeconds { get; init; }

    /// <summary>
    /// Primary evaluation metric
    /// </summary>
    public required string PrimaryMetric { get; init; }

    /// <summary>
    /// Additional evaluation metrics to track
    /// </summary>
    public List<string> AdditionalMetrics { get; init; } = new();

    /// <summary>
    /// Recommended optimization metric for AutoML
    /// </summary>
    public required string OptimizationMetric { get; init; }

    /// <summary>
    /// ML.NET trainer names to try
    /// </summary>
    public List<string> RecommendedTrainers { get; init; } = new();

    /// <summary>
    /// Confidence in this recommendation (0-1)
    /// </summary>
    public required double Confidence { get; init; }

    /// <summary>
    /// Reasoning for this recommendation
    /// </summary>
    public required string Reasoning { get; init; }

    /// <summary>
    /// Warnings about data or configuration
    /// </summary>
    public List<string> Warnings { get; init; } = new();

    /// <summary>
    /// Preprocessing recommendations
    /// </summary>
    public List<string> PreprocessingRecommendations { get; init; } = new();
}

/// <summary>
/// ML.NET trainer configuration recommendation
/// </summary>
public class TrainerRecommendation
{
    /// <summary>
    /// ML.NET trainer name (e.g., "FastTree", "LightGbm", "SdcaLogisticRegression")
    /// </summary>
    public required string TrainerName { get; init; }

    /// <summary>
    /// Expected relative performance (higher is better)
    /// </summary>
    public required double ExpectedPerformance { get; init; }

    /// <summary>
    /// Expected training speed (higher is faster)
    /// </summary>
    public required double TrainingSpeed { get; init; }

    /// <summary>
    /// Why this trainer is recommended
    /// </summary>
    public required string Rationale { get; init; }

    /// <summary>
    /// Hyperparameter suggestions
    /// </summary>
    public Dictionary<string, object> HyperparameterSuggestions { get; init; } = new();
}

/// <summary>
/// Dataset characteristics for model recommendation
/// </summary>
public class DatasetCharacteristics
{
    /// <summary>
    /// Number of training samples
    /// </summary>
    public required int SampleCount { get; init; }

    /// <summary>
    /// Number of feature columns
    /// </summary>
    public required int FeatureCount { get; init; }

    /// <summary>
    /// Number of unique target values
    /// </summary>
    public required int TargetCardinality { get; init; }

    /// <summary>
    /// Is dataset imbalanced?
    /// </summary>
    public required bool IsImbalanced { get; init; }

    /// <summary>
    /// Imbalance ratio if applicable (majority/minority)
    /// </summary>
    public double? ImbalanceRatio { get; init; }

    /// <summary>
    /// Percentage of missing values across dataset
    /// </summary>
    public required double MissingValuePercentage { get; init; }

    /// <summary>
    /// Number of categorical features
    /// </summary>
    public required int CategoricalFeatureCount { get; init; }

    /// <summary>
    /// Number of numeric features
    /// </summary>
    public required int NumericFeatureCount { get; init; }

    /// <summary>
    /// Has high cardinality categorical features?
    /// </summary>
    public required bool HasHighCardinalityFeatures { get; init; }
}
