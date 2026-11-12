using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Evaluation;

namespace MLoop.API;

// The bodies of the experiment endpoints (ExperimentEndpoints). A handler returns TypedResults of these, so the answer's
// field names and kinds are in the types - the compiler refuses an answer that is not one of them, and the OpenAPI document
// describes each instead of `object`. Property order is the order the fields are written in. The server omits a null
// property (JsonIgnoreCondition.WhenWritingNull), as it did for the anonymous objects these replace.

/// <summary>The body of an error answer that has one message: <c>{ "error": "…" }</c>.</summary>
public sealed record ErrorResponse(string? Error);

/// <summary><c>GET /experiments</c>: the rows of <c>mloop list --json --all</c> and each model's production experiment.</summary>
public sealed record ExperimentListResponse(
    int Count,
    string? Filter,
    IReadOnlyList<ExperimentListRow> Experiments,
    IReadOnlyDictionary<string, string> Production);

/// <summary><c>GET /experiments/{id}</c>.</summary>
public sealed record ExperimentDetailResponse(
    string ModelName,
    string ExperimentId,
    DateTime Timestamp,
    string Status,
    string Task,
    ExperimentConfigView Config,
    ExperimentResultView? Result,
    Dictionary<string, double>? Metrics);

/// <summary>What an experiment was trained with.</summary>
public sealed record ExperimentConfigView(
    string DataFile,
    string? DataFileHash,
    string LabelColumn,
    int TimeLimitSeconds,
    bool? AutoTime,
    string Metric,
    double TestSplit,
    string? GroupColumn,
    bool HasSchema);

/// <summary>What a finished training run found.</summary>
public sealed record ExperimentResultView(string BestTrainer, double TrainingTimeSeconds);

/// <summary><c>GET /status</c>.</summary>
public sealed record StatusResponse(
    string Project,
    string ProjectRoot,
    StatusSummary Summary,
    IReadOnlyList<ModelStatus> Models,
    DataFilesStatus DataFiles,
    IReadOnlyList<ConfiguredModel>? Config);

/// <summary>The project's totals.</summary>
public sealed record StatusSummary(
    int TotalModels,
    int TotalExperiments,
    int CompletedExperiments,
    int FailedExperiments,
    int ProductionModels);

/// <summary>One model of the project: its experiments, its production experiment and its best one.</summary>
public sealed record ModelStatus(
    string ModelName,
    int TotalExperiments,
    int Completed,
    int Failed,
    bool HasProduction,
    string? ProductionExperimentId,
    double? BestMetric,
    string? BestMetricName,
    string? BestMetricWeightColumn,
    string? BestExperimentId,
    int ExperimentsNotCompared,
    string? LatestExperiment);

/// <summary>Which of the conventional data files the project has.</summary>
public sealed record DataFilesStatus(bool TrainCsv, bool TestCsv, bool PredictCsv);

/// <summary>A model as <c>mloop.yaml</c> declares it.</summary>
public sealed record ConfiguredModel(string Name, string Task, string Label, int? TimeLimitSeconds, string Metric);

/// <summary><c>GET /compare</c>.</summary>
public sealed record CompareResponse(
    string ModelName,
    string CandidateExpId,
    string BaselineExpId,
    bool CandidateIsBetter,
    string? Metric,
    double CandidateScore,
    double BaselineScore,
    double Improvement,
    string Recommendation,
    string? PromotionRefusal,
    Dictionary<string, MetricDetail> MetricDetails);

/// <summary>One metric of a comparison.</summary>
public sealed record MetricDetail(double CandidateValue, double BaselineValue, double Difference, bool IsBetter);

/// <summary><c>POST /evaluate</c> for a model with no production experiment and none named.</summary>
public sealed record EvaluateSkippedResponse(string Status, string Reason, string Model);

/// <summary><c>POST /evaluate</c>.</summary>
public sealed record EvaluateResponse(
    string ModelName,
    string ExperimentId,
    string Task,
    string TestDataPath,
    string? WeightColumn,
    Dictionary<string, double> Metrics,
    ConfusionCounts? Confusion,
    Comparison? Comparison,
    string? SliceBy,
    IReadOnlyList<EvaluationSlices.Slice>? Slices,
    IReadOnlyList<string>? Warnings);

/// <summary><c>POST /evaluate</c> for a file that does not match the schema the experiment was trained on.</summary>
public sealed record SchemaRefusalResponse(string Error, IReadOnlyList<string> MissingColumns, IReadOnlyList<string> Suggestions);
