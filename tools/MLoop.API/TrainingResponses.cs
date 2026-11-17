namespace MLoop.API;

// The bodies of the training endpoints (TrainingEndpoints), as the experiment ones are in ExperimentResponses.cs. A handler
// returns TypedResults of these, so the answer's field names and kinds are in the types and the OpenAPI document describes each
// instead of `object`. Property order is the order the fields are written in; a null property is omitted by the server.

/// <summary><c>POST /train</c>: the job was queued (202).</summary>
public sealed record TrainStartResponse(string JobId, string ModelName, string Status, string Message);

/// <summary><c>GET /jobs</c>.</summary>
public sealed record JobListResponse(int Count, IReadOnlyList<JobSummary> Jobs);

/// <summary>A job in the list: what it trains and where it stands.</summary>
public sealed record JobSummary(
    string JobId,
    string ModelName,
    string Task,
    string Status,
    string? StatusMessage,
    DateTime CreatedAt,
    DateTime? CompletedAt,
    string? ExperimentId,
    string? BestTrainer,
    bool? Promoted);

/// <summary><c>GET /jobs/{id}</c>: what the job trains with (the project's paths, as the request wrote them) and what it found.</summary>
public sealed record JobDetailResponse(
    string JobId,
    string ModelName,
    string? DataFile,
    IReadOnlyList<string>? AdditionalDataFiles,
    string LabelColumn,
    string Task,
    int? TimeLimitSeconds,
    int? MaxTrials,
    int? Epochs,
    bool Ensemble,
    bool Portable,
    string Metric,
    double? TestSplit,
    int? MaxRows,
    string? Weight,
    string? Balance,
    bool Promote,
    string Status,
    string? StatusMessage,
    DateTime CreatedAt,
    DateTime? CompletedAt,
    string? ExperimentId,
    Dictionary<string, double>? Metrics,
    MLoop.Core.Models.ValidatedScore? Validated,
    string? BestTrainer,
    bool? Promoted,
    string? KeptInStagingReason,
    IReadOnlyList<string>? Warnings);
