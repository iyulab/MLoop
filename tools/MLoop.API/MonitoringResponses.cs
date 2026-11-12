using MLoop.Ops.Services;

namespace MLoop.API;

// The bodies of the monitoring endpoints (MonitoringEndpoints), as the experiment and model ones are in ExperimentResponses.cs
// and ModelResponses.cs.

/// <summary><c>GET /logs</c>.</summary>
public sealed record LogListResponse(int Count, string? Filter, IReadOnlyList<LogEntryView> Logs);

/// <summary>One logged prediction.</summary>
public sealed record LogEntryView(
    string? Id,
    string ModelName,
    string ExperimentId,
    IDictionary<string, object> Input,
    object Output,
    double? Confidence,
    DateTimeOffset Timestamp);

/// <summary><c>GET /feedback</c>.</summary>
public sealed record FeedbackListResponse(int Count, string ModelName, IReadOnlyList<FeedbackEntryView> Feedback);

/// <summary>One recorded feedback.</summary>
public sealed record FeedbackEntryView(
    string PredictionId,
    string ModelName,
    object PredictedValue,
    object ActualValue,
    string? Source,
    double? Weight,
    DateTimeOffset Timestamp);

/// <summary><c>POST /feedback</c>.</summary>
public sealed record FeedbackRecordedResponse(string PredictionId, bool Recorded, DateTime Timestamp);

/// <summary><c>POST /samples</c>.</summary>
public sealed record SampleResponse(
    string ModelName,
    string Path,
    int SampledCount,
    int LabeledCount,
    int TotalAvailable,
    string Strategy,
    string? LabelColumn,
    string? WeightColumn);

/// <summary><c>GET /drift</c>.</summary>
public sealed record DriftResponse(
    string ModelName,
    string Status,
    string? ExperimentId,
    string? Message,
    DriftSignal? Drift,
    string? SliceBy,
    IReadOnlyList<SliceDriftView>? Slices);

/// <summary>The drift of the rows that share one value of the slicing column.</summary>
public sealed record SliceDriftView(string Value, DriftSignal Drift);

/// <summary><c>GET /trigger</c>.</summary>
public sealed record TriggerResponse(
    string ModelName,
    bool ShouldRetrain,
    string? RecommendedAction,
    DateTimeOffset EvaluatedAt,
    IReadOnlyList<TriggerConditionView> Conditions);

/// <summary>One condition of the retraining trigger and whether it holds.</summary>
public sealed record TriggerConditionView(
    string Type,
    string Name,
    double Threshold,
    double? CurrentValue,
    bool IsMet,
    string? Details,
    DriftSignal? Drift);
