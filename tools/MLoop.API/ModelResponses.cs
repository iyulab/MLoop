using MLoop.Ops.Interfaces;

namespace MLoop.API;

// The bodies of the model endpoints (ModelEndpoints), as the experiment ones are in ExperimentResponses.cs.

/// <summary><c>GET /info</c>: the model in production and what its metadata file records.</summary>
public sealed record ModelInfoResponse(
    string ModelName,
    string ExperimentId,
    DateTime PromotedAt,
    Dictionary<string, double>? Metrics,
    string? Task,
    string? BestTrainer,
    Dictionary<string, object>? Metadata);

/// <summary><c>GET /models</c>.</summary>
public sealed record ModelListResponse(int Count, string? Filter, IReadOnlyList<ModelListItem> Models);

/// <summary>A model in production.</summary>
public sealed record ModelListItem(
    string ModelName,
    string ExperimentId,
    DateTime PromotedAt,
    string? Task,
    string? BestTrainer,
    Dictionary<string, double>? Metrics);

/// <summary><c>POST /promote</c>.</summary>
public sealed record PromoteResponse(
    string ModelName,
    string ExperimentId,
    string? PreviousExperimentId,
    DateTime PromotedAt,
    string Task,
    Dictionary<string, double>? Metrics,
    string? BestTrainer,
    string? QualityWarning,
    string? NotBetterThanProduction);

/// <summary><c>POST /promote</c> of the experiment that is in production already: nothing moved.</summary>
public sealed record PromoteUnchangedResponse(
    string ModelName,
    string ExperimentId,
    string PreviousExperimentId,
    bool Unchanged,
    string Task,
    Dictionary<string, double>? Metrics,
    string? BestTrainer);

/// <summary><c>GET /promotions</c>.</summary>
public sealed record PromotionHistoryResponse(string ModelName, IReadOnlyList<PromotionRow> Promotions);

/// <summary><c>POST /rollback</c>.</summary>
public sealed record RollbackResponse(string ModelName, string ExperimentId, string? PreviousExperimentId, string? BackupPath);
