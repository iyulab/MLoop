using Microsoft.ML;
using MLoop.Search.Tuning;

namespace MLoop.Search.Candidates;

/// <summary>The columns a trainer reads: the label, the feature vector and, when rows are weighted, the weight.</summary>
public sealed record TrainerColumns(string Label, string Features, string? Weight = null);

/// <summary>How a search runs its trainers: the seed they draw from and how many threads they may use.</summary>
/// <param name="Seed">The seed every seeded trainer and the tuner draw from.</param>
/// <param name="Threads">
/// Threads per trainer. One makes a trainer's result depend on the seed alone — SDCA and LightGBM vary
/// between runs on several threads — at the cost of time on large data.
/// </param>
public sealed record SearchSettings(int Seed = 1, int Threads = 1);

/// <summary>
/// One trainer a search may try: its name, the hyperparameters it searches, the cost estimate the
/// schema proposer starts from, and how to build it from a point of its space.
/// </summary>
/// <param name="Name">The trainer's name — what a trial reports (<c>FastTree</c>, <c>LightGbm</c>).</param>
/// <param name="Space">Its hyperparameters.</param>
/// <param name="InitialCost">The relative cost the ECI proposer starts from (AutoML's table).</param>
/// <param name="Build">The estimator for a point of <see cref="Space"/>.</param>
public sealed record TrainerCandidate(
    string Name,
    SearchSpace Space,
    double InitialCost,
    Func<MLContext, SearchPoint, TrainerColumns, SearchSettings, IEstimator<ITransformer>> Build);
