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
    Func<MLContext, SearchPoint, TrainerColumns, SearchSettings, IEstimator<ITransformer>> Build)
{
    /// <summary>
    /// Settings tried first, before the tuner starts walking from the space's cheap default: the trainer's own
    /// defaults, which a library's authors chose to work on most tables. The search starts from the best of them
    /// and the cheap default (FLAML's <c>points_to_evaluate</c>). Names are those of <see cref="Space"/>'s dimensions.
    /// </summary>
    public IReadOnlyList<IReadOnlyDictionary<string, double>> StartingPoints { get; init; } = [];

    /// <summary>
    /// Whether the trainer fits a model for each class (one-versus-all). Its cost grows with the classes: measured, one fit
    /// of an eight-leaf forest took 0.5 s on two classes and 7.9 s on a hundred, whatever the rows (4,000 or 80,000 — the
    /// per-class setup dominates), where a maximum-entropy trainer on the same rows took 0.9 s.
    /// </summary>
    public bool FitsPerClass { get; init; }

    /// <summary>
    /// The relative cost the proposer starts from on a label of <paramref name="classes"/> classes: <see cref="InitialCost"/> (AutoML's
    /// table, measured on two classes) times the classes a per-class trainer fits a model for, in halves of that — a prior known
    /// before any trial, so the search stays a function of the seed alone.
    /// </summary>
    public double CostFor(int classes) => FitsPerClass ? InitialCost * Math.Max(1, classes / 2.0) : InitialCost;

    /// <summary>
    /// The point a trainer is built from when it will be fitted on <paramref name="rows"/> rows: its leaf count, where it has one,
    /// is at most the rows. A tree cannot have more leaves than rows, so the tree is the same whatever the leaf count above that —
    /// but the trainer sizes its tree storage by the setting, and a forest of 78 trees of 32,768 leaves took 1.4 GB to fit three
    /// thousand rows. The trial still records the point the tuner proposed; only what is built is bounded.
    /// </summary>
    public static SearchPoint FittedToRows(SearchPoint point, long rows)
    {
        ArgumentNullException.ThrowIfNull(point);
        const string leaves = "NumberOfLeaves";
        if (point.Space.Dimensions.All(d => d.Name != leaves))
            return point;
        var limit = Math.Max(rows, 4);   // the dimensions start at 4 leaves
        return point[leaves] > limit ? point.With(leaves, limit) : point;
    }

    /// <summary>
    /// The size of the model a point of a space builds, read from the point alone: trees × leaves for a tree
    /// ensemble, 1 for a linear model. What a larger model costs is time to fit, save, load and score, and room
    /// on disk — a forest of a thousand trees of a thousand leaves is a model.zip of hundreds of megabytes.
    /// </summary>
    public static double ModelSize(SearchPoint point)
    {
        ArgumentNullException.ThrowIfNull(point);
        double size = 1;
        foreach (var dimension in point.Space.Dimensions)
        {
            if (dimension.Name is "NumberOfTrees" or "NumberOfLeaves")
                size *= point.Int(dimension.Name);
        }
        return size;
    }
}
